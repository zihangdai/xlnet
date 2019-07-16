from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import numpy as np
import six
from os.path import join
from six.moves import zip

from absl import flags

import tensorflow as tf


def configure_tpu(FLAGS):
  if FLAGS.use_tpu:
    tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    master = tpu_cluster.get_master()
  else:
    tpu_cluster = None
    master = FLAGS.master

  session_config = tf.ConfigProto(allow_soft_placement=True)
  # Uncomment the following line if you hope to monitor GPU RAM growth
  # session_config.gpu_options.allow_growth = True

  if FLAGS.use_tpu:
    strategy = None
    tf.logging.info('Use TPU without distribute strategy.')
  elif FLAGS.num_core_per_host == 1:
    strategy = None
    tf.logging.info('Single device mode.')
  else:
    strategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus=FLAGS.num_core_per_host)
    tf.logging.info('Use MirroredStrategy with %d devices.',
                    strategy.num_replicas_in_sync)

  per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=master,
      model_dir=FLAGS.model_dir,
      session_config=session_config,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations,
          num_shards=FLAGS.num_hosts * FLAGS.num_core_per_host,
          per_host_input_for_training=per_host_input),
      keep_checkpoint_max=FLAGS.max_save,
      save_checkpoints_secs=None,
      save_checkpoints_steps=FLAGS.save_steps,
      train_distribute=strategy
  )
  return run_config


def init_from_checkpoint(FLAGS, global_vars=False):
  tvars = tf.global_variables() if global_vars else tf.trainable_variables()
  initialized_variable_names = {}
  scaffold_fn = None
  if FLAGS.init_checkpoint is not None:
    if FLAGS.init_checkpoint.endswith("latest"):
      ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
      init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    else:
      init_checkpoint = FLAGS.init_checkpoint

    tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

    (assignment_map, initialized_variable_names
    ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    if FLAGS.use_tpu:
      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Log customized initialization
    tf.logging.info("**** Global Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
  return scaffold_fn


def get_train_op(FLAGS, total_loss, grads_and_vars=None):
  global_step = tf.train.get_or_create_global_step()

  # increase the learning rate linearly
  if FLAGS.warmup_steps > 0:
    warmup_lr = (tf.cast(global_step, tf.float32)
                 / tf.cast(FLAGS.warmup_steps, tf.float32)
                 * FLAGS.learning_rate)
  else:
    warmup_lr = 0.0

  # decay the learning rate
  if FLAGS.decay_method == "poly":
    decay_lr = tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        end_learning_rate=FLAGS.learning_rate * FLAGS.min_lr_ratio)
  elif FLAGS.decay_method == "cos":
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)
  else:
    raise ValueError(FLAGS.decay_method)

  learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                           warmup_lr, decay_lr)

  if (FLAGS.weight_decay > 0 and not FLAGS.use_tpu and
      FLAGS.num_core_per_host > 1):
    raise ValueError("Do not support `weight_decay > 0` with multi-gpu "
                     "training so far.")

  if FLAGS.weight_decay == 0:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        epsilon=FLAGS.adam_epsilon)
  else:
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        epsilon=FLAGS.adam_epsilon,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        weight_decay_rate=FLAGS.weight_decay)

  if FLAGS.use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  if grads_and_vars is None:
    grads_and_vars = optimizer.compute_gradients(total_loss)
  gradients, variables = zip(*grads_and_vars)
  clipped, gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)

  if getattr(FLAGS, "lr_layer_decay_rate", 1.0) != 1.0:
    n_layer = 0
    for i in range(len(clipped)):
      m = re.search(r"model/transformer/layer_(\d+?)/", variables[i].name)
      if not m: continue
      n_layer = max(n_layer, int(m.group(1)) + 1)

    for i in range(len(clipped)):
      for l in range(n_layer):
        if "model/transformer/layer_{}/".format(l) in variables[i].name:
          abs_rate = FLAGS.lr_layer_decay_rate ** (n_layer - 1 - l)
          clipped[i] *= abs_rate
          tf.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(
              abs_rate, l, variables[i].name))
          break

  train_op = optimizer.apply_gradients(
      zip(clipped, variables), global_step=global_step)

  # Manually increment `global_step` for AdamWeightDecayOptimizer
  if FLAGS.weight_decay > 0:
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  return train_op, learning_rate, gnorm


def clean_ckpt(_):
  input_ckpt = FLAGS.clean_input_ckpt
  output_model_dir = FLAGS.clean_output_model_dir

  tf.reset_default_graph()

  var_list = tf.contrib.framework.list_variables(input_ckpt)
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if not name.startswith("global_step") and "adam" not in name.lower():
      var_values[name] = None
      tf.logging.info("Include {}".format(name))
    else:
      tf.logging.info("Exclude {}".format(name))

  tf.logging.info("Loading from {}".format(input_ckpt))
  reader = tf.contrib.framework.load_checkpoint(input_ckpt)
  for name in var_values:
    tensor = reader.get_tensor(name)
    var_dtypes[name] = tensor.dtype
    var_values[name] = tensor

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(
      0, name="global_step", trainable=False, dtype=tf.int64)
  saver = tf.train.Saver(tf.all_variables())

  if not tf.gfile.Exists(output_model_dir):
    tf.gfile.MakeDirs(output_model_dir)

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(var_values)):
      sess.run(assign_op, {p: value})

    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, join(output_model_dir, "model.ckpt"),
               global_step=global_step)


def avg_checkpoints(model_dir, output_model_dir, last_k):
  tf.reset_default_graph()

  checkpoint_state = tf.train.get_checkpoint_state(model_dir)
  checkpoints = checkpoint_state.all_model_checkpoint_paths[- last_k:]
  var_list = tf.contrib.framework.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if not name.startswith("global_step"):
      var_values[name] = np.zeros(shape)
  for checkpoint in checkpoints:
    reader = tf.contrib.framework.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor
    tf.logging.info("Read from checkpoint %s", checkpoint)
  for name in var_values:  # Average.
    var_values[name] /= len(checkpoints)

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(
      0, name="global_step", trainable=False, dtype=tf.int64)
  saver = tf.train.Saver(tf.all_variables())

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(var_values)):
      sess.run(assign_op, {p: value})
    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, join(output_model_dir, "model.ckpt"),
        global_step=global_step)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    # tf.logging.info('original name: %s', name)
    if name not in name_to_variable:
      continue
    # assignment_map[name] = name
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"],
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.include_in_weight_decay = include_in_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])

    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    for r in self.include_in_weight_decay:
      if re.search(r, param_name) is not None:
        return True

    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          tf.logging.info('Adam WD excludes {}'.format(param_name))
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


if __name__ == "__main__":
  flags.DEFINE_string("clean_input_ckpt", "", "input ckpt for cleaning")
  flags.DEFINE_string("clean_output_model_dir", "", "output dir for cleaned ckpt")

  FLAGS = flags.FLAGS

  tf.app.run(clean_ckpt)
