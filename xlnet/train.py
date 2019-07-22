"""Pretraining on TPUs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

import tensorflow as tf
import model_utils
import tpu_estimator
import function_builder
import data_utils

# TPU parameters
flags.DEFINE_string("master", default=None,
      help="master")
flags.DEFINE_string("tpu", default=None,
      help="The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("gcp_project", default=None,
      help="Project name for the Cloud TPU-enabled project. If not specified, "
      "we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("tpu_zone",default=None,
      help="GCE zone where the Cloud TPU is located in. If not specified, we "
      "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_bool("use_tpu", default=True,
      help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1,
      help="number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="number of cores per host")
flags.DEFINE_bool("track_mean", default=False,
      help="Whether to track mean loss.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer("num_passes", default=1,
      help="Number of passed used for training.")
flags.DEFINE_string("record_info_dir", default=None,
      help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="Checkpoint path for initializing the model.")

# Optimization config
flags.DEFINE_float("learning_rate", default=1e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=1.0,
      help="Gradient clipping value.")
# lr decay
flags.DEFINE_float("min_lr_ratio", default=0.001,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")
flags.DEFINE_float("adam_epsilon", default=1e-8,
      help="Adam epsilon.")
flags.DEFINE_string("decay_method", default="poly",
      help="Poly or cos.")
flags.DEFINE_float("weight_decay", default=0.0,
      help="Weight decay rate.")

# Training config
flags.DEFINE_integer("train_batch_size", default=16,
      help="Size of the train batch across all hosts.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=1000,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=None,
      help="Number of steps for model checkpointing. "
      "None for not saving checkpoints")
flags.DEFINE_integer("max_save", default=100000,
      help="Maximum number of checkpoints to save.")

# Data config
flags.DEFINE_integer("seq_len", default=0,
      help="Sequence length for pretraining.")
flags.DEFINE_integer("reuse_len", default=0,
      help="How many tokens to be reused in the next batch. "
      "Could be half of `seq_len`.")
flags.DEFINE_bool("uncased", False,
      help="Use uncased inputs or not.")
flags.DEFINE_integer("perm_size", 0,
      help="Window size of permutation.")
flags.DEFINE_bool("bi_data", default=True,
      help="Use bidirectional data streams, i.e., forward & backward.")
flags.DEFINE_integer("mask_alpha", default=6,
      help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1,
      help="How many tokens to mask within each group.")
flags.DEFINE_integer("num_predict", default=None,
      help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer("n_token", 32000, help="Vocab size")

# Model config
flags.DEFINE_integer("mem_len", default=0,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_string("ff_activation", default="relu",
      help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


def get_model_fn():
  """doc."""
  def model_fn(features, labels, mode, params):
    """doc."""
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    assert is_training

    #### Retrieve `mems` from `params["cache"]`
    mems = {}
    idx = 0
    if FLAGS.mem_len > 0:
      mems["mems"] = params["cache"]

    #### Get loss from inputs
    total_loss, new_mems, monitor_dict = function_builder.get_loss(
        FLAGS, features, labels, mems, is_training)

    #### Turn `new_mems` into `new_cache`
    new_cache = []
    if FLAGS.mem_len > 0:
      new_cache += new_mems["mems"]

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: {}".format(num_params))

    #### Configuring the optimizer
    train_op, learning_rate, gnorm = model_utils.get_train_op(
        FLAGS, total_loss)
    monitor_dict["lr"] = learning_rate
    monitor_dict["gnorm"] = gnorm

    #### Customized initial checkpoint
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS, global_vars=True)

    #### Creating host calls
    host_call = function_builder.construct_scalar_host_call(
        monitor_dict=monitor_dict,
        model_dir=FLAGS.model_dir,
        prefix="train/",
        reduce_fn=tf.reduce_mean)

    #### Constucting training TPUEstimatorSpec with new cache.
    train_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
        scaffold_fn=scaffold_fn)

    train_spec.cache = new_cache

    return train_spec

  return model_fn


def get_cache_fn(mem_len):
  """doc."""
  tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
  def cache_fn(batch_size):
    mems = []
    if FLAGS.mem_len > 0:
      for _ in range(FLAGS.n_layer):
        zeros = tf.zeros(
            [mem_len, batch_size, FLAGS.d_model],
            dtype=tf_float)
        mems.append(zeros)

    return mems

  if mem_len > 0:
    return cache_fn
  else:
    return None


def get_input_fn(split):
  """doc."""
  assert split == "train"
  batch_size = FLAGS.train_batch_size

  input_fn, record_info_dict = data_utils.get_input_fn(
      tfrecord_dir=FLAGS.record_info_dir,
      split=split,
      bsz_per_host=batch_size // FLAGS.num_hosts,
      seq_len=FLAGS.seq_len,
      reuse_len=FLAGS.reuse_len,
      bi_data=FLAGS.bi_data,
      num_hosts=FLAGS.num_hosts,
      num_core_per_host=FLAGS.num_core_per_host,
      perm_size=FLAGS.perm_size,
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      uncased=FLAGS.uncased,
      num_passes=FLAGS.num_passes,
      use_bfloat16=FLAGS.use_bfloat16,
      num_predict=FLAGS.num_predict)

  return input_fn, record_info_dict


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  assert FLAGS.seq_len > 0
  assert FLAGS.perm_size > 0

  FLAGS.n_token = data_utils.VOCAB_SIZE
  tf.logging.info("n_token {}".format(FLAGS.n_token))

  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)

  # Get train input function
  train_input_fn, train_record_info_dict = get_input_fn("train")

  tf.logging.info("num of batches {}".format(
      train_record_info_dict["num_batch"]))

  # Get train cache function
  train_cache_fn = get_cache_fn(FLAGS.mem_len)

  ##### Get model function
  model_fn = get_model_fn()

  ##### Create TPUEstimator
  # TPU Configuration
  run_config = model_utils.configure_tpu(FLAGS)

  # TPU Estimator
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      train_cache_fn=train_cache_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params={"track_mean": FLAGS.track_mean},
      train_batch_size=FLAGS.train_batch_size,
      eval_on_tpu=FLAGS.use_tpu)

  #### Training
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)


if __name__ == "__main__":
  app.run(main)
