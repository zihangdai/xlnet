from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from absl import flags
import os
import csv
import collections
import numpy as np
import time
import math
import json
import random
from copy import copy
from collections import defaultdict as dd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import sentencepiece as spm

from data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
import model_utils
import function_builder
from classifier_utils import PaddingInputExample
from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids

# Model
flags.DEFINE_string("model_config_path", default=None,
      help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
      help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", default=False,
      help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
      help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model. "
      "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="",
      help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="",
      help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
      help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="",
      help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
      "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
      help="number of iterations per TPU training loop.")

# Training
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_steps", default=12000,
      help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=2e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
      help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0,
      help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
      help="Save the model for every save_steps. "
      "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=8,
      help="Batch size for training. Note that batch size 1 corresponds to "
      "4 sequences: one paragraph + one quesetion + 4 candidate answers.")
flags.DEFINE_float("weight_decay", default=0.00, help="weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# Evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_string("eval_split", default="dev",
      help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=32,
      help="Batch size for evaluation.")

# Data config
flags.DEFINE_integer("max_seq_length", default=512,
      help="Max length for the paragraph.")
flags.DEFINE_integer("max_qa_length", default=128,
      help="Max length for the concatenated question and answer.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
      help="Buffer size used for shuffle.")
flags.DEFINE_bool("uncased", default=False,
      help="Use uncased.")
flags.DEFINE_bool("high_only", default=False,
      help="Evaluate on high school only.")
flags.DEFINE_bool("middle_only", default=False,
      help="Evaluate on middle school only.")

FLAGS = flags.FLAGS

SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


def convert_single_example(example, tokenize_fn):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * FLAGS.max_seq_length * 4,
        input_mask=[1] * FLAGS.max_seq_length * 4,
        segment_ids=[0] * FLAGS.max_seq_length * 4,
        label_id=0,
        is_real_example=False)

  input_ids, input_mask, all_seg_ids = [], [], []
  tokens_context = tokenize_fn(example.context)
  for i in range(len(example.qa_list)):
    tokens_qa = tokenize_fn(example.qa_list[i])
    if len(tokens_qa) > FLAGS.max_qa_length:
      tokens_qa = tokens_qa[- FLAGS.max_qa_length:]

    if len(tokens_context) + len(tokens_qa) > FLAGS.max_seq_length - 3:
      tokens = tokens_context[: FLAGS.max_seq_length - 3 - len(tokens_qa)]
    else:
      tokens = tokens_context

    segment_ids = [SEG_ID_A] * len(tokens)

    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)

    tokens.extend(tokens_qa)
    segment_ids.extend([SEG_ID_B] * len(tokens_qa))

    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_B)

    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)

    cur_input_ids = tokens
    cur_input_mask = [0] * len(cur_input_ids)

    if len(cur_input_ids) < FLAGS.max_seq_length:
      delta_len = FLAGS.max_seq_length - len(cur_input_ids)
      cur_input_ids = [0] * delta_len + cur_input_ids
      cur_input_mask = [1] * delta_len + cur_input_mask
      segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(cur_input_ids) == FLAGS.max_seq_length
    assert len(cur_input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length

    input_ids.extend(cur_input_ids)
    input_mask.extend(cur_input_mask)
    all_seg_ids.extend(segment_ids)

  label_id = example.label

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=all_seg_ids,
      label_id=label_id)
  return feature


class InputExample(object):
  def __init__(self, context, qa_list, label, level):
    self.context = context
    self.qa_list = qa_list
    self.label = label
    self.level = level


def get_examples(data_dir, set_type):
  examples = []

  for level in ["middle", "high"]:
    if level == "middle" and FLAGS.high_only: continue
    if level == "high" and FLAGS.middle_only: continue

    cur_dir = os.path.join(data_dir, set_type, level)
    for filename in tf.gfile.ListDirectory(cur_dir):
      cur_path = os.path.join(cur_dir, filename)
      with tf.gfile.Open(cur_path) as f:
        cur_data = json.load(f)

        answers = cur_data["answers"]
        options = cur_data["options"]
        questions = cur_data["questions"]
        context = cur_data["article"]

        for i in range(len(answers)):
          label = ord(answers[i]) - ord("A")
          qa_list = []

          question = questions[i]
          for j in range(4):
            option = options[i][j]

            if "_" in question:
              qa_cat = question.replace("_", option)
            else:
              qa_cat = " ".join([question, option])

            qa_list.append(qa_cat)

          examples.append(InputExample(context, qa_list, label, level))

  return examples


def file_based_convert_examples_to_features(examples, tokenize_fn, output_file):
  if tf.gfile.Exists(output_file) and not FLAGS.overwrite_data:
    return

  tf.logging.info("Start writing tfrecord %s.", output_file)
  writer = tf.python_io.TFRecordWriter(output_file)

  for ex_index, example in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(example, tokenize_fn)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length * 4], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * 4], tf.float32),
      "segment_ids": tf.FixedLenFeature([seq_length * 4], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  tf.logging.info("Input tfrecord file {}".format(input_file))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    elif is_training:
      batch_size = FLAGS.train_batch_size
    elif FLAGS.do_eval:
      batch_size = FLAGS.eval_batch_size

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
      d = d.repeat()
      # d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def get_model_fn():
  def model_fn(features, labels, mode, params):
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    total_loss, per_example_loss, logits = function_builder.get_race_loss(
        FLAGS, features, is_training)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    #### load pretrained models
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      assert FLAGS.num_hosts == 1

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        eval_input_dict = {
            'labels': label_ids,
            'predictions': predictions,
            'weights': is_real_example
        }
        accuracy = tf.metrics.accuracy(**eval_input_dict)

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            'eval_accuracy': accuracy,
            'eval_loss': loss}

      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

      #### Constucting evaluation TPUEstimatorSpec with new cache.
      label_ids = tf.reshape(features['label_ids'], [-1])
      metric_args = [per_example_loss, label_ids, logits, is_real_example]

      if FLAGS.use_tpu:
        eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=(metric_fn, metric_args),
            scaffold_fn=scaffold_fn)
      else:
        eval_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_fn(*metric_args))

      return eval_spec


    #### Configuring the optimizer
    train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

    monitor_dict = {}
    monitor_dict["lr"] = learning_rate

    #### Constucting training TPUEstimatorSpec with new cache.
    if FLAGS.use_tpu:
      #### Creating host calls
      host_call = None

      train_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      train_spec = tf.estimator.EstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op)

    return train_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #### Validate flags
  if FLAGS.save_steps is not None:
    FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError(
        "At least one of `do_train` or `do_eval` must be True.")

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.spiece_model_file)
  def tokenize_fn(text):
    text = preprocess_text(text, lower=FLAGS.uncased)
    return encode_ids(sp, text)

  # TPU Configuration
  run_config = model_utils.configure_tpu(FLAGS)

  model_fn = get_model_fn()

  spm_basename = os.path.basename(FLAGS.spiece_model_file)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  if FLAGS.use_tpu:
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

  if FLAGS.do_train:
    train_file_base = "{}.len-{}.train.tf_record".format(
        spm_basename, FLAGS.max_seq_length)
    train_file = os.path.join(FLAGS.output_dir, train_file_base)

    if not tf.gfile.Exists(train_file) or FLAGS.overwrite_data:
      train_examples = get_examples(FLAGS.data_dir, "train")
      random.shuffle(train_examples)
      file_based_convert_examples_to_features(
          train_examples, tokenize_fn, train_file)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  if FLAGS.do_eval:
    eval_examples = get_examples(FLAGS.data_dir, FLAGS.eval_split)
    tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on. These do NOT count towards the metric (all tf.metrics
    # support a per-instance weight, and these get a weight of 0.0).
    #
    # Modified in XL: We also adopt the same mechanism for GPUs.

    while len(eval_examples) % FLAGS.eval_batch_size != 0:
      eval_examples.append(PaddingInputExample())

    eval_file_base = "{}.len-{}.{}.tf_record".format(
        spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)

    if FLAGS.high_only:
      eval_file_base = "high." + eval_file_base
    elif FLAGS.middle_only:
      eval_file_base = "middle." + eval_file_base

    eval_file = os.path.join(FLAGS.output_dir, eval_file_base)
    file_based_convert_examples_to_features(
        eval_examples, tokenize_fn, eval_file)

    assert len(eval_examples) % FLAGS.eval_batch_size == 0
    eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True)

    ret = estimator.evaluate(
        input_fn=eval_input_fn,
        steps=eval_steps)

    # Log current result
    tf.logging.info("=" * 80)
    log_str = "Eval | "
    for key, val in ret.items():
      log_str += "{} {} | ".format(key, val)
    tf.logging.info(log_str)
    tf.logging.info("=" * 80)


if __name__ == "__main__":
  tf.app.run()
