# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re

import numpy as np
import sentencepiece as spm
import codecs
import collections
import tensorflow as tf

import xlnet
from model_utils import init_from_checkpoint, configure_tpu
from prepro_utils import preprocess_text, encode_pieces
from run_classifier import PaddingInputExample

from data_utils import SEP_ID, CLS_ID

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", help="Input file of sentences", default="")

flags.DEFINE_string("output_file", help="Output (JSON) file", default="")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, unique_id, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          unique_id: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def _encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return pieces, ids


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b, label=0.0))
            unique_id += 1
    return examples


def model_fn_builder():
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        inp = tf.transpose(features["input_ids"], [1, 0])
        seg_id = tf.transpose(features["segment_ids"], [1, 0])
        inp_mask = tf.transpose(features["input_mask"], [1, 0])

        xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)

        # no need for dropout in prediction mode
        xlnet_config.dropout = 0.0
        xlnet_config.dropatt = 0.0

        run_config = xlnet.create_run_config(False, True, FLAGS)

        # no need for dropout in prediction mode
        run_config.dropout = 0.0
        run_config.dropatt = 0.0

        xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=inp,
            seg_ids=seg_id,
            input_mask=inp_mask)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # load pretrained models
        scaffold_fn = init_from_checkpoint(FLAGS)

        # Get a sequence output
        seq_out = xlnet_model.get_sequence_output()

        tokens = tf.transpose(seq_out, [1, 0, 2])

        predictions = {"unique_id": unique_ids,
                       'tokens': tokens}

        if FLAGS.use_tpu:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    all_is_real_example = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)
        all_is_real_example.append(feature.is_real_example)

    def input_fn(params, input_context=None):
        """The actual input function."""
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        else:
            batch_size = FLAGS.predict_batch_size

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.float32),
            "segment_ids":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(
                    all_label_ids,
                    shape=[num_examples],
                    dtype=tf.int32),
            "is_real_example":
                tf.constant(
                    all_is_real_example,
                    shape=[num_examples],
                    dtype=tf.int32),
        })

        # Shard the dataset to difference devices
        if input_context is not None:
            tf.logging.info("Input pipeline id %d out of %d",
                            input_context.input_pipeline_id, input_context.num_replicas_in_sync)
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)

        d = d.batch(batch_size=batch_size, drop_remainder=False)

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, max_seq_length, sp_model, uncased):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    features = []
    for ex_index, example in enumerate(examples):
        if isinstance(example, PaddingInputExample):
            features.append(InputFeatures(
                unique_id=ex_index,
                tokens=[''] * max_seq_length,
                input_ids=[0] * max_seq_length,
                input_mask=[1] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_id=0,
                is_real_example=False))
            continue

        tokens_a_preprocessed = preprocess_text(example.text_a, lower=uncased)
        tokens_a_unicode, tokens_a = _encode_ids(sp_model, tokens_a_preprocessed)
        tokens_a_str = [token.encode("ascii", "ignore").decode('utf-8', 'ignore') for token in tokens_a_unicode]
        tokens_b = None
        tokens_b_str = None
        if example.text_b:
            tokens_b_preprocessed = preprocess_text(example.text_b, lower=uncased)
            tokens_b_unicode, tokens_b = _encode_ids(sp_model, tokens_b_preprocessed)
            tokens_b_str = [token.encode("ascii", "ignore").decode('utf-8', 'ignore') for token in tokens_b_unicode]

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for two [SEP] & one [CLS] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for one [SEP] & one [CLS] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:max_seq_length - 2]

        tokens = []
        tokens_str = []
        segment_ids = []
        for token, token_str in zip(tokens_a, tokens_a_str):
            tokens.append(token)
            tokens_str.append(token_str)
            segment_ids.append(SEG_ID_A)
        tokens.append(SEP_ID)
        tokens_str.append("<sep>")
        segment_ids.append(SEG_ID_A)

        if tokens_b:
            for token, token_str in zip(tokens_b, tokens_b_str):
                tokens.append(token)
                tokens_str.append(token_str)
                segment_ids.append(SEG_ID_B)
            tokens.append(SEP_ID)
            tokens_str.append("<sep>")
            segment_ids.append(SEG_ID_B)

        tokens.append(CLS_ID)
        tokens_str.append("<sep>")
        segment_ids.append(SEG_ID_CLS)

        input_ids = tokens

        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        if len(input_ids) < max_seq_length:
            delta_len = max_seq_length - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % ex_index)
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: {} (id = {})".format(0.0, 0))

        features.append(InputFeatures(
            unique_id=ex_index,
            tokens=tokens_str,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=0,
            is_real_example=True))
    return features


def _round_vector(values, points):
    return [round(float(x), points) for x in values]


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.do_predict:
        predict_dir = FLAGS.predict_dir
        if not tf.gfile.Exists(predict_dir):
            tf.gfile.MakeDirs(predict_dir)

    spiece_model_file = FLAGS.spiece_model_file
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spiece_model_file)

    model_fn = model_fn_builder()

    run_config = configure_tpu(FLAGS)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    examples = read_examples(FLAGS.input_file)
    original_examples_length = len(examples)

    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on. These do NOT count towards the metric (all tf.metrics
    # support a per-instance weight, and these get a weight of 0.0).
    #
    # Modified in XL: We also adopt the same mechanism for GPUs.
    while len(examples) % FLAGS.predict_batch_size != 0:
        examples.append(PaddingInputExample())

    features = convert_examples_to_features(
        examples=examples, max_seq_length=FLAGS.max_seq_length, sp_model=sp_model, uncased=FLAGS.uncased)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    assert len(examples) % FLAGS.predict_batch_size == 0

    input_fn = input_fn_builder(
        features=features, seq_length=FLAGS.max_seq_length)

    with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file, "w")) as writer:
        for example_cnt, result in enumerate(estimator.predict(input_fn=input_fn,
                                                               yield_single_examples=True,
                                                               checkpoint_path=FLAGS.predict_ckpt)):
            if example_cnt % 1000 == 0:
                tf.logging.info("Predicting submission for example_cnt: {}".format(example_cnt))

            # output only real examples, and not padded examples
            if example_cnt < original_examples_length:
                unique_id = int(result["unique_id"])
                feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                output_json['pooled_%s' % FLAGS.summary_type] = _round_vector(
                    result['pooled_%s' % FLAGS.summary_type].flat, 6
                )
                all_features = []
                for (i, token) in enumerate(feature.tokens):
                    features = collections.OrderedDict()
                    features["token"] = token
                    features["values"] = _round_vector(result['tokens'][i].flat, 6)
                    all_features.append(features)
                output_json["features"] = all_features
                writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("model_config_path")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("spiece_model_file")
    tf.app.run()
