#-*- coding:utf-8 -*-
# pylint: disable=import-error, too-many-locals, too-many-branches
# pylint: disable=redefined-outer-name, too-many-arguments
# pylint: disable=too-few-public-methods

# Copyright 2019 The Transformer-xl Authors.
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

import math
import os
from functools import partial
import pickle
import json
import multiprocessing as mp
import numpy as np
from absl import flags

import tensorflow as tf
from tensorflow.io.gfile import exists
from tensorflow.io.gfile import makedirs
from tensorflow.io.gfile import glob

from tfxl.vocabulary import Vocab

def _preprocess(shard, train, vocab, save_dir, bsz, tgt_len, num_shuffle):
  file_names = []
  num_batch = 0

  path = train[shard]
  data_shard = vocab.encode_file(path, ordered=False,
                                 add_eos=False, add_double_eos=True)

  for shuffle in range(num_shuffle):
    basename = "train-{:03d}-{:02d}".format(shard, shuffle)
    print("Processing shard {} shuffle {}".format(shard, shuffle))

    np.random.shuffle(data_shard)
    file_name, num_batch_shuffle = create_ordered_tfrecords(
        save_dir, basename, np.concatenate(data_shard), bsz, tgt_len)
    file_names.append(file_name)
    num_batch += num_batch_shuffle

  return file_names, num_batch


class Corpus:
  def __init__(self, path, add_eos, add_beos, *args, **kwargs):
    self.vocab = Vocab(*args, **kwargs)
    self.add_eos = add_eos
    self.add_beos = add_beos
    self.vocab.build_vocab()
    pattern = os.path.join(path, "train", "train.??")
    self.train = glob(pattern)

    self.valid = \
      self.vocab.encode_file(os.path.join(path, "valid.txt"),
                             ordered=True, add_eos=self.add_eos,
                             add_beos=self.add_beos)
    self.test = \
      self.vocab.encode_file(os.path.join(path, "test.txt"),
                             ordered=True, add_eos=self.add_eos,
                             add_beos=self.add_beos)

    if FLAGS.cutoffs.split(", "):
      self.cutoffs = [0] + [int(cutoff) for cutoff in
                            FLAGS.cutoffs.split(", ")] + [len(self.vocab)]
    else:
      self.cutoffs = []

    print("cutoffs:")
    print(self.cutoffs)


  def convert_to_tfrecords(self, split, save_dir, bsz, tgt_len,
                           num_core_per_host, **kwargs):
    config_flags = kwargs.get("FLAGS")
    file_names = []
    record_name = \
      "record_info-{}.bsz-{}.tlen-{}.json".format(split, bsz, tgt_len)

    record_info_path = os.path.join(save_dir, record_name)

    bin_sizes = get_bin_sizes(self.valid, bsz // num_core_per_host, tgt_len,
                              self.cutoffs, [2.5, 2.5, 2.5])
    if split == "train":
      np.random.seed(123456)
      num_batch = 0

      if config_flags.num_procs > 1:
        #def _preprocess(shard, train, vocab, save_dir, bsz, tgt_len,num_shuffle):
        _preprocess_wrapper = \
          partial(_preprocess, train=self.train, vocab=self.vocab,
                  save_dir=save_dir,
                  bin_sizes=bin_sizes, bsz=bsz, tgt_len=tgt_len,
                  num_core_per_host=num_core_per_host,
                  num_shuffle=config_flags.num_shuffle)

        pool = mp.Pool(processes=config_flags.num_procs)
        results = pool.map(_preprocess_wrapper, range(len(self.train)))
        for res in results:
          file_names.extend(res[0])
          num_batch += res[1]
      else:
        for shard, path in enumerate(self.train):
          data_shard = self.vocab.encode_file(path, ordered=False,
                                              add_eos=self.add_eos,
                                              add_beos=self.add_beos)

          num_shuffle = config_flags.num_shuffle

          for shuffle in range(num_shuffle):
            print("Processing shard {} shuffle {}".format(shard, shuffle))
            basename = "train-{:03d}-{:02d}".format(shard, shuffle)
            np.random.shuffle(data_shard)
            file_name, num_batch_ = create_ordered_tfrecords(
                save_dir, basename, np.concatenate(data_shard), bsz, tgt_len)
            file_names.append(file_name)
            num_batch += num_batch_

    else:
      file_name, num_batch = create_ordered_tfrecords(
          save_dir, split, getattr(self, split), bsz, tgt_len)
      file_names.append(file_name)

    with open(record_info_path, "w") as fp:
      record_info = {
          "filenames": file_names,
          "bin_sizes": bin_sizes,
          "num_batch": num_batch
      }
      json.dump(record_info, fp)

def get_bin_sizes(data, batch_size, tgt_len, cutoffs, std_mult):
  """
    Note: the `batch_size` here should be per-core batch size
  """
  bin_sizes = []

  def _nearest_to_eight(arg_x): # so that it's faster on TPUs
    arg_y = arg_x - arg_x % 8
    return arg_y + 8 if arg_x % 8 >= 4 else max(8, arg_y)

  if cutoffs:
    num_batch = len(data) // batch_size // tgt_len

    data = data[:batch_size * num_batch * tgt_len]
    data = data.reshape(batch_size, num_batch, tgt_len)

    tot = batch_size * tgt_len
    for bin_idx, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
      mask = (data >= left) * (data < right)
      # pylint: disable=no-member
      percents = mask.astype(np.float64).sum(2).sum(0) / tot
      mean = np.mean(percents)
      std = np.std(percents)

      bin_size = int(math.ceil(tgt_len * batch_size * (mean + std_mult[bin_idx]
                                                       * std)))
      bin_size = _nearest_to_eight(bin_size)
      bin_sizes.append(bin_size)

  return bin_sizes

def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def batchify(data, batch_size):
  """
    Here, we use multiple randomly shifted copies to deal with this problem.
  """
  num_step = len(data) // batch_size
  data = data[:batch_size * num_step]
  data = data.reshape(batch_size, num_step)

  return data

def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len):
  file_name = "{}.bsz-{}.tlen-{}.tfrecords".format(basename, batch_size, tgt_len)
  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.io.TFRecordWriter(save_path)

  batched_data = batchify(data, batch_size)

  num_batch = 0
  # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
  for tgt_idx in range(0, batched_data.shape[1] - 1, tgt_len):
    cur_tgt_len = min(batched_data.shape[1] - 1 - tgt_idx, tgt_len)
    # drop the remainder if use tpu
    if num_batch % 500 == 0:
      print("  processing batch {}".format(num_batch))
    for idx in range(batch_size):
      inputs = batched_data[idx, tgt_idx:tgt_idx + cur_tgt_len]
      labels = batched_data[idx, tgt_idx + 1:tgt_idx + cur_tgt_len + 1]

      # features dict
      feature = {
          "inputs": _int64_feature(inputs),
          "labels": _int64_feature(labels),
      }

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      record_writer.write(example.SerializeToString())

    num_batch += 1

  record_writer.close()
  print("Done writing {}. batches: {}".format(file_name, num_batch))

  return file_name, num_batch

def get_lm_corpus(data_dir):
  cache_pkl_path = os.path.join(data_dir, "cache.pkl")

  if exists(cache_pkl_path):
    print("Loading cached dataset...")
    with open(cache_pkl_path, "rb") as fp:
      corpus = pickle.load(fp)
  else:
    print("Producing dataset...")
    kwargs = {}
    kwargs["lower_case"] = True
    kwargs["vocab_file"] = os.path.join(data_dir, FLAGS.vocab)

    corpus = Corpus(data_dir, add_eos=FLAGS.add_eos,
                    add_beos=FLAGS.add_beos, **kwargs)

    print("Saving dataset...")
    with open(cache_pkl_path, "wb") as fp:
      pickle.dump(corpus, fp, protocol=2)

    corpus_info = {
        "vocab_size" : len(corpus.vocab),
        "cutoffs" : corpus.cutoffs,
    }
    with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
      json.dump(corpus_info, fp)

  return corpus


def main(unused_argv):
  del unused_argv  # Unused
  corpus = get_lm_corpus(FLAGS.data_dir)

  save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
  if not exists(save_dir):
    makedirs(save_dir)

  # test mode
  if FLAGS.per_host_test_bsz > 0:
    corpus.convert_to_tfrecords("test", save_dir, FLAGS.per_host_test_bsz,
                                FLAGS.tgt_len, FLAGS.num_core_per_host,
                                FLAGS=FLAGS)
    return

  for split, batch_size in zip(
      ["train", "valid"],
      [FLAGS.per_host_train_bsz, FLAGS.per_host_valid_bsz]):

    if batch_size <= 0:
      continue

    print("Converting {} set...".format(split))
    corpus.convert_to_tfrecords(split, save_dir, batch_size, FLAGS.tgt_len,
                                FLAGS.num_core_per_host, FLAGS=FLAGS)


def get_input_fn(record_info_dir, split, per_host_bsz, tgt_len,
                 num_core_per_host):
  """Creates input function."""
  record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(split, per_host_bsz,
                                                            tgt_len)

  record_info_path = os.path.join(record_info_dir, record_name)
  tf.compat.v1.logging.info("loading corpus_info from {}"
                            .format(record_info_path))
  with open(record_info_path, "r") as fp:
    record_info = json.load(fp)

  file_names = record_info["filenames"]

  tf.compat.v1.logging.info("[{}] File names {}".format(split, file_names))

  def input_fn(params):
    # per-core batch size
    per_core_bsz = params["batch_size"]

    # data_dir could be a remote path, e.g., a google storage url
    data_dir = params["data_dir"]

    def parser(record):
      # whether allow the last batch with a potentially shorter length
      record_spec = {
          "inputs": tf.io.VarLenFeature(tf.int64),
          "labels": tf.io.VarLenFeature(tf.int64),
      }

      # retrieve serialized example
      example = tf.io.parse_single_example(
          serialized=record,
          features=record_spec)

      # cast int64 into int32
      # cast sparse to dense
      for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
          val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
          val = tf.cast(val, tf.int32)
        example[key] = val

      return example["inputs"], example["labels"]

    file_paths = []
    for file_name in file_names:
      file_path = os.path.join(data_dir, file_name)
      file_paths.append(file_path)

    if split == "train":
      dataset = tf.compat.v1.data.Dataset.from_tensor_slices(file_paths)
      if len(file_paths) > 1:
        dataset = dataset.shuffle(len(file_paths)).repeat()
        dataset = tf.data.TFRecordDataset(dataset)
      else:
        dataset = tf.data.TFRecordDataset(dataset)

      dataset = dataset.map(parser).cache().repeat()
      dataset = dataset.batch(per_core_bsz, drop_remainder=True)
      dataset = dataset.prefetch(num_core_per_host * per_core_bsz)
    else:
      # do not shuffle, repeat or cache in evaluation
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      dataset = tf.data.TFRecordDataset(dataset)
      dataset = dataset.map(parser)
      dataset = dataset.batch(per_core_bsz, drop_remainder=True)

    return dataset

  return input_fn, record_info

if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_string("cutoffs", default=None,
                    help="int, int, int ..")
  flags.DEFINE_bool("add_eos", default=False,
                    help="whether to add </s> symbol")
  flags.DEFINE_bool("add_beos", default=True,
                    help="whether to add <s> and </s> symbol, add_eos will be "
                         "disabled when add_beos is set to True.")
  flags.DEFINE_string("data_dir", None, help="Location of the data corpus")
  flags.DEFINE_integer("per_host_train_bsz", 60, help="train batch size each "
                                                      "host")
  flags.DEFINE_integer("per_host_valid_bsz", 60, help="valid batch size each "
                                                      "host")
  flags.DEFINE_integer("per_host_test_bsz", 0,
                       help="If > 0, enter test mode and process test set only."
                            "Otherwise, process train and dev sets only.")
  flags.DEFINE_integer("tgt_len", 70, help="number of tokens to predict")
  flags.DEFINE_integer("max_batch", -1, help="run in debug mode")
  flags.DEFINE_integer("num_core_per_host", 8, help="8 for TPU v2.")
  flags.DEFINE_integer("num_procs", 1, help="number of processes")
  flags.DEFINE_integer("num_shuffle", 4, help="number of shuffles for lm1b")
  flags.DEFINE_string("vocab", None, help="vocab filename")

  tf.compat.v1.app.run(main)
