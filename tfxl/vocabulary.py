#-*- coding:utf-8 -*-
# pylint: disable=import-error, too-many-instance-attributes
# pylint: disable=too-many-arguments

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

from collections import Counter, OrderedDict
import numpy as np

from tensorflow.io.gfile import GFile as gfile_open
from tensorflow.io.gfile import exists as exists

class Vocab(object):
  def __init__(self, min_freq=0, max_size=None, lower_case=True,
               delimiter=None, vocab_file=None):
    self.counter = Counter()
    self.min_freq = min_freq
    self.max_size = max_size
    self.lower_case = lower_case
    self.delimiter = delimiter
    self.vocab_file = vocab_file
    self.idx2sym = None
    self.sym2idx = None
    self.unk_idx = -1

  def tokenize(self, line, add_eos=False, add_beos=False):
    """

    :param line: a string line in ascii
    :param add_eos: whether to add eos or not
    :param add_beos: whether to add bos and eos or not
    :return: tokenized string list (optionally bos or eos can be added)
    """
    line = line.strip()

    # convert to lower case
    if self.lower_case:
      line = line.lower()

    # empty delimiter '' will evaluate False
    if self.delimiter == '':
      print("delimiter is empty!!")
      symbols = line
    else:
      symbols = line.split(self.delimiter)

    if add_beos:
      return ['<s>'] + symbols + ['</s>']
    elif add_eos:
      return symbols + ['</s>']

    return symbols

  def build_vocab(self):
    """
    Building idx2sym and sym2idx using self.vocab_file
    (Row numbers in a vocabulary file) - 1 is a word index.

    :return: None
    """
    if not self.vocab_file:
      print('Vocab was not set.')
      import sys
      sys.exit(1)
    print('building vocab from {}'.format(self.vocab_file))

    self.idx2sym = []
    self.sym2idx = OrderedDict()

    with gfile_open(self.vocab_file, 'r') as vocab_file:
      for line in vocab_file:
        symb = line.strip().split()[0]
        if symb not in self.sym2idx:
          self.idx2sym.append(symb)
          self.sym2idx[symb] = len(self.idx2sym) - 1
    self.unk_idx = self.sym2idx['<unk>']

    print('final vocab size {}'.format(len(self)))

  def encode_file(self, path, ordered=False, add_eos=True,
                  add_beos=False):
    """

    :param path: text file to be encoded
    :param ordered: ordering data ??
    :param add_eos: whether to add eos or not
    :param add_beos: whether to add bos and eos or not
    :return: encoded sentences per file
    """
    print('encoding file {} ...'.format(path))
    assert exists(path)

    encoded = []
    with gfile_open(path, 'r') as file_to_encode:
      for idx, line in enumerate(file_to_encode):
        if idx > 0 and idx % 500000 == 0:
          print('  line {}'.format(idx))
        symbols = self.tokenize(line, add_eos=add_eos,
                                add_beos=add_beos)
        encoded.append(np.array([self.get_idx(sym) for sym in symbols],
                                dtype=np.int64))

    if ordered:
      encoded = np.concatenate(encoded)

    return encoded

  def encode_sentence(self, line, add_eos=True, add_beos=False):
    """

    :param sentence: a sentence to be encoded
    :param ordered: ordering data ??
    :param add_eos: whether to add eos or not
    :param add_beos: whether to add bos and eos or not
    :return: encoded sentences per file
    """
    encoded = []
    symbols = self.tokenize(line, add_eos=add_eos, add_beos=add_beos)
    encoded.append(np.array([self.get_idx(sym) for sym in symbols],
                            dtype=np.int64))

    return encoded

  def get_sym(self, idx):
    """

    :param idx: symbol index
    :return: a corresponding symbol
    """
    assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
    return self.idx2sym[idx]

  def get_idx(self, sym):
    """

    :param sym: symbol
    :return: a corresponding symbol index
    """
    if sym in self.sym2idx:
      return self.sym2idx[sym]
    assert hasattr(self, 'unk_idx')
    return self.sym2idx.get(sym, self.unk_idx)

  def __len__(self):
    return len(self.idx2sym)
