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

  def tokenize(self, line, add_eos=False, add_double_eos=False):
    line = line.strip()
    # convert to lower case
    if self.lower_case:
      line = line.lower()

    # empty delimiter '' will evaluate False
    if self.delimiter == '':
      symbols = line
    else:
      symbols = line.split(self.delimiter)

    if add_double_eos: # lm1b
      return ['<s>'] + symbols + ['</s>']
    elif add_eos:
      return symbols + ['</s>']
    return symbols

  def _build_from_file(self, vocab_file_path):
    self.idx2sym = []
    self.sym2idx = OrderedDict()

    with gfile_open(vocab_file_path, 'r') as vocab_file:
      for line in vocab_file:
        symb = line.strip().split()[0]
        self.add_symbol(symb)
    self.unk_idx = self.sym2idx['<unk>']

  def build_vocab(self):
    if not self.vocab_file:
      print('Vocab was not set.')
      import sys
      sys.exit(1)
    print('building vocab from {}'.format(self.vocab_file))
    self._build_from_file(self.vocab_file)
    print('final vocab size {}'.format(len(self)))

  def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
                  add_double_eos=False):
    if verbose:
      print('encoding file {} ...'.format(path))
    assert exists(path)
    encoded = []
    with gfile_open(path, 'r') as file_to_encode:
      for idx, line in enumerate(file_to_encode):
        if verbose and idx > 0 and idx % 500000 == 0:
          print('  line {}'.format(idx))
        symbols = self.tokenize(line, add_eos=add_eos,
                                add_double_eos=add_double_eos)
        encoded.append(self.convert_to_nparray(symbols))

    if ordered:
      encoded = np.concatenate(encoded)

    return encoded

  def encode_sents(self, sents, ordered=False, verbose=False):
    if verbose:
      print('encoding {} sents ...'.format(len(sents)))
    encoded = []
    for idx, symbols in enumerate(sents):
      if verbose and idx > 0 and idx % 500000 == 0:
        print('  line {}'.format(idx))
      encoded.append(self.convert_to_nparray(symbols))

    if ordered:
      encoded = np.concatenate(encoded)

    return encoded

  def add_special(self, sym):
    if sym not in self.sym2idx:
      self.idx2sym.append(sym)
      self.sym2idx[sym] = len(self.idx2sym) - 1
      setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

  def add_symbol(self, sym):
    if sym not in self.sym2idx:
      self.idx2sym.append(sym)
      self.sym2idx[sym] = len(self.idx2sym) - 1

  def get_sym(self, idx):
    assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
    return self.idx2sym[idx]

  def get_idx(self, sym):
    if sym in self.sym2idx:
      return self.sym2idx[sym]
    assert hasattr(self, 'unk_idx')
    return self.sym2idx.get(sym, self.unk_idx)

  def get_symbols(self, indices):
    return [self.get_sym(idx) for idx in indices]

  def get_indices(self, symbols):
    return [self.get_idx(sym) for sym in symbols]

  def convert_to_nparray(self, symbols):
    nparray = np.array(self.get_indices(symbols), dtype=np.int64)
    return nparray

  def convert_to_sent(self, indices, exclude=None):
    if exclude is None:
      return ' '.join([self.get_sym(idx) for idx in indices])
    return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

  def __len__(self):
    return len(self.idx2sym)
