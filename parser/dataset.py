#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat
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

import re
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser import Configurable, Multibucket
from parser.vocabs.base_vocab import BaseVocab
from parser.misc.bucketer import Bucketer

__all__ = ['Trainset', 'Validset', 'Parseset']

#***************************************************************
class Dataset(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, vocabs, *args, **kwargs):
    """"""
    
    parser_model = kwargs.pop('parser_model', None)
    super(Dataset, self).__init__(*args, **kwargs)
    
    self._vocabs = vocabs
    self._multibuckets = [Multibucket.from_configurable(vocab, name='%s-%s'%(self.name, vocab.name)) for vocab in self.vocabs]
    
    if parser_model is not None:
      self._parser_model = parser_model.from_configurable(self, name=self.name)
    else:
      self._parser_model = None
    
    with Bucketer.from_configurable(self, self.n_buckets, name='bucketer-%s'%self.name) as bucketer:
      splits = bucketer.compute_splits(len(sent) for sent in self.iterfiles())
      for i in xrange(len(splits)):
        splits[i] += 1
    for multibucket, vocab in self.iteritems():
      multibucket.open(splits, depth=vocab.depth)
    for sent in self.iterfiles():
      for multibucket, vocab in self.iteritems():
        multibucket.add([vocab.ROOT] + [vocab.index(line[vocab.conll_idx]) for line in sent])
    for multibucket in self:
      multibucket.close()
    self._multibucket = Multibucket.from_dataset(self)
    return
  
  #=============================================================
  def __call__(self, moving_params=None):
    """"""
    
    return self._parser_model(self.vocabs, moving_params=moving_params)
  
  #=============================================================
  def iterfiles(self):
    """"""
    
    for data_file in self.data_files:
      with codecs.open(data_file, encoding='utf-8', errors='ignore') as f:
        buff = []
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            if not re.match('[0-9]+[-.][0-9]+', line):
              buff.append(line.split('\t'))
          elif buff:
            yield buff
            buff = []
        yield buff
  
  #=============================================================
  def iterbatches(self, shuffle=True):
    """"""
    
    batch_size = self.batch_size
    batch_by = self.batch_by 
    batches = []
    for bkt_idx, bucket in enumerate(self.multibucket):
      if batch_size == 0:
        n_splits = 1
      else:
        if batch_by == 'tokens':
          n_tokens = bucket.indices.shape[0] * bucket.indices.shape[1]
          n_splits = max(n_tokens // batch_size, 1)
        elif batch_by == 'seqs':
          n_seqs = bucket.indices.shape[0]
          n_splits = max(n_seqs // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      splits = np.array_split(range_func(bucket.indices.shape[0]), n_splits)
      for split in splits:
        batches.append( (bkt_idx, split) )
    if shuffle:
      np.random.shuffle(batches)
    for bkt_idx, batch in batches:
      feed_dict = {}
      for multibucket, vocab in self.iteritems():
        bucket = multibucket[bkt_idx]
        vocab.set_feed_dict(bucket.indices[batch], feed_dict)
      yield feed_dict
  
  #=============================================================
  def iteritems(self):
    for i in xrange(len(self)):
      yield (self[i], self._vocabs[i])
  
  #=============================================================
  def update_history(self, history, accumulators):
    self._parser_model.update_history(history, accumulators)
    return
  
  def print_accuracy(self, accumulators, time):
    self._parser_model.print_accuracy(accumulators, time, prefix=self.PREFIX.title())
    return
  
  def write_probs(self, input_file, output_file, probs):
    self._parser_model.write_probs(input_file, output_file, probs, self.multibucket.inv_idxs())
    return
  
  def plot(self, history):
    self._parser_model.plot(history)
    return
  
  #=============================================================
  @property
  def data_files(self):
    return getattr(self, '{0}_files'.format(self.PREFIX.lower()))
  @property
  def multibucket(self):
    return self._multibucket
  @property
  def vocabs(self):
    return self._vocabs
  @property
  def train_keys(self):
    return self._parser_model.train_keys
  @property
  def parse_keys(self):
    return self._parser_model.parse_keys
  
  #=============================================================
  def __len__(self):
    return len(self._multibuckets)
  def __iter__(self):
    return (multibucket for multibucket in self._multibuckets)
  def __getitem__(self, key):
    return self._multibuckets[key]

#***************************************************************
class Trainset(Dataset):
  PREFIX = 'train'
class Parseset(Dataset):
  PREFIX = 'parse'

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser.vocabs import *
  from parser.dataset import Trainset
  
  configurable = Configurable()
  dep_vocab = DepVocab.from_configurable(configurable)
  word_vocab = WordVocab.from_configurable(configurable)
  lemma_vocab = LemmaVocab.from_configurable(configurable)
  pretrained_vocab = PretrainedVocab.from_vocab(word_vocab)
  char_vocab = NgramMultivocab.from_vocab(word_vocab)
  word_multivocab = Multivocab.from_configurable(configurable, [word_vocab, pretrained_vocab, char_vocab], name='words')
  tag_vocab = TagVocab.from_configurable(configurable)
  xtag_vocab = XTagVocab.from_configurable(configurable)
  head_vocab = HeadVocab.from_configurable(configurable)
  rel_vocab = RelVocab.from_configurable(configurable)
  trainset = Trainset.from_configurable(configurable, [dep_vocab, word_multivocab, lemma_vocab, tag_vocab, xtag_vocab, head_vocab, rel_vocab])
  trainset()
  print('Dataset passes')
  
