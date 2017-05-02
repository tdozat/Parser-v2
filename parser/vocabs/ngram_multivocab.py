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

import os
import sys
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser import Configurable, Multibucket
from parser.vocabs.base_vocab import BaseVocab
from parser.vocabs import SubtokenVocab, NgramVocab, Multivocab
from parser.misc.bucketer import Bucketer

__all__ = ['NgramMultivocab']

#***************************************************************
class NgramMultivocab(Multivocab, SubtokenVocab):
  """"""
  
  #=============================================================
  def __init__(self, token_vocab, *args, **kwargs):
    """"""
    
    super(BaseVocab, self).__init__(*args, **kwargs)
    self._cased = super(BaseVocab, self).cased
    
    SubtokenVocab.__setattr__(self, '_token_vocab', token_vocab)
    self._multibucket = Multibucket.from_configurable(self, embed_model=self.embed_model, name=self.name)
    self._vocabs = [NgramVocab.from_vocab(self.token_vocab, i+1, cased=self.cased) for i in xrange(self.max_n)]
    self._special_tokens = super(BaseVocab, self).special_tokens
    self._special_tokens_set = set(self._special_tokens)
    SubtokenVocab._set_special_tokens(self)
    self._tok2idx = {}
    
    for vocab in self:
      assert vocab.token_vocab is self.token_vocab
    return
  
  #=============================================================
  def add_files(self, conll_files):
    """"""
    
    self.token_vocab.count(conll_files)
    return
  
  #=============================================================
  def index_tokens(self):
    """"""
    
    n_buckets = self.n_buckets
    tok2idxs = {token: [vocab.subtoken_indices(token) for vocab in self] for token in self.token_vocab.counts}
    with Bucketer.from_configurable(self, self.n_buckets, name='bucketer-%s'%self.name) as bucketer:
      splits = bucketer.compute_splits(len(indices[0]) for indices in tok2idxs.values())
      bucketer.plot()
    with self.multibucket.open(splits, depth=len(self)):
      for index, special_token in enumerate(self.special_tokens):
        self.tok2idx[special_token] = self.multibucket.add([[index]*len(self)])
      for token, _ in self.sorted_counts(self.token_vocab.counts):
        indices = tok2idxs[token]
        sequence = [[indices[i][j] for i in xrange(len(indices)) if j < len(indices[i])] for j in xrange(len(indices[0]))]
        self.tok2idx[token] = self.multibucket.add(sequence)
    return
  
  #=============================================================
  def __call__(self, placeholder, keep_prob=None, moving_params=None):
    return SubtokenVocab.__call__(self, placeholder, keep_prob=keep_prob, moving_params=moving_params)
  
  def index(self, token):
    return SubtokenVocab.index(self, token)
  
  def generate_placeholder(self):
    return SubtokenVocab.generate_placeholder(self)
  
  #=============================================================
  def embedding_lookup(self, placeholders, embed_keep_prob=None, moving_params=None):
    """"""
    
    if moving_params is None:
      shape = tf.shape(placeholders)
      shape = tf.stack([shape[0], 1, shape[2]])
      placeholders = la.random_where(embed_keep_prob, placeholders, self.UNK, shape=shape)
    embeddings = [vocab.embedding_lookup(placeholders[:,:,i], embed_keep_prob=1, moving_params=moving_params) for i, vocab in enumerate(self)]
    return tf.stack(embeddings, axis=2)
  
  #=============================================================
  def __iter__(self):
    return (vocab for vocab in self._vocabs)
  def __getitem__(self, key):
    return self._vocabs[key]
  def __len__(self):
    return len(self._vocabs)

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser import Configurable
  from parser.vocabs import WordVocab, NgramMultivocab
  
  configurable = Configurable()
  token_vocab = WordVocab.from_configurable(configurable)
  ngram_multivocab = NgramMultivocab.from_vocab(token_vocab)
  ngram_multivocab.add_files(configurable.valid_files)
  ngram_multivocab.index_tokens()
  print("Indices for '<PAD>': %s" % str(ngram_multivocab.index('<PAD>')))
  print("Indices for 'the': %s" % str(ngram_multivocab.index('the')))
  print("Indices for 'The': %s" % str(ngram_multivocab.index('The')))
  print('NgramMultivocab passes')
  