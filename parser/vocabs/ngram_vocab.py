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
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.vocabs import TokenVocab, SubtokenVocab, CharVocab
from parser import Multibucket

__all__ = ['NgramVocab']

#***************************************************************
class NgramVocab(SubtokenVocab):
  """"""
  
  #=============================================================
  def __init__(self, n, token_vocab, *args, **kwargs):
    """"""
    
    recount = kwargs.pop('recount', False)
    initialize_zero = kwargs.pop('initialize_zero', False)
    super(TokenVocab, self).__init__(*args, **kwargs)
    
    self._n = n
    self._token_vocab = token_vocab
    self._token_counts = Counter()
    self._subtoken_vocab = CharVocab.from_vocab(self.token_vocab)
    self._multibucket = Multibucket.from_configurable(self, embed_model=self.embed_model, name=self.name)
    
    if recount:
      self.count()
    else:
      if os.path.isfile(self.filename):
        self.load()
      else:
        self.count()
        self.dump()
    self.index_vocab()
    
    embed_dims = [len(self), self.embed_size]
    if initialize_zero:
      self.embeddings = np.zeros(embed_dims)
    else:
      self.embeddings = np.random.randn(*embed_dims)
    return
  
  #=============================================================
  def count(self):
    """"""
    
    special_tokens = set(self.token_vocab.special_tokens)
    for token in self.token_vocab:
      if token not in special_tokens:
        idxs = self.subtoken_vocab.subtoken_indices(token)
        idxs = [self.subtoken_vocab.START] + idxs + [self.subtoken_vocab.STOP]
        if len(idxs) > self.n:
          for i in xrange(len(idxs) - self.n):
            subtoken = ''.join(self.subtoken_vocab[idxs[i:i+self.n]])
            self.counts[subtoken] += 1
            self.token_counts[subtoken] += self.token_vocab.counts[token]
    return
  
  #=============================================================
  def subtoken_indices(self, token):
    """"""
    
    idxs = self.subtoken_vocab.subtoken_indices(token)
    idxs = [self.subtoken_vocab.START] + idxs + [self.subtoken_vocab.STOP]
    if len(idxs) <= self.n:
      return [self.PAD]
    else:
      subtokens = []
      for i in xrange(len(idxs) - self.n):
        subtokens.append(''.join(self.subtoken_vocab[idxs[i:i+self.n]]))
      return self[subtokens]
  
  #=============================================================
  @property
  def n(self):
    return self._n
  @property
  def subtoken_vocab(self):
    return self._subtoken_vocab
  @property
  def name(self):
    return '%d-%s' % (self.n, super(NgramVocab, self).name)
  
  #=============================================================
  def __setattr__(self, name, value):
    if name == '_subtoken_vocab':
      self._conll_idx = value.conll_idx
      if self.cased is None:
        self._cased = value.cased
      elif self.cased != value.cased:
        cls = value.__class__
        value = cls.from_configurable(value, value.token_vocab,
                                      cased=self.cased,
                                      recount=True)
    super(NgramVocab, self).__setattr__(name, value)
    return

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser import Configurable
  from parser.vocabs import WordVocab, CharVocab, NgramVocab
  
  configurable = Configurable()
  token_vocab = WordVocab.from_configurable(configurable, 1)
  if os.path.isfile('saves/defaults/2-ngrams.txt'):
    os.remove('saves/defaults/2-ngrams.txt')
  ngram_vocab = NgramVocab.from_vocab(token_vocab, 2)
  ngram_vocab = NgramVocab.from_vocab(token_vocab, 2)
  ngram_vocab.token_vocab.count(conll_files = configurable.valid_files)
  ngram_vocab.index_tokens()
  ngram_vocab.fit_to_zipf()
  print('NgramVocab passes')