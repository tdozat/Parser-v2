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
import re
from collections import Counter

import numpy as np
import tensorflow as tf

import parser.neural.linalg as linalg
from parser import Configurable

#***************************************************************
class BaseVocab(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(BaseVocab, self).__init__(*args, **kwargs)
    
    self._cased = super(BaseVocab, self).cased
    self._special_tokens = super(BaseVocab, self).special_tokens
    self._special_tokens_set = set(self._special_tokens)
    self._set_special_tokens()
    # NOTE: __setattr__ turns these into dicts
    self._str2idx = zip(self.special_tokens, range(len(self.special_tokens)))
    self._idx2str = zip(range(len(self.special_tokens)), self.special_tokens)
    self._tok2idx = self._str2idx
    self._counts = None
    self._embeddings = None
    # NOTE this placeholder stores the token data indices
    # I.e. the token's index in the word/tag/glove embedding matrix
    # CharVocab will by default be "char"
    self.placeholder = None
  
  #=============================================================
  def _set_special_tokens(self):
    pattern = re.compile('\W+', re.UNICODE)
    for i, token in enumerate(self.special_tokens):
      token = token.lstrip('<')
      token = token.rstrip('>')
      token = token.upper()
      token = pattern.sub('', token)
      assert token not in self.__dict__
      self.__dict__[token] = i
    return
  
  #=============================================================
  @classmethod
  def from_vocab(cls, vocab, *args, **kwargs):
    """"""
    
    args += (vocab,)
    return cls.from_configurable(vocab, *args, **kwargs)
 
  #=============================================================
  def generate_placeholder(self):
    """"""
    
    if self.placeholder is None:
      self.placeholder = tf.placeholder(tf.int32, shape=[None, None], name=self.name)
    return self.placeholder
  
  #=============================================================
  def __call__(self, placeholder=None, moving_params=None):
    """"""
    
    placeholder = self.generate_placeholder() if placeholder is None else placeholder
    embeddings = self.embeddings if moving_params is None else moving_params.average(self.embeddings)
    return tf.nn.embedding_lookup(embeddings, placeholder)
  
  #=============================================================
  def set_feed_dict(self, data, feed_dict):
    """"""
    
    feed_dict[self.placeholder] = data
    return
  
  #=============================================================
  def load(self):
    raise NotImplementedError()
  def dump(self):
    raise NotImplementedError()
  def count(self):
    raise NotImplementedError()
  
  #=============================================================
  def strings(self):
    return self._str2idx.keys()
  def indices(self):
    return self._str2idx.values()
  def iteritems(self):
    return self._str2idx.iteritems()
  def most_common(self, n=None):
    return self._counts.most_common(n)
  def index(self, token):
    if not self.cased and token not in self._special_tokens_set:
      token = token.lower()
    return self._tok2idx.get(token, self.UNK)
  
  #=============================================================
  @property
  def depth(self):
    return None
  @property 
  def special_tokens(self):
    return self._special_tokens
  @property 
  def cased(self):
    return self._cased
  @property
  def counts(self):
    return self._counts
  @property
  def embeddings(self):
    return self._embeddings
  @embeddings.setter
  def embeddings(self, matrix):
    if matrix.shape[1] != self.embed_size:
      raise ValueError("Matrix shape[1] of %d doesn't match expected shape of %d" % (matrix.shape[1], self.embed_size))
    with tf.device('/cpu:0'):
      with tf.variable_scope(self.name.title()):
        self._embeddings = tf.Variable(matrix, name='Embeddings', dtype=tf.float32, trainable=True)
    return
  
  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, basestring):
      if not self.cased and key not in self._special_tokens_set:
        key = key.lower()
      return self._str2idx.get(key, self.UNK)
    elif isinstance(key, (int, long, np.int32, np.int64)):
      return self._idx2str.get(key, self.special_tokens[self.UNK])
    elif hasattr(key, '__iter__'):
      return [self[k] for k in key]
    else:
      raise ValueError('key to BaseVocab.__getitem__ must be (iterable of) string or integer')
    return
  
  def __setitem__(self, key, value):
    if isinstance(key, basestring):
      if not self.cased and key not in self._special_tokens_set:
        key = key.lower()
      self._str2idx[key] = value
      self._idx2str[value] = key
    elif isinstance(key, (int, long)):
      if not self.cased and value not in self._special_tokens_set:
        value = value.lower()
      self._idx2str[key] = value
      self._str2idx[value] = key
    elif hasattr(key, '__iter__') and hasattr(value, '__iter__'):
      for k, v in zip(key, value):
        self[k] = v
    else:
      raise ValueError('keys and values to BaseVocab.__setitem__ must be (iterable of) string or integer')
  
  def __contains__(self, key):
    if isinstance(key, basestring):
      if not self.cased and key not in self._special_tokens_set:
        key = key.lower()
      return key in self._str2idx
    elif isinstance(key, (int, long)):
      return key in self._idx2str
    else:
      raise ValueError('key to BaseVocab.__contains__ must be string or integer')
    return
  
  def __len__(self):
    return len(self._str2idx)
  
  def __iter__(self):
    return (key for key in sorted(self._str2idx, key=self._str2idx.get))

  def __setattr__(self, name, value):
    if name in ('_str2idx', '_idx2str', '_str2idxs'):
      value = dict(value)
    elif name == '_counts':
      value = Counter(value)
    super(BaseVocab, self).__setattr__(name, value)
    return
  
#***************************************************************
if __name__ == '__main__':
  """"""
  
  base_vocab = BaseVocab()
  print('BaseVocab passes')
