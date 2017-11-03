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
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser import Configurable
from parser.neural import linalg
from parser.vocabs import TokenVocab, SubtokenVocab

__all__ = ['Multivocab']

#***************************************************************
class Multivocab(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, vocabs, *args, **kwargs):
    """"""
    
    super(Multivocab, self).__init__(*args, **kwargs)
    
    self._vocabs = vocabs
    self._set_special_tokens()
    # NOTE Don't forget to run index_tokens() after adding test/validation files!
    self.placeholder = None
    return
  
  #=============================================================
  def __call__(self, placeholder=None, moving_params=None):
    """"""
    # TODO check to see if a word is all unk, and if so, replace it with a random vector
    
    embeddings = [vocab(moving_params=moving_params) for vocab in self]
    return tf.add_n(embeddings)
  
  #=============================================================
  def setup(self):
    """"""

    self.placeholder = None
    for vocab in self:
      vocab.setup()
    return

  #=============================================================
  def generate_placeholder(self):
    """"""
    
    if self.placeholder is None:
      self.placeholder = tf.stack([vocab.generate_placeholder() for vocab in self], axis=2)
    return self.placeholder
  
  #=============================================================
  def _set_special_tokens(self):
    pattern = re.compile('\W+', re.UNICODE)
    self._special_tokens = zip(*[vocab.special_tokens for vocab in self])
    for i, token in enumerate(self.special_tokens):
      n = len(token)
      assert len(set(token)) == 1
      token = token[0]
      token = token.lstrip('<')
      token = token.rstrip('>')
      token = token.upper()
      token = pattern.sub('', token)
      assert token not in self.__dict__
      self.__dict__[token] = tuple(i for _ in xrange(n))
    return
  
  #=============================================================
  def add_files(self, conll_files):
    """"""
    
    conll_files = list(conll_files)
    token_vocabs = []
    for vocab in self:
      if hasattr(vocab, 'token_vocab'):
        if vocab.token_vocab not in token_vocabs:
          vocab.token_vocab.count(conll_files)
          token_vocabs.append(vocab.token_vocab)
    return
  
  #=============================================================
  def index_tokens(self):
    """"""
    
    for vocab in self:
      if hasattr(vocab, 'index_tokens'):
        vocab.index_tokens()
    return
  
  #=============================================================
  def set_feed_dict(self, data, feed_dict):
    """"""
    
    for i, vocab in enumerate(self):
      vocab.set_feed_dict(data[:,:,i], feed_dict)
    return
  
  #=============================================================
  def index(self, token):
    return tuple(vocab.index(token) for vocab in self)
  
  #=============================================================
  @property
  def depth(self):
    return len(self)
  @property
  def special_tokens(self):
    return self._special_tokens
  @property
  def conll_idx(self):
    return self._conll_idx
  
  #=============================================================
  def __iter__(self):
    return (vocab for vocab in self._vocabs)
  def __getitem__(self, key):
    return self._vocabs[key]
  def __len__(self):
    return len(self._vocabs)
  def __setattr__(self, key, value):
    if key == '_vocabs':
      conll_idxs = set([vocab.conll_idx for vocab in value if hasattr(vocab, 'conll_idx')]) 
      assert len(conll_idxs) == 1
      self._conll_idx = list(conll_idxs)[0]
    super(Multivocab, self).__setattr__(key, value)

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser.vocabs import PretrainedVocab, WordVocab, CharVocab, Multivocab
  
  configurable = Configurable()
  token_vocab = WordVocab.from_configurable(configurable)
  pretrained_vocab = PretrainedVocab.from_vocab(token_vocab)
  subtoken_vocab = CharVocab.from_vocab(token_vocab)
  multivocab = Multivocab.from_configurable(configurable, [pretrained_vocab, token_vocab, subtoken_vocab])
  multivocab.add_files(configurable.valid_files)
  multivocab.index_tokens()
  print("Indices for '<PAD>': %s" % str(multivocab.index('<PAD>')))
  print("Indices for 'the': %s" % str(multivocab.index('the')))
  print("Indices for 'The': %s" % str(multivocab.index('The')))
  print('Multivocab passes')
  
