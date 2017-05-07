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

import numpy as np
import tensorflow as tf

from parser.vocabs import TokenVocab, Multivocab
from parser.neural.models import NN

#***************************************************************
class BaseEmbed(NN):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(BaseEmbed, self).__init__(*args, **kwargs)
    # This placeholder represents the data in the bucket that called BaseEmbed.__init__
    self.placeholder = None
    return
  
  #=============================================================
  def __call__(self, vocab, keep_prob=None, moving_params=None):
    """"""
    
    self.moving_params = moving_params
    if isinstance(vocab, Multivocab):
      multivocab = vocab
      self.generate_placeholder([None,None,None])
      embeddings = [TokenVocab.__call__(vocab, self.placeholder[:,:,i]) for i, vocab in enumerate(multivocab)]
      embeddings = tf.stack(embeddings, axis=2)
      # (n x b x g x d) -> (n x b x d)
      with tf.variable_scope('Pre-Attn'):
        embeddings = self.linear_attention(embeddings)
      self._tokens_to_keep = tf.to_float(tf.greater(self.placeholder[:,:,0], vocab.PAD))
    else:
      self.generate_placeholder([None,None])
      # (n x b x d)
      embeddings = TokenVocab.__call__(vocab, self.placeholder)
      self._tokens_to_keep = tf.to_float(tf.greater(self.placeholder, vocab.PAD))
    self._batch_size = tf.shape(self.placeholder)[0]
    self._bucket_size = tf.shape(self.placeholder)[1]
    self._sequence_lengths = tf.to_int32(tf.reduce_sum(self.tokens_to_keep, axis=1))
    self._n_tokens = tf.reduce_sum(self.sequence_lengths)
    return embeddings
  
  #=============================================================
  def generate_placeholder(self, shape):
    if self.placeholder is None:
      self.placeholder = tf.placeholder(tf.int32, shape=shape, name='%s-bkt' % self.name)
    return self.placeholder
