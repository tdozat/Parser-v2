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
import scipy.linalg as la
import tensorflow as tf

from parser.vocabs.base_vocab import BaseVocab

#***************************************************************
class RetrainedVocab(BaseVocab):
  """"""
  
  #=============================================================
  def __init__(self, pretrained_vocab, *args, **kwargs):
    """"""
    
    super(RetrainedVocab, self).__init__(*args, **kwargs)
    
    self._pretrained_vocab = pretrained_vocab
    return
  
  #=============================================================
  def __call__(self):
    """"""
    
    embed_size = self.embed_size
    row_idxs = tf.placeholder(tf.int32, shape=(None,), name='row_idxs')
    col_idxs = tf.placeholder(tf.int32, shape=(None,), name='col_idxs')
    S, U, _ = tf.svd(self.pretrained_vocab.embeddings)
    self.embeddings = U[:,:embed_size] * S[:embed_size]
    
    old_rows = tf.gather(self.pretrained_vocab.embeddings, row_idxs)
    old_cols = tf.gather(self.pretrained_vocab.embeddings, col_idxs)
    new_rows = tf.gather(self.embeddings, row_idxs)
    new_cols = tf.gather(self.embeddings, col_idxs)
    old_matmul = tf.matmul(old_rows, old_cols, transpose_b=True)
    new_matmul = tf.matmul(new_rows, new_cols, transpose_b=True)
    
    if self.embed_loss == 'cross_entropy':
      old_matmul = tf.expand_dims(tf.nn.softmax(old_matmul), axis=1)
      new_matmul = tf.expand_dims(tf.nn.softmax(new_matmul), axis=2)
      loss = -tf.reduce_sum(tf.matmul(old_matmul, tf.log(new_matmul))) / tf.to_float(tf.shape(row_idxs)[0])
    elif self.embed_loss == 'l2_loss':
      loss = tf.reduce_sum((old_matmul - new_matmul)**2 / 2) / tf.to_float(tf.shape(row_idxs)[0])
    else:
      raise ValueError('embed_loss must be in "(cross_entropy, l2_loss)"')
    
    return {'row_idxs': row_idxs,
            'col_idxs': col_idxs,
            'loss': loss}
  
  #=============================================================
  def dump(self):
    """"""
    
    matrix = self.embeddings.eval()
    with codecs.open(self.name+'.txt', 'w') as f:
      for idx in xrange(self.START_IDX, len(self)):
        f.write('%s %s\n' % (self[idx], ' '.join(matrix[idx])))
    return
  
  #=============================================================
  @property
  def pretrained_vocab(self):
    return self._pretrained_vocab
  
  #=============================================================
  def __setattr__(self, name, value):
    if name == '_pretrained_vocab':
      self._str2idx = value._str2idx
      self._idx2str = value._idx2str
      self._counts = value._counts
    super(RetrainedVocab, self).__setattr__(name, value)

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser import Configurable
  from parser.vocabs import PretrainedVocab
  configurable = Configurable(retrained_vocab={'embed_loss':'cross_entropy', 'retrained_embed_size':50})
  pretrained_vocab = PretrainedVocab.from_configurable(configurable)
  retrained_vocab = RetrainedVocab.from_vocab(pretrained_vocab)
  retrain_loss = retrained_vocab(pretrained_vocab)
  print('RetrainedVocab passes')
