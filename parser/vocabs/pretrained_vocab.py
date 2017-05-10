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
import gzip
from backports import lzma
from collections import Counter

import numpy as np
import tensorflow as tf

import parser.neural.linalg as linalg
from parser.vocabs.base_vocab import BaseVocab

#***************************************************************
class PretrainedVocab(BaseVocab):
  """"""
  
  #=============================================================
  def __init__(self, token_vocab, *args, **kwargs):
    """"""
    
    super(PretrainedVocab, self).__init__(*args, **kwargs)
    
    self._token_vocab = token_vocab
    
    self.load()
    self.count()
    return
  
  #=============================================================
  def __call__(self, placeholder=None, moving_params=None):
    """"""
    
    embeddings = super(PretrainedVocab, self).__call__(placeholder, moving_params=moving_params)
    # (n x b x d') -> (n x b x d)
    with tf.variable_scope(self.name.title()):
      matrix = linalg.linear(embeddings, self.token_embed_size, moving_params=moving_params)
      if moving_params is None:
        with tf.variable_scope('Linear', reuse=True):
          weights = tf.get_variable('Weights')
          tf.losses.add_loss(tf.nn.l2_loss(tf.matmul(tf.transpose(weights), weights) - tf.eye(self.token_embed_size)))
    return matrix
    #return embeddings # changed in saves2/test8
  
  #=============================================================
  def load(self):
    """"""
    
    embeddings = []
    cur_idx = len(self.special_tokens)
    max_rank = self.max_rank
    if self.filename.endswith('.xz'):
      open_func = lzma.open
    else:
      open_func = codecs.open
    with open_func(self.filename, 'rb') as f:
      reader = codecs.getreader('utf-8')(f, errors='ignore')
      if self.skip_header == True:
        reader.readline()
      for line_num, line in enumerate(reader):
        if (not max_rank) or line_num < max_rank:
          line = line.rstrip().split(' ')
          if len(line) > 1:
            embeddings.append(np.array(line[1:], dtype=np.float32))
            self[line[0]] = cur_idx
            cur_idx += 1
        else:
          break
    try:
      embeddings = np.stack(embeddings)
      embeddings = np.pad(embeddings, ( (len(self.special_tokens),0), (0,0) ), 'constant')
      self.embeddings = np.stack(embeddings)
    except:
      shapes = set([embedding.shape for embedding in embeddings])
      raise ValueError("Couldn't stack embeddings with shapes in %s" % shapes)
    return
  
  #=============================================================
  def count(self):
    """"""
    
    if self.token_vocab is not None:
      zipf = self.token_vocab.fit_to_zipf(plot=False)
      zipf_freqs = zipf.predict(np.arange(len(self))+1)
    else:
      zipf_freqs = -np.log(np.arange(len(self))+1)
    zipf_counts = zipf_freqs / np.min(zipf_freqs)
    for count, token in zip(zipf_counts, self.strings()):
      self.counts[token] = int(count)
    return
  
  #=============================================================
  @property
  def token_vocab(self):
    return self._token_vocab
  @property
  def token_embed_size(self):
    return (self.token_vocab or self).embed_size
  @property
  def embeddings(self):
    return super(PretrainedVocab, self).embeddings
  @embeddings.setter
  def embeddings(self, matrix):
    self._embed_size = matrix.shape[1]
    with tf.device('/cpu:0'):
      with tf.variable_scope(self.name.title()):
        self._embeddings = tf.Variable(matrix, name='Embeddings', trainable=False)
    return

#***************************************************************
if __name__ == '__main__':
  """"""
  
  pretrained_vocab = PretrainedVocab(None)
  print('PretrainedVocab passes')
