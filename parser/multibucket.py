#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from parser import Configurable
from parser import Bucket
from parser.misc.colors import ctext

#***************************************************************
class Multibucket(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    self._embed_model = kwargs.pop('embed_model', None)
    super(Multibucket, self).__init__(*args, **kwargs)
    
    self._indices = []
    self._buckets = []
    self._len2idx = {}
    self.placeholder = None
    return
  
  #=============================================================
  def __call__(self, vocab, keep_prob=None, moving_params=None):
    """"""
    
    # This placeholder is used to ensure the bucket data is in the right order
    reuse = None if moving_params is None else True
    self.generate_placeholder()
    embeddings = []
    for i, bucket in enumerate(self):
      if i > 0:
        reuse = True
      with tf.variable_scope(self.name+'-multibucket', reuse=reuse):
        embeddings.append(bucket(vocab, keep_prob=keep_prob, moving_params=moving_params))
    return tf.nn.embedding_lookup(tf.concat(embeddings, axis=0), self.placeholder)
  
  #=============================================================
  def reset_placeholders(self):
    self.placeholder = None
    for bucket in self:
      bucket.reset_placeholders()
    return

  #=============================================================
  def generate_placeholder(self):
    """"""
    
    if self.placeholder is None:
      self.placeholder = tf.placeholder(tf.int32, shape=(None,), name=self.name+'-multibucket')
    return self.placeholder
  
  #=============================================================
  def open(self, maxlens, depth=None):
    """"""
    
    self._indices = [(0,0)]
    self._buckets = []
    self._len2idx = {}
    prevlen = -1
    for idx, maxlen in enumerate(maxlens):
      self._buckets.append(Bucket.from_configurable(self, embed_model=self.embed_model, name='%s-%d' % (self.name, idx)).open(maxlen, depth=depth))
      self._len2idx.update(zip(range(prevlen+1, maxlen+1), [idx]*(maxlen-prevlen)))
      prevlen = maxlen
    return self
  
  #=============================================================
  def add(self, idxs, tokens=None):
    """"""
    
    if isinstance(self.indices, np.ndarray):
      raise TypeError("The buckets have already been closed, you can't add to them")
    
    idx = self._len2idx.get(len(idxs), len(self)-1)
    bkt_idx = self[idx].add(idxs, tokens=tokens)
    self.indices.append( (idx, bkt_idx) )
    return len(self.indices) - 1
  
  #=============================================================
  def close(self):
    """"""
    
    for bucket in self:
      bucket.close()
    
    self._indices = np.array(self.indices, dtype=[('bkt_idx', 'i4'), ('idx', 'i4')])
    return
  
  #=============================================================
  def inv_idxs(self):
    """"""
    
    return np.argsort(np.concatenate([np.where(self.indices['bkt_idx'][1:] == bkt_idx)[0] for bkt_idx in xrange(len(self))]))
  
  #=============================================================
  def get_tokens(self, bkt_idx, batch):
    """"""

    return self[bkt_idx].get_tokens(batch)

  #=============================================================
  @classmethod
  def from_dataset(cls, dataset, *args, **kwargs):
    """"""
    
    multibucket = cls.from_configurable(dataset, *args, **kwargs)
    indices = []
    for multibucket_ in dataset:
      indices.append(multibucket_.indices)
    #for i in xrange(1, len(indices)):
    #  assert np.equal(indices[0].astype(int), indices[i].astype(int)).all()
    multibucket._indices = np.array(multibucket_.indices)
    buckets = [Bucket.from_dataset(dataset, i, *args, **kwargs) for i in xrange(len(multibucket_))]
    multibucket._buckets = buckets
    if dataset.verbose:
      for bucket in multibucket:
        print('Bucket {name} is {shape}'.format(name=bucket.name, shape=ctext(' x '.join(str(x) for x in bucket.indices.shape), 'bright_blue')))
    return multibucket
  
  #=============================================================
  @property
  def indices(self):
    return self._indices
  @property
  def embed_model(self):
    return self._embed_model
  
  #=============================================================
  def __str__(self):
    return str(self._buckets)
  def __iter__(self):
    return (bucket for bucket in self._buckets)
  def __getitem__(self, key):
    return self._buckets[key]
  def __len__(self):
    return len(self._buckets)
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, trace):
    if exception_type is not None:
      raise exception_type(exception_value)
    self.close()
    return
