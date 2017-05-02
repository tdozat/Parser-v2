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

from parser.configurable import Configurable

#***************************************************************
class Bucket(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    embed_model = kwargs.pop('embed_model', None)
    super(Bucket, self).__init__(*args, **kwargs)
    
    self._indices = []
    self._maxlen = 0
    self._depth = 1
    if embed_model is not None:
      self._embed_model = embed_model.from_configurable(self, name=self.name)
    else:
      self._embed_model = None
    return
  
  #=============================================================
  def __call__(self, vocab, keep_prob=None, moving_params=None):
    """"""
    
    return self.embed_model(vocab, keep_prob=keep_prob, moving_params=moving_params)
  
  #=============================================================
  def open(self, maxlen, depth=None):
    """"""
    
    if depth is None:
      self._indices = [[0]]
    else:
      self._indices = [[[0]*depth]]
    self._maxlen = maxlen
    self._depth = depth
    return self
  
  #=============================================================
  def add(self, idxs):
    """"""
    
    if isinstance(self.indices, np.ndarray):
      raise TypeError("The bucket has already been closed, you can't add to it")
    if len(idxs) > len(self) and len(self) != -1:
      raise ValueError('Bucket of max len %d received sequence of len %d' % (len(self), len(idxs)))
    
    self.indices.append(idxs)
    return len(self.indices) - 1
  
  #=============================================================
  def close(self):
    """"""
    
    if self.depth is None:
      indices = np.zeros((len(self.indices), len(self)), dtype=np.int32)
      for i, sequence in enumerate(self.indices):
        indices[i,0:len(sequence)] = sequence 
    else:
      indices = np.zeros((len(self.indices), len(self), self.depth), dtype=np.int32)
      for i, sequence in enumerate(self.indices):
        for j, index in enumerate(sequence):
          indices[i,j,0:len(index)] = index
    self._indices = indices
    #if self.verbose:
    #  print('Bucket %s is %s' % (self.name, ' x '.join(str(x) for x in self.indices.shape)))
  
  #=============================================================
  @classmethod
  def from_dataset(cls, dataset, bkt_idx, *args, **kwargs):
    """"""
    
    kwargs = dict(kwargs)
    kwargs['name'] = '{name}-{bkt_idx}'.format(name=dataset.name, bkt_idx=bkt_idx)
    bucket = cls.from_configurable(dataset, *args, **kwargs)
    indices = []
    for multibucket in dataset:
      indices.append(multibucket[bkt_idx].indices)
    for i in xrange(len(indices)):
      if len(indices[i].shape) == 2:
        indices[i] = indices[i][:,:,None]
    bucket._indices = np.concatenate(indices, axis=2)
    bucket._maxlen = bucket.indices.shape[1]
    bucket._depth = bucket.indices.shape[2]
    return bucket
    
  #=============================================================
  @property
  def indices(self):
    return self._indices
  @property
  def embed_model(self):
    return self._embed_model
  @property
  def depth(self):
    return self._depth
  @property
  def placeholder(self):
    return self.embed_model.placeholder

  #=============================================================
  def __len__(self):
    return self._maxlen
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, trace):
    if exception_type is not None:
      raise exception_type(exception_value)
    self.close()
    return
