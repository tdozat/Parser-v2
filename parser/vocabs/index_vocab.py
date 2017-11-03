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
import sys
from collections import Counter

import numpy as np
import tensorflow as tf

from parser import Configurable

__all__ = ['DepVocab', 'HeadVocab']

#***************************************************************
class IndexVocab(Configurable):
  """"""
  
  ROOT = 0
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(IndexVocab, self).__init__(*args, **kwargs)
    self.placeholder = None
  
  #=============================================================
  def generate_placeholder(self):
    """"""
    
    if self.placeholder is None:
      self.placeholder = tf.placeholder(tf.int32, shape=[None, None], name=self.name)
    return self.placeholder
  
  #=============================================================
  def set_feed_dict(self, data, feed_dict):
    """"""
    
    feed_dict[self.placeholder] = data
    return
  
  #=============================================================
  def setup(self):
    self.placeholder = None
    return

  #=============================================================
  def index(self, token):
    return 0 if token == '_' else int(token)
  
  #=============================================================
  @property
  def depth(self):
    return None
  @property
  def conll_idx(self):
    return self._conll_idx

  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, basestring):
      return int(key)
    elif isinstance(key, (int, long, np.int32, np.int64)):
      return str(key)
    elif hasattr(key, '__iter__'):
      return [self[k] for k in key]
    else:
      raise ValueError('key to BaseVocab.__getitem__ must be (iterable of) string or integer')
    return

#***************************************************************
class DepVocab(IndexVocab):
  _conll_idx = 0
class HeadVocab(IndexVocab):
  _conll_idx = 6
