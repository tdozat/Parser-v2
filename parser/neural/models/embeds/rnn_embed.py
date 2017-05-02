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

from parser.neural.models.embeds.base_embed import BaseEmbed

#***************************************************************
class RNNEmbed(BaseEmbed):
  """"""
  
  #=============================================================
  def __call__(self, *args, **kwargs):
    """"""
    
    # (n x b x d)
    embeddings = super(RNNEmbed, self).__call__(*args, **kwargs)
    # (n x b x d) -> (n x b x h)
    with tf.variable_scope('RNN'):
      recur = self.RNN(embeddings, self.recur_size)
    # (n x b x h) -> (n x h)
    with tf.variable_scope('MLP'):
      hidden = self.linear_attention(recur)
    # (n x h) -> (n x o)
    linear = self.linear(hidden, vocab.token_embed_size)
    return linear 