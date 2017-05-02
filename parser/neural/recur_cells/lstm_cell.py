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

import tensorflow as tf

from parser.neural.recur_cells.base_cell import BaseCell
from parser.neural.linalg import linear
from parser.neural.functions import gate, tanh

#***************************************************************
class LSTMCell(BaseCell):
  """"""
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    with tf.variable_scope(scope or type(self).__name__):
      cell_tm1, hidden_tm1 = tf.split(state, 2, axis=1)
      input_list = [inputs, hidden_tm1]
      lin = linear(input_list,
                   self.output_size,
                   add_bias=True,
                   n_splits=4,
                   moving_params=self.moving_params)
      cell_act, input_act, forget_act, output_act = lin
      
      cell_tilde_t = tanh(cell_act)
      input_gate =  gate(input_act)
      forget_gate = gate(forget_act-self.forget_bias)
      output_gate = gate(output_act)
      cell_t = input_gate * cell_tilde_t + (1-forget_gate) * cell_tm1
      hidden_tilde_t = self.recur_func(cell_t)
      hidden_t = hidden_tilde_t * output_gate
      
      return hidden_t, tf.concat([cell_t, hidden_t], 1)
  
  #=============================================================
  @property
  def state_size(self):
    return self.output_size * 2
