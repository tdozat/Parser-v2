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
from parser.neural.functions import gate

#***************************************************************
class GRUCell(BaseCell):
  """"""
  
  #=============================================================
  def __call__(self, inputs, state, scope=None):
    """"""
    
    with tf.variable_scope(scope or type(self).__name__):
      cell_tm1, hidden_tm1 = tf.split(state, 2, axis=1)
      input_list = [inputs, hidden_tm1]
      with tf.variable_scope('Gates'):
        gates = linear(inputs_list,
                       self.output_size,
                       add_bias=True,
                       n_splits=2,
                       moving_params=self.moving_params)
        update_act, reset_act = gates
        update_gate = gate(update_act-self.forget_bias)
        reset_gate = gate(reset_act)
        reset_state = reset_gate * hidden_tm1
      input_list = [inputs, reset_state]
      with tf.variable_scope('Candidate'):
        hidden_act = linear(input_list,
                            self.output_size,
                            add_bias=True,
                            moving_params=self.moving_params)
        hidden_tilde = self.recur_func(hidden_act)
      cell_t = update_gate * cell_tm1 + (1-update_gate) * hidden_tilde
    return cell_t, tf.concat([cell_t, cell_t], 1)
  
  #=============================================================
  @property
  def state_size(self):
    return self.output_size * 2
