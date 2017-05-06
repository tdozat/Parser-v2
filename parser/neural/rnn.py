# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""RNN helpers for TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import parser.neural.linalg as linalg

#===============================================================
def birnn(cell, inputs, sequence_length, initial_state_fw=None, initial_state_bw=None, ff_keep_prob=1., recur_keep_prob=1., dtype=tf.float32, scope=None):
  """"""
  
  # Forward direction
  with tf.variable_scope(scope or 'BiRNN_FW') as fw_scope:
    output_fw, output_state_fw = rnn(cell, inputs, sequence_length, initial_state_fw, ff_keep_prob, recur_keep_prob, dtype, scope=fw_scope)

  # Backward direction
  rev_inputs = tf.reverse_sequence(inputs, sequence_length, 1, 0)
  with tf.variable_scope(scope or 'BiRNN_BW') as bw_scope:
    output_bw, output_state_bw = rnn(cell, rev_inputs, sequence_length, initial_state_bw, ff_keep_prob, recur_keep_prob, dtype, scope=bw_scope)
  output_bw = tf.reverse_sequence(output_bw, sequence_length, 1, 0)
  # Concat each of the forward/backward outputs
  outputs = tf.concat([output_fw, output_bw], 2)

  return outputs, tf.tuple([output_state_fw, output_state_bw])

#===============================================================
def rnn(cell, inputs, sequence_length=None, initial_state=None, ff_keep_prob=1., recur_keep_prob=1., dtype=tf.float32, scope=None):
  """"""
  
  inputs = tf.transpose(inputs, [1, 0, 2])  # (B,T,D) => (T,B,D)
  
  parallel_iterations = 32
  if sequence_length is not None:
    sequence_length = tf.to_int32(sequence_length)
  
  with tf.variable_scope(scope or 'RNN') as varscope:
    #if varscope.caching_device is None:
    #  varscope.set_caching_device(lambda op: op.device)
    input_shape = tf.shape(inputs)
    time_steps, batch_size, _ = tf.unstack(input_shape, 3)
    const_time_steps, const_batch_size, const_depth = inputs.get_shape().as_list()
    
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError('If no initial_state is provided, dtype must be.')
      state = cell.zero_state(batch_size, dtype)
    
    zero_output = tf.zeros(tf.stack([batch_size, cell.output_size]), inputs.dtype)
    if sequence_length is not None:
      min_sequence_length = tf.reduce_min(sequence_length)
      max_sequence_length = tf.reduce_max(sequence_length)
    
    time = tf.constant(0, dtype=tf.int32, name='time')
    
    output_ta = tf.TensorArray(dtype=inputs.dtype,
                               size=time_steps,
                               tensor_array_name='dynamic_rnn_output')
    
    input_ta = tf.TensorArray(dtype=inputs.dtype,
                              size=time_steps,
                              tensor_array_name='dynamic_rnn_input')
    
    if ff_keep_prob < 1:
      noise_shape = tf.stack([1, batch_size, const_depth])
      inputs = tf.nn.dropout(inputs, ff_keep_prob, noise_shape=noise_shape)
      
    if recur_keep_prob < 1:
      ones = tf.ones(tf.stack([batch_size, cell.output_size]))
      state_dropout = tf.nn.dropout(ones, recur_keep_prob)
      state_dropout = tf.concat([ones] * (cell.state_size // cell.output_size - 1) + [state_dropout], 1)
    else:
      state_dropout = 1
      
    input_ta = input_ta.unstack(inputs)
    
    #-----------------------------------------------------------
    def _time_step(time, state, output_ta_t):
      """"""
      
      input_t = input_ta.read(time)
      
      #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      def _empty_update():
        return zero_output, state
      #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      def _call_cell():
        return cell(input_t, state * state_dropout)
      #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      def _maybe_copy_some_through():
        new_output, new_state = _call_cell()
        
        return tf.cond(
          time < min_sequence_length,
          lambda: (new_output, new_state),
          lambda: (tf.where(time >= sequence_length, zero_output, new_output),
                   tf.where(time >= sequence_length, state, new_state)))
      #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      
      if sequence_length is not None:
        output, new_state = tf.cond(
          time >= max_sequence_length,
          _empty_update,
          _maybe_copy_some_through)
      else:
        (output, new_state) = _call_cell()
      
      output_ta_t = output_ta_t.write(time, output)
      
      return (time + 1, new_state, output_ta_t)
    #-----------------------------------------------------------
    
    _, final_state, output_final_ta = tf.while_loop(
      cond=lambda time, _1, _2: time < time_steps,
      body=_time_step,
      loop_vars=(time, state, output_ta),
      parallel_iterations=parallel_iterations)
    
    final_outputs = output_final_ta.stack()
    
    outputs = tf.transpose(final_outputs, [1, 0, 2])  # (T,B,D) => (B,T,D)
    return outputs, final_state
