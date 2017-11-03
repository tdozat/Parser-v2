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

from parser.neural import linalg
from parser.configurable import Configurable

#***************************************************************
class NN(Configurable):
  """"""
  
  ZERO = tf.convert_to_tensor(0.)
  ONE = tf.convert_to_tensor(1.)
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(NN, self).__init__(*args, **kwargs)
    
    self._tokens_to_keep = None
    self._sequence_lengths = None
    self._n_tokens = None
    self._batch_size = None
    self._bucket_size = None
    self.moving_params = None
    return
  
  #=============================================================
  def embed_concat(self, vocabs, vocabs_to_merge=[['words', 'lemmas'], ['tags', 'xtags']]):
    """"""
    
    merge_dict = {vocab.name:vocab.name for vocab in vocabs}
    for vocab1, vocab2 in vocabs_to_merge:
      merge_dict[vocab2] = vocab1
    if self.moving_params is None:
      placeholders = []
      drop_masks = []
      for vocab in vocabs:
        placeholder = vocab.generate_placeholder()
        if len(placeholder.get_shape().as_list()) == 3:
          placeholder = tf.unstack(placeholder, axis=2)[0]
        placeholders.append(placeholder)
        if merge_dict[vocab.name] == vocab.name:
          drop_mask = tf.expand_dims(linalg.random_mask(vocab.embed_keep_prob, tf.shape(placeholder)), 2)
          drop_masks.append(drop_mask)
      for placeholder in placeholders:
        print(placeholder.graph)
      total_masks = tf.add_n(drop_masks)
      scale_mask = len(drop_masks) / tf.maximum(total_masks, 1.)
    embed_dict = {}
    for vocab in vocabs:
      if merge_dict[vocab.name] in embed_dict:
        embed_dict[merge_dict[vocab.name]] += vocab(moving_params=self.moving_params)
      else:
        embed_dict[merge_dict[vocab.name]] = vocab(moving_params=self.moving_params)
    embeddings = []
    i = 0
    for vocab in vocabs:
      if vocab.name in embed_dict:
        embedding = embed_dict[vocab.name]
        if self.moving_params is None:
          embedding *= drop_masks[i]*scale_mask
        embeddings.append(embedding)
        i += 1
    return tf.concat(embeddings, 2)
  
  #=============================================================
  def linear(self, inputs, output_size, keep_prob=None, n_splits=1, add_bias=True, initializer=tf.zeros_initializer()):
    """"""
    
    if isinstance(inputs, (list, tuple)):
      n_dims = len(inputs[0].get_shape().as_list())
      inputs = tf.concat(inputs, n_dims-1)
    else:
      n_dims = len(inputs.get_shape().as_list())
    input_size = inputs.get_shape().as_list()[-1]
    
    if self.moving_params is None:
      keep_prob = keep_prob or self.mlp_keep_prob
    else:
      keep_prob = 1
    if keep_prob < 1:
      noise_shape = tf.stack([self.batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    lin = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits,
                        add_bias=add_bias,
                        initializer=initializer,
                        moving_params=self.moving_params)
    
    if output_size == 1:
      if isinstance(lin, list):
        lin = [tf.squeeze(x, axis=(n_dims-1)) for x in lin]
      else:
        lin = tf.squeeze(lin, axis=(n_dims-1))
    return lin
  
  #=============================================================
  def bilinear(self, inputs1, inputs2, output_size, keep_prob=None, n_splits=1, add_bias1=True, add_bias2=True, initializer=tf.zeros_initializer()):
    """"""
    
    if isinstance(inputs1, (list, tuple)):
      n_dims1 = len(inputs1[0].get_shape().as_list())
      inputs1 = tf.concat(inputs1, n_dims-1)
    else:
      n_dims1 = len(inputs1.get_shape().as_list())
    inputs1_size = inputs1.get_shape().as_list()[-1]
    
    if isinstance(inputs2, (list, tuple)):
      n_dims2 = len(inputs2[0].get_shape().as_list())
      inputs2 = tf.concat(inputs2, n_dims-1)
    else:
      n_dims2 = len(inputs2.get_shape().as_list())
    inputs2_size = inputs2.get_shape().as_list()[-1]
    try:
      assert n_dims1 == n_dims2
    except AssertionError:
      raise ValueError('Inputs1 and Inputs2 to bilinear have different no. of dims')
    
    if self.moving_params is None:
      keep_prob = keep_prob or self.mlp_keep_prob
    else:
      keep_prob = 1
    if keep_prob < 1:
      noise_shape1 = tf.stack([self.batch_size] + [1]*(n_dims1-2) + [inputs1_size])
      noise_shape2 = tf.stack([self.batch_size] + [1]*(n_dims2-2) + [inputs2_size])
      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape1)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape2)
    
    bilin = linalg.bilinear(inputs1, inputs2, output_size,
                            n_splits,
                            add_bias1=add_bias1,
                            add_bias2=add_bias2,
                            initializer=initializer,
                            moving_params=self.moving_params)
    
    if output_size == 1:
      if isinstance(bilin, list):
        bilin = [tf.squeeze(x, axis=(n_dims1-1)) for x in bilin]
      else:
        bilin = tf.squeeze(bilin, axis=(n_dims1-1))
    return bilin

  #=============================================================
  def convolutional(self, inputs, window_size, output_size, keep_prob=None, n_splits=1, add_bias=True, initializer=None):
    """"""
    
    if isinstance(inputs, (list, tuple)):
      n_dims = len(inputs[0].get_shape().as_list())
      inputs = tf.concat(inputs, n_dims-1)
    else:
      n_dims = len(inputs.get_shape().as_list())
    input_size = inputs.get_shape().as_list()[-1]
    
    if self.moving_params is None:
      keep_prob = keep_prob or self.conv_keep_prob
    else:
      keep_prob = 1
      
    if keep_prob < 1:
      inputs = tf.nn.dropout(inputs, keep_prob)
    
    conv = linalg.convolutional(inputs,
                                window_size,
                                output_size,
                                n_splits=n_splits,
                                add_bias=add_bias,
                                initializer=initializer,
                                moving_params=self.moving_params)
    
    if output_size == 1:
      if isinstance(conv, list):
        conv = [tf.squeeze(x, axis=(n_dims-1)) for x in conv]
      else:
        conv = tf.squeeze(conv, axis=(n_dims-1))
    return conv
  
  #=============================================================
  def MLP(self, inputs, output_size=None, keep_prob=None, n_splits=1, add_bias=True):
    """"""
    
    if output_size is None:
      output_size = self.mlp_size
    
    if self.mlp_func.__name__.startswith('gated'):
      output_size *= 2
    linear = self.linear(inputs, output_size, keep_prob=keep_prob, n_splits=n_splits, add_bias=add_bias, initializer=None)
    
    if isinstance(linear, list):
      return [self.mlp_func(lin) for lin in linear]
    else:
      return self.mlp_func(linear)
  
  #=============================================================
  def biMLP(self, inputs1, inputs2, output_size=None, keep_prob=None, n_splits=1, add_bias1=True, add_bias2=True):
    """"""
    
    if output_size is None:
      output_size = self.mlp_size
    
    if self.mlp_func.__name__.startswith('gated'):
      output_size *= 2
    bilinear = self.bilinear(inputs1, inputs2, output_size, keep_prob=keep_prob, n_splits=n_splits, add_bias1=add_bias, add_bias2=add_bias, initializer=None)
    
    if isinstance(bilinear, list):
      return [self.mlp_func(bilin) for bilin in bilinear]
    else:
      return self.mlp_func(bilinear)
  
  #=============================================================
  def CNN(self, inputs, window_size, output_size, keep_prob=None, n_splits=1, add_bias=True):
    """"""
    
    if window_size is None:
      window_size = self.window_size
    if output_size is None:
      output_size = self.mlp_size
    
    if self.conv_func.__name__.startswith('gated'):
      output_size *= 2
    convolutional = self.convolutional(inputs, window_size, output_size, keep_prob=keep_prob, n_splits=n_splits, add_bias=add_bias, initializer=None)
    
    if isinstance(convolutional, list):
      return [self.conv_func(conv) for conv in convolutional]
    else:
      return self.conv_func(convolutional)
  
  #=============================================================
  def linear_attention(self, inputs):
    """"""
    
    attn_logits = self.linear(inputs, 1, add_bias=False)
    # (n x b)
    attn_prob = tf.nn.softmax(attn_logits)
    attn_prob = tf.expand_dims(attn_prob, -1)
    # (n x b x d).T (n x b x 1) -> (n x d x 1)
    outputs = tf.matmul(inputs, attn_prob, transpose_a=True)
    outputs = tf.squeeze(outputs, axis=-1)
    return outputs
  
  #=============================================================
  def bilinear_attention(self, inputs):
    """"""
    
    attn_logits = self.bilinear(inputs, inputs, 1, add_bias2=False, add_bias=False)
    # (n x b x b)
    attn_prob = tf.nn.softmax(attn_logits)
    # (n x b x b) (n x b x d) -> (n x b x d)
    outputs = tf.matmul(attn_prob, inputs)
    return outputs
  
  #=============================================================
  def RNN(self, inputs, output_size):
    """"""
    
    input_size = inputs.get_shape().as_list()[-1]
    cell = self.recur_cell.from_configurable(self, output_size, input_size=input_size, moving_params=self.moving_params)
    
    if self.moving_params is None:
      ff_keep_prob = self.ff_keep_prob
      recur_keep_prob = self.recur_keep_prob
    else:
      ff_keep_prob = 1
      recur_keep_prob = 1
    
    top_recur, end_state = self.rnn_func(cell, inputs, self.sequence_lengths,
                                 ff_keep_prob=ff_keep_prob,
                                 recur_keep_prob=recur_keep_prob)
    return top_recur, end_state
  
  #=============================================================
  def __call__(self, inputs, targets, moving_params=None):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  @property
  def tokens_to_keep(self):
    return self._tokens_to_keep
  @property
  def batch_size(self):
    return self._batch_size
  @property
  def bucket_size(self):
    return self._bucket_size
  @property
  def sequence_lengths(self):
    return self._sequence_lengths
  @property
  def n_tokens(self):
    return self._n_tokens
