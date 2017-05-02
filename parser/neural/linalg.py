#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#***************************************************************
def orthonormal_initializer(input_size, output_size):
  """"""
  
  if not tf.get_variable_scope().reuse:
    print(tf.get_variable_scope().name)
    I = np.eye(output_size)
    lr = .1
    eps = .05/(output_size + input_size)
    success = False
    while not success:
      Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
      for i in xrange(100):
        QTQmI = Q.T.dot(Q) - I
        loss = np.sum(QTQmI**2 / 2)
        Q2 = Q**2
        Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
        if np.isnan(Q[0,0]):
          lr /= 2
          break
      if np.isfinite(loss) and np.max(Q) < 1e6:
        success = True
      eps *= 2
    print('Orthogonal pretrainer loss: %.2e' % loss)
  else:
    Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
  return Q.astype(np.float32)

#===============================================================
def linear(inputs, output_size, n_splits=1, add_bias=True, initializer=None, moving_params=None):
  """"""
  
  # Prepare the input
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  n_dims = len(inputs[0].get_shape().as_list())
  all_inputs = tf.concat(inputs, n_dims-1)
  input_size = all_inputs.get_shape().as_list()[-1]
  
  # Prepare the output
  output_size *= n_splits
  output_shape = []
  shape = tf.shape(all_inputs)
  for i in xrange(n_dims-1):
    output_shape.append(shape[i])
  output_shape.append(output_size)
  output_shape = tf.stack(output_shape)
  
  all_inputs = tf.reshape(all_inputs, [-1, input_size])
  with tf.variable_scope('Linear'):
    # Get the matrix
    if initializer is None and not tf.get_variable_scope().reuse:
      mat = orthonormal_initializer(input_size, output_size//n_splits)
      mat = np.concatenate([mat]*n_splits, axis=1)
      initializer = tf.constant_initializer(mat)
    matrix = tf.get_variable('Weights', [input_size, output_size], initializer=initializer)
    if moving_params is not None:
      matrix = moving_params.average(matrix)
    else:
      tf.add_to_collection('Weights', matrix)
    
    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
      if moving_params is not None:
        bias = moving_params.average(bias)
    else:
      bias = 0
    
    # Do the multiplication
    lin = tf.matmul(all_inputs, matrix) + bias
    lin = tf.reshape(lin, output_shape)
    if n_splits > 1:
      return tf.split(lin, n_splits, n_dims-1)
    else:
      return lin
  
#===============================================================
def bilinear(inputs1, inputs2, output_size, n_splits=1, add_bias1=True, add_bias2=True, initializer=None, moving_params=None):
  """"""
  
  # Prepare the input
  if not isinstance(inputs1, (list, tuple)):
    inputs1 = [inputs1]
  n_dims1 = len(inputs1[0].get_shape().as_list())
  all_inputs1 = tf.concat(inputs1, n_dims1-1)
  inputs1_size = all_inputs1.get_shape().as_list()[-1]
  inputs1_bucket_size = tf.shape(all_inputs1)[-2]
  
  if not isinstance(inputs2, (list, tuple)):
    inputs2 = [inputs2]
  n_dims2 = len(inputs2[0].get_shape().as_list())
  all_inputs2 = tf.concat(inputs2, n_dims2-1)
  inputs2_size = all_inputs2.get_shape().as_list()[-1]
  inputs2_bucket_size = tf.shape(all_inputs2)[-2]
  
  # Prepare the output
  output_size *= n_splits
  output_shape = []
  shape1 = tf.shape(all_inputs1)
  for i in xrange(n_dims1-1):
    output_shape.append(shape1[i])
  output_shape.append(output_size)
  output_shape.append(inputs2_bucket_size)
  output_shape = tf.stack(output_shape)
  
  all_inputs1 = tf.reshape(all_inputs1, tf.stack([-1, inputs1_bucket_size, inputs1_size]))
  if add_bias1:
    bias1 = tf.ones(tf.stack([tf.shape(all_inputs1)[0], inputs1_bucket_size, 1]))
    all_inputs1 = tf.concat([all_inputs1, bias1], 2)
    inputs1_size += 1
  all_inputs2 = tf.reshape(all_inputs2, tf.stack([-1, inputs2_bucket_size, inputs2_size]))
  if add_bias2:
    bias2 = tf.ones(tf.stack([tf.shape(all_inputs2)[0], inputs2_bucket_size, 1]))
    all_inputs2 = tf.concat([all_inputs2, bias2], 2)
    inputs2_size += 1
  with tf.variable_scope('Bilinear'):
    # Get the matrix
    if initializer is None and tf.get_variable_scope().reuse is None:
      mat = orthonormal_initializer(inputs1_size, inputs2_size)[:,None,:]
      mat = np.concatenate([mat]*output_size, axis=1)
    weights = tf.get_variable('Weights', [inputs1_size, output_size, inputs2_size], initializer=initializer)
    if moving_params is not None:
      weights = moving_params.average(weights)
    else:
      tf.add_to_collection('Weights', weights)
    
    # Do the multiplication
    # (bn x d) (d x rd) -> (bn x rd)
    lin = tf.matmul(tf.reshape(all_inputs1, [-1, inputs1_size]),
                        tf.reshape(weights, [inputs1_size, -1]))
    # (b x nr x d) (b x n x d)T -> (b x nr x n)
    bilin = tf.matmul(tf.reshape(lin, tf.stack([-1, inputs1_bucket_size*output_size, inputs2_size])),
                      all_inputs2, transpose_b=True)
    # (bn x r x n)
    bilin = tf.reshape(bilin, output_shape)
    
    if n_splits > 1:
      return tf.split(bilin, n_splits, n_dims-2)
    else:
      return bilin

#===============================================================
def convolutional(inputs, window_size, output_size, n_splits=1, add_bias=True, initializer=None, moving_params=None):
  """"""
  
  # Prepare the input
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  n_dims = len(inputs[0].get_shape().as_list())
  all_inputs = tf.concat(inputs, n_dims-1)
  input_size = all_inputs.get_shape().as_list()[-1]
  bucket_size = tf.shape(all_inputs)[-2]
  
  # Prepare the output
  output_size *= n_splits
  output_shape = []
  shape = tf.shape(all_inputs)
  for i in xrange(n_dims-1):
    output_shape.append(shape[i])
  output_shape.append(output_size)
  output_shape = tf.stack(output_shape)
  
  all_inputs = tf.reshape(all_inputs, tf.stack([-1, bucket_size, input_size]))
  with tf.variable_scope('Convolutional'):
    # Get the matrix
    if initializer is None and not tf.get_variable_scope().reuse:
      mat = orthonormal_initializer(input_size*window_size, output_size//n_splits)
      mat = np.concatenate([mat]*n_splits, axis=1)
      mat = np.reshape(mat, [window_size, input_size, output_size])
      initializer = tf.constant_initializer(mat)
    matrix = tf.get_variable('Weights', [window_size, input_size, output_size], initializer=initializer)
    if moving_params is not None:
      matrix = moving_params.average(matrix)
    else:
      tf.add_to_collection('Weights', matrix)
    
    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
      if moving_params is not None:
        bias = moving_params.average(bias)
    else:
      bias = 0
      
    # Do the multiplication
    conv = tf.nn.conv1d(all_inputs, matrix, 1, 'SAME') + bias
    conv = tf.reshape(conv, output_shape)
    if n_splits > 1:
      return tf.split(conv, n_splits, n_dims-1)
    else:
      return conv
  
#===============================================================
def random_mask(prob, mask_shape, dtype=tf.float32):
  """"""
  
  rand = tf.random_uniform(mask_shape)
  ones = tf.ones(mask_shape, dtype=dtype)
  zeros = tf.zeros(mask_shape, dtype=dtype)
  prob = tf.ones(mask_shape) * prob
  return tf.where(rand < prob, ones, zeros)

#===============================================================
def random_where(prob, success, fail, keep_dims=None): 
  """"""
  
  mask_shape = tf.shape(success)
  keep_dims = np.zeros(len(success.get_shape().as_list()), dtype=np.int32)
  keep_dims[dims] = np.int32(1)
  mask_shape = mask_shape ** keep_dims
  mask = random_mask(prob, mask_shape, dtype=success.dtype)
  return mask * success + (1-mask) * fail
