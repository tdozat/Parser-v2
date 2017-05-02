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

import os
import cPickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from parser import Configurable
from parser.neural.optimizers import RadamOptimizer

#***************************************************************
class Zipf(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, counts, *args, **kwargs):
    """"""
    
    super(Zipf, self).__init__(*args, **kwargs)
    
    self._counts = counts
    self._ranks = np.arange(len(counts))[:,None]+1
    self._counts = np.array([count for _, count in counts.most_common()])[:,None]
    self._freqs = self.counts / np.sum(self.counts)
    self._params = {}
    self._preds = None
    self._error = None
    if os.path.isfile(os.path.join(self.save_dir, '%s.pkl' % self.name)):
      self.load()
      self.preds = np.exp(self.predict(self.ranks)[:,None])
      self.error = np.log(self.freqs) - np.log(self.preds)
    else:
      self.preds, self.error = self.fit()
      self.dump()
    return
  
  #=============================================================
  def __call__(self):
    """"""
    
    radam_optimizer = RadamOptimizer.from_configurable(self, learning_rate=1e-1, decay_steps=500)
    x = tf.placeholder(tf.float32, shape=(None,1), name='x')
    y = tf.placeholder(tf.float32, shape=(None,1), name='y')
    def affine(a, b):
      return a * tf.log(x) + b
    a = tf.get_variable('a', shape=self.n_zipfs, dtype=tf.float32, initializer=tf.random_normal_initializer())
    b = tf.get_variable('b', shape=self.n_zipfs, dtype=tf.float32, initializer=tf.random_normal_initializer())
    s = tf.get_variable('s', shape=self.n_zipfs, dtype=tf.float32, initializer=tf.random_uniform_initializer(-2,-.5))
    t = tf.get_variable('t', shape=self.n_zipfs, dtype=tf.float32, initializer=tf.random_normal_initializer())
    w = tf.expand_dims(tf.nn.softmax(affine(a, b)), axis=1, name='w')
    z = tf.expand_dims(affine(s, t), axis=2, name='z')
    yhat = tf.squeeze(tf.matmul(w, z), axis=2, name='yhat')
    ell = tf.reduce_mean((tf.log(y) - yhat)**2 / 2, name='ell')
    ell += tf.reduce_mean((tf.reduce_max(w, axis=0) - 1)**2 / 2)
    minimize = radam_optimizer.minimize(ell, name='minimize')
    return x, y, ell, minimize
    
  #=============================================================
  def dump(self):
    """"""
    
    a = self.params['a']
    b = self.params['b']
    s = self.params['s']
    t = self.params['t']
    params = np.stack([a, b, s, t])
    with open(os.path.join(self.save_dir, '%s.pkl' % self.name.lower()), 'w') as f:
      pkl.dump(params, f)
    return
  
  #=============================================================
  def load(self):
    """"""
    
    with open(os.path.join(self.save_dir, '%s.pkl' % self.name.lower())) as f:
      params = pkl.load(f)
    self.params['a'] = params[0]
    self.params['b'] = params[1]
    self.params['s'] = params[2]
    self.params['t'] = params[3]
    assert len(params[0]) == self.n_zipfs
    return
  
  #=============================================================
  def fit(self):
    """"""
    
    batch_size = self.batch_size 
    print_every = self.print_every
    verbose = self.verbose
    
    losses = []
    with tf.Graph().as_default() as graph:
      x, y, ell, minimize = self()
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if verbose:
          print('Fitting multi-zipfian distribution')
        for i in xrange(self.max_train_iters):
          idxs = np.exp(np.random.uniform(-1, np.log(len(self.counts)), size=batch_size)).astype(np.int64)
          feed_dict = {x:self.ranks[idxs], y:self.freqs[idxs]}
          loss, _ = sess.run([ell, minimize], feed_dict=feed_dict)
          losses.append(loss)
          if verbose and ((i+1) % print_every) == 0:
            print('%5d) loss: %.2e' % (i+1, np.mean(losses)))
            losses = []
        yhat = tf.exp(graph.get_tensor_by_name('yhat:0'))
        a = graph.get_tensor_by_name('a:0')
        b = graph.get_tensor_by_name('b:0')
        s = graph.get_tensor_by_name('s:0')
        t = graph.get_tensor_by_name('t:0')
        yhat, a, b, s, t = sess.run([yhat, a, b, s, t], feed_dict={x:self.ranks})
    self.params['a'] = a
    self.params['b'] = b
    self.params['s'] = s
    self.params['t'] = t
    return yhat, (np.log(self.freqs) - yhat)
  
  #=============================================================
  def plot(self):
    """"""
    
    a, b = self.params['a'], self.params['b']
    s, t = self.params['s'], self.params['t']
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(figsize=(10,10), ncols=2, nrows=2)
    ax0.plot(self.ranks.flatten(), self.preds.flatten(), label='Best fit')
    ax0.plot(self.ranks.flatten(), self.freqs.flatten(), label='Data')
    ax0.grid()
    ax0.legend(loc='best')
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.set_title('Rank-Frequency')
    ax0.set_xlabel('Log token rank')
    ax0.set_ylabel('Log token frequency')
    ax0.set_ylim(ymax=1,ymin=1e-7)
    
    ax1.set_title('Error of best fit')
    ax1.plot(self.ranks.flatten(), self.error.flatten(), 'r')
    ax1.set_xscale('log')
    ax1.grid()
    ax1.set_xlabel('Log token rank')
    ax1.set_ylabel('Error')
    
    ax2.plot(self.ranks.flatten(), self.preds.flatten())
    for z in self.zipf(self.ranks).T:
      ax2.plot(self.ranks.flatten(), np.exp(z), '--k')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_title('Zipf distributions')
    ax2.set_xlabel('Log token rank')
    ax2.set_ylabel('Log token frequency')
    ax2.set_ylim(ymax=1,ymin=1e-7)
    ax2.grid()
    
    ax3.plot(self.ranks.flatten(), self.preds.flatten())
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_xlabel('Log token rank')
    ax3.set_ylabel('Log token frequency')
    ax3.set_ylim(ymax=1,ymin=1e-7)
    ax3.grid()
    
    ax4 = ax3.twinx()
    for w in self.weight(self.ranks).T:
      ax4.plot(self.ranks.flatten(), w, '--k')
    ax4.set_xscale('log')
    ax4.set_title('Zipf weights')
    ax4.set_ylabel('Weight')
    
    fig.suptitle('Multi-Zipf Distribution', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=.925)
    plt.savefig(os.path.join(self.save_dir, '%s.png' % (self.name)))
    return
  
  #=============================================================
  def affine(self, x, a, b):
    return a*np.log(x) + b
  def zipf(self, x):
    s, t = self.params['s'], self.params['t']
    return self.affine(x, s, t)
  def softmax(self, x):
    x -= np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)
  def weight(self, x):
    a, b = self.params['a'], self.params['b']
    return self.softmax(self.affine(x, a, b))
  def predict(self, x):
    if x.shape[-1] != 1:
      x = x[...,None]
    w = self.weight(x)
    z = self.zipf(x)
    return np.einsum('...i,...i->...', w, z)
    
  #=============================================================
  @property
  def ranks(self):
    return self._ranks
  @property
  def counts(self):
    return self._counts
  @property
  def freqs(self):
    return self._freqs
  @property
  def preds(self):
    return self._preds
  @property
  def error(self):
    return self._error
  @property
  def params(self):
    return self._params
  @preds.setter
  def preds(self, value):
    assert self.preds is None
    self._preds = value
  @error.setter
  def error(self, value):
    assert self.error is None
    self._error = value
