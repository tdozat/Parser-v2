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
from collections import Counter

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson, nbinom
import matplotlib.pyplot as plt
import tensorflow as tf

from parser import Configurable
from parser.misc.colors import ctext, color_pattern

#***************************************************************
class Bucketer(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, k, *args, **kwargs):
    """"""
    
    super(Bucketer, self).__init__(*args, **kwargs)
    
    self._k = k
    self.__enter__()
    return
  
  #=============================================================
  def compute_splits(self, data, plot=True):
    """"""
    
    len2cnt = Counter(data)
    
    # Error checking
    if len(len2cnt) < self.k:
      raise ValueError('Trying to sort %d lengths into %d buckets' % (len(len2cnt), self.k))
    
    # Initialize
    self._len2cnt = len2cnt
    self._lengths = sorted(self.len2cnt.keys())
    
    # Initialize the splits evenly
    lengths = sorted([l for length, count in len2cnt.items() for l in [length]*count])
    self._splits = [np.max(split) for split in np.array_split(lengths, self.k)]
    
    # Make sure all the splits are ordered correctly and present in the len2cnt
    idx = len(self)-1
    while idx > 0:
      while self[idx] > self.lengths[0] and (self[idx] <= self[idx-1] or self[idx] not in self.len2cnt):
        self[idx] -= 1
      idx -= 1
    
    idx = 1
    while idx < len(self)-1:
      while self[idx] < self.lengths[-1] and (self[idx] <= self[idx-1] or self[idx] not in self.len2cnt):
        self[idx] += 1
      idx += 1
    
    # Reindex
    self.reindex()
    
    # Iterate
    old_splits = None
    i = 0
    if self.verbose:
      print(color_pattern('Initial # of tokens in buckets:', str(self.size()), 'bright_red'))
    while self != old_splits:
      old_splits = list(self)
      self.recenter()
      i += 1
    if self.verbose:
      print(color_pattern('Final # of tokens in buckets:', str(self.size()), 'bright_white'))
    
    self.reindex()
    return self._splits
  
  #=============================================================
  def reindex(self):
    """"""
    
    idx = 0
    self._counts = [0 for _ in self]
    self._lidxs = []
    for lidx, length in enumerate(self.lengths):
      self.counts[idx] += self.len2cnt[length]
      if length == self[idx]:
        self.lidxs.append(lidx)
        idx += 1
    return
    
  #=============================================================
  def recenter(self):
    """"""
    
    for idx in xrange(len(self)-1):
      split = self[idx]
      lidx = self.lidxs[idx]
      old_size = self.size()
      count = self.len2cnt[self.lengths[lidx]]
      
      if lidx > 0 and self.lengths[lidx-1] not in self:
        self[idx] = self.lengths[lidx-1]
        new_size = self.size()
        if old_size > new_size:
          self.lidxs[idx] = lidx-1
          self.counts[idx] -= count
          self.counts[idx-1] += count
          continue
        else:
          self[idx] = self.lengths[lidx]
      
      if lidx < len(self.lengths)-1 and self.lengths[lidx+1] not in self:
        self[idx] = self.lengths[lidx+1]
        new_size = self.size()
        if old_size > new_size:
          self.lidxs[idx] = lidx+1
          self.counts[idx] -= count
          self.counts[idx+1] += count
        else:
          self[idx] = self.lengths[lidx]
    return
  
  #=============================================================
  def size(self):
    """"""
    
    size = 0
    idx = 0
    for lidx, length in enumerate(self.lengths):
      size += self[idx] * self.len2cnt[length]
      if length == self[idx]:
        idx += 1
    return size
  
  #=============================================================
  def plot(self, use_poisson=False):
    """"""
    
    x = np.array(self.len2cnt.keys(), dtype=np.float32)
    y = np.array(self.len2cnt.values(), dtype=np.float32)
    y /= np.sum(y)
    mean = np.sum(x*y)
    var = np.sum((x-mean)**2*y)
    p = (1 - mean/var)
    r = mean * (1-p) / p
    if use_poisson:
      yhat = poisson.pmf(x, mean)
    else:
      yhat = nbinom.pmf(x, r, 1-p)
    error = yhat - y
    
    fig, (ax0, ax1) = plt.subplots(figsize=(10,5), ncols=2)
    ax0.plot(x, yhat.flatten(), label='Best fit')
    ax0.plot(x, y, label='Data')
    ax0.grid()
    ax0.legend(loc='best')
    ax0.set_title('Lengths')
    ax0.set_xlabel('Length')
    ax0.set_ylabel('Frequency')
    ax0.set_xlim(xmin=np.min(x)-.5, xmax=np.max(x)+.5)
    for split in self:
      ax0.axvline(split, color='k', ls='--')
    
    ax1.set_title('Error of best fit')
    ax1.plot(x, y - yhat.flatten(), 'r')
    ax1.grid()
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Error')
    ax1.set_xlim(xmin=np.min(x)-.5, xmax=np.max(x)+.5)
    for split in self:
      ax1.axvline(split, color='k', ls='--')
    
    fig.suptitle('Negative Binomial Distribution', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=.875)
    plt.savefig(os.path.join(self.save_dir, '%s.png' % (self.name)))
  
  #=============================================================
  @property
  def k(self):
    return self._k
  @property
  def len2cnt(self):
    return self._len2cnt
  @property
  def lidxs(self):
    return self._lidxs
  @property
  def counts(self):
    return self._counts
  @property
  def lengths(self):
    return self._lengths
  
  #=============================================================
  def __len__(self):
    return len(self._splits)
  def __str__(self):
    return str(self._splits)
  def __iter__(self):
    return (split for split in self._splits)
  def __contains__(self, key):
    return key in self._splits
  def __getitem__(self, key):
    return self._splits[key]
  def __setitem__(self, key, value):
    self._splits[key] = value
    return
  def __eq__(self, value):
    return self._splits == value
  def __ne__(self, value):
    return self._splits != value
  def __enter__(self):  
    self._len2cnt = {}
    self._lengths = []
    self._counts = []
    self._splits = []
    self._lidxs = []
    return self
  def __exit__(self, exception_type, exception_value, traceback):
    if exception_type is not None:
      raise exception_type(exception_value)
    return True
  
#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser import Configurable
  from parser.misc.bucketer import Bucketer
  
  from scipy.stats import truncnorm
  with Bucketer(5) as bucketer:
    print(bucketer.compute_splits([[0] * np.int(truncnorm(0, 10, scale=5).rvs()) for _ in xrange(1000)]))
    bucketer.plot()
