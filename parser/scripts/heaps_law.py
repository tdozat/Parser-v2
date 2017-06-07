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
import argparse

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from collections import defaultdict

#***************************************************************
if __name__ == '__main__':
  """"""
  
  parser = argparse.ArgumentParser()
  parser.add_argument('files', nargs='+')
  
  args = parser.parse_args()
  words = []
  types = set()
  n_types = []
  for filename in args.files:
    with open(filename) as f:
      for line in f:
        line = line.strip()
        if line:
          if not re.match('#|[0-9]+[-.][0-9]+', line):
            words.append(line.split('\t')[1])
  np.random.shuffle(words)
  for word in words:
    types.add(word)
    n_types.append(len(types))
  
  K = 1
  b = .75
  y = n_types
  logy = np.log(y)
  x = np.arange(len(n_types))+1
  logx = np.log(x)
  d2ell = np.array([[1, np.mean(logx)],[np.mean(logx), np.mean(logx**2)]])
  d2ellinv = inv(d2ell)
  ell = np.mean((logy - b*logx-K)**2 / 2)
  dell = np.array([np.mean(K+b*logx-logy), np.mean((K+b*logx-logy)*logx)])
  updates = d2ellinv.dot(dell)
  K -= updates[0]
  b -= updates[1]
  print(b)
  #K_ = 5
  #b_ = .74
  #for i in xrange(20):
  #  ell = np.mean((y - K_*x**b_)**2 / 2)
  #  K_ -= 2*np.mean((K_*x**b_-y)*x**b_) / np.mean(x**(2*b_))
  #  b_ -= 2*np.mean((K_*x**b_-y)*K_*x**b_*logx) / np.mean((2*K_*x**b_ - y)*K_*x**b_*logx**2)
  #  print(ell, K_, b_)
  #plt.figure()
  #plt.grid()
  #plt.plot(x, y)
  #plt.plot(x, np.exp(b*logx+K))
  #plt.show()
  #plt.figure()
  #plt.grid()
  #plt.plot(x, logy - b*logx-K)
  #plt.show()
  #plt.figure()
  #plt.grid()
  #plt.plot(x, y - K_*x**b_)
  #plt.show()
