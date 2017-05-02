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
from collections import defaultdict

#***************************************************************
class DepTree:
  """"""
  
  #=============================================================
  def __init__(self, buff):
    """"""
    
    self._head2deps = defaultdict(list)
    self._dep2head = dict()
    self._str = []
    for line in buff:
      dep_idx = int(line[0])
      head_idx = int(line[6])
      self.head2deps[head_idx].append(dep_idx)
      self.dep2head[dep_idx] = head_idx
      self._str.append(line[1])
    return
  
  #=============================================================
  def count_nonprojective(self):
    """"""
    
    nonproj = []
    for dep in self:
      head = self.dep2head[dep]
      span_min = min(dep, head)
      span_max = max(dep, head)
      for mid_dep in xrange(span_min+1, span_max):
        mid_head = self.dep2head[mid_dep]
        nonproj.append(mid_head < span_min or mid_head > span_max)
    return nonproj
  
  #=============================================================
  @property
  def head2deps(self):
    return self._head2deps
  @property
  def dep2head(self):
    return self._dep2head
  
  #=============================================================
  def __iter__(self):
    return (dep for dep in self.dep2head)
  def __len__(self):
    return len(self.dep2head)
  def __str__(self):
    return ' '.join(self._str)+'\n'
  
#***************************************************************
if __name__ == '__main__':
  """"""
  
  parser = argparse.ArgumentParser()
  parser.add_argument('files', nargs='+')
  
  args = parser.parse_args()
  for filename in args.files:
    nonproj = []
    with open(filename) as f:
      buff = []
      for line in f:
        line = line.strip()
        if line:
          if not re.match('#|[0-9]+[-.][0-9]+', line):
            buff.append(line.split('\t'))
        else:
          tree = DepTree(buff)
          nonproj.extend(tree.count_nonprojective())
          buff = []
    print(filename, np.mean(nonproj)*100)
