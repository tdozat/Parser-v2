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
import codecs
from backports import lzma

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from collections import defaultdict

#***************************************************************
if __name__ == '__main__':
  """"""
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-k', '--k_trials', type=int, default=100)
  parser.add_argument('-n', '--n_words', type=int, default=1000)
  parser.add_argument('files', nargs='+')
  
  args = parser.parse_args()
  words = []
  types = set()
  n_types = []
  for filename in args.files:
    with codecs.open(filename, encoding='utf-8', errors='ignore') as f:
      for line in f:
        line = line.strip()
        if line:
          if not re.match('#|[0-9]+[-.][0-9]+', line):
            words.append(line.split('\t')[1])
  trials = []
  n_words = args.n_words or len(words)
  for _ in xrange(args.k_trials):
    np.random.shuffle(words)
    tokens = words[:args.n_words]
    with codecs.open('uncompressed.txt', 'w', encoding='utf-8', errors='ignore') as f:
      f.write('\n'.join(tokens))
    with lzma.open('compressed.txt.xz', 'wb') as f:
      f.write('\n'.join(tokens).encode('utf-8', 'ignore'))
    trials.append(os.path.getsize('compressed.txt.xz')/os.path.getsize('uncompressed.txt'))
  os.remove('uncompressed.txt')
  os.remove('compressed.txt.xz')
  print(len(words), np.mean(trials))
