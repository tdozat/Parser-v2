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

import codecs

#***************************************************************
encodings = ['utf-8', 'ascii']

def get_encoding(filename):
  """"""
  
  success = False
  for encoding in encodings:
    with codecs.open(filename, encoding=encoding) as f:
      try:
        for i, line in enumerate(f):
          pass
        success = True
        break
      except ValueError as e:
        print('Encoding {0} failed for file {1} at line {2}: {3}\n{4}'.format(encoding, filename, i, line, e))
        continue

  if success:
    return encoding
  else:
    raise ValueError('No valid encoding found for file {0}'.format(filename))
