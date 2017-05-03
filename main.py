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

import re
import os
import sys
import codecs
from argparse import ArgumentParser

from parser import Configurable
from parser import Network

#***************************************************************
# Set up the argparser
argparser = ArgumentParser('Network')
argparser.add_argument('--save_dir', required=True)
subparsers = argparser.add_subparsers()

#===============================================================
# Train
#---------------------------------------------------------------
def train(load, **kwargs):
  """"""
  
  try:
    if not load and os.path.isdir(kwargs['default']['save_dir']):
      raw_input('Save directory already exists. Press <enter> to continue or <ctrl-z> to abort.')
  except KeyboardInterrupt:
    sys.exit(0)
  configurable = Configurable(**kwargs)
  network = Network.from_configurable(configurable)
  network.train(load=load)
  return
#---------------------------------------------------------------
# TODO make the pretrained vocab names a list given to TokenVocab
train_parser = subparsers.add_parser('train')
train_parser.set_defaults(action=train)
train_parser.add_argument('--load', action='store_true')
with codecs.open('config/defaults.cfg') as f:
  section_regex = re.compile('\[(.*)\]')
  for line in f:
    match = section_regex.match(line)
    if match:
      # Section Name -> --section_name
      section_name = '--'+match.group(1).lower().replace(' ', '_')
      # --section_name opt1=value1 opt2=value2 opt3=value3
      train_parser.add_argument(section_name, nargs='+')

#***************************************************************
# Parse the arguments
kwargs = vars(argparser.parse_args())
action = kwargs.pop('action')
save_dir = kwargs.pop('save_dir')
load = bool(kwargs.pop('load'))
kwargs = {key: value for key, value in kwargs.iteritems() if value is not None}
for section, values in kwargs.iteritems():
  values = [value.split('=', 1) for value in values]
  kwargs[key] = {opt: value for opt, value in values}
if 'default' not in kwargs:
  kwargs['default'] = {}
kwargs['default']['save_dir'] = save_dir
action(load, **kwargs)  
