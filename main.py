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

# TODO make the pretrained vocab names a list given to TokenVocab
#***************************************************************
# Set up the argparser
argparser = ArgumentParser('Network')
argparser.add_argument('--save_dir', required=True)
subparsers = argparser.add_subparsers()
section_names = set()
# --section_name opt1=value1 opt2=value2 opt3=value3
with codecs.open('config/defaults.cfg') as f:
  section_regex = re.compile('\[(.*)\]')
  for line in f:
    match = section_regex.match(line)
    if match:
      section_names.add(match.group(1).lower().replace(' ', '_'))

#===============================================================
# Train
#---------------------------------------------------------------
def train(save_dir, **kwargs):
  """"""
  
  load = kwargs.pop('load')
  try:
    if not load and os.path.isdir(save_dir):
      raw_input('Save directory already exists. Press <enter> to continue or <ctrl-z> to abort.')
  except KeyboardInterrupt:
    sys.exit(0)
  
  network = Network(**kwargs)
  network.train(load=load)
  return
#---------------------------------------------------------------

train_parser = subparsers.add_parser('train')
train_parser.set_defaults(action=train)
train_parser.add_argument('--load', action='store_true')
for section_name in section_names:
  train_parser.add_argument('--'+section_name, nargs='+')

#===============================================================
# Parse
#---------------------------------------------------------------
def parse(save_dir, **kwargs):
  """"""
  
  files = kwargs.pop('files')
  network = Network(**kwargs)
  network.parse(files)
  return
#---------------------------------------------------------------

parse_parser = subparsers.add_parser('parse')
parse_parser.set_defaults(action=parse)
for section_name in section_names:
  parse_parser.add_argument('--'+section_name, nargs='+')
parse_parser.add_argument('files', nargs='+')

#***************************************************************
# Parse the arguments
kwargs = vars(argparser.parse_args())
action = kwargs.pop('action')
save_dir = kwargs.pop('save_dir')
kwargs = {key: value for key, value in kwargs.iteritems() if value is not None}
for section, values in kwargs.iteritems():
  if section in section_names:
    values = [value.split('=', 1) for value in values]
    kwargs[section] = {opt: value for opt, value in values}
if 'default' not in kwargs:
  kwargs['default'] = {}
kwargs['default']['save_dir'] = save_dir
action(save_dir, **kwargs)  
