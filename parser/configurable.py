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
import re
import multiprocessing as mp
import glob
import numpy as np
import tensorflow as tf

try:
  from ConfigParser import SafeConfigParser, NoOptionError
except ImportError:
  from configparser import SafeConfigParser, NoOptionError
  

#***************************************************************
class Configurable(object):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    if args:
      if len(args) > 1:
        raise TypeError('Configurables take at most one argument')
    
    self._name = kwargs.pop('name', None)
    if args and not kwargs:
      self._config = args[0]
    else:
      self._config = self._configure(*args, **kwargs)
    
    if not args:
      if not os.path.isdir(self.save_dir):
        os.makedirs(self.save_dir)
      
      config_file = os.path.join(self.save_dir, 'config.cfg')
      if not os.path.isfile(config_file):
        with open(config_file, 'w') as f:
          self._config.write(f)
    return
  
  #=============================================================
  def _configure(self, *args, **kwargs):
    """"""
    
    config = SafeConfigParser()
    if args:
      old_config = args[0]
      for section in old_config.sections():
        config.add_section(section)
        for option in old_config.options(section):
          config.set(section, option, old_config.get(section, option))
    config_files = [os.path.join('config', 'defaults.cfg'),
                    kwargs.pop('config_file', '')]
    config.read(config_files)
    config_sections = {section.lower().replace(' ', '_'): section for section in config.sections()}
    # argparse syntax should be --section_name option1=value1 option2=value2
    # kwarg syntax should be section_name={option1: value1, option2: value2}
    # or option1=value1, option2=value2
    for kw, arg in kwargs.iteritems():
      if isinstance(arg, dict):
        section = config_sections[kw]
        for option, value in arg.iteritems():
          config.set(section, option, str(value))
      else:
        option, value = kw, arg
        section = re.sub('\B([A-Z][a-z])', r' \1', self.__class__.__name__)
        config.set(section, option, str(value))
    return config
  
  #=============================================================
  @classmethod
  def from_configurable(cls, configurable, *args, **kwargs):
    """"""
    
    args += (configurable._config,)
    return cls(*args, **kwargs)
  
  #=============================================================
  def _get(self, config_func, option):
    superclasses = [superclass.__name__ for superclass in self.__class__.__mro__]
    superclasses[-1] = 'DEFAULT'
    for superclass in superclasses:
      superclass = re.sub('\B([A-Z][a-z])', r' \1', superclass)
      if self._config.has_section(superclass) and self._config.has_option(superclass, option):
        return None if self._config.get(superclass, option) == 'None' else config_func(superclass, option)
    raise NoOptionError(option, superclasses)
  
  def _getlist(self, lst):
    lst = lst.split(':')
    i = 0
    while i < len(lst):
      if lst[i].endswith('\\'):
        lst[i] = ':'.join([lst[i].rstrip('\\'), lst.pop(i+1)])
      else:
        i += 1
    return lst
  
  def _globlist(self, lst):
    globbed = []
    for elt in lst:
      globbed.extend(glob.glob(elt))
    return globbed
  
  def get(self, option):
    return self._get(self._config.get, option)
  def getint(self, option):
    return self._get(self._config.getint, option)
  def getfloat(self, option):
    return self._get(self._config.getfloat, option)
  def getboolean(self, option):
    return self._get(self._config.getboolean, option)
  def getlist(self, option):
    return self._getlist(self._get(self._config.get, option))
  def getfiles(self, option):
    return self._globlist(self.getlist(option))
  
  #=============================================================
  # [Default]
  @property
  def save_dir(self):
    return self.get('save_dir')
  @property
  def data_dir(self):
    return self.get('data_dir')
  
  #=============================================================
  # [Configurable]
  @property
  def name(self):
    if self._name is None:
      return self.get('name')
    else:
      return self._name
  @property
  def train_files(self):
    return self.getfiles('train_files')
  @property
  def valid_files(self):
    return self.getfiles('valid_files')
  @property
  def test_files(self):
    return self.getfiles('test_files')
  @property
  def verbose(self):
    return self.getboolean('verbose')
  
  #=============================================================
  # [Vocab]
  @property
  def filename(self):
    return self.get('filename')
  @property
  def embed_size(self):
    return self.getint('embed_size')
  @property
  def cased(self):
    return self.getboolean('cased')
  #@property
  #def mlp_keep_prob(self):
  #  return self.getfloat('attn_keep_prob')
  @property
  def embed_keep_prob(self):
    return self.getfloat('embed_keep_prob')
  @property
  def min_occur_count(self):
    return self.getint('min_occur_count')
  @property
  def max_rank(self):
    return self.getint('max_rank')
  @property
  def special_tokens(self):
    return self.getlist('special_tokens')
  @property
  def subtoken_vocab(self):
    from parser import vocabs
    return getattr(vocabs, self.get('subtoken_vocab'))
  @property
  def embed_model(self):
    from parser.neural.models import embeds
    return getattr(embeds, self.get('embed_model'))
  @property
  def parser_model(self):
    from parser.neural.models import parsers 
    return getattr(parsers, self.get('parser_model'))
  
  #=============================================================
  # [Pretrained Vocab]
  @property
  def max_entries(self):
    return self.getint('max_entries')
  
  #=============================================================
  # [Retrained Vocab]
  @property
  def embed_loss(self):
    return self.get('embed_loss')
  
  #=============================================================
  # [Ngram Multivocab]
  @property
  def max_n(self):
    return self.getint('max_n')
  
  #=============================================================
  # [Bytepair Vocab]
  @property
  def n_bytepairs(self):
    return self.getint('n_bytepairs')
  
  #=============================================================
  # [Bucket / NN]
  @property
  def n_layers(self):
    return self.getint('n_layers')
  @property
  def conv_size(self):
    return self.getint('conv_size')
  @property
  def window_size(self):
    return self.getint('window_size')
  @property
  def mlp_size(self):
    return self.getint('mlp_size')
  @property
  def recur_size(self):
    return self.getint('recur_size')
  @property
  def recur_func(self):
    from parser.neural import functions
    return getattr(functions, self.get('recur_func'))
  @property
  def recur_cell(self):
    from parser.neural import recur_cells
    return getattr(recur_cells, self.get('recur_cell'))
  @property
  def rnn_func(self):
    from parser.neural import rnn
    return getattr(rnn, self.get('rnn_func'))
  @property
  def forget_bias(self):
    return self.getfloat('forget_bias')
  @property
  def conv_keep_prob(self):
    return self.getfloat('conv_keep_prob')
  @property
  def recur_keep_prob(self):
    return self.getfloat('recur_keep_prob')
  @property
  def ff_keep_prob(self):
    return self.getfloat('ff_keep_prob')
  
  #=============================================================
  # [Metabucket / Parser / Joint Parser]
  @property
  def mlp_keep_prob(self):
    return self.getfloat('mlp_keep_prob')
  @property
  def mlp_func(self):
    from parser.neural import functions
    return getattr(functions, self.get('mlp_func'))
  @property
  def conv_func(self):
    from parser.neural import functions
    return getattr(functions, self.get('conv_func'))
  @property
  def tag_mlp_size(self):
    return self.getint('tag_mlp_size')
  @property
  def arc_mlp_size(self):
    return self.getint('arc_mlp_size')
  @property
  def rel_mlp_size(self):
    return self.getint('rel_mlp_size')
  @property
  def input_vocabs(self):
    return self.getlist('input_vocabs')
  @property
  def output_vocabs(self):
    return self.getlist('output_vocabs')
  
  #=============================================================
  # [Dataset]
  @property
  def data_files(self):
    return self.get('data_files')
  @property
  def n_buckets(self):
    return self.getint('n_buckets')
  
  #=============================================================
  # [Network]
  @property
  def max_train_iters(self):
    return self.getint('max_train_iters')
  @property
  def print_every(self):
    return self.getint('print_every')
  @property
  def validate_every(self):
    return self.getint('validate_every')
  @property
  def batch_by(self):
    return self.get('batch_by')
  @property
  def batch_size(self):
    return self.getint('batch_size')
  @property
  def save_every(self):
    return self.getint('save_every')
  @property
  def per_process_gpu_memory_fraction(self):
    return self.getfloat('per_process_gpu_memory_fraction')
  
  #=============================================================
  # [Zipf / Bucketer]
  @property
  def n_zipfs(self):
    return self.getint('n_zipfs')
  @property
  def n_poissons(self):
    return self.getint('n_poissons')

  #=============================================================
  # [Radam Optimizer]
  @property
  def learning_rate(self):
    return self.getfloat('learning_rate')
  @property
  def decay(self):
    return self.getfloat('decay')
  @property
  def decay_steps(self):
    return self.getfloat('decay_steps')
  @property
  def clip(self):
    return self.getfloat('clip')
  @property
  def mu(self):
    return self.getfloat('mu')
  @property
  def nu(self):
    return self.getfloat('nu')
  @property
  def gamma(self):
    return self.getfloat('gamma')
  @property
  def chi(self):
    return self.getfloat('chi')
  @property
  def epsilon(self):
    return self.getfloat('epsilon')

#***************************************************************
if __name__ == '__main__':
  """"""
  
  config = Configurable()
  print('Configurable passes')
