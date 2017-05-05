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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from parser.misc.colors import ctext, color_pattern
from parser.neural.models.nn import NN

#***************************************************************
class BaseParser(NN):
  """"""
  
  PAD = 0
  ROOT = 1
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """"""
    
    self.moving_params = moving_params
    if isinstance(vocabs, dict):
      self.vocabs = vocabs
    else:
      self.vocabs = {vocab.name: vocab for vocab in vocabs}
    
    input_vocabs = [self.vocabs[name] for name in self.input_vocabs]
    #embed = tf.concat([vocab(moving_params=self.moving_params) for vocab in input_vocabs], 2)
    embed = self.embed_concat(input_vocabs)
    for vocab in self.vocabs.values():
      if vocab not in input_vocabs:
        vocab.generate_placeholder()
    placeholder = self.vocabs['words'].placeholder
    if len(placeholder.get_shape().as_list()) == 3:
      placeholder = placeholder[:,:,0]
    self._tokens_to_keep = tf.to_float(tf.greater(placeholder, self.ROOT))
    self._batch_size = tf.shape(placeholder)[0]
    self._bucket_size = tf.shape(placeholder)[1]
    self._sequence_lengths = tf.reduce_sum(tf.to_int32(tf.greater(placeholder, self.PAD)), axis=1)
    self._n_tokens = tf.to_int32(tf.reduce_sum(self.tokens_to_keep))
    
    top_recur = embed
    for i in xrange(self.n_layers):
      with tf.variable_scope('RNN%d' % i):
        top_recur = self.RNN(top_recur, self.recur_size)
    return top_recur
  
  #=============================================================
  def process_accumulators(self, accumulators, time=None):
    """"""
    
    n_tokens, n_seqs, loss, rel_corr, arc_corr, corr, seq_corr = accumulators
    acc_dict = {
      'Loss': loss,
      'LS': rel_corr/n_tokens*100,
      'UAS': arc_corr/n_tokens*100,
      'LAS': corr/n_tokens*100,
      'SS': seq_corr/n_seqs*100,
    }
    if time is not None:
      acc_dict.update({
        'Token_rate': n_tokens / time,
        'Seq_rate': n_seqs / time,
      })
    return acc_dict
  
  #=============================================================
  def update_history(self, history, accumulators):
    """"""
    
    acc_dict = self.process_accumulators(accumulators)
    for key, value in acc_dict.iteritems():
      history[key].append(value)
    return
  
  #=============================================================
  def print_accuracy(self, accumulators, time, prefix='Train'):
    """"""
    
    acc_dict = self.process_accumulators(accumulators, time=time)
    strings = []
    strings.append(color_pattern('Loss:', '{Loss:7.3f}', 'bright_red'))
    strings.append(color_pattern('LS:', '{LS:5.2f}%', 'bright_cyan'))
    strings.append(color_pattern('UAS:', '{UAS:5.2f}%', 'bright_cyan'))
    strings.append(color_pattern('LAS:', '{LAS:5.2f}%', 'bright_cyan'))
    strings.append(color_pattern('SS:', '{SS:5.2f}%', 'bright_green'))
    strings.append(color_pattern('Speed:', '{Seq_rate:6.1f} seqs/sec', 'bright_yellow'))
    string = ctext('{0}  ', 'bold') + ' | '.join(strings)
    print(string.format(prefix, **acc_dict))
    return
  
  #=============================================================
  def plot(self, history, prefix='Train'):
    """"""
    
    pass
  
  #=============================================================
  def write_probs(self, input_file, output_file, probs, inv_idxs):
    """"""
    
    # TODO implement argmax, projective, nonprojective
    # Store the algorithms in a separate file that gets imported in Configurable
    #parse_algorithm = self.parse_algorithm 
    
    # Turns list of tuples of tensors into list of matrices
    print(probs[0][0].shape, probs[1][0].shape)
    arc_probs = [arc_prob for arc_batch in probs[0] for arc_prob in arc_batch]
    rel_probs = [rel_prob for rel_batch in probs[1] for rel_prob in rel_batch]
    print(arc_probs[0])
    print(rel_probs[0])
    
    with open(output_file, 'w') as f:
      with open(input_file) as g:
        for i in xrange(len(inv_idxs)):
          arc_prob, rel_prob = arc_probs[i], rel_probs[i]
          arc_preds = np.argmax(arc_prob, axis=0)
          arc_preds_one_hot = np.zeros([rel_prob.shape[0], rel_prob.shape[2]])
          arc_preds_one_hot[np.arange(len(arc_preds)), arc_preds] = 1.
          print(rel_prob.shape, arc_preds_one_hot.shape)
          rel_preds = np.argmax(np.einsum('nrb,nb->nr', rel_prob, arc_preds_one_hot), axis=0)
          for arc_pred, rel_pred in zip(arc_preds, rel_preds):
            line = g.readline()
            while not re.match('[0-9]+\t', line):
              line = g.readline()
            line = line.strip().split('\t')
            line[6] = str(arc_pred)
            line[7] = self.vocabs['rels'][rel_pred]
            f.write('\t'.join(line)+'\n')
          f.write('\n')
    return
  
  #=============================================================
  @property
  def train_keys(self):
    return ('n_tokens', 'n_seqs', 'loss', 'n_rel_correct', 'n_arc_correct', 'n_correct', 'n_seqs_correct')
  
  #=============================================================
  @property
  def parse_keys(self):
    return ('arc_probs', 'rel_probs')
