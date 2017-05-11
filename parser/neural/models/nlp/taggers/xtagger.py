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

import numpy as np
import tensorflow as tf

from parser.neural.models.nlp.taggers.base_xtagger import BaseXTagger

#***************************************************************
class XTagger(BaseXTagger):
  """"""
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """"""
    
    top_recur = super(XTagger, self).__call__(vocabs, moving_params=moving_params)
    int_tokens_to_keep = tf.to_int32(self.tokens_to_keep)
    
    with tf.variable_scope('MLP'):
      tag_mlp, xtag_mlp = self.MLP(top_recur, self.mlp_size, n_splits=2)
    
    with tf.variable_scope('Tag'):
      tag_logits = self.linear(tag_mlp, len(self.vocabs['tags']))
      tag_probs = tf.nn.softmax(tag_logits)
      tag_preds = tf.to_int32(tf.argmax(tag_logits, axis=-1))
      tag_targets = self.vocabs['tags'].placeholder
      tag_correct = tf.to_int32(tf.equal(tag_preds, tag_targets))*int_tokens_to_keep
      tag_loss = tf.losses.sparse_softmax_cross_entropy(tag_targets, tag_logits, self.tokens_to_keep)
    
    with tf.variable_scope('XTag'):
      xtag_logits = self.linear(xtag_mlp, len(self.vocabs['xtags']))
      xtag_probs = tf.nn.softmax(xtag_logits)
      xtag_preds = tf.to_int32(tf.argmax(xtag_logits, axis=-1))
      xtag_targets = self.vocabs['xtags'].placeholder
      xtag_correct = tf.to_int32(tf.equal(xtag_preds, xtag_targets))*int_tokens_to_keep
      xtag_loss = tf.losses.sparse_softmax_cross_entropy(xtag_targets, xtag_logits, self.tokens_to_keep)
    
    correct = tag_correct * xtag_correct
    n_correct = tf.reduce_sum(correct)
    n_tag_correct = tf.reduce_sum(tag_correct)
    n_xtag_correct = tf.reduce_sum(xtag_correct)
    n_seqs_correct = tf.reduce_sum(tf.to_int32(tf.equal(tf.reduce_sum(correct, axis=1), self.sequence_lengths-1)))
    loss = tag_loss + xtag_loss
    
    outputs = {
      'tag_logits': tag_logits,
      'tag_probs': tag_probs,
      'tag_preds': tag_preds,
      'tag_targets': tag_targets,
      'tag_correct': tag_correct,
      'tag_loss': tag_loss,
      'n_tag_correct': n_tag_correct,

      'xtag_logits': xtag_logits,
      'xtag_probs': xtag_probs,
      'xtag_preds': xtag_preds,
      'xtag_targets': xtag_targets,
      'xtag_correct': xtag_correct,
      'xtag_loss': xtag_loss,
      'n_xtag_correct': n_xtag_correct,
      
      'n_tokens': self.n_tokens,
      'n_seqs': self.batch_size,
      'tokens_to_keep': self.tokens_to_keep,
      'n_correct': n_correct,
      'n_seqs_correct': n_seqs_correct,
      'loss': loss
    }
    
    return outputs
