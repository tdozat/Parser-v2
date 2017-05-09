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

from parser.neural.models.parsers.base_parser import BaseParser

#***************************************************************
class XbarParser(BaseParser):
  """"""
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """"""
    
    top_recur = super(Parser, self).__call__(vocabs, moving_params=moving_params)
    
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = self.MLP(top_recur, self.arc_mlp_size + self.rel_mlp_size + self.p_mlp_size,
                                   n_splits=2)
      arc_dep_mlp, rel_dep_mlp, p_dep_mlp = tf.split(dep_mlp, [self.arc_mlp_size, self.rel_mlp_size, self.p_mlp_size], axis=2)
      arc_head_mlp, rel_head_mlp, p_head_mlp = tf.split(head_mlp, [self.arc_mlp_size, self.rel_mlp_size, self.p_mlp_size], axis=2)
    
    with tf.variable_scope('p'):
      # (n x b x d) o (d x 1 x d) o (n x b x d).T -> (n x b x b)
      arc_ps = tf.nn.sigmoid(self.bilinear(p_dep_mlp, p_head_mlp, 1, add_bias2=False, add_bias=False))
    with tf.variable_scope('Arc'):
      # (n x b x d) o (d x 1 x d) o (n x b x d).T -> (n x b x b)
      arc_logits = self.bilinear(arc_dep_mlp, arc_head_mlp, 1, add_bias2=False, add_bias=False)
      # (b x 1)
      i_mat = tf.expand_dims(tf.range(self.batch_size), 1)
      # (1 x b)
      j_mat = tf.expand_dims(tf.range(self.batch_size), 0)
      # (b x 1) > (1 x b) -> (b x b)
      k_mat = j_mat > i_mat
      # (b x 1)
      n_mat = tf.expand_dims(self.sequence_lengths, 1) - 1 - i_mat
      # (n x b x b) + (b x b) * (n x b x b) + (b x b) * (n x b x b) -> (n x b x b)
      arc_logits += (
        tf.to_float(j_mat > i_mat)*(tf.log(arc_ps)) + 
        tf.to_float(j_mat < i_mat)*(tf.log(1-arc_ps)) )
      # (n x b x b) - (b x b) * (b x b) -> (n x b x b)
      arc_logits -= tf.log(n_mat*(1-tf.eye(self.batch_size)))
      # (n x b x b)
      arc_probs = tf.nn.softmax(arc_logits)
      # (n x b)
      arc_preds = tf.argmax(arc_logits, axis=-1)
      # (n x b)
      arc_targets = self.vocabs['heads'].placeholder
      # (n x b)
      arc_correct = tf.to_float(tf.equal(arc_preds, arc_targets))*self.tokens_to_keep
      # ()
      arc_loss = tf.losses.sparse_softmax_cross_entropy(arc_targets, arc_logits, self.tokens_to_keep)
    
    with tf.variable_scope('Rel'):
      # (n x b x d) o (d x r x d) o (n x b x d).T -> (n x b x r x b)
      rel_logits = self.bilinear(rel_dep_mlp, rel_head_mlp, len(self.vocabs['rels']))
      # (n x b x r x b)
      rel_probs = tf.nn.softmax(rel_logits)
      # (n x b x b)
      one_hot = tf.one_hot(arc_preds if moving_params is not None else arc_targets, self.bucket_size)
      # (n x b x b) -> (n x b x b x 1)
      one_hot = tf.expand_dims(one_hot, axis=3)
      # (n x b x r x b) o (n x b x b x 1) -> (n x b x r x 1)
      select_rel_logits = tf.matpl(rel_logits, one_hot)
      # (n x b x r x 1) -> (n x b x r)
      select_rel_logits = tf.squeeze(select_rel_logits, axis=3)
      # (n x b)
      rel_preds = tf.argmax(select_rel_logits, axis=-1)
      # (n x b)
      rel_targets = self.vocabs['rels'].placeholder
      # (n x b)
      rel_correct = tf.to_float(tf.equal(rel_preds * rel_targets))*self.tokens_to_keep
      # ()
      rel_loss = tf.losses.sparse_softmax_cross_entropy(rel_targets, select_rel_logits, self.tokens_to_keep)
    
    n_correct = tf.reduce_sum(arc_correct * rel_correct)
    n_arc_correct = tf.reduce_sum(arc_correct)
    n_rel_correct = tf.reduce_sum(rel_correct)
    loss = arc_loss + rel_loss
    
    outputs = {
      'arc_logits': arc_logits,
      'arc_probs': arc_probs,
      'arc_pred': arc_preds,
      'arc_targets': arc_targets,
      'arc_correct': arc_correct,
      'arc_loss': arc_loss,
      'n_arc_correct': n_arc_correct,
      
      'rel_logits': rel_logits,
      'rel_probs': rel_probs,
      'rel_preds': rel_preds,
      'rel_targets': rel_targets,
      'rel_correct': rel_correct,
      'rel_loss': rel_loss,
      'n_rel_correct': n_rel_correct,
      
      'n_tokens': self.n_tokens,
      'tokens_to_keep': self.tokens_to_keep,
      'n_correct': n_correct,
      'loss': loss
    }
    
    return outputs
