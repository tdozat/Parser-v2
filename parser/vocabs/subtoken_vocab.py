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
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.vocabs import TokenVocab
from parser import Multibucket
from parser.misc.bucketer import Bucketer

__all__ = ['CharVocab']

#***************************************************************
class SubtokenVocab(TokenVocab):
  """"""
  
  #=============================================================
  def __init__(self, token_vocab, *args, **kwargs):
    """"""
    
    recount = kwargs.pop('recount', False)
    initialize_zero = kwargs.pop('initialize_zero', False)
    super(TokenVocab, self).__init__(*args, **kwargs)
    
    self._token_vocab = token_vocab
    self._token_counts = Counter()
    self._multibucket = Multibucket.from_configurable(self, embed_model=self.embed_model, name=self.name)
    self._tok2idx = {}
    
    if recount:
      self.count()
    else:
      if os.path.isfile(self.filename):
        self.load()
      else:
        self.count()
        self.dump()
    self.index_vocab()
    
    embed_dims = [len(self), self.embed_size]
    if initialize_zero:
      self.embeddings = np.zeros(embed_dims)
    else:
      self.embeddings = np.random.randn(*embed_dims)
    return
  
  #=============================================================
  def __call__(self, placeholder=None, moving_params=None):
    """"""
    
    placeholder = self.generate_placeholder() if placeholder is None else placeholder
    embeddings = self.multibucket(self, keep_prob=self.embed_keep_prob, moving_params=moving_params)
    return tf.nn.embedding_lookup(embeddings, placeholder)
  
  #=============================================================
  def count(self):
    """"""
    
    special_tokens = set(self.token_vocab.special_tokens)
    for token in self.token_vocab.counts:
      for subtoken in token:
        self.counts[subtoken] += 1
        self.token_counts[subtoken] += self.token_vocab.counts[token]
    return
  
  #=============================================================
  def load(self):
    """"""
    
    with codecs.open(os.path.join(self.save_dir, self.name+'.txt'), encoding='utf-8') as f:
      for line_num, line in enumerate(f):
        try:
          line = line.strip()
          if line:
            line = line.split('\t')
            token, count, token_count = line
            self.counts[token] = int(count)
            self.token_counts[token] = int(token_count)
        except:
          raise ValueError('File %s is misformatted at line %d' % (train_file, line_num+1))
    return
  
  #=============================================================
  def dump(self):
    """"""
    
    with codecs.open(os.path.join(self.save_dir, self.name+'.txt'), 'w', encoding='utf-8') as f:
      for token, count in self.sorted_counts(self._counts):
        f.write('%s\t%d\t%d\n' % (token, count, self.token_counts[token]))
    return
  
  #=============================================================
  def subtoken_indices(self, token):
    """"""
    
    return self[list(token)]
  
  #=============================================================
  def index_tokens(self):
    """"""
    
    self._tok2idx = {}
    tok2idxs = {token: self.subtoken_indices(token) for token in self.token_vocab.counts}
    with Bucketer.from_configurable(self, self.n_buckets, name='bucketer-%s'%self.name) as bucketer:
      splits = bucketer.compute_splits(len(indices) for indices in tok2idxs.values())
    with self.multibucket.open(splits):
      for index, special_token in enumerate(self.token_vocab.special_tokens):
        index = index if index != self.token_vocab.UNK else self.META_UNK
        self.tok2idx[special_token] = self.multibucket.add([index])
      for token, _ in self.sorted_counts(self.token_vocab.counts):
        self.tok2idx[token] = self.multibucket.add(tok2idxs[token])
    self._idx2tok = {idx: tok for tok, idx in self.tok2idx.iteritems()}
    self._idx2tok[0] = self[self.PAD]
    return
  
  #=============================================================
  def set_feed_dict(self, data, feed_dict):
    """"""
    
    uniq, inv = np.unique(data, return_inverse=True)
    # this placeholder stores the indices into the new, on-the-fly embedding matrix
    feed_dict[self.placeholder] = inv.reshape(data.shape)
    unsorted = []
    indices = self.multibucket.indices[uniq]
    for bkt_idx, bucket in enumerate(self.multibucket):
      where = np.where(indices['bkt_idx'] == bkt_idx)[0]
      idxs = indices[where]['idx']
      bucket_data = bucket.indices[idxs]
      # these placeholders store the bucket data's index into the vocab's subtoken matrix
      if bucket_data.shape[0]:
        unsorted.append(where)
        feed_dict[bucket.placeholder] = bucket_data
      else:
        feed_dict[bucket.placeholder] = bucket.indices[0:1]
    # this placeholder makes sure the on-the-fly embedding matrix is in the right order
    feed_dict[self.multibucket.placeholder] = np.argsort(np.concatenate(unsorted))
    return
  
  #=============================================================
  def index(self, token):
    if not self.cased and token not in self._special_tokens_set:
      token = token.lower()
    return self._tok2idx.get(token, self.META_UNK)
  
  #=============================================================
  @property
  def multibucket(self):
    return self._multibucket
  @property
  def token_counts(self):
    return self._token_counts
  @property
  def token_vocab(self):
    return self._token_vocab
  @property
  def token_embed_size(self):
    return (self.token_vocab or self).embed_size
  @property
  def conll_idx(self):
    return self.token_vocab.conll_idx
  @property
  def tok2idx(self):
    return self._tok2idx
  @property
  def idx2tok(self):
    return self._idx2tok
  
  #=============================================================
  def __setattr__(self, name, value):
    if name == '_token_vocab':
      if self.cased is None:
        self._cased = value.cased
      elif self.cased != value.cased:
        cls = value.__class__
        value = cls.from_configurable(value,
                                      cased=self.cased,
                                      recount=True)
    super(SubtokenVocab, self).__setattr__(name, value)
    return

#***************************************************************
class CharVocab(SubtokenVocab):
  pass

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser import Configurable
  from parser.vocabs import WordVocab, CharVocab
  
  configurable = Configurable()
  token_vocab = WordVocab.from_configurable(configurable, 1)
  token_vocab.fit_to_zipf()
  if os.path.isfile('saves/defaults/chars.txt'):
    os.remove('saves/defaults/chars.txt')
  subtoken_vocab = CharVocab.from_vocab(token_vocab)
  subtoken_vocab = CharVocab.from_vocab(token_vocab)
  subtoken_vocab.token_vocab.count(configurable.valid_files)
  subtoken_vocab.index_tokens()
  subtoken_vocab.fit_to_zipf()
  print('SubtokenVocab passes')
