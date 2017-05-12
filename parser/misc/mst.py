# !/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#***************************************************************
#===============================================================
def find_cycles(edges):
  """"""
  
  vertices = np.arange(len(edges))
  indices = np.zeros_like(vertices) - 1
  lowlinks = np.zeros_like(vertices) - 1
  stack = []
  onstack = np.zeros_like(vertices, dtype=np.bool)
  current_index = 0
  cycles = []
  
  #-------------------------------------------------------------
  def strong_connect(vertex, current_index):
    """"""
    
    indices[vertex] = current_index
    lowlinks[vertex] = current_index
    stack.append(vertex)
    current_index += 1
    onstack[vertex] = True
    
    for vertex_ in np.where(edges == vertex)[0]:
      if indices[vertex_] == -1:
        current_index = strong_connect(vertex_, current_index)
        lowlinks[vertex] = min(lowlinks[vertex], lowlinks[vertex_])
      elif onstack[vertex_]:
        lowlinks[vertex] = min(lowlinks[vertex], indices[vertex_])
    
    if lowlinks[vertex] == indices[vertex]:
      cycle = []
      vertex_ = -1
      while vertex_ != vertex:
        vertex_ = stack.pop()
        onstack[vertex_] = False
        cycle.append(vertex_)
      if len(cycle) > 1:
        cycles.append(np.array(cycle))
    return current_index
  #-------------------------------------------------------------
  
  for vertex in vertices:
    if indices[vertex] == -1:
      current_index = strong_connect(vertex, current_index)
  return cycles

#===============================================================
def find_roots(edges):
  """"""
  
  return np.where(edges[1:] == 0)[0]
  
#***************************************************************
def argmax(probs):
  """"""
  
  edges = np.argmax(probs, axis=1)
  return edges
  
#===============================================================
def greedy(probs):
  """"""
  
  edges = np.argmax(probs, axis=1)
  cycles = True
  while cycles:
    cycles = find_cycles(edges)
    for cycle_vertices in cycles:
      # Get the best heads and their probabilities
      cycle_edges = edges[cycle_vertices]
      cycle_probs = probs[cycle_vertices, cycle_edges]
      # Get the second-best edges and their probabilities
      probs[cycle_vertices, cycle_edges] = 0
      backoff_edges = np.argmax(probs[cycle_vertices], axis=1)
      backoff_probs = probs[cycle_vertices, backoff_edges]
      probs[cycle_vertices, cycle_edges] = cycle_probs
      # Find the node in the cycle that the model is the least confident about and its probability
      new_root_in_cycle = np.argmax(backoff_probs/cycle_probs)
      new_cycle_root = cycle_vertices[new_root_in_cycle]
      # Set the new root
      probs[new_cycle_root, cycle_edges[new_root_in_cycle]] = 0
      edges[new_cycle_root] = backoff_edges[new_root_in_cycle]
  return edges

#===============================================================
def chu_liu_edmonds(probs):
  """"""
  
  vertices = np.arange(len(probs))
  edges = np.argmax(probs, axis=1)
  cycles = find_cycles(edges)
  if cycles:
    print("found cycle, fixing...")
    # (c)
    cycle_vertices = cycles.pop()
    # (nc)
    non_cycle_vertices = np.delete(vertices, cycle_vertices)
    #-----------------------------------------------------------
    # (c)
    cycle_edges = edges[cycle_vertices]
    # get rid of cycle nodes
    # (nc x nc)
    non_cycle_probs = np.array(probs[non_cycle_vertices,:][:,non_cycle_vertices])
    # add a node representing the cycle
    # (nc+1 x nc+1)
    non_cycle_probs = np.pad(non_cycle_probs, [[0,1], [0,1]], 'constant')
    # probabilities of heads outside the cycle
    # (c x nc) / (c x 1) = (c x nc)
    backoff_cycle_probs = probs[cycle_vertices][:,non_cycle_vertices] / probs[cycle_vertices,cycle_edges][:,None]
    # probability of a node inside the cycle depending on something outside the cycle
    # max_0(c x nc) = (nc)
    non_cycle_probs[-1,:-1] = np.max(backoff_cycle_probs, axis=0)
    # probability of a node outside the cycle depending on something inside the cycle
    # max_1(nc x c) = (nc)
    non_cycle_probs[:-1,-1] = np.max(probs[non_cycle_vertices][:,cycle_vertices], axis=1)
    #-----------------------------------------------------------
    # (nc+1)
    non_cycle_edges = chu_liu_edmonds(non_cycle_probs)
    # This is the best source vertex into the cycle
    non_cycle_root, non_cycle_edges = non_cycle_edges[-1], non_cycle_edges[:-1] # in (nc)
    source_vertex = non_cycle_vertices[non_cycle_root] # in (v)
    # This is the vertex in the cycle we want to change
    cycle_root = np.argmax(backoff_cycle_probs[:,non_cycle_root]) # in (c)
    target_vertex = cycle_vertices[cycle_root] # in (v)
    edges[target_vertex] = source_vertex
    # update edges with any other changes
    mask = np.where(non_cycle_edges < len(non_cycle_probs)-1)
    edges[non_cycle_vertices[mask]] = non_cycle_vertices[non_cycle_edges[mask]]
    mask = np.where(non_cycle_edges == len(non_cycle_probs)-1)
    edges[non_cycle_vertices[mask]] = cycle_vertices[np.argmax(probs[non_cycle_vertices][:,cycle_vertices], axis=1)]
  return edges

#===============================================================
def nonprojective(probs, sequence_length):
  """"""
  
  probs *= 1-np.eye(len(probs)).astype(np.float32)
  probs[0] = 0
  probs[0,0] = 1
  probs[:,sequence_length:] = 0
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  #edges = chu_liu_edmonds(probs)
  edges = greedy(probs)
  roots = find_roots(edges)
  best_edges = edges
  best_score = -np.inf
  if len(roots) > 1:
    for root in roots:
      probs_ = make_root(probs, root)
      #edges_ = chu_liu_edmonds(probs_)
      edges_ = greedy(probs_)
      score = score_edges(probs_, edges_)
      if score > best_score:
        best_edges = edges_
        best_score = score
  return best_edges

#===============================================================
def make_root(probs, root):
  """"""
  
  probs = np.array(probs)
  probs[1:,0] = 0
  probs[root,:] = 0
  probs[root,0] = 1
  probs /= np.sum(probs, axis=1, keepdims=True)
  return probs

#===============================================================
def score_edges(probs, edges):
  """"""
  
  return np.sum(np.log(probs[np.arange(1,len(probs)), edges[1:]]))

#***************************************************************
if __name__ == '__main__':
  def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=1, keepdims=True)
  probs = softmax(np.random.randn(100,100))
  probs *= 1-np.eye(len(probs)).astype(np.float32)
  probs[0] = 0
  probs[0,0] = 1
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  edges = nonprojective(probs)
  roots = find_roots(edges)
  best_edges = edges
  best_score = -np.inf
  if len(roots) > 1:
    for root in roots:
      probs_ = make_root(probs, root)
      edges_ = nonprojective(probs_)
      score = score_edges(probs_, edges_)
      print(score)
      if score > best_score:
        best_edges = edges_
        best_score = score
  edges = best_edges
  print(edges)
  print(np.arange(len(edges)))
  print(find_cycles(edges))
  print(find_roots(edges))
