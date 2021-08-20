#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 11:36:26 2021

@author: ngoc

Utility functions. 
"""


import numpy as np
import networkx as nx
import os
import pickle
import random


#naming convention files
def _appendModelInfo(fname,q,smallR):
  return fname+"_q_"+ str(q) + "_r_" + str(smallR) + ".pk"

def saveTo(obj,save_folder,fname): 
  """ Save obj to save_folder/fname with pickle
  Create save_folder if it doesn't exist yet."""
  if not os.path.exists(save_folder):    
    os.makedirs(save_folder)
  with open(os.path.join(save_folder, fname), 'wb') as file:      
    pickle.dump(obj, file)   


def getReachability(A):
  """Returns the reachability matrix from the adjacency matrix A. 
  Param: 
    A: (d,d) np.array
  """
  #Amat = reshapeToMatrix(A)  
  G = nx.DiGraph(A)
  length = dict(nx.all_pairs_shortest_path_length(G))
  R = np.array([[length.get(m, {}).get(n, 0) > 0 for m in G.nodes] for n in G.nodes], dtype=np.int32)
  return R.T  
    
def getSupport(G):
  """Returns the support of a networkx graph as an np array """    
  B = nx.to_numpy_array(G)
  return(B != 0)

def getRiver(labels,river='lower-colorado',prefix=None):
  """Returns the support of the true river as an np array from given labels"""
  B = np.zeros((len(labels),len(labels)))
  label_inv = dict([(labels[i],i) for i in range(len(labels))])  
  edges = np.loadtxt(prefix+'data/' + river +  '/adjacency.txt', delimiter=' ',dtype=np.str)
  for e in edges:
    i,j = e
    B[label_inv[i],label_inv[j]] = 1
  return B    

def spanTree(W,isMin=True):
  """Wrapper to find min root-directed spanning tree in a graph with adjacency matrix W
  if isMin = False then find the max spanning tree instead
  """
  if isMin:
    G = nx.DiGraph(W.T*1.0)
  else:
    G = nx.DiGraph(W.T*-1.0)
  try:
    tree = nx.minimum_spanning_arborescence(G)
  except:
    #try returning the spanning arborescence anyway, without the is_arboresence test. 
    import networkx.algorithms.tree.branchings as bb
    ed = bb.Edmonds(G)
    tree = ed.find_optimum('weight', default=1, kind="min", style="arborescence", preserve_attrs=False)    
  return tree.reverse(copy=False)


def _count_accuracy(G_true, G_est, name_suffix = ''):
    """Compute various accuracy metrics for G_est.
    
    This code is a small modification of the count_accuracy function in
      https://github.com/xunzheng/notears/blob/master/notears/utils.py 
    Unlike Xunzheng et al, we use normalized SHD instead of SHD (see definition below)   

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        G_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        G_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
        score_suffix: str
          string to append to each key name in the return matrix
          eg: str = '_r' would return keys 'fdr_r' instead of 'fdr'

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: (undirected extra + undirected missing + reverse) / (worst case = number of edges in true graph + number of edges in the estimated graph)
    """
    if not ((G_est == 0) | (G_est == 1)).all():
      raise ValueError('G_est should take value in {0,1}')
    d = G_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(G_est == -1)
    pred = np.flatnonzero(G_est == 1)
    cond = np.flatnonzero(G_true)
    cond_reversed = np.flatnonzero(G_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(G_est + G_est.T))
    cond_lower = np.flatnonzero(np.tril(G_true + G_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    #normalize shd
    shd = shd / (np.sum(G_true) + np.sum(G_est))
    return {'fdr'+name_suffix: fdr, 'tpr'+name_suffix: tpr, 'fpr'+name_suffix: fpr, 'shd'+name_suffix: shd}

def count_accuracy(G_true, G_list): 
  """Compute the accuracy scores of G_true vs each tree in G_list, 
  for both the tree *and* the reachability graphs. 
  
  Args: 
    G_true (np.ndarray): [d, d] ground truth graph, {0, 1}
    G_list: list
      each entry is an (np.ndarray): [d, d] estimated graph, {0, 1}. 

  Returns: 
    list of scores
    each entry is a dictionary with keys
    ['fdr', 'tpr', 'fpr', 'shd', 'fdr_r', 'tpr_r', 'fpr_r', 'shd_r']
    
  """
  R_true = getReachability(G_true)
  scores_dict = []
  for G_est in G_list: 
    R_est = getReachability(G_est)
    scores = _count_accuracy(G_true,G_est)  
    _scores_reachable = _count_accuracy(R_true,R_est,name_suffix='_r')
    scores.update(_scores_reachable)
    scores_dict += [scores]
  return scores_dict

def create_sa(d,c1,c2):
    """Generates a random spanning arborescence with d nodes and edge weights uniformly between c1 and c2
    Args: 
      d (int): Number of nodes
      c1 (float): minimum edge weight
      c2 (float): maximum edge weight
    Returns: 
      C (d x d numpy array): Edge weight matrix
    """
    G=nx.generators.trees.random_tree(d)    
    v=[random.choice(list(G.nodes))]
    A=np.zeros((d,d))
    while len(v):
        v2=[]
        for i in v:
            n=list(G.neighbors(i))
            for n_k in n:
                G.remove_edge(n_k,i)
            A[n,i]=1
            v2+=n
        v=np.copy(v2)
    C=np.multiply(np.random.uniform(c1, high=c2, size=(d,d)),A)           
    return(C)

def kleene(C):
    """Calculates the Kleene Star Matrix B from C (Max-times)
    Args: 
      C (d x d numpy array): Edge weight matrix
    
    Returns: 
      B (d x d numpy array): Kleene star matrix
      
    """
    d=np.size(C,0)
    G=nx.DiGraph(C)
    top_sort=list(nx.topological_sort(G))  
    B=np.copy(C)+np.eye(d)
    #different distance in top. order
    for idx in range(2,d):        
        for idy, val in enumerate(top_sort[0:d-idx]):
            B[val,top_sort[idy+idx]]=np.amax(np.multiply(B[val,top_sort[idy:(idy+idx)]],B[top_sort[idy:(idy+idx)],top_sort[idx+idy]]))
    return(B)

def MLMatrixMult(Z,B):

    """Calculates max-linear data, returns log-data    
    Args: 
      Z (n x d numpy array): Array of all sources, where n is the number of observations, d the number of nodes
      B (d x d numpy array): Kleene star matrix
    
    Returns: 
      X (n x d numpy array): max-linear log-data, ie: X = \log(B \odot Z)
    """
    (n,d) = np.shape(Z) 
    X=np.zeros((n,d))
    for i in range(n):
        Y=[np.multiply(Z[i,:],B[:,j]) for j in range(d)]
        X[i,:]=np.amax(Y, axis=1)
    return(np.log(X))   
