#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:35:30 2020

@author: ngoc

The quantile estimator
for the max-linear Bayesian trees (qtree)
without automated parameter selection. 

"""

import numpy as np
import pandas as pd
from utils import spanTree, getSupport, getReachability, saveTo,_appendModelInfo
import os
import pickle

def quantileDispersion(i, df, smallR =0.05, q = 0.8, weights=None):
  """
  Internal function. 
  Returns the i-th row of the dispersion matrix D, ie, (D_{.i}) (i as the source). 
  """   
  select = df[i] >= df[i].quantile(q)
  df_subset = df[select]
  dfi = df_subset.subtract(df_subset[i],axis='rows')
  quantile_lower = dfi.quantile(smallR)
  quantile_lower[i] = np.nan
  means = dfi.mean()
  scores = (quantile_lower-means)**2
  scores[i] = np.nan
  if weights is None:
    return np.array(scores)
  else:
    return np.array([scores[j]/weights[j] for j in range(len(scores))])  
    
    
class QTree:
  """ The quantile estimator for max-linear Bayesian Trees. 

  Parameters
  ----------
  df: pandas DataFrame, shape (n_samples, n_nodes)
    The input data. Each row is an observation where some of the nodes have extreme values. 
    
  Returns
  -------
  tree: networkX DiGraph
    The estimated tree whose nodes correspond to the columns of df. 

  Examples
  --------
  >>> from qtree import qtree
  >>> import numpy as np
  >>> import pandas as pd
  >>> X = np.arange(100).reshape(20, 5)
  >>> df = pd.DataFrame(X)
  >>> estimator = QTree()
  >>> estimator.fit(df)
  """
  
  def __init__(self, smallR =0.05, q = 0.8, weights = 'nan'):
    """
    Parameters
    ----------  
    smallR: a number in [0,1].
      the lower quantile to be computed
      default = 0.05 
    q: a number in [0,1]. 
      the q-bias.
      default = 0.8
    weights: str. None, 'nan'
      Default is 'nan'. Each node is weighted by (number of non-nan obs), as an indicator of reliability
      If None, all nodes and all edges are weighted equally
      nan = 
    """
    self.smallR = smallR
    self.q = q
    self.weights = weights
    self.centroid = None #{0,1} d x d matrix. 
    self.var = None #float
          
  def fit(self, df):
    """Fit the model to data.
    Returns the adjacency matrix of the computed tree, flattened as a vector of length d^2. (TODO: remove the flattening). 

    Parameters
    ----------
    df: pandas DataFrame, shape (n_samples, n_nodes)
      The input data. Each row is an observation where some of the nodes have extreme values. 
    """
    dispersion = []
    V = df.shape[1]
    if self.weights == 'nan':
      weights = df.shape[0]-(df.isna()).sum()  
    else:
      weights = None
    for i in range(V):
      dispersion += [quantileDispersion(i,df,self.smallR,self.q, weights)]
    D = np.vstack(dispersion)
    tree = getSupport(spanTree(D,isMin=True))
    self.centroid = tree
    return tree
    
  def fitToSamples(self, samples = None, save_folder = None):
    """ 
    Fit QTree to multiple subdatasets and 
    return the centroid tree and variability score. 
    
    Parameters
    ----------
    samples: list 
    list of pandas DataFrame with SAME column set.   
    Each can be a random subsample from a dataset.
    save_folder: str
      if is not none, 
        estimated trees for each subsample
        reachability trees for each subsample
        centroid tree
        variability score
      are saved to save_folder/trees_on_subsamples.pk
         
    Returns
    -------
    e_T: networkX DiGraph
      The centroid tree E(T)
    var_T: float 
      The variability score Var(T)       
    """
    #fit tree to samples
    nrep = len(samples)    
    trees = []
    trees_reach = []
    for i in range(nrep):
      X = samples[i]
      tree_i = self.fit(X)
      reach_i = getReachability(tree_i)
      trees += [tree_i]
      trees_reach += [reach_i]
  
    W = np.sum(trees, axis=0)/nrep #agreement matrix
    e_T = getSupport(spanTree(W,isMin=False)) #centroid
    R_T = getReachability(e_T) #reachability of centroid
      
   
    var_T = np.mean([np.sum(R_T != trees_reach[i]) for i in range(nrep)])/np.sum(R_T) + np.mean([np.sum(e_T != trees[i]) for i in range(nrep)])/np.sum(e_T)
    
    if save_folder is not None:
      obj = (trees,trees_reach,e_T, var_T)  
      fname = _appendModelInfo("trees_on_subsamples",self.q,self.smallR)
      saveTo(obj, save_folder,fname)
        
    self.centroid = e_T
    self.var = var_T
    
    return (e_T,var_T)     
