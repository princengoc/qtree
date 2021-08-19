#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:36:26 2021

@author: Johannes

QTree Simulationstudy

Testing QTree without Parameter Selection for various Parameters


  Parameters
  ----------
  rep_n: Number of repetitions
  d_n= Number of Nodes
  n_n= Number of Observations
  noise_to_signal_n: Noise to Signal Ratio
  c1: Minimum Edge weight (in max-times model)
  c2: Maximum Edge weight (in max-times model)
  v: Shape Parameter of the Frechet Distribution
    
  Returns
  -------
  Saves Results in the folder output/QTreeSim:
  
  (G_true,G_est,score): (numpy array,numpy array,dict)
  
  G_true: True underlying Graph
  G_est: Estimated underlying Graph
  score: Dict of metrics between G_true and G_est: fdr (False Discovery Rate), tpr (True Positive Rate)
         fpr (False Positive Rate), shd (Normalized Structural Hamming Distance) as well as 
         fdr_r, tpr_r, fpr_r and shd_r for the metrics of the respective reachability graphs


"""

import os
from QTree import QTree
from utils import saveTo, count_accuracy
import pandas as pd
import scipy.stats as st
import numpy as np
import networkx as nx
import random
 
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

  def fromCtoB(C):
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
    for i in range(n):
        Y=[np.multiply(Z[i,:],B[:,j]) for j in range(d)]
        X[i,:]=np.amax(Y, axis=1)
    return(np.log(X))   

if __name__ == "__main__":
  
  np.random.seed(1) #fix seed
  #set parameters
  rep_n=100
  
  d_n=[10,30,50,100]
  n_n=[10,25,50, 100,200, 500, 1000]
  noise_to_signal_n=[0.1,0.2,0.3]
  
  c1=0.2
  c2=1
  
  #shape
  v=1
  for rep in range(rep_n):
      print("Repetition: " + str(rep+1) + " of " + str(rep_n))
      for idx_d,d in enumerate(d_n):
          C=create_sa(d,c1,c2)
          B=fromCtoB(C)
          G_true=C>0

          for idx_n,n in enumerate(n_n):
              Z=st.invweibull.rvs(v, 0, 1, size=(n,d))
              X=MLMatrixMult(Z,B)
              sample_std=np.median(np.std(X,0))

              for idx_noise_to_signal, noise_to_signal in enumerate(noise_to_signal_n):
                  noise=np.random.normal(loc=0.0, scale=(noise_to_signal*sample_std), size=(n,d))
                  Y=X+noise
                  
                  estimator = QTree(q=0)
                  G_est=estimator.fit(pd.DataFrame(Y))
                  
                  score=count_accuracy(G_true,[G_est])[0]
                  
                  save_folder=os.path.join('output/QTreeSim','qtree_rep_'+str(rep)+'_d_'+str(d)+'_n_'+str(n)+'_noise_'+str(noise_to_signal))
                  if save_folder is not None:
                      obj = (G_true,G_est,score)  
                      fname = 'output'+ ".pk"
                      saveTo(obj, save_folder,fname)
