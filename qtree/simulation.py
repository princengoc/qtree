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
      
  1. Scores.pk: list of dictionaries of all mean scores
  
  2. fdr.png/tpr.png/fpr.png/shd.png Plots the respective metrics for increasing sample size n
  
  3. For every ieration, every sample size, graph size and noise, creates a folder and saves the output

     (G_true,G_est,score): (numpy array,numpy array,dict)  
  
     G_true: True underlying Graph
     G_est: Estimated underlying Graph
     score: Dict of metrics between G_true and G_est: fdr (False Discovery Rate), tpr (True Positive Rate)
         fpr (False Positive Rate), shd (Normalized Structural Hamming Distance) as well as 
         fdr_r, tpr_r, fpr_r and shd_r for the metrics of the respective reachability graphs
"""

import os
from QTree import QTree
from utils import saveTo, count_accuracy, create_sa, kleene, MLMatrixMult
from simulationplot import plotMetric
import pandas as pd
import scipy.stats as st
import numpy as np




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
  scores=[]
  keys=['fdr', 'tpr', 'fpr', 'shd', 'fdr_r', 'tpr_r', 'fpr_r', 'shd_r'] 
  
  for rep in range(rep_n):
      
      for idx_d,d in enumerate(d_n):
          C=create_sa(d,c1,c2)
          B=kleene(C)
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

                  save_folder=os.path.join('output/QTreeSim','qtree_rep_'+str(rep)+'_d_'+str(d)+'_n_'+str(n)+'_noise_'+str(noise_to_signal))
                  if save_folder is not None:
                      obj = (G_true,G_est,count_accuracy(G_true,[G_est])[0])  
                      fname = 'output'+ ".pk"
                      saveTo(obj, save_folder,fname)
                      
                      
                  score = {k: v / rep_n for k, v in count_accuracy(G_true,[G_est])[0].items()}
                  score.update({'d':d,'n':n,'noise': noise_to_signal})
                  
                  if not [s for s in scores if s['d'] == d and s['n']==n and s['noise']==noise_to_signal]:
                      scores+=[score]
                      continue
                                   
                  for s in scores:
                      if s['d']==d and s['n']==n and s['noise']==noise_to_signal:
                          for k in keys:
                              s.update({k:s[k]+score[k]})
    
  save_folder='output/QTreeSim'     
  if save_folder is not None:
      fname = 'Scores'+ ".pk"
      saveTo(scores, save_folder,fname)
      
  for metric in ['fdr', 'tpr', 'fpr', 'shd']:
       plotMetric(scores,0.3,metric)