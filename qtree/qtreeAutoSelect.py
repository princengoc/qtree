#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:36:26 2021

@author: ngoc

QTree algorithm with automated parameter selection. 
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
from QTree import QTree
from utils import count_accuracy,spanTree, getReachability, getSupport, saveTo, _appendModelInfo, _count_accuracy, count_accuracy
from collections import Counter as Ctr

def generateSubsamples(df,save_folder, nrep = 100, frac = 0.75, seed_entropy = 237262676468864319646780408567402854442,resample=True):
  """
  Subsample a fraction frac of data, repeat nrep times
  Save the generated samples as pickled list to save/(river)/samples.pk.   
  
  Parameters
  ----------
  df: pandas DataFrame, shape (n_samples, n_nodes)
    The input data. Each row is an observation where some of the nodes have extreme values. 
  frac: a number in [0,1]
    fraction of data to be sampled without replacement each time
  nrep: int
    number of times data to be subsampled
  seed_entropy: int
    for reproducibility of data subsampling. 
  river: 
    subfolder name. (Recommended to be the river name). 
  resample: 
    default True.Generate new samples. 
    if False: try to load existing samples from
    save/(river)/samples.pk
  """
  if not resample:
    with open(os.path.join(save_folder, 'samples.pk'), 'rb') as file_name:
      samples = pickle.load(file_name)          
    return samples

  #set a deterministic sequence of random seeds for reproducibility  
  sq1 = np.random.SeedSequence(seed_entropy)
  seeds = sq1.generate_state(nrep) 
  
  samples = [df.sample(frac = frac,random_state =seeds[i]) for i in range(nrep)]
  
  saveTo(samples,save_folder,'samples.pk')    

  return samples


def qtreeAutoSelect(samples, save_folder, q_range = [0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9], smallR_range = [0.05], weights = 'nan', refit = True,saveAll=True): 
  """
  QTree algorithm with automated parameter selection. 
  
  Parameters
  ----------
  samples: list 
    list of pandas DataFrame with SAME column set.   
    Each can be a random subsample from a dataset.  
  q_range: list
    list of q values (numbers from 0 to 1)
  smallR_range: list-like
    list of smallR values (numbers from 0 to 1)
  weights: str.
    Default is 'nan'. Each node is weighted by (number of non-nan obs), as an indicator of reliability
    If None, all nodes and all edges are weighted equally
  refit: Boolean
    if True:  fit model to the samples
    if False: load the fitted models in save_folder/trees_on_subsamples[q][tau].pk
  save_folder: str
    path to folder that contains the fitted models
    if refit is True: the function saves the fitted models after fitting. 
  saveAll: Boolean
    if True: save the centroids for all parameters to save_folder/centroids_all_params.pk
    
  Returns
  -------
    centroidAndVars: pandas DataFrame
      of var_T and e_T for each qtree model with params (q,smallR) 
    idx_best: 
      index of the best model 
  """
  centroidAndVars = []
    
  for smallR in smallR_range:
    for q in q_range:  
      if not refit:       
        fname = _appendModelInfo("trees_on_subsamples", q,smallR)
        with open(os.path.join(save_folder, fname), 'rb') as file:      
          (trees,trees_reach,e_T, var_T) = pickle.load(file)        
      else: 
        model =  QTree(smallR = smallR, q = q,weights = weights)  
        e_T,var_T = model.fitToSamples(samples,save_folder=save_folder)  
      centroidAndVars += [{'q': q, 'smallR': smallR, 'var_T': var_T,'e_T': e_T}]

  centroidAndVars = pd.DataFrame(centroidAndVars)  
  idx_best = centroidAndVars['var_T'].idxmin()
  output = (centroidAndVars,idx_best)
  
  if saveAll:
    saveTo(output,save_folder,'centroids_all_params.pk')
  
  return output


def saveScores(G_true, **kwargs): 
  """
  Compute the score of the auto select procedure
  when G_true is known and save to save_folder/scores.pk
  
  Saved objects: 
  scores_subtrees
    pandas dataframe, consist of scores all trees fitted to subsamples, for each model. 
  scores_centroid
    pandas dataframe, consists of scores the centroid trees. 
  
  Parameters
  ----------
  G_true: (d,d) np.array {0,1}
    the true tree
  kwargs: parameters used to generate the fitted model. 
      
  Returns
  -------
  None
  """
  scores_subtrees = []
  scores_centroid = []
  save_folder =kwargs['save_folder']
  for smallR in kwargs['smallR_range']:
    for q in kwargs['q_range']:  
      fname = _appendModelInfo("trees_on_subsamples", q,smallR)
      with open(os.path.join(save_folder, fname), 'rb') as file:      
        (trees,trees_reach,e_T, var_T) = pickle.load(file)                
      scores = count_accuracy(G_true,trees)
      #update scores with model info
      model_params = {'q': q, 'smallR': smallR}
      for score_dict in scores:
        score_dict.update(model_params) 
      scores_subtrees += scores
      
      score_eT = count_accuracy(G_true,[e_T])[0]
      score_eT.update(model_params)
      scores_centroid += [score_eT]

  #save to files
  with open(os.path.join(save_folder, 'scores.pk'), 'wb') as file:      
    output = (pd.DataFrame(scores_subtrees),pd.DataFrame(scores_centroid))
    pickle.dump(output, file)  
  

if __name__ == "__main__":
  #sample run on the Danube
  river = 'danube' 
  save_folder = os.path.join('save',river)
  #set parameters
  sampling_parameters = {'nrep': 10, 'frac': 0.75, 'resample': True, 'save_folder': save_folder} 
  model_parameters = {'q_range': [0.6,0.7,0.8], 'smallR_range': [0.05], 'weights': 'nan', 'refit': True, 'saveAll': True, 'save_folder': save_folder} 

  print('processing river', river)  
  data_file = os.path.join('data',river,'data.pk')
  if not os.path.exists(data_file): 
    raise ValueError('data not yet generated for ', river)  
  
  #load data
  with open(data_file, 'rb') as file_name:
      df,labels,G_true = pickle.load(file_name)                  
  
  #generate samples
  samples = generateSubsamples(df, **sampling_parameters)
  #fit model, compute centroid and variability, and find best model
  (centroidAndVars,idx_best) = qtreeAutoSelect(samples, **model_parameters)
  if G_true is not None:
    saveScores(G_true,**model_parameters)
  print('done fitting river', river, '- please run plotting codes to visualize the results.')  
