#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:36:26 2021

@author: ngoc

QTree algorithm with automated parameter selection. 

Sample run on the Danube and the Middle Colorado datasets. 


  Parameters
  ----------
  df: pandas DataFrame, shape (n_samples, n_nodes)
    The input data. Each row is an observation where some of the nodes have extreme values. 
  frac: a number in [0,1]
    fraction of data to be sampled without replacement each time
  nrep: int
    number of times data to be subsampled
  q_range: list-like
    list of q values to try
  smallR_range: list-like
    list of smallR values to try
  seed_entropy: int
    for reproducibility of data subsampling. This ensures that all parameters are fitted to the same subsample. 
    See the generateSubSamples function. 
  saveSamples: boolean
    default = True
    if True: save the generated samples as pickled list to save/river/samples.pk. 
  saveAll: boolean
    default = False
    if True: save the estimated trees for all parameters
  river: str
    name of the folder to dump output files. 
    
  Returns
  -------
  (q*,smallR*,tree): (float,float,networkX DiGraph)
    The optimal parameters and the estimated tree. 

  Examples
  --------
  >>> from qtree import qtreeAutoselect
  >>> import numpy as np
  >>> import pandas as pd
  >>> X = np.arange(100).reshape(100, 1)
  >>> df = pd.DataFrame(X)
  >>> tree = qtree(X)
  >>> tree_optimal = qtreeAutoSelect(XXX)


"""

import networkx as nx
import pickle
import os
from qtreeAutoSelect import generateSubsamples, qtreeAutoSelect,saveScores

import matplotlib.pyplot as plt     
import seaborn as sns
from plotCodes import plotComparison       

def fit(sampling_parameters, model_parameters, river='danube'):
  """  Fit qtree with automated parameter selection to river. 
  Compute the scores of the performance if the true river network G_true is supplied in data/(river)/data.pk, 
  but does not plot the performance. 
  
  Return all information needed to plot. 
  """
  print('processing river', river)  
  data_file = os.path.join('data',river,'data.pk')
  if not os.path.exists(data_file): 
    raise ValueError('data not yet generated for ', river)  
  #make output folder
  out_folder = os.path.join('output',river)
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)    
  
  #load raw data
  with open(data_file, 'rb') as file_name:
      df,labels,G_true = pickle.load(file_name)                  
  
  print('generating subsamples')
  samples = generateSubsamples(df, **sampling_parameters)
  print('fit model, compute centroid and find best model')
  (centroidAndVars,idx_best) = qtreeAutoSelect(samples, **model_parameters)
  #save scores if G_true is available
  if G_true is not None: 
    saveScores(G_true,**model_parameters)
  
  fitResults = {'G_true': G_true, 'labels': labels, 'centroidAndVars': centroidAndVars, 'idx_best': idx_best}
  return fitResults

def plot(fitResults, plotOptions, river='danube'):
  """Plot qtree with automated parameter selection
  to river after fitting
  """
  G_true = fitResults['G_true']
  labels = fitResults['labels']
  centroidAndVars = fitResults['centroidAndVars']
  idx_best = fitResults['idx_best']
  
  #make output folder
  out_folder = os.path.join('output',river)
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)    
  
  print('plotting results')
  G = nx.DiGraph(G_true)
  G = nx.relabel_nodes(G,labels)    

  if plotOptions['trueTree']:
    plt.close('all')
    plt.figure(figsize=(20,20))
    plotComparison(G,G,nodesize=5)
    plt.savefig(os.path.join(out_folder,'true.png'))
    plt.close('all')      

  if plotOptions['estimatedTree']:
    plt.close('all')
    tree = nx.DiGraph(centroidAndVars['e_T'][idx_best])
    tree = nx.relabel_nodes(tree,labels)
    plt.figure(figsize=(15,15))
    plotComparison(tree,G,nodesize=6)
    plt.savefig(os.path.join(out_folder,'estimated.png'))
    plt.close('all')
            
  if plotOptions['scores']: 
    with open(os.path.join(save_folder, 'scores.pk'), 'rb') as file_name:
      scores_subtrees,scores_centroid = pickle.load(file_name)

    metrics = {'fdr': 'False Discovery Rates', 'tpr': 'True Positive Rates', 'fpr': 'False positive rates', 'shd': 'Normalized Structural Hamming Distance'}
    
    for metric in metrics.keys():
      ax = scores_subtrees.boxplot(column=[metric], by = ['q'],meanline=True, showmeans=True, showcaps=True,showbox=True)
      scores_subtrees.boxplot(column=[metric+'_r'], by = ['q'],ax=ax,showmeans=True,meanline=True)
      plt.suptitle(metrics[metric] + ' for ' + river)
      plt.title('')
      sns.pointplot(x='q', y=metric, data=scores_centroid, ax=ax)    
      sns.pointplot(x='q', y=metric+'_r', data=scores_centroid, ax=ax, color = 'green')    
      plt.axvline(idx_best,color='red')
      plt.xlabel('q')
      plt.ylabel(metrics[metric])        
      plt.savefig(os.path.join(out_folder,metric+'.png'))
    
    plt.close('all')
    


if __name__ == "__main__":
  #set parameters
  rivers = ['danube','lower-colorado','middle-colorado','upper-colorado','lower-colorado150']
  q_range = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]
  
  for river in rivers: 
    save_folder = os.path.join('save',river)
    sampling_parameters = {'nrep': 1000, 'frac': 0.75, 'resample': True, 'save_folder': save_folder}
    model_parameters = {'q_range': q_range, 'smallR_range': [0.05], 'weights': 'nan', 'refit': True, 'saveAll': True, 'save_folder': save_folder}
    plotOptions = {'trueTree': True, 'estimatedTree': True, 'scores': True}

    fitResults = fit(sampling_parameters,model_parameters,river)  
    plot(fitResults, plotOptions, river)
