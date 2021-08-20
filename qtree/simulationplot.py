#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:36:26 2021
@author: Johannes
Plot the Results of the QTree Simulationstudy
  Parameters
  ----------
  metric: Metric to plot - can be 'fdr', 'tpr', 'fpr' and 'shd'
  noise= Noise to Signal Ratio; Default values used in the simulation study are 0.1, 0.2 and 0.3
    
  Returns
  -------
  Saves Plot in the folder output/QTreeSim:
      
  metric.png: Figure plotting (n,metric) for all sample sizes n. Different graph sizes d are plotted 
              separately, solid lines plot the metric of the graph, dashed lines of the reachability graph
"""

import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import sys

if __name__ == "__main__":
    
   metric='shd'   
   noise=0.3    
   
   save_folder='output/QTreeSim'

   with open(os.path.join(save_folder, 'Scores.pk'), 'rb') as file_name:
       scores  = pickle.load(file_name)

   d_n= list(dict.fromkeys([s['d'] for s in scores]))
   n_n= list(dict.fromkeys([s['n'] for s in scores]))
   noise_to_signal_n= list(dict.fromkeys([s['noise'] for s in scores]))
   
   
   if noise not in noise_to_signal_n:
       print("Value of noise not found in scores")
       sys.exit() 

   fig, axs = plt.subplots(len(d_n)//2, 2)
   
   for idx_d,d in enumerate(d_n):
       
       graph=[]
       reach=[]
       
       for idx_n,n in enumerate(n_n):
                 graph+=[[s for s in scores if s['d'] == d and s['n']==n and s['noise']==noise][0][metric]] 
                 reach+=[[s for s in scores if s['d'] == d and s['n']==n and s['noise']==noise][0][metric+'_r']]
                       
       axs[idx_d//2, idx_d%2].plot(n_n, graph,color='midnightblue', linestyle='solid', marker='o')
       axs[idx_d//2, idx_d%2].plot(n_n, reach,color='midnightblue', linestyle='dashed', marker='o')
       axs[idx_d//2, idx_d%2].set_xscale('log')
       axs[idx_d//2, idx_d%2].set_xticks(n_n)
       axs[idx_d//2, idx_d%2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
       axs[idx_d//2, idx_d%2].set_xlabel('Sample Size',fontweight="bold")
       axs[idx_d//2, idx_d%2].legend([metric, metric+'_r'])
       axs[idx_d//2, idx_d%2].set_title(metric + " for d = " + str(d), fontweight="bold")
   plt.tight_layout()
   plt.close(fig)
   fig.savefig(save_folder+'/'+metric+'.png', dpi=300) 


