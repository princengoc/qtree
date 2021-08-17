# QTree

Implementation of the QTree algorithm as appeared in the paper
*Latent Trees for Extremes* by Tran, Buck and Kluppelberg. 

## Setup


Easiest way is to clone the directory

```
git clone https://github.com/princengoc/qtree
cd qtree 
```

The included data files are the 5 rivers presented in the paper. These are: the Danube, three sections of the Lower Colorado, and a subset of the Lower Colorado for stations with at least 150 observations in the raw time-series.

Alternatively, you can do

`python -m pip install git+git://github.com/princengoc/qtree` 

### Reproduce all the results in the paper
To reproduce all the results in the paper, including the plots, do

`python main.py`

*Warning*: this takes time, as the paper uses 1000 repetitions for each model parameter for each river. Similarly good results can be obtained by changing `nrep` to 100 in `main.py`. 

## Apply QTree for a particular river

1. Make a data file for your river and put it at
`qtree/data/your-river-name/data.pk`. 

`data.pk` is a Python `pickle` file that contains the tuple
`(data_frame, labels,G_true)`

where 
* `data_frame` is a pandas dataframe (number-of-observations number-of-nodes)
* `labels` is a dictionary of labels for the columns of `data_frame`
* `G_true` is a V times V numpy array of {0,1} that is the adjacency matrix for the true river network. If the true river network is unknown, `G_true` should be `None`. 

An example format is in `qtree/data/danube/data.pk` 

2. Run `fit` in `main.py`. 

Modify the `__main__` block of `main.py` to select the parameters and river you want. The default fits qtree with automated parameter selection on all rivers in the list `rivers`. For each river, results (eg: estimated trees) are saved to the folder `save/(river)/`. 

The parameters you generally want to modify are: 
* river: your river name
* nrep: number of repetitions of subsampling
* frac: fraction of data subsampled for fitting
* q_range : range of q parameters to search over
* smallR_range: range of smallR parameters to search over

These are described in details in Algorithm 2 of the paper and in the documentation of `generateSubsamples` and `qtreeAutoSelect` in `qtreeAutoSelect.py`. These functions together make up Algorithm 2. They do the following

  1. `generateSubsamples`: Subsample a fraction `frac` of data, repeat `nrep` times. Save the generated samples as pickled list to `save/(river)/samples.pk`.
  2. `qtreeAutoSelect`:  fits the QTree algorithm (Algorithm 1 of paper) to each subsample, compute the centroid eT and the variability varT, and find the best parameters and the estimated tree eT
  
For a debug run, you can also modify the `__main__` block of `qtreeAutoSelect.py`, which defaults to fitting qtree on the Danube only.   

### Do performance plots if the true river network is known. 

If the true network is known and you just want to check how well QTree works, after running `fit` in `main.py`, you can run `plot`. Plot options are given in the `__main__` block of `main.py`. 







