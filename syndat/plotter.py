import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples
import pandas as pd

def bhatta_coef(x1, x2, n_bins = 10):
    cx = np.concatenate((x1, x2))
    h1 = np.histogram(x1, bins = n_bins, range=(np.min(cx), np.max(cx)), density=True)[0]
    h2 = np.histogram(x2, bins = n_bins, range=(np.min(cx), np.max(cx)), density=True)[0]
    bhatta_coef = 0
    for i in range(n_bins):
        bhatta_coef += np.sqrt(h1[i]*h2[i]) * (np.max(cx)-np.min(cx))/n_bins
    return bhatta_coef

def bhatta_dist(x1, x2, n_bins = 10):
    return -np.log(bhatta_coef(x1, x2, n_bins=n_bins))

def compare_hists(orig, syn, cols):
    nbin = 50
    for c in cols:
        plt.figure()
        _max = np.nanmax(np.concatenate((orig[c],syn[c])), )
        _min = np.nanmin(np.concatenate((orig[c],syn[c])))
        orig[c].hist(bins = nbin, alpha=0.5, range=(_min,_max), density=True, label='Original')
        syn[c].hist(bins = nbin, alpha=0.5, range=(_min,_max), density=True, label='Synthetic')
        
        d = bhatta_coef(orig[c].to_numpy(), syn[c].to_numpy())
        
        plt.title(c+'\nBhattacharyya coef = {:.2f}'.format(d))
        plt.legend()
        plt.show()

def compare_cdf(orig, syn, cols):
    for c in cols:
        plt.figure()
        orig_sorted = np.sort(orig[c])
        p_orig = 1.0 * np.arange(len(orig_sorted)) / (len(orig_sorted)-1)

        syn_sorted = np.sort(syn[c])
        p_syn = 1.0 * np.arange(len(syn_sorted)) / (len(syn_sorted)-1)

        ks,p = ss.ks_2samp(syn[c], orig[c])
        
        plt.plot(orig_sorted, p_orig, label='Original')
        plt.plot(syn_sorted, p_syn, label='Synthetic')
        plt.title(c+'\nKS = {:.2f}, p = {:.2f}'.format(ks,p))
        plt.legend()
        plt.show()
        
def compare_qqplot(orig, syn, cols):
    for c in cols:
        plt.figure()
        if len(syn[c]) < len(orig[c]):
            qqplot_2samples(syn[c], orig[c])
        else:
            qqplot_2samples(orig[c], syn[c])
        plt.title(c)
        plt.show()

