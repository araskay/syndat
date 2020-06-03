import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples
import scipy
import pandas as pd

def compare_quant(df1, df2, col_name):
    qqplot_2samples(
        df1[col_name], df2[col_name],
        xlabel='original data', ylabel='synthetic data', line='45'
    )
    plt.title(col_name)
    plt.show()
    # KS test
    ks_stats, p = scipy.stats.ks_2samp(df1[col_name], df2[col_name])
    print('KS p-value =', p)
    
def compare_dt(df1, df2, col_name, dt1_format=None, dt2_format=None):
    df1_ord_dt = df1.copy()
    df1_ord_dt[col_name] = (
        pd.to_datetime(df1_ord_dt[col_name], format=dt1_format).
        apply(lambda x: x.toordinal())
    )
    
    df2_ord_dt = df2.copy()
    df2_ord_dt[col_name] = (
        pd.to_datetime(df2_ord_dt[col_name], format=dt2_format).
        apply(lambda x: x.toordinal())
    )
    
    compare_quant(df1_ord_dt, df2_ord_dt, col_name)
    
    
def qq_categ(df1, df2, sd, col):
    df1_le = sd.categ_to_label(df1, sd.cols)[0].copy()
    df2_le = sd.categ_to_label(df2, sd.cols)[0].copy()
    compare_quant(df1_le, df2_le, col)
    
    
def compare_cat(df1, df2, col_name):
    fig, ax = plt.subplots(1,2)
    bar_dat = df1[col_name].value_counts().reset_index().sort_values('index')
    ax[0].bar(bar_dat['index'],bar_dat[col_name])
    ax[0].set_xticklabels(bar_dat['index'], rotation=90)

    bar_dat = df2[col_name].value_counts().reset_index().sort_values('index')
    ax[1].bar(bar_dat['index'],bar_dat[col_name])
    ax[1].set_xticklabels(bar_dat['index'], rotation=90)
    fig.suptitle(col_name)
    plt.show()
    
def comparison_plots(
    df1, df2, cols, sd, dt1_format = None, dt2_format=None):
    for c in cols:
        if cols[c] == 'float' or cols[c] == 'int' or cols[c] == 'ord':
            compare_quant(df1, df2, c)
        if cols[c] == 'unord':
            qq_categ(df1, df2, sd, c)
        if cols[c] == 'dt':
            compare_dt(
                df1, df2, c, dt1_format=dt1_format, dt2_format=dt2_format
            )