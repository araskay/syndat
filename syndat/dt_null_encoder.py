import pandas as pd
import numpy as np

def encode(df, cols, null_date=pd.Timestamp.min):
    for c in cols:
        if cols[c] == 'dt':
            df.loc[df[c].isnull(),c] = null_date # use this old date to indicate nulls
    return df

def decode(df_samp, df_orig, cols):
    for c in cols:
        if cols[c] == 'dt':
            df_samp.loc[df_samp[c] < np.min(df_orig[c]), c] = pd.NaT
    return df_samp

