import pandas as pd
import numpy as np
from datetime import datetime

def encode(df, cols, null_date=datetime.strptime('1900-01-01', '%Y-%m-%d')):
    for c in cols:
        if cols[c] == 'dt':
            df.loc[df[c].isnull(),c] = null_date # use this old date to indicate nulls
    return df

def decode(df_samp, df_orig, cols):
    for c in cols:
        if cols[c] == 'dt':
            df_samp.loc[df_samp[c] < np.min(df_orig[c]), c] = pd.NaT
    return df_samp

