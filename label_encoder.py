import sklearn.preprocessing as skp
from functools import reduce
import numpy as np
import math

def homogeneous_type(grp):
    return (
        True if len(set([x.dtype for x in grp])) == 1 else False
    )

class LabelEncoder:
    
    def __init__(self, grp_dict, verbose = False):
        self.grp_dict = grp_dict
        self.le_dict = dict()
        self.get_encoders(verbose=verbose)
        

    def get_encoders(self, verbose=False):
        if verbose:
            print('Getting encoders...')
        for grp in self.grp_dict:
            if verbose:
                print(grp)
            if not homogeneous_type(self.grp_dict[grp]):
                raise ValueError('{} contains mixed type data'.format(grp))                
            le = skp.LabelEncoder()
            concat = reduce(
                lambda x,y: np.concatenate((np.array(x),np.array(y))),
                self.grp_dict[grp],
                []
            )
            le.fit(concat)
            self.le_dict[grp] = le

            
    def encode(self, df, encode_dict, verbose = False, exclude_na=True):
        out_df = df.copy()
        if verbose:
            print('Encoding...')
        for grp in encode_dict:
            if verbose:
                print(grp)
            for c in encode_dict[grp]:
                if verbose:
                    print('   ',c)
                if exclude_na:
                    out_df.loc[df[c].notnull(),c] = (
                        self.le_dict[grp].transform(df.loc[df[c].notnull(),c])
                    )
                    out_df.loc[df[c].isnull(),c] = df.loc[df[c].isnull(),c]
                else:
                    out_df[c] = self.le_dict[grp].transform(df[c])
        return out_df

def add_prefix(df, prefix_dict, verbose=False):
    for c in prefix_dict:
        if verbose:
            print(c)
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: prefix_dict[c]+str(int(x)) if not math.isnan(x) else np.nan
            )
            if verbose:
                print('    Done!')
        else:
            if verbose:
                print('    Not a column. Ignoring!')
    return(df)

def get_ndigits(x):
    '''
    helper fx to calculate number of digits
    '''
    x = abs(x)
    n = 0
    while x > 0:
        n += 1
        x = x // 10
    return n

def le2id(a, n_digits, prefix):
    '''
    convert label encodings to numeric ids
    
    parameters
    ----------
    a: array like
        input array
    n_digit: int
        number of digits for ids (including prefix)
    prefix: int
        numeric prefix to use for ids
        
    returns
    -------
    numpy array
        ids
    '''
    n_zeros = n_digits - get_ndigits(prefix)

    if np.nanmax(a) > 10**n_zeros:
        raise ValueError("n_digits is not large enough to cover the range of the given array. Consider increasing n_digits")

    a = np.array(a) + prefix * 10**n_zeros
    
    return a

def covnert_le_to_id(df, le2id_dict, verbose=False):
    for col in le2id_dict:
        if verbose:
            print(col)
        if col in df.columns:
            df[col] = le2id(
                df[col],
                n_digits=le2id_dict[col]['n_digits'],
                prefix=le2id_dict[col]['prefix']
            )
            if verbose:
                print('    Done!')
        else:
            if verbose:
                print('    Not a column. Ignoring!')
    return df         