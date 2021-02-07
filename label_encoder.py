import sklearn.preprocessing as skp
from functools import reduce
import numpy as np

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

    def add_prefix(self, df, prefix_dict):
        for c in prefix_dict:
            df[c] = df[c].apply(lambda x: prefix_dict[c]+str(x))
        return(df)