import pandas as pd
import numpy as np
import statsmodels.nonparametric.kernel_density as smkd
import sklearn.neighbors as skn
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import datetime as dt
from scipy import optimize
import sys
import parser
import json

class SynDat:
    '''
    generate synthetic data from an existing data set
    '''

    def __init__(
        self, data: pd.DataFrame, cols: dict,
        dt_format: str = None, calc_kde: bool = True
    ):
        '''
        Parameters
        ----------
        data: data frame
            original data to generate synthetic data from
        cols: dict
            dictionary of var names and their type.
            allowed types: quant, categ, ord, dt
            cols should be entered in order (requires python 3.6+)
        dt_format: str, default = None
        calc_kde: bool, default = True
            whether to calculate KDE at object instantiation

        Returns
        -------
        None
        '''
        self.df = data[cols].copy()
        self.cols = cols
        self.dt_format = dt_format
        self.kde = None
        self.var_type = None
        self.cat_le = None

        if calc_kde:
            self.df = self.to_dt(self.df, self.cols, format=self.dt_format)
            self.df = self.dt_to_ordinal(self.df, self.cols)
            self.df, self.cat_le = self.categ_to_label(self.df, self.cols)            
            self.var_type = self.get_var_type(self.cols)
            self.kde = self.run_kde()

    def get_var_type(self, cols: dict) -> list:
        '''
        create a list of var types from cols dict for statsmodel
        KDEMultivariate:
        'c': continuous,
        'u': unoderded categorical,
        'o': ordered categorical

        Parameters
        ----------
        cols: dict
            dictionary of column names and their type

        Returns
        -------
        list of var types
        '''
        var_type = []
        for c in cols:
            if cols[c] == 'quant' or cols[c] == 'dt':
                var_type.append('c')
            elif cols[c] == 'categ':
                var_type.append('u')
            elif cols[c] == 'ord':
                var_type.append('o')
        return var_type

    
    def to_dt(self, df: pd.DataFrame, cols: dict, dt_format=None) -> pd.DataFrame:
        '''
        convet dt cols to datetime type

        Parameters
        ----------
        df: data frame
            input data
        cols: dict
            dictionary of var names and their type.
            allowed types: quant, categ, ord, dt
            cols should be entered in order (requires python 3.6+)

        Returns
        -------
        data frame with updated date cols
        '''
        for c in cols:
            if cols[c] == 'dt':
                df[c] = pd.to_datetime(df[c], format=dt_format)
        return df

    def dt_to_ordinal(self, df: pd.DataFrame, cols: dict) -> pd.DataFrame:
        '''
        convert dates to ordinal

        Parameters
        ----------
        df: data frame
            input data
        cols: dict
            dictionary of var names and their type.
            allowed types: quant, categ, ord, dt
            cols should be entered in order (requires python 3.6+)

        Returns
        -------
        data frame with updated date cols

        '''
        for c in cols:
            if cols[c] == 'dt':
                df[c] = df[c].apply(lambda x: x.toordinal())
        return df


    def ordinal_to_dt(self, df: pd.DataFrame, cols: dict) -> pd.DataFrame:
        '''
        convert ordinal to date

        Parameters
        ----------
        df: data frame
            input data
        cols: dict
            dictionary of var names and their type.
            allowed types: quant, categ, ord, dt
            cols should be entered in order (requires python 3.6+)

        Returns
        -------
        data frame with updated date cols        
        '''
        for c in cols:
            if cols[c] == 'dt':
                df[c] = df[c].apply(
                    lambda x: dt.date.fromordinal(int(x))
                )
        return df
    

    def categ_to_label(self, df: pd.DataFrame, cols: dict) -> pd.DataFrame:
        '''
        convert categories to numeric labels

        Parameters
        ----------
        df: data frame
            input data
        cols: dict
            dictionary of var names and their type.
            allowed types: quant, categ, ord, dt
            cols should be entered in order (requires python 3.6+)

        Returns
        -------
        data frame, dict of label encoding            
        '''
        cat_le = dict()
        for c in cols:
            if cols[c] == 'categ':
                le = skp.LabelEncoder()
                le.fit(df[c])
                df[c] = le.transform(df[c])
                cat_le[c] = le
        return df, cat_le


    def label_to_categ(
        self, df: pd.DataFrame, cols: dict, cat_le: dict
    ) -> pd.DataFrame:
        '''
        convert numeric labels to original categories

        Parameters
        ----------
        df: data frame
            input data
        cols: dict
            dictionary of var names and their type.
            allowed types: quant, categ, ord, dt
            cols should be entered in order (requires python 3.6+)
        cat_le: dict
            dict of LabelEncoders

        Returns
        -------
        data frame with updated categorical columns            
        '''        
        for c in cols:
            if cols[c] == 'categ':
                df[c] = cat_le[c].inverse_transform(df[c].astype(int))
        return df


    def run_kde(self) -> smkd.KDEMultivariate:
        '''
        run kernel density estimation and return the estimated density

        Parameters
        ----------
        None

        Returns
        -------
        estimated KDE
        '''
        kde = smkd.KDEMultivariate(self.df, var_type=self.var_type)
        return kde


    def rejection_sampling(
        self, kde: smkd.KDEMultivariate, rng: np.ndarray,
        var_type: list, M: int = 1, n: int = 1000
    ) -> np.ndarray:
        '''
        rejection sampling

        Parameters
        ----------
        kde: statsmodel KDEMultivariate
            estimated kde
        rng: array_like
            range of variables in the form of a 2D array of (min, max)
        var_type: list
            list of variable types for statsmodel KDEMultivariate
            'c': continuous,
            'u': unoderded categorical,
            'o': ordered categorical
        M: int, default = 1
            pdf max value
        n: int, default = 1000
            number of sample to generate     

        Returns
        -------
        2D array of samples        
        '''

        samp = []
        i = 0
        while (i<n):
            x = (
                np.random.rand(len(var_type))
                * (np.max(rng, axis=-1) - np.min(rng, axis=-1))
                + np.min(rng, axis=-1)
            )

            # round categorical vars to the nearest int
            x = [round(xi) if var_type[i] != 'c' else xi
                 for i,xi in enumerate(x)]

            u = np.random.rand()

            if u < kde.pdf(x) / M:
                samp.append(x)
                i += 1

        return np.array(samp)

    
    def get_sample(self, n=1000) -> pd.DataFrame:
        '''
        draw random samples from the estimated distribution

        Parameters
        ----------
        n: int, default = 1000
            number of samples to generate

        Returns
        -------
        data frame of synthetic data
        '''
        mins = np.array(self.df.min())
        maxs = np.array(self.df.max())
        rng = np.stack((mins,maxs), axis=1)

        # find pdf max
        '''
        res = optimize.minimize(
            lambda x: -self.kde.pdf(x), x0=np.array(self.df.median()),
            method='Nelder-Mead'
        )
        M = -res.fun
        '''
        # use value at median as an approximate to pdf's max
        M = self.kde.pdf(self.df.median())

        # rejection sampling
        samp = self.rejection_sampling(self.kde, rng, self.var_type, M=M, n=n)

        # create df from sample
        df_samp = pd.DataFrame(samp, columns=self.cols)

        # convert ordinal back to dt
        df_samp = self.ordinal_to_dt(df_samp, self.cols)

        # convert categorical labels back to original categories
        df_samp = self.label_to_categ(df_samp, self.cols, self.cat_le)

        return df_samp

def load_json(fname: str) -> dict:
    '''
    helper function to load json files

    Parameters
    ----------
    fname: str
        json file name

    Returns
    -------
    dictionary created from the json file
    '''
    with open(fname, 'r') as fid:
        return json.load(fid)   

if __name__ == '__main__':
    arg_list = ['data=', 'cols=', 'out=']
    help_msg = (
        '\n--------\n'
        + 'Usage:\n'
        + 'python syndat.py --data <csv> --cols <json> --out <csv>'
        + '\n--------\n'
    )
    prs = parser.ParseArgs(arg_list, help_msg)
    params = prs.get_args(sys.argv[1:])

    if params['data'] == '' or params['cols'] == '' or params['out'] == '':
        prs.printhelp()
        sys.exit()

    data = pd.read_csv(params['data'])
    cols = load_json(params['cols'])

    samp = SynDat(data, cols).get_sample()

    samp.to_csv(params['out'], index=False)

