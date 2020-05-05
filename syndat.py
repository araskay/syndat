import pandas as pd
import numpy as np
import statsmodels.nonparametric.kernel_density as smkd
import sklearn.neighbors as skn
import sklearn.model_selection as skms
import datetime as dt
from scipy import optimize

class SynDat:
    '''
    generate synthetic data from an existing data set
    '''

    def __init__(
        self, data: pd.DataFrame, cols: dict, categs: dict,
        calc_kde: bool = True
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
        categs: dict
            dictionary of categories and their label mapping
            example: {'x_categ': {'c1':1, 'c2':2, 'c3':3}}
        calc_kde: bool, default = True
            whether to calculate KDE at object instantiation

        Returns
        -------
        None
        '''
        self.df = data
        self.cols = cols
        self.categs = categs
        self.kde = None
        self.var_type = None

        if calc_kde:
            self.df = self.dt_to_ordinal(self.df, self.cols)
            self.df = self.categ_to_lablel(self.df, self.categs)            
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
    

    def categ_to_lablel(self, df: pd.DataFrame, categs: dict) -> pd.DataFrame:
        '''
        convert categories to numeric labels

        Parameters
        ----------
        df: data frame
            input data
        categs: dict
            dictionary of categories and their label mapping
            example: {'x_categ': {'c1':1, 'c2':2, 'c3':3}}

        Returns
        -------
        data frame with updated categorical columns            
        '''
        for c in categs:
            df[c] = df[c].apply(lambda x: categs[c][x])
        return df


    def label_to_categ(self, df: pd.DataFrame, categs: dict) -> pd.DataFrame:
        '''
        convert numeric labels to original categories

        Parameters
        ----------
        df: data frame
            input data
        categs: dict
            dictionary of categories and their label mapping
            example: {'x_categ': {'c1':1, 'c2':2, 'c3':3}}

        Returns
        -------
        data frame with updated categorical columns            
        '''        
        for c in categs:
            inv_mapping = {v:k for k,v in categs[c].items()}
            df[c] = df[c].apply(lambda x: inv_mapping[x])
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
        var_type: list, n: int = 1000
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
        n: int, default = 1000
            number of sample to generate     

        Returns
        -------
        2D array of samples        
        '''
        # find pdf max
        res = optimize.minimize(
            lambda x: -kde.pdf(x), x0=np.array(self.df.median()),
            method='Nelder-Mead'
        )
        M = -res.fun

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

        # rejection sampling
        samp = self.rejection_sampling(self.kde, rng, self.var_type, n=n)

        # create df from sample
        df_samp = pd.DataFrame(samp, columns=self.cols)

        # convert ordinal back to dt
        df_samp = self.ordinal_to_dt(df_samp, self.cols)

        # convert categorical labels back to original categories
        df_samp = self.label_to_categ(df_samp, self.categs)

        return df_samp



