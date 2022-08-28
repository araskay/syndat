import pandas as pd
import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kde import KDEUnivariate
import sklearn.preprocessing as skp
import datetime as dt
from scipy import optimize

class DataGenerator:
    '''
    generate synthetic data from an existing data set
    '''

    def __init__(
        self, data: pd.DataFrame, cols: dict,
        convert_dt: bool = False, dt_format: str = None,
        multivariate: bool = False, calc_kde: bool = True,
        verbose=False
    ):
        '''
        Parameters
        ----------
        data: data frame
            original data to generate synthetic data from
        cols: dict
            dictionary of var names and their type.
            allowed types:
                - float
                - int
                - ord (ordered discrete)
                - unord (unordered discrete)
                - dt (datetime)
            cols should be entered in order (requires python 3.6+)
        convert_dt: bool, default = False
            whether to convert dt cols to pandas datetime format
        dt_format: str, default = None
            date format to use for coverting strings into pandas datetime
        multivariate: bool, default = False
            whether to use multivariate kernel density estimation
        calc_kde: bool, default = True
            whether to calculate KDE at object instantiation
        verbose: bool, default = False
            whether to print verbose info

        Returns
        -------
        None
        '''
        self.df = data[list(cols.keys())].copy()
        self.cols = cols
        self.dt_format = dt_format
        self.multivariate = multivariate
        self.kde = None
        self.var_type = None
        self.cat_le = None
        self.verbose = verbose


        if calc_kde:
            if convert_dt:
                self.df = self.to_dt(
                    self.df, self.cols, dt_format=self.dt_format
                )
            self.df = self.dt_to_ordinal(self.df, self.cols)
            self.df, self.cat_le = self.categ_to_label(self.df, self.cols)            
            self.var_type = self.get_var_type(self.cols)
            if self.multivariate:
                self.kde = self.run_kde_mv()
            else:
                self.kde = self.run_kde_uv(verbose=self.verbose)
                

    def get_var_type(self, cols: dict) -> list:
        '''
        create a list of var types from cols dict for statsmodel
        KDEMultivariate

        Parameters
        ----------
        cols: dict
            dictionary of column names and their type

        Returns
        -------
        list of var types
        '''
        if self.verbose:
            print('getting var types')
        var_type = []
        for c in cols:
            if cols[c] == 'float' or cols[c] == 'int' or cols[c] == 'dt':
                var_type.append('c')
            elif cols[c] == 'unord':
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

        Returns
        -------
        data frame with updated date cols

        '''
        if self.verbose:
            print('converting dt to ordinal')
        for c in cols:
            if cols[c] == 'dt':
                if self.verbose:
                    print(c)
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
    
    def round_int(self, df: pd.DataFrame, cols: dict) -> pd.DataFrame:
        '''
        convert ordinal to date

        Parameters
        ----------
        df: data frame
            input data
        cols: dict
            dictionary of var names and their type.

        Returns
        -------
        data frame with updated date cols        
        '''
        for c in cols:
            if cols[c] == 'int':
                df.loc[df[c].notnull(), c] = (
                    df.loc[df[c].notnull(), c].apply(lambda x: int(x))
                )
        return df


    def categ_to_label(
        self, df: pd.DataFrame, cols: dict, convert_to_str: bool = True
    ) -> pd.DataFrame:
        '''
        convert categories to numeric labels

        Parameters
        ----------
        df: data frame
            input data
        cols: dict
            dictionary of var names and their type.
        convert_to_str: bool, default = True
            whether to convert unordered categorical data to str -
            especially useful when there are mixed types and/or Nulls

        Returns
        -------
        data frame, dict of label encoding            
        '''
        if self.verbose:
            print('converting categorical to label')
        cat_le = dict()
        for c in cols:
            if cols[c] == 'unord':
                if self.verbose:
                    print(c)
                if convert_to_str:
                    df[c] = df[c].astype(str)
                le = skp.LabelEncoder()
                df[c] = le.fit_transform(df[c])
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
        cat_le: dict
            dict of LabelEncoders

        Returns
        -------
        data frame with updated categorical columns            
        '''        
        for c in cols:
            if cols[c] == 'unord':
                df[c] = cat_le[c].inverse_transform(df[c].astype(int))
        return df


    def run_kde_mv(self) -> KDEMultivariate:
        '''
        run kernel density estimation and return the estimated density

        Parameters
        ----------
        None

        Returns
        -------
        estimated KDE
        '''
        kde = KDEMultivariate(self.df, var_type=self.var_type)
        return kde

    def run_kde_uv(self, verbose=False) -> dict:
        '''
        run KDE independtly on each col and return dict of estimated KDEs

        Parameters
        ----------
        None

        Returns
        -------
        dict of estimated KDEs
        '''
        if verbose:
            print('calculating UV KDE')
        kde_dict = dict()
        for c in self.cols:
            if verbose:
                print(c)
            if self.df[c].std() == 0:
                bw = 1
            else:
                bw = 'normal_reference'
            kde = KDEUnivariate(self.df[c].astype(float))
            kde.fit(bw=bw)
            kde_dict[c] = kde
        return kde_dict


    def rejection_sampling(
        self, kde: KDEMultivariate, rng: np.ndarray,
        cols: dict, M: int = 1, n: int = 1000, verbose: bool = False
    ) -> np.ndarray:
        '''
        rejection sampling

        Parameters
        ----------
        kde: statsmodel KDEMultivariate
            estimated kde
        rng: array_like
            range of variables in the form of a 2D array of (min, max)
        cols: dict
            dictionary of var names and their type.
        M: int, default = 1
            pdf max value
        n: int, default = 1000
            number of sample to generate
        verbose: bool, default = False
            whether to show verbose outputs 

        Returns
        -------
        2D array of samples        
        '''

        samp = []
        i = 0
        while (i<n):
            x = (
                np.random.rand(len(cols))
                * (np.max(rng, axis=-1) - np.min(rng, axis=-1))
                + np.min(rng, axis=-1)
            )

            # round categorical vars to the nearest int
            x = [
                x[i] if cols[c] == 'float'
                else round(x[i])
                for i,c in enumerate(cols)
            ]

            u = np.random.rand()

            verbose_interval = int(n/100)

            if u < kde.pdf(x) / M:
                samp.append(x)
                i += 1
                if verbose:
                    if i % verbose_interval == 0:
                        print(
                            'sampled {0} out of {1} ({2}%)'.
                            format(i,n,round(i/n*100.0))
                        )

        return np.array(samp)

    
    def get_sample(
        self, n: int = 1000, use_med_approx: bool = False,
        verbose: bool = False
    ) -> pd.DataFrame:
        '''
        draw random samples from the estimated distribution(s)

        Parameters
        ----------
        n: int, default = 1000
            number of samples to generate
        use_med_approx: bool, default = False
            whether to use pdf value at median as a estimate for max -
            only applicable to multivariate data (i.e., dependent cols)
        verbose: bool, default = False
            whether to show verbose outputs 

        Returns
        -------
        data frame of synthetic data
        '''
        if not self.multivariate:
            df_samp = pd.DataFrame()
            for c in self.cols:
                df_samp[c] = np.random.choice(self.kde[c].icdf, n)
        else:
            # check for NAs in quantitative cols
            quant_cols = [
                x for x in self.cols
                if self.cols[x] in ['int','float']
            ]
            na_count = self.df[quant_cols].isnull().sum()
            na_cols = [x for x in na_count.index if na_count[x]>0]
            if na_count.sum()>0:
                raise ValueError(
                    ('The following quantitative (float/int) columns have NAs: '
                    + str(na_cols))
                    + 'Consider imputing NAs or using the univariate estimate.'
                    
                )            

            mins = np.array(self.df.min())
            maxs = np.array(self.df.max())
            rng = np.stack((mins,maxs), axis=1)

            # find pdf max
            if use_med_approx:
                # use value at median as an approximate to pdf's max
                M = self.kde.pdf(self.df.median())
            else:
                res = optimize.minimize(
                    lambda x: -self.kde.pdf(x), x0=np.array(self.df.median()),
                    method='Nelder-Mead'
                )
                M = -res.fun

            if verbose:
                print('M =', M)

            # rejection sampling
            samp = self.rejection_sampling(
                self.kde, rng, self.cols, M=M, n=n, verbose = verbose
            )

            # create df from sample
            df_samp = pd.DataFrame(samp, columns=self.cols)

        # convert ordinal back to dt
        df_samp = self.ordinal_to_dt(df_samp, self.cols)

        # convert categorical labels back to original categories
        df_samp = self.label_to_categ(df_samp, self.cols, self.cat_le)

        # round int vars to the nearest integer
        df_samp = self.round_int(df_samp, self.cols)

        return df_samp


