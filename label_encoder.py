import sklearn.preprocessing as skp

class LabelEncoder:
    
    def __init__(self, grp_dict, df):
        self.grp_dict = grp_dict
        self.df = df
        self.le_dict = dict()
        self.get_encoders()

    def get_encoders(self):
        for grp in self.grp_dict:
            le = skp.LabelEncoder()
            le.fit(self.df[self.grp_dict[grp]].stack().ravel())
            self.le_dict[grp] = le

            
    def encode(self, df):
        out_df = df.copy()
        for grp in self.grp_dict:
            for c in self.grp_dict[grp]:
                out_df[c] = self.le_dict[grp].transform(df[c])
        return out_df