from cuml.preprocessing import LabelEncoder, OneHotEncoder
import cudf
from cudf.core.reshape import get_dummies

class CategoricalPreprocess:
    def __init__(self, df, lbl_enc_feats, ohe_feats, target_enc_feats, handle_na = False):
      """
      df: pandas dataframe
      categorical_features: list of column names
      encoding_type: label, binary, ohe
      handle_na: True/False
      """
      self.df = df
      self.lbl_enc_feats = lbl_enc_feats
      self.ohe_feats = ohe_feats
      self.target_enc_feats = target_enc_feats
      self.handle_na = handle_na
      self.columns = self.df.columns

      if self.handle_na:
          for feat in self.df.columns:
              self.df.loc[:, feat] = self.df.loc[:, feat].astype(str).fillna("-9999999")
        
      self.output_df = self.df.copy()

    def one_hot_encoder(self):
        self.output_df = cudf.get_dummies(self.output_df, columns = self.ohe_feats)
# FIX is to update the df in the method

    def label_encoder(self):
      for feat in self.lbl_enc_feats:
        le = LabelEncoder()
        le.fit(self.df[feat])
        self.output_df[feat] = le.transform(self.output_df[feat])

    def preprocess(self):
      return self.output_df
    #def target_encoder(self, target):
