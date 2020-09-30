from cuml.preprocessing import LabelEncoder, OneHotEncoder
import cudf
from cudf.core.reshape import get_dummies
from pandas import read_csv

class CategoricalFeatures:
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
      self.encoders = {"one hot encoded": self.ohe_feats,
                       "label encoded": self.lbl_enc_feats,
                       "target encoded": self.target_enc_feats}

      if self.handle_na:
          if self.lbl_enc_feats != None:
            for feat in self.lbl_enc_feats:
              self.df[feat] = self.df[feat].astype(str).fillna("MISSING")

          if self.ohe_feats != None:
            for feat in self.ohe_feats:
              self.df[feat] = self.df[feat].astype(str).fillna("MISSING")

          if self.target_enc_feats != None:
            for feat in self.target_enc_feats:
              self.df[feat] = self.df[feat].astype(str).fillna("MISSING")
    
        
      self.output_df = self.df.copy()

    def one_hot_encoder(self, dummy_nas = None):
      """
      Takes the output_df and creates dummifies any features in ohe_feats list

      By default it won't dummy any NAs in the features but this can be tweaked to True to handle them
      Params
      - dummy_nas = True/False (default to False), used to indicate if get_dummies will dummy NAs
      """

      # Check if dummy_nas if the default (None), if it is then set dummy_nas to False (i.e. don't dummy NAs) 
      if dummy_nas == None:
        dummy_nas = False

      # Otherwise set to True (dummy NAs)  
      else:
        dummy_nas = True

      self.output_df = cudf.get_dummies(self.output_df, 
                                          columns = self.ohe_feats, 
                                          dummy_na = dummy_nas)

    def label_encoder(self):
      """
      Takes the output_df and label encode any featrues in lbl_enc_feats list
      """  
      for feat in self.lbl_enc_feats:
        le = LabelEncoder()
        le.fit(self.output_df[feat])
        self.output_df[feat] = le.transform(self.output_df[feat])

    

    # Target encoder

    def preprocess(self):
      """
      Returns the preprocessed output dataframe (output_df)
      """
      return self.output_df

    def get_encoders(self):
      """
      Returns a dictionary of the features and which encoder they fall under
      """ 
      return self.encoders
