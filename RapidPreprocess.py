from cuml.preprocessing import LabelEncoder, OneHotEncoder
import cudf

# Preprocess categorical features using various encodings and fill missings
class CategoricalFeatures:
    def __init__(self, df, ohe_feats, lbl_enc_feats, ord_feats, target_enc_feats, handle_na = False):
      """
      Used to handle missing values, rare values and multiple encoding types with categorical features

      Params:
      - df = cuDF dataframe
      - ohe_feats = list of features to one hot encode
      - lbl_enc_feats = list of features to label encode
      - ord_feats = dictionary of {"feature": [order of unique values] to ordinally encode}
      - target_enc_feats = list of features to use for target encoding
      - handle_na = True/False to indicate if you want to fill NAs
      """
      self.df = df
      self.ohe_feats = ohe_feats
      self.lbl_enc_feats = lbl_enc_feats
      self.ord_feats = ord_feats
      self.target_enc_feats = target_enc_feats
      self.handle_na = handle_na
      self.encoders = {"one hot encoded" : self.ohe_feats,
                       "label encoded" : self.lbl_enc_feats,
                       "ordinal encoded" : self.ord_feats, 
                       "target encoded" : self.target_enc_feats}

      # If handle_na is True then fill NAs with "MISSING"
      if self.handle_na:

          # If there are features to ohe then fill any NASs with "MISSING"
          if self.ohe_feats != None:
            for feat in self.ohe_feats:
              self.df[feat] = self.df[feat].astype(str).fillna("MISSING")

          # If there are features to label encode then fill any NASs with "MISSING"
          if self.lbl_enc_feats != None:
            for feat in self.lbl_enc_feats:
              self.df[feat] = self.df[feat].astype(str).fillna("MISSING")

          # If there are ordinal features then fill any NASs with "MISSING"
          if self.ohe_feats != None:
            for feat in self.ohe_feats:
              self.df[feat] = self.df[feat].astype(str).fillna("MISSING")

          # If there are features to target encode then fill any NASs with "MISSING"
          if self.target_enc_feats != None:
            for feat in self.target_enc_feats:
              self.df[feat] = self.df[feat].astype(str).fillna("MISSING")
    
        
      self.output_df = self.df.copy()

    def rare_values(self, feat, min_percent = 0):
      """
      Merge rare values into one category called "RARE" for feature selected based upon percentage they make up in a column
      Params:
      - min_percent = the minimum percent a level has to have to be considered not "RARE"
      - feat = feature in df to check if rare values exist
      """

      self.output_df.loc[self.output_df[feat].value_counts()[self.output_df[feat]].values / self.output_df[feat].count() <= min_percent, feat] = "RARE"

    def one_hot_encoder(self, dummy_nas = None):
      """
      Takes the output_df and creates dummifies any features in ohe_feats list

      By default it won't dummy any NAs in the features but this can be tweaked to True to handle them
      Params:
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
      Takes the output_df and label encode any features in lbl_enc_feats list
      """  
      # Loop through each feature in lbl_enc_feats and label encode it
      for feat in self.lbl_enc_feats:
        le = LabelEncoder()
        le.fit(self.output_df[feat])
        self.output_df[feat] = le.transform(self.output_df[feat])

    def ordinal_encoder(self):
      """
      Takes the output_df and ordinaly encodes the features in ord_feats using the order provided per feature
      """
      # Loop through each ordinal feature in ord_feats (the keys are the features) and perform ordinal encoding
      for feat in list(self.ord_feats.keys()):
        self.output_df[feat] = self.df[feat].label_encoding(cats = self.ord_feats[feat])  

    # Target encoder (when released)

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

# Preprocess continuous features using fill missings
class ContinuousFeatures:
  def __init__(self, df, cont_feats, handle_na = False):
    """
    Used to fill missing values of continous feats with -999999

    Params:
    - df = cuDF dataframe
    - cont_feats = list of continuous features
    - handle_na = True/False on whether or not to fill NAs
    """
    self.df = df
    self.cont_feats = cont_feats

    # If handle_na is True then fill NAs with -999999
    if handle_na:
      for feat in self.cont_feats:
        self.df[feat] = self.df[feat].astype(str).fillna(-999999)

    # Possible scaler here using .scale
        
    self.output_df = self.df.copy()

  def preprocess(self):
    return self.output_df   

# Preprocess datetime features by exploding datetimes
class DatetimeFeatures:
  def __init__(self, df, date_or_datetime_feats):
    """
    Used to explode dates and convert any non converted dates to datetime64 ns
    
    Params:
    - df = cuDF dataframe
    - date_or_datetime_feats = list of date or datetime features
    """
    self.df = df
    self.date_or_datetime_feats = date_or_datetime_feats
    
    # Convert all date_or_datetime_feats to datetimes in ns
    for feat in date_or_datetime_feats:
      self.df[feat] = self.df[feat].astype("datetime64[ns]")
    
    self.output_df = self.df.copy()

  def explode_date(self, date_feat, date_part, sin_cos_transform = False):
    """
    Explodes each date_feats into the date_parts

    Potential date parts are
    ['year', 'month', 'week', 'day', 'weekday', 'dayofyear', 'is_month_end', 'is_month_start', 
    'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start', 'elapsed']
    """
    # Get the date column and initialise the units need for sin/cos transforms
    date_feat = self.df[date_feat]
    units = 0

    # Below each if looks at the date_part and returns the appropriate date part
    # YEAR, raise exception if sin_cos_transform is True
    if date_part == "year":
      if sin_cos_transform == True:
        raise Exception("Can't have cyclical years.")
      
      else:
        add_date_col = date_feat.dt.year

    # QUARTER

    # MONTH
    elif date_part == "month":
      add_date_col = date_feat.dt.month
      units = 12

    # WEEK OF THE YEAR
    elif date_part == "week":
      add_date_col = date_feat.dt.week
      units = 52
    
    # DAY OF THE MONTH
    elif date_part == "day":
      add_date_col = date_feat.dt.day
      # Depends on month again
      #units = dependent, cyclical ness will have to be considered

    # WEEKDAY
    elif date_part == "weekday":
      add_date_col = date_feat.dt.weekday
      units = 7

    # IF ITS THE START OF THE MONTH
    elif date_part == "is_month_start":
      add_date_col = date_feat.dt.day.replace(to_replace = [2, 31], value = 0)
      units = 12

    # END OF THE MONTH
    #elif date_part == "is_month_end":
    #  print(date_feat.dt.is_month_end)

    # If sin_cos_transform is True then create two new columns sin_/cos_ suffixed with the date_part
    # Transform use cudfs sin/cos
    if sin_cos_transform == True:
      self.output_df["sin_" + date_part] = cudf.sin((2 * 3.14 * (add_date_col)) / units)
      self.output_df["cos_" + date_part] = cudf.cos((2 * 3.14 * (add_date_col)) / units)
    
    # If False then don't do the transform
    else:
      self.output_df[date_part] = add_date_col
      

  def preprocess(self):
    return self.output_df
