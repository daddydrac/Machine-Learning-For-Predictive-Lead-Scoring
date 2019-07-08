import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import clone


NUMERIC_FEATURES = [
    'age', 
    'campaign', 
    'previous', 
    'emp.var.rate', 
    'cons.price.idx', 
    'cons.conf.idx', 
    'euribor3m', 
    'nr.employed',
    'campaign_to_previous'
]

CATEGORICAL_FEATURES =  [
  'job',
  'marital',
  'education',
  'default',
  'housing',
  'loan',
  'contact',
  'month',
  'day_of_week',
  'poutcome'
]

NEW_CATEGORICAL_FEATURES = [
  'pcontacted_last_campaign',
  'pcampaign',
  'previous',
  'campaign_gte10'
]

def ft_pcontacted_last_campaign(X):
  pcontacted = ~(X == 999)
  return pcontacted.values.reshape(-1,1)

def ft_pcampaign(X):
  pcampaign = ~(X == 'nonexistent')
  return pcampaign.values.reshape(-1,1)

def ft_previous(X):
  previous = X.astype(str)
  return previous.values.reshape(-1,1)

def ft_campaign_gte10(X):
  campaign_gte10 = X >= 10
  return campaign_gte10.values.reshape(-1,1)

def ft_campaign_to_previous(X):
  ratio = lambda x: 0 if x.previous == 0 else x.campaign / x.previous
  campaign_to_previous = X[['campaign', 'previous']].apply(ratio, axis=1)
  return campaign_to_previous.values.reshape(-1,1)

def get_categorical_ct():
  # Create the transformers for categorical features
  add_pcontacted_last_campaign = FunctionTransformer(ft_pcontacted_last_campaign, validate=False)
  add_pcampaign = FunctionTransformer(ft_pcampaign, validate=False)
  add_previous = FunctionTransformer(ft_previous, validate=False)
  add_campaign_gte10 = FunctionTransformer(ft_campaign_gte10, validate=False)
  
  cat_features = [
    ('categoricals', 'passthrough', CATEGORICAL_FEATURES),
    ('pcontacted_last_campaign', add_pcontacted_last_campaign, 'pdays'),
    ('pcampaign', add_pcampaign, 'poutcome'),
    ('previous', add_previous, 'previous'),
    ('campaign_gte10', add_campaign_gte10, 'campaign')
  ]
  cat_ct = ColumnTransformer(cat_features)
  
  return cat_ct

def get_categorical_pipeline():
  cat_cts = get_categorical_ct()

  # Create the pipeline to transform categorical features
  cat_pipeline = Pipeline([
    ('cat_ct', cat_cts),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
  ])

  return cat_pipeline

def get_numeric_pipeline():
  binning_pipeline = Pipeline([
    ('log', FunctionTransformer(np.log, validate=True)),
    ('kbins', KBinsDiscretizer())
  ])

  # Create the transformers for numeric features
  # num_ct = ColumnTransformer([('numerics', 'passthrough', numerics)])

  # new_num_features = [
  #   ('num_ct', num_ct),
  #   ('ft_campaign_to_previous', FunctionTransformer(ft_campaign_to_previous, validate=False))
  # ]
  # num_union = FeatureUnion(new_num_features)

  # # Create the pipeline to transform numeric features
  # num_pipeline = Pipeline([
  #   ('num_union', num_union),
  #   ('scaler', RobustScaler())
  # ])

  age_campaign_ct = ColumnTransformer([
    ('age_pipeline', clone(binning_pipeline), ['age']),
    ('campaign_pipeline', clone(binning_pipeline), ['campaign'])
  ])
  
  return age_campaign_ct

def get_pipeline():
  # Create the categorical and numeric pipelines
  cat_pipeline = get_categorical_pipeline()
  num_pipeline = get_numeric_pipeline()

  # Create the feature union of categorical and numeric attributes
  ft_union = FeatureUnion([
    ('cat_pipeline', cat_pipeline),
    # ('num_pipeline', num_pipeline)
  ])

  pipeline = Pipeline([
    ('ft_union', ft_union)
  ])

  return pipeline