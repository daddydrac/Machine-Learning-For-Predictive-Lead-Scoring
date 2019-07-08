import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def build_tuned_model(name, base_model, X_train, y_train, hparams, scorer=None, cv_folds=5, pipeline=None):
  from time import time
  start = time()
  print('==> Starting {}-fold cross validation for {} model, {} examples'.format(str(cv_folds), name, len(X_train)))
  model = TunedModel(hparams, name=name, model=base_model, pipeline=pipeline)
  model.train(X_train, y_train, scorer, cv_folds)
  elapsed = time() - start
  print("==> Elapsed seconds: {:.3f}".format(elapsed))
  
  print('Best {} model: {}'.format(model.name, model.model))
  print('Best {} score: {:.3f}'.format(
    model.name,
    model.results.sort_values('mean_test_score', ascending=False
  ).head(1).mean_test_score.values[0]))

  return model

# ============================================================================================================
# Model
# ============================================================================================================
class Model(object):
  def __init__(self, name, model, pipeline=None):
    self.name = name
    self.model = model
    self.pipeline = pipeline

  def train(self, X, y):
    """ Fits the model and builds the full pipeline """
    if self.pipeline is None:
      X_transformed = X
      self.model_pipeline = make_pipeline(self.model)
    else:
      X_transformed = self.pipeline.fit_transform(X)
      self.model_pipeline = make_pipeline(self.pipeline, self.model)
      
    self.model.fit(X_transformed, y)
    
    return self

  def predict(self, X):
    """ Fits the model and builds the full pipeline 
    TODO: Make sure the model was fitted
    """
    # if self.pipeline is None:
    #   X_transformed = X
    # else:
    #   X_transformed = self.pipeline.fit_transform(X)
      
    preds = self.model_pipeline.predict(X)
    
    return preds

  def get_model_pipeline(self):
    """ Useful for cross validation to refit the pipeline on every round """
    full_pipeline = clone(self.pipeline)
    full_pipeline.steps.append((self.name, self.model))
    return full_pipeline

  def score(self, X, y, scorer):
    """ Scores the model using the scorer
  
    Postcondititions: 
      - score should not be 0 
      - model.predictions should have elements
    """
    score = 0
    
    if self.pipeline is None:
      model.predictions = self.model.predict(X)
      score = scorer(self.model, X, y)
    else:
      model_predictions = self.model_pipeline.predict(X)
      score = scorer(self.model_pipeline, X, y)

    return score

  def save(self, file_path):
    from joblib import dump
    dump(self, file_path)
  
  @staticmethod
  def load(file_path):
    from joblib import load
    model = load(file_path)
    return model
    
      
  def score_cv(self, X, y, scorer, k=5):
    """ Scores the model using the scorer
  
    Postcondititions: 
      - score should not be 0 
      - model.predictions should have elements
    """
    from sklearn.model_selection import cross_val_score
    
    score = 0
    
    if self.pipeline is None:
      score = cross_val_score(self.model, X, y, scoring=scorer, cv=k, n_jobs=-1)
    else:
      score = scorer(self.model_pipeline, X, y)
      
    return (score.mean(), score.std())

# ============================================================================================================
# TunedModel
# ============================================================================================================
class TunedModel(Model):
  """ A class used to optimize the hyperparameters for a machine learning algorithm

  Parameters
  ----------
  name : string
      The name of a model
      
  param_grid : dict
      A dict of (parameter, values) pairs to optimize
      
  pipeline : object
      A pipeline to apply to the data before fitting the model
  """

  def __init__(self, param_grid, **kwargs):
      Model.__init__(self, **kwargs)
      self.param_grid = param_grid

  def train(self, X, y, scorer, cv_folds=5):
      """ Tunes a model using the parameter grid that this class was initialized with.
      
      Parameters
      ----------
      X : array-like, matrix
          Input data
          
      y : array-like
          Targets for input data
          
      cv_folds : int, optional, default: 5
          The number of cross-validation folds to use in the optimization process.
      """
      if not self.pipeline:
        # Setup
        grid_search = GridSearchCV(
            self.model, 
            self.param_grid, 
            cv=cv_folds,
            scoring=scorer, 
            return_train_score=True, 
            n_jobs=-1)
        
        # Run it
        grid_search.fit(X, y)
        
        # Save the model
        self.model = grid_search.best_estimator_
      else:
        # Setup
        grid_search = GridSearchCV(
            self.get_model_pipeline(), 
            self.param_grid, 
            cv=cv_folds,
            scoring=scorer, 
            return_train_score=True, 
            n_jobs=-1)
        
        # Run it
        grid_search.fit(X, y)
        
        # Save the model and pipeline
        self.model = grid_search.best_estimator_.steps[-1][1]
        self.pipeline = Pipeline(grid_search.best_estimator_.steps[:-1])

      self.results = pd.DataFrame(grid_search.cv_results_)