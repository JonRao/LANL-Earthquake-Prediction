import logging
import lightgbm as lgb
import pandas as pd

from model.model_train import ModelTrain

logger = logging.getLogger('LANL.train.lgb')

LGB_PARAMS = {
          'num_leaves': 64,
          'min_data_in_leaf': 10,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          "random_state": 42,
          'n_estimators': 20000,
          'silent': True,
         }

class LGBModel(ModelTrain):
    def __init__(self, feature_version=None, params=LGB_PARAMS):
        super().__init__(feature_version=feature_version, params=params, logger=logger)
        self.feature_rank = []
    
    def train(self, X, y, X_valid, y_valid):
        """ Train model output model for prediction"""
        model = lgb.LGBMRegressor(**self.params)
        
        eval_set = [(X, y), (X_valid, y_valid)]

        model.fit(X, y, eval_set=eval_set, eval_metric='mae', verbose=1000, early_stopping_rounds=200)
        self.feature_importance(X.columns, model)

        def predict(X):
            """ wrapper for prediction"""
            y_pred = model.predict(X, num_iteration=model.best_iteration_)
            return y_pred
        return predict

    def feature_importance(self, column_names, model):
        model.feature_importances_
        df = pd.DataFrame({'feature': column_names, 'importance': model.feature_importances_})
        self.feature_rank.append(df)
    
    def rank_feature(self):
        df = pd.concat(self.feature_rank, axis=0)
        df['importance'] /= len(self.feature_rank)
        df = df.groupby('feature').mean().sort_values(by='importance', ascending=False)
        return df

