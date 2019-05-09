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
          'learning_rate': 0.005,
          "boosting": "gbdt",
          "bagging_freq": 9,
          "bagging_fraction": 0.91,
        #   "feature_fraction_fraction": 0.38314,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          "random_state": 42,
          'n_estimators': 20000,
          'silent': True,
          "lambda_l2": 1.545969040487455e-05,
          "lambda_l1": 0.2030,
         }
# tune based on feature version 5
TUNED = {
    # 'num_leaves': hp.choice('num_leaves', np.arange(30, 100, 4, dtype=int)),
    # 'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
    # 'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(10, 100, 5, dtype=int)),
    'num_leaves': 30,
    'learning_rate': 0.01,
    'min_data_in_leaf': 10,

    # "bagging_freq": hp.choice('bagging_freq', np.arange(1, 10, 1, dtype=int)),
    # "bagging_fraction": hp.uniform('bagging_fraction', 0, 1),
    # "feature_fraction_fraction": hp.uniform('feature_fraction_fraction', 0, 1),
    "bagging_freq": 9,
    "bagging_fraction": 0.44024,
    "feature_fraction_fraction": 0.38314,
    "bagging_seed": 11,
    "feature_fraction_seed": 11,
    "lambda_l2": 1.545969040487455e-05,
    # "lambda_l1": hp.uniform('lambda_l1', 0, 1),
    "lambda_l1": 0.2030,
}

class LGBModel(ModelTrain):
    def __init__(self, feature_version=None, params=LGB_PARAMS):
        super().__init__(feature_version=feature_version, params=params, logger=logger)
        self.feature_rank = []
    
    def train(self, X, y, X_valid, y_valid):
        """ Train model output model for prediction"""
        self.update(TUNED)
        model = lgb.LGBMRegressor(**self.params)
        
        eval_set = [(X, y), (X_valid, y_valid)]

        model.fit(X, y, eval_set=eval_set, eval_metric='mae', verbose=200, early_stopping_rounds=200)
        self.feature_importance(X.columns, model)

        def predict(X):
            """ wrapper for prediction"""
            y_pred = model.predict(X, num_iteration=model.best_iteration_)
            return y_pred
        return predict, model

    def feature_importance(self, column_names, model):
        df = pd.DataFrame({'feature': column_names, 'importance': model.feature_importances_})
        self.feature_rank.append(df)
    
    def rank_feature(self):
        df = pd.concat(self.feature_rank, axis=0)
        df['importance'] /= len(self.feature_rank)
        df = df.groupby('feature').mean().sort_values(by='importance', ascending=False)
        return df

