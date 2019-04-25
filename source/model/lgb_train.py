import logging
import lightgbm as lgb
from model.model_train import ModelTrain

logger = logging.getLogger('LANL.train.lgb')

LGB_PARAMS = {
          'num_leaves': 51,
          'min_data_in_leaf': 10,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 1,
          "bagging_fraction": 0.91,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          "random_state": 42,
          'n_estimators': 20000,
          'silent': True,
        #   'reg_alpha': 0.1302650970728192,
        #   'reg_lambda': 0.3603427518866501,
        #   'num_threads': 5,
        #   'device': 'gpu',
        #   'gpu_platform_id': 0,
        #   'gpu_device_id':  0,
         }

class LGBModel(ModelTrain):
    def __init__(self, params=LGB_PARAMS):
        super().__init__(params, logger)
    
    def train(self, X, y, X_valid, y_valid):
        """ Train model output model for prediction"""
        model = lgb.LGBMRegressor(**self.params)
        
        eval_set = [(X, y), (X_valid, y_valid)]

        model.fit(X, y, eval_set=eval_set, eval_metric='mae', verbose=1000, early_stopping_rounds=200)

        def predict(X):
            """ wrapper for prediction"""
            y_pred = model.predict(X, num_iteration=model.best_iteration_)
            return y_pred
        return predict