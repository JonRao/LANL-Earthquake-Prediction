
import logging
import numpy as np
import xgboost as xgb
from model.model_train import ModelTrain

logger = logging.getLogger('LANL.train.xgb')

XGB_PARAMS = {'eta': 0.03,
              'max_depth': 9,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
            #   'feature_selector': 'thrifty',
            #   'top_k': 100,
            
              }

class XGBModel(ModelTrain):
    def __init__(self, feature_version=None, params=XGB_PARAMS):
        super().__init__(feature_version=feature_version, params=params, logger=logger)
    
    def train(self, X, y, X_valid, y_valid):
        """ Train model output model for prediction"""

        train_data = xgb.DMatrix(data=X, label=y, feature_names=X.columns)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_valid.columns)
        watchlist = [(train_data, 'train'), (valid_data, 'valid'),]

        # TODO: runtime parameters will be allowed to customize?
        # reasonable start point
        self.params['base_score'] = np.mean(y.values)
        model = xgb.train(dtrain=train_data, num_boost_round=20000, early_stopping_rounds=200, 
                          evals=watchlist, verbose_eval=500, params=self.params)

        def predict(X):
            """ wrapper for prediction"""
            data = xgb.DMatrix(data=X, feature_names=X.columns)
            y_pred = model.predict(data, ntree_limit=model.best_ntree_limit)
            return y_pred

        return predict, model