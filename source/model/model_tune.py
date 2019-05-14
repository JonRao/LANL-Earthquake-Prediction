import pandas as pd
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK 
import numpy as np
import mock
import logging
from sklearn.metrics import mean_absolute_error

# from xgb_train import XGBModel
from model.lgb_train import LGBModel

class ModelTune(object):
    MAX_EVALS = 200

    def __init__(self, X, y, X_test, fold_iter, model, space):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.fold_iter = fold_iter

        self.model = model
        self.model.logger = mock.MagicMock()
        self.space = space
        self.logger = logging.getLogger(f'LANL.optimize.{model.model_name}')
        self.tune_params = self.parse_tune_param()
    
    def parse_tune_param(self):
        dump = []
        for key, val in self.space.items():
            if callable(val):
                dump.append(key)
        return dump
    
    def objective(self, params):
        """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
        self.model.update(params)
        self.model.train_CV_test(self.X, self.y, self.X_test, self.fold_iter)

        dump = [f"loss: {self.model.mean_score:.5f}",]
        for name in self.tune_params:
            dump.append(f'{name}: {params[name]}')
        log = ','.join(dump)
        self.logger.info(log)

        return {'loss': self.model.oof_score, 'params': self.model.params, 'status': STATUS_OK}

    def tune(self):
        best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, 
                    max_evals=self.MAX_EVALS)
        return best


def tune_lgb(X, y, X_test, fold_iter, feature_version=5):
    """ Tune lgb model"""
    fold_iter = list(fold_iter)
    space = {
        # 'num_leaves': hp.choice('num_leaves', np.arange(30, 100, 4, dtype=int)),
        # 'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
        # 'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(10, 100, 5, dtype=int)),
        'num_leaves': 30,
        'learning_rate': 0.09,
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
        "lambda_l2": 0.2030,
    }
    model = LGBModel(feature_version=feature_version)
    obj = ModelTune(X, y, X_test, fold_iter, model, space)
    return obj.tune()

# def tune_randomforest(X, y, X_test, fold_iter):
#     random_grid = {
#         'n_estimators': n_estimators,
#         'max_features': max_features,
#         'max_depth': max_depth,
#         'min_samples_split': min_samples_split,
#         'min_samples_leaf': min_samples_leaf,
#         'bootstrap': bootstrap}



