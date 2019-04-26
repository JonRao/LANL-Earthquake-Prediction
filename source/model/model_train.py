from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import datetime
import os.path
import pickle
import json

import data_loader

class ModelTrain(metaclass=ABCMeta):

    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)
    
    def __init__(self, feature_version=None, params=None, logger=None):
        self.params = params
        self.logger = logger
        self.columns, self.feature_version = data_loader.load_feature_names(feature_version)
        self.model_name = type(self).__name__

    def update(self, params):
        self.params.update(params)

    @abstractmethod
    def train(self, X_train, y_train, X_valid, y_valid):
        pass
    
    def store_model(self):
        prefix = datetime.datetime.now().strftime(r'%m%d_%H%M')
        name_model = f'{prefix}_{self.model_name}_{self.feature_version}_CV_{self.mean_score:.2f}_{self.std_score:.2f}'

        data = {}
        data['prediction'] = self.prediction
        data['oof'] = self.oof

        pickle.dump(data, open(os.path.join('./data/prediction', name_model), 'wb'))
        
        params = {}
        params['feature_version'] = self.feature_version
        params['params'] = self.params
        
        with open(os.path.join('./data/params', name_model), 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

        self.logger.info('Model stored!')

    def train_CV_test(self, X, y, X_test, fold_iter):
        """ Return predicted values as well as oof"""
        dump = []
        prediction = np.zeros(len(X_test))
        oof = np.zeros(len(y))
        X = X[self.columns]
        X_test = X_test[self.columns]

        for fold_n, (train_index, valid_index) in enumerate(fold_iter):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            predictor = self.train(X_train, y_train, X_valid, y_valid)
            y_pred = predictor(X_valid)

            oof[valid_index] = y_pred
            prediction += predictor(X_test)

            score = mean_absolute_error(y_pred, y_valid)
            self.logger.info(f"fold: {fold_n}, score: {score:.2f}")
            dump.append(score)

        # store all necessary info
        self.mean_score = np.mean(dump)
        self.std_score = np.std(dump)
        self.prediction = prediction / fold_n
        self.oof = oof


        self.logger.info(f"mean_score: {self.mean_score:.2f}, std: {self.std_score:.2f}")
        return self.prediction, self.oof