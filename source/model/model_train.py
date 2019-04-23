from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

class ModelTrain(metaclass=ABCMeta):

    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)
    
    def __init__(self, params=None, logger=None):
        self.params = params
        self.logger = logger

    def update(self, params):
        self.params.update(params)

    @abstractmethod
    def train(self, X_train, y_train, X_valid, y_valid):
        pass
    
    def train_CV_test(self, X, y, X_test, fold_iter):
        """ Return predicted values as well as oof"""
        dump = []
        prediction = np.zeros(len(X_test))
        oof = np.zeros(len(y))

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

        self.logger.info(f"mean_score: {np.mean(dump):.2f}, std: {np.std(dump):.2f}")
        return prediction / fold_n, oof