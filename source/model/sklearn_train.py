
import logging
import lightgbm as lgb
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

from model.model_train import ModelTrain

logger = logging.getLogger('LANL.train.sklearn')

package = {
            'NuSVR': (NuSVR, {'gamma': 'scale', 'nu': 0.9, 'C': 10.0, 'tol': 0.01}),
            'SVR': (SVR, {'gamma': 'scale', 'C': 1.0, 'epsilon': 0.2}),
            'KernelRidge': (KernelRidge, {'kernel': 'rbf', 'alpha': 0.1, 'gamma': 0.01}),
            'RandomForest': (RandomForestRegressor, {'n_estimators': 100, 'min_samples_leaf': 2, 'max_features': 0.5})
          }

class SklearnModel(ModelTrain):
    def __init__(self, model_name='SVR', feature_version=None, params=None):
        logger = logging.getLogger(f'LANL.train.{model_name}')
        super().__init__(feature_version=feature_version, params=params, logger=logger)
        self.model, self.params = package[model_name]
        self.model_name = model_name

        if params is not None:
            self.params = params
    
    def train(self, X, y, X_valid, y_valid):
        """ Train model output model for prediction"""
        model = self.model(**self.params)

        model.fit(X, y)

        def predict(X):
            """ wrapper for prediction"""
            y_pred = model.predict(X)
            return y_pred
        return predict