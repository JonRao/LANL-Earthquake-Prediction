import logging
from catboost import CatBoostRegressor
from model.model_train import ModelTrain

logger = logging.getLogger('LANL.train.cat')

CAT_PARAMS = {
              'iterations': 20000, 
              'eval_metric':'MAE',
              # early stopping related
              'od_type': 'Iter',
              'od_wait': 40
             }

class CatModel(ModelTrain):
    def __init__(self, feature_version=None, params=CAT_PARAMS):
        super().__init__(feature_version=feature_version, params=params, logger=logger)
    
    def train(self, X, y, X_valid, y_valid):
        """ Train model output model for prediction"""
        model = CatBoostRegressor(**self.params)
        eval_set = [(X_valid, y_valid)]
        model.fit(X, y, eval_set=eval_set, cat_features=[], use_best_model=True, verbose=500)

        def predict(X):
            """ wrapper for prediction"""
            y_pred = model.predict(X)
            return y_pred

        return predict, model