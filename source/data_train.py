import pandas as pd
import numpy as np
import logging
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

import source.data_loader as data_loader

logger = logging.getLogger('seismic_prediction.train')


XGB_PARAMS = {'eta': 0.03,
              'max_depth': 9,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': True,
            #   'feature_selector': 'thrifty',
            #   'top_k': 100,
            
              }

LGB_PARAMS = {
          'num_leaves': 51,
          'min_data_in_leaf': 10,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.03,
          'max_depth': 9,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          "random_state": 42,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
        #   'device': 'gpu',
        #   'gpu_platform_id': 0,
        #   'gpu_device_id':  0,
         }

def train_CV(X, y, fold_iter, model_choice='xgb', params=XGB_PARAMS):
    dump = []
    for fold_n, (train_index, valid_index) in enumerate(fold_iter):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        predictor = train_model(model_choice, X_train, y_train, params, X_valid, y_valid)
        y_pred = predictor(X_valid)

        score = mean_absolute_error(y_pred, y_valid)
        logger.info(f"model: {model_choice}, fold: {fold_n}, score: {score:.2f}")
        dump.append(score)
    logger.info(f"model: {model_choice}, mean_score: {np.mean(dump):.2f}, std: {np.std(dump):.2f}")
    return dump

def train_CV_test(X, y, X_test, fold_iter, model_choice='xgb', params=XGB_PARAMS):
    dump = []
    prediction = np.zeros(len(X_test))

    for fold_n, (train_index, valid_index) in enumerate(fold_iter):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # lda = LDA(n_components=20)
        # X_train = lda.fit_transform(X_train, y_train)
        # X_valid = lda.transform(X_valid)
        predictor = train_model(model_choice, X_train, y_train, params, X_valid, y_valid)
        y_pred = predictor(X_valid)
        prediction += predictor(X_test)

        score = mean_absolute_error(y_pred, y_valid)
        logger.info(f"model: {model_choice}, fold: {fold_n}, score: {score:.2f}")
        dump.append(score)
    logger.info(f"model: {model_choice}, mean_score: {np.mean(dump):.2f}, std: {np.std(dump):.2f}")
    return prediction / fold_n

def fold_maker(X, n_fold=10, fold_choice='default'):
    if fold_choice == 'default':
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
        fold_iter = folds.split(X)
    elif fold_choice == 'earthquake':
        earthquake_id = data_loader.load_earthquake_id()
        group_kfold = LeaveOneGroupOut()
        fold_iter = group_kfold.split(X, groups=earthquake_id)
    else:
        raise AttributeError(f"Not support CV {fold_choice} yet...")

    return fold_iter

def train_model(model_choice, X, y, params=None, X_valid=None, y_valid=None):
    if model_choice == 'xgb':
        if params is None:
            params = XGB_PARAMS
        predictor = train_xgb(X, y, params, X_valid, y_valid)
    elif model_choice == 'lgb':
        if params is None:
            params = LGB_PARAMS
        predictor = train_lgb(X, y, params, X_valid, y_valid)
    elif model_choice == 'rnn':
        predictor = train_rnn(X, y, params, X_valid, y_valid)
    else:
        raise AttributeError(f"Not support {model_choice} yet...")
    return predictor

def train_xgb(X, y, params=XGB_PARAMS, X_valid=None, y_valid=None):

    train_data = xgb.DMatrix(data=X, label=y, feature_names=X.columns)
    watchlist = [(train_data, 'train'),]
    if X_valid is not None:
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_valid.columns)
        watchlist += [(valid_data, 'valid'),]

    model = xgb.train(dtrain=train_data, num_boost_round=20000, early_stopping_rounds=500, 
                      evals=watchlist, verbose_eval=500, params=params)

    def predict(X):
        """ wrapper for prediction"""
        data = xgb.DMatrix(data=X, feature_names=X.columns)
        y_pred = model.predict(data, ntree_limit=model.best_ntree_limit)
        return y_pred
    return predict

def train_lgb(X, y, params=LGB_PARAMS, X_valid=None, y_valid=None):

    model = lgb.LGBMRegressor(**params, n_estimators=50000, silent=True)
    if X_valid is not None:
        eval_set = [(X_valid, y_valid),]
    else:
        eval_set = None

    model.fit(X, y, eval_set=eval_set, eval_metric='rmse', verbose=False, early_stopping_rounds=500)

    def predict(X):
        """ wrapper for prediction"""
        y_pred = model.predict(X, ntree_limit=model.best_iteration_)
        return y_pred
    return predict

def train_rnn(X, y, params, X_valid=None, y_valid=None):
    cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]

    model = Sequential()
    model.add(CuDNNGRU(48, input_shape=(None, X.shape[1])))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.summary()
    model.compile(optimizer=adam(lr=0.0005), loss="mae")

    history = model.fit(X, y,
                        batch_size=2048,
                        steps_per_epoch=1000,
                        epochs=50,
                        verbose=0,
                        callbacks=cb)

    def predict(X):
        """ wrapper for prediction"""
        y_pred = model.predict(X, batch_size=2048)
        return y_pred
    return predict
