import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneGroupOut
import pickle
import os


import data_loader
import data_transform
from model.xgb_train import XGBModel
from model.lgb_train import LGBModel
from model.cat_train import CatModel
from model.sklearn_train import SklearnModel, package
from model.model_tune import tune_lgb

logger = logging.getLogger('LANL.train')

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

def cv_predict(fold_choice):
    X_tr, y_tr, X_test, file_group = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice=fold_choice)

    # model = SklearnModel('RandomForest')
    model = XGBModel(feature_version='5')
    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    model.store_model()
    # df = model.rank_feature()
    # df.to_csv('./feature_tmp.csv')
    # predicted_result = data_train.train_CV_test(X_tr, y_tr, X_test, fold_iter, model_choice='lgb', params=data_train.LGB_PARAMS)
    return predicted_result, oof, file_group, model.oof_score

def cv_predict_all(fold_choice, feature_version):
    """ Generate prediction packages"""
    X_tr, y_tr, X_test, file_group = data_loader.load_data()
    fold_iter = list(fold_maker(X_tr, fold_choice=fold_choice))
    for model in LGBModel.subclasses:
        if model is SklearnModel:
            for name in package:
                if name != 'RandomForest':
                    obj = model(feature_version=feature_version, model_name=name)
                    obj.train_CV_test(X_tr, y_tr, X_test, fold_iter)
                    obj.store_model()
        else:
            continue
            obj = model(feature_version=feature_version)
            obj.train_CV_test(X_tr, y_tr, X_test, fold_iter)
            obj.store_model()

def ensemble(fold_choice='earthquake'):
    """ Stack/blend/ensemble from existing outputs"""
    folder = r'./data/prediction'
    X_tr, y_tr = data_loader.load_transfrom_train()
    X_test = data_loader.load_transfrom_test()
    file_group = X_test.index
    fold_iter = list(fold_maker(X_tr, fold_choice=fold_choice))

    train_stack = []
    test_stack = []
    for name in os.listdir(folder):
        if 'CV' in name:
            path = os.path.join(folder, name)
            data = pickle.load(open(path, 'rb'))
            test_stack.append(data['prediction'])
            train_stack.append(data['oof'])

    train_stack = np.vstack(train_stack).transpose()
    train_stack = pd.DataFrame(train_stack)
    test_stack = np.vstack(test_stack).transpose()
    test_stack = pd.DataFrame(test_stack)

    model = SklearnModel(model_name='RandomForest', feature_version='stack')
    predicted_result, oof = model.train_CV_test(train_stack, y_tr, test_stack, fold_iter)
    return predicted_result, oof, file_group, model.oof_score

def tune_model():
    X_tr, y_tr, X_test, _ = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice='earthquake')

    result = tune_lgb(X_tr, y_tr, X_test, fold_iter)
    return result