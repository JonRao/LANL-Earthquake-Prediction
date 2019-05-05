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
        # fold_iter = shuffle_group(fold_iter)
    else:
        raise AttributeError(f"Not support CV {fold_choice} yet...")

    return fold_iter

def shuffle_group(fold_iter):
    """ Wrong for shuffling across different earthquakes"""
    for train_index, valid_index in fold_iter:
        np.random.shuffle(train_index)
        np.random.shuffle(valid_index)
        yield train_index, valid_index

def cv_predict(fold_choice, feature_version=None):
    X_tr, y_tr, X_test, file_group = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice=fold_choice)

    model = LGBModel(feature_version=feature_version)
    # model = SklearnModel('RandomForest')
    # model = SklearnModel('KernelRidge', feature_version=feature_version)
    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    model.store_prediction()
    df = model.rank_feature()
    df.to_csv('./feature_tmp.csv')
    return predicted_result, oof, file_group, model.oof_score


def cv_predict_all(fold_choice, feature_version):
    """ Generate prediction packages"""
    X_tr, y_tr, X_test, _ = data_loader.load_data()
    fold_iter = list(fold_maker(X_tr, fold_choice=fold_choice))
    cv_predict_all_helper(feature_version, fold_iter, X_tr, y_tr, X_test)


def cv_predict_all_helper(feature_version, fold_iter, X_tr, y_tr, X_test):
    for model in LGBModel.subclasses:
        if model is SklearnModel:
            for name in package:
                obj = model(feature_version=feature_version, model_name=name)
                obj.train_CV_test(X_tr, y_tr, X_test, fold_iter)
                obj.store_model()
                obj.store_prediction()
        else:
            obj = model(feature_version=feature_version)
            obj.train_CV_test(X_tr, y_tr, X_test, fold_iter)
            obj.store_model()
            obj.store_prediction()


def ensemble(X_tr, y_tr, X_test, fold_choice='earthquake'):
    fold_iter = list(fold_maker(X_tr, fold_choice=fold_choice))
    model = LGBModel(feature_version='stack')

    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    return predicted_result, oof, model.oof_score


def prepare_ensemble(feature_group=None, lower=0.3, upper=5):
    """ Only select model within feature group"""
    folder = r'./data/prediction'
    _, y_tr = data_loader.load_transfrom_train()
    X_test = data_loader.load_transfrom_test()
    file_group = X_test.index
    filter_pipe = filter_model(feature_group, lower, upper)
    next(filter_pipe)   # prime it

    train_stack = []
    test_stack = []
    column_name = []
    column_name_unique = set()
    excluded_count = 0
    for name in os.listdir(folder):
        model_name = name.split('_', 2)[-1]
        if filter_pipe.send(name) and (model_name not in column_name_unique):
            column_name_unique.add(model_name)
            column_name.append(model_name)

            path = os.path.join(folder, name)
            data = pickle.load(open(path, 'rb'))
            test_stack.append(data['prediction'])
            train_stack.append(data['oof'])
        else:
            excluded_count += 1
    
    logger.info(f'Excluded {excluded_count} model packages!')
    logger.info(f'{len(column_name)} model packages for ensemble!')

    train_stack = np.vstack(train_stack).transpose()
    train_stack = pd.DataFrame(train_stack, columns=column_name)
    test_stack = np.vstack(test_stack).transpose()
    test_stack = pd.DataFrame(test_stack, columns=column_name)
    return train_stack, y_tr, test_stack, file_group

def filter_model(feature_group, lower, upper):
    """Pipeline to filter out unwanted predictions: 
        1. not in feature group, and 2. not in score range """
    keep = False
    name = None
    while True:
        name = yield keep
        keep = False
        if 'CV' in name:
            score = float(name.rsplit('_', 3)[-2])
            # model_name = name.split('_', 2)[-1]
            feature_version = int(name.split('_')[3])
            if lower <= score <= upper:
                if (feature_group is None) or (feature_version in feature_group):
                    keep = True

def tune_model():
    X_tr, y_tr, X_test, _ = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice='earthquake')

    result = tune_lgb(X_tr, y_tr, X_test, fold_iter)
    return result