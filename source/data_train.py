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

def cv_predict(fold_choice, feature_version=None):
    X_tr, y_tr, X_test, file_group = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice=fold_choice)

    model = LGBModel(feature_version=feature_version)
    # model = SklearnModel('RandomForest')
    # model = SklearnModel('KernelRidge', feature_version=feature_version)
    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    model.store_model()
    # df = model.rank_feature()
    # df.to_csv('./feature_all.csv')
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


def ensemble(fold_choice='earthquake', lower=0.8, upper=44):
    """ Stack/blend/ensemble from existing outputs"""
    folder = r'./data/prediction'
    X_tr, y_tr = data_loader.load_transfrom_train()
    X_test = data_loader.load_transfrom_test()
    file_group = X_test.index
    fold_iter = list(fold_maker(X_tr, fold_choice=fold_choice))

    train_stack = []
    test_stack = []
    column_name = []
    column_name_unique = set()
    for name in os.listdir(folder):
        if 'CV' in name:
            score = float(name.rsplit('_', 3)[-2])
            if (score > lower) and (score < upper):
                model_name = name.split('_', 2)[-1]
                if model_name not in column_name_unique:
                    column_name_unique.add(model_name)
                    column_name.append(model_name)

                    path = os.path.join(folder, name)
                    data = pickle.load(open(path, 'rb'))
                    test_stack.append(data['prediction'])
                    train_stack.append(data['oof'])
                else:
                    logger.info(f'Ensemble excluded: {name}')
            else:
                logger.info(f'Ensemble excluded: {name}')

    train_stack = np.vstack(train_stack).transpose()
    train_stack = pd.DataFrame(train_stack, columns=column_name)
    test_stack = np.vstack(test_stack).transpose()
    test_stack = pd.DataFrame(test_stack, columns=column_name)

    # TODO: ensemble hyperparameter setup
    # model = SklearnModel(model_name='RandomForest', feature_version='stack')
    # model = SklearnModel(model_name='RandomForest', feature_version='stack')
    model = LGBModel(feature_version='stack')
    # model = CatModel(feature_version='stack')

    predicted_result, oof = model.train_CV_test(train_stack, y_tr, test_stack, fold_iter)
    df = model.rank_feature()
    df.to_csv('./feature_ensemble.csv')
    return predicted_result, oof, file_group, model.oof_score

def ensemble_filter(feature_group, fold_choice='earthquake'):
    folder = r'./data/prediction'
    X_tr, y_tr = data_loader.load_transfrom_train()
    X_test = data_loader.load_transfrom_test()
    file_group = X_test.index
    fold_iter = list(fold_maker(X_tr, fold_choice=fold_choice))

    train_stack = []
    test_stack = []
    column_name = []
    column_name_unique = set()
    for name in os.listdir(folder):
        if 'CV' in name:
            score = float(name.rsplit('_', 3)[-2])
            model_name = name.split('_', 2)[-1]
            if model_name not in column_name_unique:
                if model_name in feature_group:
                    column_name_unique.add(model_name)
                    column_name.append(model_name)

                    path = os.path.join(folder, name)
                    data = pickle.load(open(path, 'rb'))
                    test_stack.append(data['prediction'])
                    train_stack.append(data['oof'])
                else:
                    logger.info(f'Ensemble excluded: {name}')
            else:
                logger.info(f'Ensemble excluded: {name}')

    train_stack = np.vstack(train_stack).transpose()
    train_stack = pd.DataFrame(train_stack, columns=column_name)
    test_stack = np.vstack(test_stack).transpose()
    test_stack = pd.DataFrame(test_stack, columns=column_name)

    # TODO: ensemble hyperparameter setup
    # model = SklearnModel(model_name='RandomForest', feature_version='stack')
    # model = SklearnModel(model_name='RandomForest', feature_version='stack')
    model = LGBModel(feature_version='stack')
    # model = CatModel(feature_version='stack')

    predicted_result, oof = model.train_CV_test(train_stack, y_tr, test_stack, fold_iter)
    # df = model.rank_feature()
    # df.to_csv('./feature_tmp.csv')
    return predicted_result, oof, file_group, model.oof_score

def tune_model():
    X_tr, y_tr, X_test, _ = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice='earthquake')

    result = tune_lgb(X_tr, y_tr, X_test, fold_iter)
    return result