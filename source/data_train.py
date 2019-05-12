import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeavePGroupsOut
import pickle
import os
import random


import data_loader
import data_transform
from model.xgb_train import XGBModel
from model.lgb_train import LGBModel
from model.cat_train import CatModel
from model.sklearn_train import SklearnModel, package
from model.model_tune import tune_lgb

logger = logging.getLogger('LANL.train')

def fold_maker(X, fold_choice='default', n_fold=5, n_groups=2):
    if fold_choice == 'default':
        folds = KFold(n_splits=n_fold, shuffle=False)
        fold_iter = folds.split(X)
        fold_iter = shuffle_group(fold_iter)
    elif fold_choice == 'earthquake':
        earthquake_id = data_loader.load_earthquake_id()
        group_kfold = LeaveOneGroupOut()
        fold_iter = group_kfold.split(X, groups=earthquake_id)
        # fold_iter = shuffle_group(fold_iter)
        # fold_iter = min_valid_filter(fold_iter)
    elif fold_choice == f'eqCombo':
        earthquake_id = eqComboMaker(n_fold)
        group_kfold = LeaveOneGroupOut()
        fold_iter = group_kfold.split(X, groups=earthquake_id)
        fold_iter = shuffle_group(fold_iter)
    elif fold_choice == 'k-earthquake':
        earthquake_id = data_loader.load_earthquake_id()
        group_kfold = LeavePGroupsOut(n_groups=n_groups)
        fold_iter = group_kfold.split(X, groups=earthquake_id)
        fold_iter = min_valid_filter(fold_iter)
    elif fold_choice == 'customize':
        fold = CVPipe()
        fold_iter = fold.fold_iter(num_fold=n_fold, mini_quake_prob=0.3)
    else:
        raise AttributeError(f"Not support CV {fold_choice} yet...")

    return (list(fold_iter), fold_choice)

def eqComboMaker(n_fold):
    earthquake_id = data_loader.load_earthquake_id()
    for i in range(17):
        earthquake_id[earthquake_id == i] = i % n_fold
    return earthquake_id



def min_valid_filter(fold_iter, n=100):
    for train_index, valid_index in fold_iter:
        if len(valid_index) > n:
            yield train_index, valid_index
        else:
            logger.warn(f'Drop set with {len(valid_index)} less than minimum {n}')


class CVPipe(object):
    """ Finally, customized cv strategy
        Taking mini-quake into consideration
        Guess the ratio (mini-quake in test set)
    """
    MINI_QUAKE_ID = {2, 7, 14}
    def __init__(self):
        earthquake_id = data_loader.load_earthquake_id()
        self.mini = self.MINI_QUAKE_ID
        self.noMini = set(range(16)) - self.mini
        self.index_map, self.count_map = self.process_earthquake(earthquake_id)
    
    def process_earthquake(self, df):
        earthquake_index_map = {}
        earthquake_count_map = {}
        for i, tmp in df.groupby(df):
            earthquake_index_map[i] = tmp.index.tolist()
            earthquake_count_map[i] = len(tmp)
        return earthquake_index_map, earthquake_count_map
    
    def fold_iter(self, num_fold=20, mini_quake_prob=0):
        count = 0
        logger.info(f'Customize {num_fold}-fold, mini_quake_prob: {mini_quake_prob:.2f}')
        while (count < num_fold):
            train_index = random.sample(list(self.noMini), 10)
            valid_index = list(self.noMini - set(train_index))

            if np.random.uniform() < mini_quake_prob:
                #include mini-quake
                train_mini_index = random.sample(self.mini, 2)
                valid_mini_index = self.mini - set(train_mini_index)

                train_index = list(train_index) + list(train_mini_index)
                valid_index = list(valid_index) + list(valid_mini_index)
            else:
                # all goto train data
                train_index = list(train_index) + list(self.mini)

            random.shuffle(train_index)
            random.shuffle(valid_index)

            logger.info(f'Train: {train_index}, Valid: {valid_index}')
            train_index = self.get_index(train_index)
            valid_index = self.get_index(valid_index)
            count += 1
            yield train_index, valid_index
    
    def get_index(self, indexGroup):
        dump = []
        for i in indexGroup:
            dump.extend(self.index_map[i])
        return dump
    

def shuffle_group(fold_iter):
    """ Wrong for shuffling across different earthquakes"""
    for train_index, valid_index in fold_iter:
        np.random.shuffle(train_index)
        # np.random.shuffle(valid_index)
        yield train_index, valid_index

def cv_predict(fold_choice, feature_version=None, num_fold=10, feature_save=False):
    """ Use average fold prediction for testing data"""
    X_tr, y_tr, X_test, file_group = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice=fold_choice, n_fold=num_fold)

    model = LGBModel(feature_version=feature_version)
    # model = SklearnModel('RandomForest')
    # model = SklearnModel('KernelRidge', feature_version=feature_version)
    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    # model.store_prediction()
    if feature_save:
        df = model.rank_feature()
        df.to_csv('./feature_tmp.csv')
    return predicted_result, oof, file_group, model.oof_score


def cv_predict_all(fold_choice, feature_version):
    """ Generate prediction packages"""
    X_tr, y_tr, X_test, _ = data_loader.load_data()
    fold_iter = fold_maker(X_tr, fold_choice=fold_choice)
    cv_predict_all_helper(feature_version, fold_iter, X_tr, y_tr, X_test)


def cv_predict_all_helper(feature_version, fold_iter, X_tr, y_tr, X_test):
    for model in LGBModel.subclasses:
        if model is SklearnModel:
            for name in package:
                obj = model(feature_version=feature_version, model_name=name)
                obj.train_CV_test(X_tr, y_tr, X_test, fold_iter)
                # obj.store_model()
                obj.store_prediction()
        else:
            obj = model(feature_version=feature_version)
            obj.train_CV_test(X_tr, y_tr, X_test, fold_iter)
            # obj.store_model()
            obj.store_prediction()


def ensemble(X_tr, y_tr, X_test, fold_choice):
    fold_iter = fold_maker(X_tr, fold_choice=fold_choice)
    model = LGBModel(feature_version='stack')
    params = {"feature_fraction_fraction": 0.38,
              "bagging_freq": 5,}
        
    model.update(params)
    # model = SklearnModel('RandomForest', feature_version='stack')
    model = LGBModel(feature_version='stack')

    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    return predicted_result, oof, model.oof_score


def prepare_ensemble(feature_group=None, lower=0.3, upper=5):
    """ Only select model within feature group"""
    folder = r'./data/prediction'
    # _, y_tr = data_loader.load_transform_train()
    # X_test = data_loader.load_transform_test()
    # file_group = X_test.index
    _, y_tr, _, file_group = data_loader.load_data()
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
            score = float(name.split('_')[5])
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