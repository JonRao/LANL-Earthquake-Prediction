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
    X_tr, y_tr = data_loader.load_transfrom_train()
    X_tr, means_dict = data_transform.missing_fix_tr(X_tr)

    X_test = data_loader.load_transfrom_test()
    file_group = X_test.index

    X_test = X_test[X_tr.columns]
    X_test = data_transform.missing_fix_test(X_test, means_dict)

    X_tr = X_tr.clip(-1e8, 1e8)
    X_test = X_test.clip(-1e8, 1e8)

    # scaler = StandardScaler()
    # scaler.fit(X_tr)
    # scaled_train_X = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
    # scaled_test_X = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    fold_iter = fold_maker(X_tr, fold_choice=fold_choice)
    # model = SklearnModel('RandomForest')
    model = LGBModel()
    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    model.store_model()
    df = model.rank_feature()
    df.to_csv('./feature.csv')
    # predicted_result = data_train.train_CV_test(X_tr, y_tr, X_test, fold_iter, model_choice='lgb', params=data_train.LGB_PARAMS)
    return predicted_result, oof, file_group

def blend(fold_choice):
    X_tr, y_tr = data_loader.load_transfrom_train()
    X_tr, means_dict = data_transform.missing_fix_tr(X_tr)

    X_test = data_loader.load_transfrom_test()
    file_group = X_test.index

    X_test = X_test[X_tr.columns]
    X_test = data_transform.missing_fix_test(X_test, means_dict)

    X_tr = X_tr.clip(-1e8, 1e8)
    X_test = X_test.clip(-1e8, 1e8)

    fold_iter = list(fold_maker(X_tr, fold_choice=fold_choice))
    prediction = np.zeros(len(X_test))

    for name in package:
        model = SklearnModel(name)
        predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
        prediction += predicted_result
        model.store_model()


    # for model in XGBModel.subclasses:
    #     try:
    #         if model is SklearnModel:
    #             for name in package:
    #                 model = model(name)
    #                 predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    #                 prediction += predicted_result
    #                 model.store_model()
    #         else:
    #             model = model()
    #             predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    #             prediction += predicted_result
    #             model.store_model()
    #     except:
    #         continue
    # predicted_result = data_train.train_CV_test(X_tr, y_tr, X_test, fold_iter, model_choice='lgb', params=data_train.LGB_PARAMS)
    return prediction / len(XGBModel.subclasses), oof, file_group

def stack(fold_choice='earthquake'):
    """ Stack from existing models"""
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

    model = LGBModel(feature_version='stack')
    predicted_result, oof = model.train_CV_test(train_stack, y_tr, test_stack, fold_iter)
    return predicted_result, oof, file_group