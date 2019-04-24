import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneGroupOut


import data_loader
import data_transform
from model.xgb_train import XGBModel
from model.lgb_train import LGBModel
from model.cat_train import CatModel

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

def cv_predict(fold_choice='earthquake'):
    X_tr, y_tr = data_loader.load_transfrom_train()
    X_tr, means_dict = data_transform.missing_fix_tr(X_tr)

    X_test = data_loader.load_transfrom_test()
    file_group = X_test.index

    X_test = X_test[X_tr.columns]
    X_test = data_transform.missing_fix_test(X_test, means_dict)

    # X_tr = X_tr.clip(-1e6, 1e6)
    # X_test = X_test.clip(-1e6, 1e6)

    # scaler = StandardScaler()
    # scaler.fit(X_tr)
    # scaled_train_X = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
    # scaled_test_X = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # fold_iter = fold_maker(X_tr, fold_choice=fold_choice)
    # model = CatModel()
    # predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    # # predicted_result = data_train.train_CV_test(X_tr, y_tr, X_test, fold_iter, model_choice='lgb', params=data_train.LGB_PARAMS)
    # generateSubmission(predicted_result, file_group, file_name='submission_lgb_mae_reg_more')

def generateSubmission(predicted_result, file_group, file_name='submission.csv'):
    df = pd.Series(predicted_result, index=file_group).to_frame()
    df = df.rename(columns={0: 'time_to_failure'})
    df.index.name = 'seg_id'
    df['time_to_failure'] = df['time_to_failure'].clip(0, 16)
    df.to_csv(f'./test_result/{file_name}.csv')