import pickle
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import logging

import data_loader
import data_transform
import data_train
from model.xgb_train import XGBModel
from model.lgb_train import LGBModel
from model.cat_train import CatModel

warnings.filterwarnings("ignore")

def main():
    # X_tr, y_tr = data_loader.load_transfrom_train()
    # X_tr, _ = data_transform.missing_fix_tr(X_tr)
    # fold_iter = data_train.fold_maker(X_tr, fold_choice='earthquake')
    # data_train.train_CV(X_tr, y_tr, fold_iter, model_choice='xgb', params=data_train.XGB_PARAMS)
    # X_test = data_loader.load_transfrom_test()
    # file_group = X_test.index
    # X_test = X_test[X_tr.columns]

    # data_train.train_CV_test(X_tr, y_tr, )
    # concat and scaling
    # X_tr, X_test = data_transform.concat_scaling(X_tr, X_test)
    # predictor = data_train.train_xgb(X_tr, y_tr)
    # generateSubmission(predictor, X_test, file_group=file_group)

    # predictor = blend()
    # generateSubmission(predictor, test_data)

    logger = logging.getLogger('LANL')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('./seismic.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('Begin Logging:')
    print(XGBModel.subclasses)
    cv_predict('earthquake')


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

    fold_iter = data_train.fold_maker(X_tr, fold_choice=fold_choice)
    model = CatModel()
    predicted_result, oof = model.train_CV_test(X_tr, y_tr, X_test, fold_iter)
    # predicted_result = data_train.train_CV_test(X_tr, y_tr, X_test, fold_iter, model_choice='lgb', params=data_train.LGB_PARAMS)
    generateSubmission(predicted_result, file_group, file_name='submission_lgb_mae_reg_more')

def blend():
    X_tr, y_tr = data_loader.load_transfrom_train()
    # # test_data = X_tr

    dump = []
    for model_choice in ('lgb', 'xgb'):
    # for model_choice in ('xgb',):
        print(f"Working on {model_choice}")
        predictor = data_train.train_model(model_choice, X_tr, y_tr)
        dump.append(predictor)

    def predict(X):
        tmp = []
        for model in dump:
            result = model(X)
            tmp.append(result)
        return np.mean(tmp, axis=0)
    return predict

def generateSubmission(predicted_result, file_group, file_name='submission.csv'):
    df = pd.Series(predicted_result, index=file_group).to_frame()
    df = df.rename(columns={0: 'time_to_failure'})
    df.index.name = 'seg_id'
    df['time_to_failure'] = df['time_to_failure'].clip(0, 16)
    df.to_csv(f'./test_result/{file_name}.csv')

def hypo_train_xgb():
    X_tr, y_tr = data_loader.load_transfrom_train()
    fold_iter = data_train.fold_maker(X_tr)

    XGB_PARAMS = {'eta': 0.03,
              'max_depth': 9,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
    }

    result = {}
    for eta in (0.02, 0.03, 0.05):
        for max_depth in (7, 9, 11):
            for subsample in (0.8, 0.9, 0.95):
                key = eta, max_depth, subsample
                params = dict(XGB_PARAMS)
                params['eta'] = eta
                params['max_depth'] = max_depth 
                params['subsample'] = subsample

                tmp = data_train.train_CV(X_tr, y_tr, fold_iter, params=params)
                result[key] = analyze_score(tmp)
    return result

def analyze_score(scoreGroup):
    result = {}
    result['mean'] = np.mean(scoreGroup)
    result['std'] = np.std(scoreGroup)
    result['max'] = np.max(scoreGroup)
    result['min'] = np.min(scoreGroup)
    return result

if __name__ == '__main__':
    main()
    # result = hypo_train_xgb()
    # print(result)