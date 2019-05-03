import pickle
import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging

import data_loader
import data_transform
import data_train
import data_transfer

warnings.filterwarnings("ignore")

def main():
    # predictor = blend()
    # generateSubmission(predictor, test_data)

    logger = log_prep()
    logger.info('Begin Logging:')
    data_transfer.prepare_model(5)
    # predicted_result, _, file_group = data_train.stack()
#     predicted_result, _, file_group, score = data_train.cv_predict('earthquake')
    # generateSubmission(predicted_result, file_group, file_name='all_lgb')
#     feature_selection_iterative()
    # result = data_train.tune_model()
    # print(result)
#     predicted_result, _, file_group, score = data_train.cv_predict('earthquake')
#     for v in range(41, 48):
#         data_train.cv_predict_all('earthquake', feature_version=v)
    # predicted_result, _, file_group, score = data_train.ensemble_filter(col)
    # generateSubmission(predicted_result, file_group, file_name=f'ensemble_{score:.2f}')
    # feature_ensemble_iterative()
    
def feature_ensemble_iterative():
    logger = logging.getLogger('LANL.train.feature_select')
    df = pd.read_csv('./feature_ensemble.csv')
    num_feature_group = [250, 150, 100, 50, 25, 20, 15, 10]
    for i, num in enumerate(num_feature_group):
        logger.info(f'Iteration {i} - features - {num}')
        col = set(df['feature'].tolist()[:num])
        predicted_result, _, file_group, score = data_train.ensemble_filter(col)
        generateSubmission(predicted_result, file_group, file_name=f'top_{num}_ensemble_{score:.2f}')

def feature_selection_iterative():
    logger = logging.getLogger('LANL.train.feature_select')
    num_feature_group = [500, 400, 250, 150, 100, 50, 25, 20, 15, 10]
    for i, num in enumerate(num_feature_group):
        df = pd.read_csv('./feature_tmp.csv')
        col = df['feature'].tolist()[:num]
        data_loader.store_feature_names(col)
        logger.info(f'Iteration {i} - features - {num}')
        predicted_result, _, file_group, score = data_train.cv_predict('earthquake')
        generateSubmission(predicted_result, file_group, file_name=f'top_feature_{num}_score_{score:.2f}_lgb')

def generateSubmission(predicted_result, file_group, file_name='submission.csv'):
    df = pd.Series(predicted_result, index=file_group).to_frame()
    df = df.rename(columns={0: 'time_to_failure'})
    df.index.name = 'seg_id'
    df['time_to_failure'] = df['time_to_failure'].clip(0, 16)
    df.to_csv(f'./test_result/{file_name}.csv')

def log_prep():
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

    return logger

if __name__ == '__main__':
    main()
    # result = hypo_train_xgb()
    # print(result)