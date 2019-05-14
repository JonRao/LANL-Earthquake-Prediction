import logging
import joblib
import os

logger = logging.getLogger('LANL.data_transfer')

import data_loader
import data_train

def load_all_features():
    X_tr, _ = data_loader.load_transform_train()
    return set(X_tr.columns)

def load_feature(start=0):
    all_feature = load_all_features()
    while True:
        try:
            col_names, feature_version = data_loader.load_feature_names(start)
            if not (len(set(col_names) - all_feature) > 0):
                yield col_names, feature_version
            start += 1
        except:
            break

def load_unique_feature():
    dump = set()
    for col_names, feature_version in load_feature():
        key = frozenset(col_names)
        if key not in dump:
            dump.add(key)
            yield col_names, feature_version
        else:
            print(f'Duplicated feature version: {feature_version}')

def load_undone_feature_old():
    name_group = os.listdir(r'./data/transfer')
    try:
        current_feature = int(max(name_group, default=0, key=lambda x: int(x.split('_')[3])).split('_')[3])
    except:
        # empty folder case
        current_feature = -1 
    for col, feature_version in load_unique_feature():
        if feature_version > current_feature:
            yield col, feature_version

def load_undone_feature():
    name_group = os.listdir(r'./data/prediction')
    try:
        current_feature = int(max(name_group, default=0, key=lambda x: int(x.split('_')[3])).split('_')[3])
    except:
        # empty folder case
        current_feature = -1 
    for col, feature_version in load_unique_feature():
        if feature_version > current_feature:
            yield col, feature_version

def load_limited_feature(n):
    for col, feature_version in load_undone_feature():
        if len(col) <= n:
            yield col, feature_version


def load_model():
    X_test = data_loader.load_transform_test()
    dump = joblib.load(r'./data/transfer/0430_2239_LGBModel_48_CV_2.01_1.91_0.71.p')
    col, _ = data_loader.load_feature_names()
    return dump[0][0].predict(X_test[col])

 
def prepare_model(fold_choice, number_rounds=99, n=250):
    X_tr, y_tr, X_test, _ = data_loader.load_data()
    fold_iter = list(data_train.fold_maker(X_tr, fold_choice))

    for i, (_, feature_version) in enumerate(load_limited_feature(n)):
        if i >= number_rounds:
            break
        logger.info(f'Working on: {feature_version}')
        data_train.cv_predict_all_helper(feature_version, fold_iter, X_tr, y_tr, X_test)


if __name__ == '__main__':
    # print(len(load_all_features()))
    # for i, col in enumerate(load_unique_feature()):
    #     print(i, col[1])
    # print(load_model())
    for i, j in load_limited_feature(n=250):
        print(j)
