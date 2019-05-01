import logging
import joblib

logger = logging.getLogger('LANL.data_transfer')

import data_loader
import data_train

def load_all_features():
    X_tr, _ = data_loader.load_transfrom_train()
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


def load_model():
    X_test = data_loader.load_transfrom_test()
    dump = joblib.load(r'./data/transfer/0430_2239_LGBModel_48_CV_2.01_1.91_0.71.p')
    col, _ = data_loader.load_feature_names()
    return dump[0][0].predict(X_test[col])

 
def prepare_model():
    X_tr, y_tr, X_test, _ = data_loader.load_data()
    fold_iter = list(data_train.fold_maker(X_tr, fold_choice='earthquake'))

    for _, feature_version in load_unique_feature():
        logger.info(f'Working on: {feature_version}')
        data_train.cv_predict_all_helper(feature_version, fold_iter, X_tr, y_tr, X_test)


if __name__ == '__main__':
    # print(len(load_all_features()))
    # for i, col in enumerate(load_unique_feature()):
    #     print(i, col[1])
    print(load_model())
