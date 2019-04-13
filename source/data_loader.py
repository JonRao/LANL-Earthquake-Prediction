import pandas as pd
import numpy as np
import os
import pickle
import pandas as pd
from tqdm import tqdm

import data_transform

def load_train():
    train = pickle.load(open('../data/train.p', 'rb'))
    return train

def load_transfrom_train(update=False):
    cache_path = '../data/transform.p'
    existed = True
    if os.path.exists(cache_path):
        X_tr, y_tr = pickle.load(open(cache_path, 'rb'))
    else:
        existed = False
    
    if (not existed) or update:
        train = load_train()
        # train = train.head(15_000_000)
        X_tr_new, y_tr_new = data_transform.transform_train(train)
        if existed:
            # upload cache
            X_tr_new = pd.concat([X_tr, X_tr_new], axis=1)

        dump = X_tr_new, y_tr_new
        pickle.dump(dump, open(cache_path, 'wb'))
        X_tr, y_tr = dump
    

    return X_tr, y_tr

def load_transfrom_test(update=False):
    cache_path = '../data/test_transform.p'
    existed = True
    if os.path.exists(cache_path):
        result = pickle.load(open(cache_path, 'rb'))
    else:
        existed = False

    if (not existed) or update:
        raw = load_test()
        dump = {}
        for name, df in tqdm(raw):
            tmp = data_transform.transform(df)
            dump[name] = dict(tmp)
        result_new = pd.DataFrame(dump).T
        if existed:
            # upload cache
            result_new = pd.concat([result, result_new], axis=1)
        pickle.dump(result_new, open(cache_path, 'wb'))
        result = result_new

    return result

def load_test():
    cache_path = '../data/test_submission.p'
    test_path = '../data/test'
    if os.path.exists(cache_path):
        result = pickle.load(open(cache_path, 'rb'))
    else:
        result = []
        for r, d, f in os.walk(test_path):
            for file in f:
                if file.endswith('.csv'):
                    name = file.split('.', 1)[0]
                    df = pd.read_csv(os.path.join(test_path, file), dtype={'acoustic_data': np.int16})
                    result.append((name, df['acoustic_data']))

        pickle.dump(result, open(cache_path, 'wb'))
    return result

def load_earthquake_id():
    cache_path = '../data/earthquake_id.p'
    if os.path.exists(cache_path):
        earthquake_id = pickle.load(open(cache_path, 'rb'))
    else:
        train = load_train()
        earthquake_id = data_transform.transform_earthquake_id(train)
        pickle.dump(earthquake_id, open(cache_path, 'wb'))

    return earthquake_id


if __name__ == '__main__':
    load_transfrom_train()
    load_transfrom_test()