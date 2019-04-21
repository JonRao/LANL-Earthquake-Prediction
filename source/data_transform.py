# feature engineering
import pandas as pd
import numpy as np
import tqdm
import librosa  # MFCC feature
import warnings
from collections import ChainMap, defaultdict
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.signal import firls, convolve, decimate, hilbert, hann
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")

def transform_train(train, rows=150_000):
    """ Transfrom train data into X, y with lower frequency"""

    segments = int(np.floor(train.shape[0] / rows))
    X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)
    y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

    for segment in tqdm.tqdm(range(segments)):
        seg = train.iloc[segment*rows : segment*rows+rows]
        # x_denoised = resample(seg['acoustic_data'].values)
        # x = pd.Series(x_denoised.real)
        x = pd.Series(seg['acoustic_data'].values)
        y = seg['time_to_failure'].values[-1]

        y_tr.loc[segment, 'time_to_failure'] = y
        result_one = transform(x)
        for key, val in result_one.items():
            X_tr.loc[segment, key] = val
    
    return X_tr, y_tr

def transform_earthquake_id(train, rows=150_000):
    segments = int(np.floor(train.shape[0] / rows))
    earthquake_id = pd.Series()

    current_quake_id = 0
    last_time_to_failure = train.iloc[0]['time_to_failure'] 
    
    for segment in range(segments):
        seg = train.iloc[segment*rows : segment*rows+rows]
        times_to_failure = seg['time_to_failure'].values

        # Ignore segments with an earthquake in it.
        if np.abs(times_to_failure[0]-times_to_failure[-1])>1:
            earthquake_id.loc[segment] = current_quake_id
            continue

        if np.abs(times_to_failure[-1]-last_time_to_failure)>1:
            current_quake_id += 1

        earthquake_id.loc[segment] = current_quake_id
        last_time_to_failure = times_to_failure[-1]
    return earthquake_id


def transform(df):
    """ augment X to more features"""
    transform_pack = [
        transform_pack1,
        transform_pack2,
    ]
    dump = []
    for func in transform_pack:
        x = func(df)
        dump.append(x)
    
    return ChainMap(*dump)

def transform_pack2(df):
    """ augment X to more features until 04/15 MFCC related"""
    output = {}
    data = df.values.astype(np.float32)
    mfcc = librosa.feature.mfcc(data)

    output = {}
    for i, each_mfcc in enumerate(mfcc):
        output[f'mfcc_mean_{i}'] = np.mean(each_mfcc)
        output[f'mfcc_std_{i}'] = np.std(each_mfcc)
    
    # for i, each_mfcc in enumerate(librosa.feature.delta(mfcc)):
    #     output[f'delta_mfcc_mean_{i}'] = np.mean(each_mfcc)
    #     output[f'delta_mfcc_std_{i}'] = np.std(each_mfcc)

    # for i, each_mfcc in enumerate(librosa.feature.delta(mfcc, order=2)):
    #     output[f'accelerate_mfcc_mean_{i}'] = np.mean(each_mfcc)
    #     output[f'accelerate_mfcc_std_{i}'] = np.std(each_mfcc)
    

    melspec = librosa.feature.melspectrogram(data)
    logmel = librosa.core.power_to_db(melspec)
    # delta = librosa.feature.delta(logmel)
    # accelerate = librosa.feature.delta(logmel, order=2)

    for i, each_logmel in enumerate(logmel):
        output[f'logmel_mean_{i}'] = np.mean(each_logmel)
        output[f'logmel_std_{i}'] = np.std(each_logmel)
    
    # for i, each_logmel in enumerate(delta):
    #     output[f'delta_logmel_mean_{i}'] = np.mean(each_logmel)
    #     output[f'delta_logmel_std_{i}'] = np.std(each_logmel)

    # for i, each_logmel in enumerate(accelerate):
    #     output[f'accelerate_logmel_mean_{i}'] = np.mean(each_logmel)
    #     output[f'accelerate_logmel_std_{i}'] = np.std(each_logmel)

    return output


def transform_pack1(df):
    """ augment X to more features """
    output = {}
    output['mean'] = df.mean()
    output['std'] = df.std()
    output['max'] = df.max()
    output['min'] = df.min()
    output['mad'] = df.mad()
    output['kurt'] = df.kurtosis()
    output['skew'] = df.skew()
    output['med'] = df.median()

    output['max_to_abs_min'] = df.max() / np.abs(df.min())
    output['max_to_min_abs_diff'] = df.max() - np.abs(df.min())
    
    # abs
    tmp = np.abs(df)
    output['abs_min'] = tmp.min()
    output['abs_std'] = tmp.std()
    output['abs_med'] = tmp.median()

    output['Hilbert_mean'] = np.abs(hilbert(df.values)).mean()
    output['Hann_window_mean'] = (convolve(df.values, hann(150), mode='same') / sum(hann(150))).mean()

    x = df.values
    output['F_test'], output['p_test'] = stats.f_oneway(x[:30000],x[30000:60000],x[60000:90000],x[90000:120000],x[120000:])
    output['av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    # split into segments
    for first in (50_000, 100_000):
        tmp = df[first:]
        output[f'std_first_{first}'] = tmp.std()
        output[f'mean_first_{first}'] = tmp.mean()
        output[f'mad_first_{first}'] = tmp.mad()
        output[f'kurtosis_first_{first}'] = tmp.kurtosis()
        output[f'skew_first_{first}'] = tmp.skew()
        output[f'med_first_{first}'] = tmp.median()
        output[f'std_change_rate_first_{first}'] = tmp.diff().std()
        output[f'trend_first_{first}'] = feature_trend(tmp)
        output[f'abs_trend_first_{first}'] = feature_trend(tmp, abs_value=True)

    
    for last in (50_000, 100_000):
        tmp = df[-last:]
        output[f'std_last_{last}'] = tmp.std()
        output[f'mean_last_{last}'] = tmp.mean()
        output[f'mad_last_{last}'] = tmp.mad()
        output[f'kurtosis_last_{last}'] = tmp.kurtosis()
        output[f'skew_last_{last}'] = tmp.skew()
        output[f'med_last_{last}'] = tmp.median()
        output[f'std_change_rate_last_{last}'] = tmp.diff().std()
        output[f'trend_last_{last}'] = feature_trend(tmp)
        output[f'abs_trend_last_{last}'] = feature_trend(tmp, abs_value=True)

    tmp = np.abs(df)
    output['count_big'] = np.sum(tmp > 100)
    output['count_med'] = np.sum((tmp <= 100) & (tmp > 10))

    x = np.cumsum(df ** 2)
    # Convert to float
    x = np.require(x, dtype=np.float)
    n_sta_group = (500, 5000, 3333, 10000, 50, 333, 4000)
    n_lta_group = (10000, 100000, 6666, 25000, 1000, 666, 10000)
    for i, (n_sta, n_lta) in enumerate(zip(n_sta_group, n_lta_group)):
        tmp = feature_sta_lta_ratio(x, n_sta, n_lta)
        output[f'classic_sta_lta{i}_mean'] = np.nanmean(tmp)


    for window in (500, 5000, 10000):
        output[f'exp_moving_average_{window}_mean'] = pd.Series.ewm(df, span=window).mean().mean(skipna=True)

    # quantile related
    for q in (0.001, 0.1, 0.3, 0.7, 0.9, 0.999):
        output[f'q{q}'] = np.quantile(df, q)

    output['ave_trim_tail_0.1'] = stats.trim_mean(df, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = df.rolling(windows).std().dropna().values
        x_roll_mean = df.rolling(windows).mean().dropna().values
        
        output[f'ave_roll_std_{windows}'] = x_roll_std.mean()
        output[f'std_roll_std_{windows}'] = x_roll_std.std()
        output[f'max_roll_std_{windows}'] = x_roll_std.max()
        output[f'min_roll_std_{windows}'] = x_roll_std.min()
        output[f'q01_roll_std_{windows}'] = np.quantile(x_roll_std, 0.01)
        output[f'q05_roll_std_{windows}'] = np.quantile(x_roll_std, 0.05)
        output[f'q95_roll_std_{windows}'] = np.quantile(x_roll_std, 0.95)
        output[f'q99_roll_std_{windows}'] = np.quantile(x_roll_std, 0.99)
        output[f'trend_roll_mean_{windows}'] = feature_trend(x_roll_std)
        output[f'av_change_roll_std_{windows}'] = np.mean(np.diff(x_roll_std))
        
        output[f'std_roll_mean_{windows}'] = x_roll_mean.std()
        output[f'max_roll_mean_{windows}'] = x_roll_mean.max()
        output[f'min_roll_mean_{windows}'] = x_roll_mean.min()
        output[f'q01_roll_mean_{windows}'] = np.quantile(x_roll_mean, 0.01)
        output[f'q05_roll_mean_{windows}'] = np.quantile(x_roll_mean, 0.05)
        output[f'q95_roll_mean_{windows}'] = np.quantile(x_roll_mean, 0.95)
        output[f'q99_roll_mean_{windows}'] = np.quantile(x_roll_mean, 0.99)
        output[f'av_change_roll_mean_{windows}'] = np.mean(np.diff(x_roll_mean))

        # abs related
        tmp_mean = np.abs(x_roll_mean)
        output[f'std_abs_roll_mean_{windows}'] = tmp_mean.std()
        output[f'skew_abs_roll_mean_{windows}'] = stats.skew(tmp_mean)
        output[f'kurt_abs_roll_mean_{windows}'] = stats.kurtosis(tmp_mean)
        output[f'trend_abs_roll_mean_{windows}'] = feature_trend(tmp_mean)
        output[f'av_change_abs_roll_mean_{windows}'] = np.mean(np.diff(tmp_mean))

        tmp_std = np.abs(x_roll_std)
        output[f'std_abs_roll_std_{windows}'] = tmp_std.std()
        output[f'skew_abs_roll_std_{windows}'] = stats.skew(tmp_std)
        output[f'kurt_abs_roll_std_{windows}'] = stats.kurtosis(tmp_std)
        output[f'trend_abs_roll_std_{windows}'] = feature_trend(tmp_std)
        output[f'av_change_abs_roll_std_{windows}'] = np.mean(np.diff(tmp_std))
    

    return output


def resample(xs):
    """ Resample out the noise, seems not good..."""
    filt = firls(2001, bands=[0,240e3,245e3,250e3,255e3,2e6], desired=[0,0,1,1,0,0], fs=4e6)
    xs = convolve(xs.astype(float), filt, mode='valid')
    t = 2*np.pi*250e3/4e6*np.arange(len(xs))
    xs = xs*(np.cos(t) + 1j*np.sin(t))
    # xs = decimate(xs, 150, ftype='fir')
    return xs

def feature_trend(y, abs_value=False):
    """ linear regression """
    if not isinstance(y, np.ndarray):
        y = y.values
    if abs_value:
        y = np.abs(y)
    x = np.array(range(len(y)))
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def feature_sta_lta_ratio(x, length_sta, length_lta):
    """ short-term change over long-term change ratio (not rolling mean)"""

    sta = x
    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

def concat_scaling(X_tr, X_test):
    raw = pd.concat([X_tr, X_test], sort=False)
    df = preprocess_features(raw)
    train_cutoff = len(X_tr)
    test_cutoff = len(X_test)
    return df[:train_cutoff], df[:test_cutoff]


def preprocess_features(X):
    # scaling 
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return X

def missing_fix_tr(X_tr):
    means_dict = {}
    for col in X_tr.columns:
        if X_tr[col].isnull().any():
            print(col)
            mean_value = X_tr.loc[X_tr[col] != -np.inf, col].mean()
            X_tr.loc[X_tr[col] == -np.inf, col] = mean_value
            if np.abs(mean_value) > 1e10:
                mean_value = 0
            X_tr[col] = X_tr[col].fillna(mean_value)
            means_dict[col] = mean_value
    return X_tr, means_dict

def missing_fix_test(X_test, means_dict):
    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
            X_test[col] = X_test[col].fillna(means_dict[col])
    return X_test