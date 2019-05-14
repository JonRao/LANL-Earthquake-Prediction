# feature engineering
import pandas as pd
import numpy as np
import tqdm
import librosa  # MFCC feature
import warnings
import multiprocessing
import scipy.signal as sg
from collections import ChainMap, defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from tsfresh.feature_extraction import feature_calculators


import data_loader

warnings.filterwarnings("ignore")


NY_FREQ_IDX = 75_000
CUTOFF = 18_000


def transform_test(raw):
    """ Transform test data in parallel"""
    num_process = multiprocessing.cpu_count()
    ctx = multiprocessing.get_context('spawn')

    dump = {}
    with ctx.Pool(num_process) as p:
        for name, result_one in p.imap_unordered(transform_helper, tqdm.tqdm(raw)):
            dump[name] = result_one 

    X_test = pd.DataFrame(index=dump, dtype=np.float64)
    for name, result_one in tqdm.tqdm(dump.items()):
        for key, val in result_one.items():
            X_test.loc[name, key] = val

    return X_test 


def transform_helper(data):
    """ Helper for test data case"""
    name, df = data
    tmp = transform(df)
    return name, tmp


def transform_train(train, rows=150_000):
    # num_process = 6
    num_process = multiprocessing.cpu_count()
    ctx = multiprocessing.get_context('spawn')

    segments = int(np.floor(train.shape[0] / rows)) # missing last part
    def train_getter(segments):
        for segment in range(segments):
            seg = train.iloc[segment*rows : segment*rows + rows]
            yield seg


    dump = []
    with ctx.Pool(num_process) as p:
        for result in p.imap_unordered(transform_train_helper, tqdm.tqdm(enumerate(train_getter(segments)), total=segments)):
            dump.append(result)
    
    count = len(dump)
    X_tr = pd.DataFrame(index=range(count), dtype=np.float64)
    y_tr = pd.DataFrame(index=range(count), dtype=np.float64, columns=['time_to_failure'])
    for segment, result_one, y in tqdm.tqdm(dump):
        y_tr.loc[segment, 'time_to_failure'] = y
        for key, val in result_one.items():
            X_tr.loc[segment, key] = val

    return X_tr, y_tr


def transform_train_helper(args):
    """ Transform in parallel helper"""
    segmentId, seg = args
    x = pd.Series(seg['acoustic_data'].values)
    y = seg['time_to_failure'].values[-1]

    result_one = transform(x)
    return segmentId, result_one, y


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
        # transform_pack1,
        # transform_pack2,
        # transform_pack3,
        # transform_pack4,
        # transform_pack5,
        # transform_pack6,
        # transform_pack7,
        transform_pack8,
    ]
    dump = []
    for func in transform_pack:
        x = func(df)
        dump.append(x)
    
    return ChainMap(*dump)

def transform_random(df):
    """ Only a handful features for multiple training data"""
    output = {}
    output1 = transform_pack2(df)

    percentiles = [25,]

    # In absolute space
    df = np.abs(df)
    length = len(df)

    x = np.sort(df.values)[::-1]
    for p in percentiles:
        bound = int((p / 100) * length)
        other_bound = length - bound
        tmp1 = np.mean(np.power(2, x[:bound]))
        tmp2 = np.mean(np.power(2, x[other_bound:]))
        output[f'ampl_p{p}_ratio'] = tmp1 / tmp2

def transform_pack8(df):
    """Useful to tell mini-quakes"""
    output = {}

    x = df.astype('float').values
    contrast = librosa.feature.spectral_contrast(x, hop_length=1501, fmin=200.0, n_bands=6, quantile=0.005, linear=True)
    output['contrast'] = contrast[0].max()
    return output

def transform_pack7(df):
    """Features from Vettejeep"""
    MAX_FREQ_IDX = 20_000
    FREQ_STEP = 2500
    output = {}
    xc = df.values.astype(np.float64)
    xcdm = xc - np.mean(xc)
    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = sg.lfilter(b, a, xcdm)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ_IDX]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)
    phzFFT = np.arctan(imagFFT / realFFT)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in freq_bands:
        output[f'FFT_Mag_01q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        output[f'FFT_Mag_10q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        output[f'FFT_Mag_90q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        output[f'FFT_Mag_99q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
        output[f'FFT_Mag_mean_{freq}'] = np.mean(magFFT[freq: freq + FREQ_STEP])
        output[f'FFT_Mag_std_{freq}'] = np.std(magFFT[freq: freq + FREQ_STEP])
        output[f'FFT_Mag_max_{freq}'] = np.max(magFFT[freq: freq + FREQ_STEP])

        output[f'FFT_Phz_mean_{freq}'] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        output[f'FFT_Phz_std_{freq}'] = np.std(phzFFT[freq: freq + FREQ_STEP])

    output['FFT_Rmean'] = realFFT.mean()
    output['FFT_Rstd'] = realFFT.std()
    output['FFT_Rmax'] = realFFT.max()
    output['FFT_Rmin'] = realFFT.min()
    output['FFT_Imean'] = imagFFT.mean()
    output['FFT_Istd'] = imagFFT.std()
    output['FFT_Imax'] = imagFFT.max()
    output['FFT_Imin'] = imagFFT.min()

    output['FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    output['FFT_Rstd__first_6000'] = realFFT[:6000].std()
    output['FFT_Rmax_first_6000'] = realFFT[:6000].max()
    output['FFT_Rmin_first_6000'] = realFFT[:6000].min()
    output['FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    output['FFT_Rstd_first_18000'] = realFFT[:18000].std()
    output['FFT_Rmax_first_18000'] = realFFT[:18000].max()
    output['FFT_Rmin_first_18000'] = realFFT[:18000].min()    
    return output

def transform_pack6(df):
    """ 0427 paper amplitude features"""
    output = {}

    percentiles = [1, 5, 10, 20, 25, 30]

    # In absolute space
    df = np.abs(df)
    length = len(df)

    x = np.sort(df.values)[::-1]
    for p in percentiles:
        bound = int((p / 100) * length)
        other_bound = length - bound
        output[f'ampl_p{p}'] = np.mean(np.power(2, x[:bound]))
        output[f'ampl_p{100-p}'] = np.mean(np.power(2, x[other_bound:]))
        output[f'ampl_p{p}_ratio'] = output[f'ampl_p{p}']  / output[f'ampl_p{100-p}']
        output[f'ampl_p{p}_diff'] = output[f'ampl_p{p}'] - output[f'ampl_p{100-p}']

    tmp = x
    output['num_peaks_1'] = feature_calculators.number_peaks(tmp, 1)
    output['num_peaks_5'] = feature_calculators.number_peaks(tmp, 5)
    output['binned_entropy_5'] = feature_calculators.binned_entropy(x, 5)

    
    return output


def transform_pack5(df):
    """ 0425 more features"""
    output = {}

    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    for w in [50, 500, 5000]:
        x_roll_std = df.rolling(w).std().dropna().values
        x_roll_mean = df.rolling(w).mean().dropna().values

        for p in percentiles:
            output[f'percentile_roll_{w}_mean_{p}'] = np.percentile(x_roll_mean, p)
            output[f'percentile_roll_{w}_std_{p}'] = np.percentile(x_roll_std, p)

        output[f'mean_abs_change_mean_{w}'] = feature_calculators.mean_abs_change(x_roll_mean)
        output[f'mean_change_roll_mean_{w}'] = feature_calculators.mean_change(x_roll_mean)
        output[f'mean_abs_change_std_{w}'] = feature_calculators.mean_abs_change(x_roll_std)
        output[f'mean_change_roll_std_{w}'] = feature_calculators.mean_change(x_roll_std)
    return output

def transform_pack4(df):
    x = df.values.astype(np.float32)
    output = {}
    output['spectral_rolloff'] = librosa.feature.spectral_rolloff(x)[0][0]
    output['spectral_centroid'] = librosa.feature.spectral_centroid(x)[0][0]
    output['spectral_contrast'] = librosa.feature.spectral_contrast(x)[0][0]
    output['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(x)[0][0]
    return output

def transform_pack3(df):
    """ augment X from tsfresh features"""
    x = df.values
    output = {}

    output['kstat_1'] = stats.kstat(x, 1)
    output['kstat_2'] = stats.kstat(x, 2)
    output['kstat_3'] = stats.kstat(x, 3)
    output['kstat_4'] = stats.kstat(x, 4)
    output['abs_energy'] = feature_calculators.abs_energy(x)
    output['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
    output['count_above_mean'] = feature_calculators.count_above_mean(x)
    output['count_below_mean'] = feature_calculators.count_below_mean(x)
    output['range_minf_m4000'] = feature_calculators.range_count(x, -np.inf, -4000)
    output['range_m4000_m3000'] = feature_calculators.range_count(x, -4000, -3000)
    output['range_m3000_m2000'] = feature_calculators.range_count(x, -3000, -2000)
    output['range_m2000_m1000'] = feature_calculators.range_count(x, -2000, -1000)
    output['range_m1000_0'] = feature_calculators.range_count(x, -1000, 0)
    output['range_0_p1000'] = feature_calculators.range_count(x, 0, 1000)
    output['range_p1000_p2000'] = feature_calculators.range_count(x, 1000, 2000)
    output['range_p2000_p3000'] = feature_calculators.range_count(x, 2000, 3000)
    output['range_p3000_p4000'] = feature_calculators.range_count(x, 3000, 4000)
    output['range_p4000_pinf'] = feature_calculators.range_count(x, 4000, np.inf)

    output['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
    output['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
    output['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
    output['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
    output['last_loc_max'] = feature_calculators.last_location_of_maximum(x)
    output['time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
    output['time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)
    output['time_rev_asym_stat_1000'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1000)

    output['autocorrelation_10'] = feature_calculators.autocorrelation(x, 10)
    output['autocorrelation_100'] = feature_calculators.autocorrelation(x, 100)
    output['autocorrelation_1000'] = feature_calculators.autocorrelation(x, 1000)
    output['autocorrelation_5000'] = feature_calculators.autocorrelation(x, 5000)

    output['c3_5'] = feature_calculators.c3(x, 5)
    output['c3_10'] = feature_calculators.c3(x, 10)
    output['c3_100'] = feature_calculators.c3(x, 100)

    output['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
    output['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
    output['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
    output['cid_ce_1'] = feature_calculators.cid_ce(x, 1)
    output['binned_entropy_10'] = feature_calculators.binned_entropy(x, 10)
    output['binned_entropy_50'] = feature_calculators.binned_entropy(x, 50)
    output['binned_entropy_80'] = feature_calculators.binned_entropy(x, 80)
    output['binned_entropy_100'] = feature_calculators.binned_entropy(x, 100)

    tmp = np.abs(x)
    output['num_crossing_0'] = feature_calculators.number_crossing_m(tmp, 0)
    output['num_crossing_10'] = feature_calculators.number_crossing_m(tmp, 10)
    output['num_crossing_100'] = feature_calculators.number_crossing_m(tmp, 100)
    output['num_peaks_10'] = feature_calculators.number_peaks(tmp, 10)
    output['num_peaks_50'] = feature_calculators.number_peaks(tmp, 50)
    output['num_peaks_100'] = feature_calculators.number_peaks(tmp, 100)
    output['num_peaks_500'] = feature_calculators.number_peaks(tmp, 500)

    output['spkt_welch_density_1'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 1}]))[0][1]
    output['spkt_welch_density_10'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 10}]))[0][1]
    output['spkt_welch_density_50'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 50}]))[0][1]
    output['spkt_welch_density_100'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 100}]))[0][1]

    output['time_rev_asym_stat_1'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1)
    output['time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
    output['time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)     

    return output


def transform_pack2(df):
    """ augment X to more features until 04/15 MFCC related"""
    output = {}
    data = df.values.astype(np.float32)
    mfcc = librosa.feature.mfcc(data)

    output = {}
    for i, each_mfcc in enumerate(mfcc):
        output[f'mfcc_mean_{i}'] = np.mean(each_mfcc)
        output[f'mfcc_std_{i}'] = np.std(each_mfcc)

    melspec = librosa.feature.melspectrogram(data)
    logmel = librosa.core.power_to_db(melspec)

    for i, each_logmel in enumerate(logmel):
        output[f'logmel_mean_{i}'] = np.mean(each_logmel)
        output[f'logmel_std_{i}'] = np.std(each_logmel)
    
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

    output['Hilbert_mean'] = np.abs(sp.hilbert(df.values)).mean()
    output['Hann_window_mean'] = (sp.convolve(df.values, sp.hann(150), mode='same') / sum(sp.hann(150))).mean()

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

    n_sta_group = (500, 5000, 3333, 10000, 50, 333, 4000)
    n_lta_group = (10000, 100000, 6666, 25000, 1000, 666, 10000)
    for i, (n_sta, n_lta) in enumerate(zip(n_sta_group, n_lta_group)):
        tmp = feature_sta_lta_ratio(df.values, n_sta, n_lta)
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
    x = np.cumsum(x ** 2)
    # Convert to float
    x = np.require(x, dtype=np.float)

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

def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)
    return b, a

def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')
    return b, a

def des_bw_filter_bp(low, high):  # band pass filter
    b, a = sg.butter(4, Wn=(low/NY_FREQ_IDX, high/NY_FREQ_IDX), btype='bandpass')
    return b, a

def preprocess_features(X_tr, X_test):
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_tr_scaled, X_test_scaled


def missing_fix_tr(X_tr):
    means_dict = {}
    for col in X_tr.columns:
        if X_tr[col].isnull().any():
            print('train', col)
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
            X_test.loc[X_test[col] == -np.inf, col] = means_dict.get(col, 0)
            X_test[col] = X_test[col].fillna(means_dict.get(col, 0))
            print('test', col)
    return X_test

def on_the_fly_features(X_tr, X_test, n=10):
    """ NN features, generate based on input"""
    neigh = NearestNeighbors(n, n_jobs=-1)
    neigh.fit(X_tr)

    dists, _ = neigh.kneighbors(X_tr, n_neighbors=n)
    mean_dist = dists.mean(axis=1)
    max_dist = dists.max(axis=1)
    min_dist = dists.min(axis=1)

    X_tr['mean_dist'] = mean_dist
    X_tr['max_dist'] = max_dist
    X_tr['min_dist'] = min_dist

    test_dists, _ = neigh.kneighbors(X_test, n_neighbors=n)

    test_mean_dist = test_dists.mean(axis=1)
    test_max_dist = test_dists.max(axis=1)
    test_min_dist = test_dists.min(axis=1)

    X_test['mean_dist'] = test_mean_dist
    X_test['max_dist'] = test_max_dist
    X_test['min_dist'] = test_min_dist

    return X_tr, X_test


if __name__ == '__main__':
    # X_tr, y_tr = data_loader.load_transform_train()
    # X_test = data_loader.load_transform_test()
    # X_tr, means_dict = missing_fix_tr(X_tr)
    # X_test = missing_fix_test(X_test, means_dict)
    # preprocess_features(X_tr, X_test)
    data_loader.load_data(10)