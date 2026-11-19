"""Faithful re-implementation of the authors' preprocessing (attachment code), with switches for ablations."""
import numpy as np, pandas as pd, os, pickle

PAST_HOURS, HORIZON = 24, 6
MONSOON_MONTHS = [6, 7, 8, 9, 10, 11, 12]
CUTOFF = pd.Timestamp('2026-11-02 08:00')   # frozen snapshot for the revision


def ceil_to_next_hour(ts):
    ts = pd.to_datetime(ts)
    if ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        return ts
    return (ts + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def align(df, tcol, vcol, idx):
    df = df.dropna(subset=[tcol, vcol]).copy()
    df[tcol] = pd.to_datetime(df[tcol], errors='coerce')
    df = df.dropna(subset=[tcol])
    df['h'] = df[tcol].apply(ceil_to_next_hour)
    return df.groupby('h')[vcol].sum().reindex(idx, fill_value=0.0)


def kernel(L):
    x = np.arange(1, L + 1); k = 1.0 / (np.abs(x - (L + 1) / 2.0) + 1.0)
    return k / k.sum()


def route(r, k):
    return np.convolve(r, k, mode='full')[:len(r)]


def load_raw(path, cutoff=CUTOFF):
    cache = path + f'.{cutoff:%Y%m%d%H}.pkl'
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    df = pd.read_excel(path)
    df['Date Neel_level'] = pd.to_datetime(df['Date Neel_level'], errors='coerce')
    lv = df[['Date Neel_level', 'Neel_level']].dropna()
    lv = lv[lv['Date Neel_level'] <= cutoff]
    start = lv['Date Neel_level'].min().floor('h'); end = lv['Date Neel_level'].max().ceil('h')
    idx = pd.date_range(start, end, freq='h')
    hw = pd.DataFrame(index=idx)
    u = lv.groupby('Date Neel_level')['Neel_level'].mean()
    raw_level = pd.to_numeric(u.reindex(idx), errors='coerce')
    hw['level_raw'] = raw_level.values
    hw['level'] = raw_level.interpolate('time').bfill().ffill().values
    hw['rain'] = align(df[['Date Neel_rain', 'Neel_rain']], 'Date Neel_rain', 'Neel_rain', idx).values
    hw['discharge'] = align(df[['Date Discharge', 'Discharge']], 'Date Discharge', 'Discharge', idx).values
    hw['rain_illithode'] = align(df[['Date Illithode_rain', 'Illithode_rain']], 'Date Illithode_rain', 'Illithode_rain', idx).values
    hw['rain_kuttampuzha'] = align(df[['Date Kuttampuzha_rain', 'Kuttampuzha_rain']], 'Date Kuttampuzha_rain', 'Kuttampuzha_rain', idx).values
    hw = hw[hw.index.month.isin(MONSOON_MONTHS)].copy()
    pickle.dump(hw, open(cache, 'wb'))
    return hw


def hours_since(arr):
    last, res = -9999, []
    for i, v in enumerate(arr):
        if v > 0:
            last = i; res.append(0)
        else:
            res.append(i - last if last > -9999 else 999)
    return np.array(res)


PAST_FEATURES = ["level", "level_diff_1h", "level_ma_3h", "level_ma_6h", "routed_rain",
                 "routed_lag_1", "routed_lag_2", "routed_lag_3", "routed_6h_sum", "rain_6h_sum", "rain_12h_sum",
                 "discharge_arrival", "dis_lag_1", "dis_lag_3", "dis_lag_6", "hours_since_rain", "hours_since_discharge",
                 "hour_sin", "hour_cos", "month", "monsoon_flag",
                 "rain_illithode", "routed_rain_illithode", "rain_illithode_arrival", "rain_illithode_6h_sum",
                 "rain_illithode_12h_sum", "rain_illithode_lag_1", "rain_illithode_lag_2", "rain_illithode_lag_3",
                 "rain_illithode_lag_6",
                 "rain_kuttampuzha", "routed_rain_kuttampuzha", "rain_kuttampuzha_arrival", "rain_kuttampuzha_6h_sum",
                 "rain_kuttampuzha_12h_sum", "rain_kuttampuzha_lag_1", "rain_kuttampuzha_lag_2",
                 "rain_kuttampuzha_lag_3", "rain_kuttampuzha_lag_6"]
CATEGORY = {}
for f in PAST_FEATURES:
    if f.startswith('level'): CATEGORY[f] = 'level'
    elif 'illithode' in f or 'kuttampuzha' in f: CATEGORY[f] = 'upstream'
    elif f.startswith(('routed', 'rain')): CATEGORY[f] = 'local'
    elif f.startswith(('dis', 'discharge')): CATEGORY[f] = 'discharge'
    else: CATEGORY[f] = 'temporal'


def features(hw, L_loc=4, L_up=6, tau_up=2, tau_q=3, routed=True, shift=True):
    hw = hw.copy()
    tq = tau_q if shift else 0; tu = tau_up if shift else 0
    k = kernel(L_loc) if routed else np.array([1.0])
    ku = kernel(L_up) if routed else np.array([1.0])
    hw['routed_rain'] = route(hw['rain'].values, k)
    hw['discharge_arrival'] = hw['discharge'].shift(tq).fillna(0.0)
    for n in ['rain_illithode', 'rain_kuttampuzha']:
        hw[f'routed_{n}'] = route(hw[n].values, ku)
        hw[f'{n}_arrival'] = hw[f'routed_{n}'].shift(tu).fillna(0.0)
        hw[f'{n}_6h_sum'] = hw[n].rolling(6, min_periods=1).sum()
        hw[f'{n}_12h_sum'] = hw[n].rolling(12, min_periods=1).sum()
        for lag in [1, 2, 3, 6]:
            hw[f'{n}_lag_{lag}'] = hw[n].shift(lag).fillna(0.0)
    hw['level_diff_1h'] = hw['level'].diff(1).fillna(0.0)
    hw['level_ma_3h'] = hw['level'].rolling(3, min_periods=1).mean()
    hw['level_ma_6h'] = hw['level'].rolling(6, min_periods=1).mean()
    hw['rain_6h_sum'] = hw['rain'].rolling(6, min_periods=1).sum()
    hw['rain_12h_sum'] = hw['rain'].rolling(12, min_periods=1).sum()
    hw['routed_6h_sum'] = hw['routed_rain'].rolling(6, min_periods=1).sum()
    for lag in [1, 2, 3, 6, 12]:
        hw[f'routed_lag_{lag}'] = hw['routed_rain'].shift(lag).fillna(0.0)
        hw[f'dis_lag_{lag}'] = hw['discharge'].shift(lag).fillna(0.0)
    hw['hours_since_rain'] = hours_since(hw['rain'].values)
    hw['hours_since_discharge'] = hours_since(hw['discharge'].values)
    hw['hour_sin'] = np.sin(2 * np.pi * hw.index.hour / 24)
    hw['hour_cos'] = np.cos(2 * np.pi * hw.index.hour / 24)
    hw['month'] = hw.index.month
    hw['monsoon_flag'] = hw['month'].isin(MONSOON_MONTHS).astype(int)
    return hw


def windows(hw, mask_mode='masked', n_fp=2, noise_sd=0.12, noise_seed=12345):
    """mask_mode: 'masked' (proposed), 'zerofill' (no mask channel; absent hours = 0)."""
    P = hw[PAST_FEATURES].values.astype('float32')
    fr = hw['routed_rain'].values
    lv = hw['level'].values
    n = len(hw) - PAST_HOURS - HORIZON
    Xp = np.lib.stride_tricks.sliding_window_view(P, (PAST_HOURS, P.shape[1]))[:n, 0]
    Xf = np.zeros((n, HORIZON, 2), 'float32'); Y = np.zeros((n, HORIZON, 1), 'float32')
    nrng = np.random.default_rng(noise_seed)   # authors' code used unseeded np.random here
    for j, i in enumerate(range(PAST_HOURS, len(hw) - HORIZON)):
        f = fr[i:i + HORIZON].copy()
        rng = np.random.default_rng(seed=(i % 100000))
        ch = rng.choice(HORIZON, size=n_fp, replace=False)
        m = np.zeros(HORIZON); m[ch] = 1.0
        m[np.where(f > (f.mean() + f.std()))[0]] = 1.0
        noise = nrng.normal(scale=noise_sd, size=HORIZON) * f
        g = f.copy(); g[m == 1] = np.maximum(0.0, g[m == 1] + noise[m == 1])
        Xf[j, :, 0] = g * m; Xf[j, :, 1] = m
        Y[j, :, 0] = lv[i:i + HORIZON]
    times = hw.index[PAST_HOURS:len(hw) - HORIZON]
    if mask_mode == 'zerofill':
        Xf = Xf[:, :, :1]
    return np.ascontiguousarray(Xp), Xf, Y, times


def split(n):
    return int(0.8 * n)
