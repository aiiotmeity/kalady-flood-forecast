"""Collect every number needed for the revised manuscript into results.json and results.txt."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipe, evaluate as ev
from scipy import stats

E = ev.E
R = {}
z = np.load(f'{E}/data/win_base.npz'); t = pd.DatetimeIndex(z['t']); s = pipe.split(len(t)); tv = t[s:]
yv, last = ev.yv, ev.last


def pack(S):
    return {k: {'h_mean': S[k]['h_mean'].tolist(), 'h_sd': S[k]['h_sd'].tolist(), 'mean': float(S[k]['all_mean']),
                'sd': float(S[k]['all_sd']), 'runs': S[k]['run_means'].tolist()} for k in ['mae', 'rmse', 'nse', 'p95', 'p99']} | {'n_runs': S['n_runs']}


CFG = ['persistence', 'arima', 'xgb', 'zerofill_r', 'proposed', 'proposed_r', 'raw_r', 'noshift_r',
       'L2_r', 'L4_r', 'L8_r', 'L10_r']
R['overall'] = {}
for c in CFG:
    S = ev.summary(c)
    if S: R['overall'][c] = pack(S)
R['tests'] = {}
for c in CFG:
    if c != 'proposed_r' and ev.summary(c):
        w, tt = ev.compare('proposed_r', c)
        R['tests'][c] = {'wilcoxon_p': float(w), 'ttest_p': None if np.isnan(tt) else float(tt)}

# subgroups
lv0 = yv[:, 0]; rise = yv[:, 5] - last
groups = {'lt2': lv0 <= 2, '2to5': (lv0 > 2) & (lv0 <= 5), 'gt5': lv0 > 5, 'ge8': lv0 >= 8,
          'rising03': rise > 0.3, 'no_interp': ~ev.interp,
          'pre_dis_end': tv < pd.Timestamp('2025-07-26 18:00'), 'post_dis_end': tv >= pd.Timestamp('2025-07-26 18:00')}
R['groups'] = {}
for g, sel in groups.items():
    R['groups'][g] = {'n': int(sel.sum())}
    if sel.sum() == 0: continue
    for c in ['persistence', 'arima', 'xgb', 'zerofill_r', 'proposed', 'proposed_r']:
        S = ev.summary(c, sel)
        if S: R['groups'][g][c] = {'mae_h': S['mae']['h_mean'].tolist(), 'rmse_h': S['rmse']['h_mean'].tolist(),
                                   'mae': float(S['mae']['all_mean']), 'mae_sd': float(S['mae']['all_sd']),
                                   'rmse': float(S['rmse']['all_mean']), 'rmse_sd': float(S['rmse']['all_sd'])}
    for c in ['persistence', 'arima', 'xgb', 'zerofill_r']:
        if ev.summary(c, sel) and sel.sum() > 10:
            R['groups'][g].setdefault('p_vs', {})[c] = float(ev.compare('proposed_r', c, sel)[0])

# RMSE decomposition: share of squared error from top-1% errors and from high levels
P = ev.summary('proposed_r')['avgpred']; e2 = (yv - P) ** 2
R['tail'] = {'share_sq_err_top1pct': [float(np.sort(e2[:, h])[-len(e2) // 100:].sum() / e2[:, h].sum()) for h in range(6)],
             'share_sq_err_rising03': [float(e2[rise > 0.3, h].sum() / e2[:, h].sum()) for h in range(6)],
             'n_rising03': int((rise > 0.3).sum())}

# 2018 flood hold-out
evy = np.load(f'{E}/ev_y.npy') if os.path.exists(f'{E}/ev_y.npy') else None
if evy is not None:
    R['event2018'] = {'n': int(len(evy)), 'max_level': float(evy.max()), 'hours_gt8': int((evy[:, 0] > 8).sum())}
    def m(q, y=evy):
        e = y - q; return {'mae_h': np.abs(e).mean(0).tolist(), 'rmse_h': np.sqrt((e ** 2).mean(0)).tolist(),
                          'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean()))}
    R['event2018']['persistence'] = m(np.load(f'{E}/ev_persist.npy')); R['event2018']['arima'] = m(np.load(f'{E}/ev_arima.npy'))
    runs = [np.load(f) for f in sorted(glob.glob(f'{E}/runs/ev2018_r_s*/pred_event.npy'))]
    if runs:
        ms = [m(r) for r in runs]
        R['event2018']['proposed_r'] = {'mae': float(np.mean([x['mae'] for x in ms])), 'mae_sd': float(np.std([x['mae'] for x in ms], ddof=1)) if len(ms) > 1 else 0,
                                        'rmse': float(np.mean([x['rmse'] for x in ms])), 'rmse_sd': float(np.std([x['rmse'] for x in ms], ddof=1)) if len(ms) > 1 else 0,
                                        'mae_h': np.mean([x['mae_h'] for x in ms], 0).tolist(), 'rmse_h': np.mean([x['rmse_h'] for x in ms], 0).tolist(), 'n_runs': len(ms)}
        hi = evy[:, 0] > 8
        R['event2018']['gt8'] = {'n': int(hi.sum()), 'persistence': m(np.load(f'{E}/ev_persist.npy')[hi], evy[hi]),
                                 'arima': m(np.load(f'{E}/ev_arima.npy')[hi], evy[hi]),
                                 'proposed_r': m(np.mean(runs, 0)[hi], evy[hi])}

# training info
R['train_info'] = {}
for c in ['proposed_r', 'zerofill_r', 'proposed', 'raw_r', 'noshift_r']:
    inf = [json.load(open(f)) for f in sorted(glob.glob(f'{E}/runs/{c}_s*/info.json'))]
    if inf: R['train_info'][c] = {'best_epoch': [i['best_epoch'] for i in inf], 'epochs': [i['epochs'] for i in inf],
                                  'seconds': [i['seconds'] for i in inf], 'params': inf[0]['params']}

# LIME
if os.path.exists(f'{E}/lime_subset.npz'):
    zz = np.load(f'{E}/lime_subset.npz'); SH = zz['shares']; ix = zz['idx']; r = yv[ix, 5] - last[ix]
    mm, sd = SH.mean(0), SH.std(0, ddof=1)
    R['lime'] = {'cats': ['level', 'upstream', 'local', 'discharge', 'temporal'], 'mean': mm.tolist(), 'sd': sd.tolist(),
                 'cv': (sd / np.abs(mm)).tolist(), 'rising_n': int((r > 0.3).sum()), 'rising_mean': SH[r > 0.3].mean(0).tolist(),
                 'rising_sd': SH[r > 0.3].std(0, ddof=1).tolist(), 'alloc': json.load(open(f'{E}/lime_subset_alloc.json'))}

R['data'] = {'n_windows': int(len(t)), 'n_train': int(s), 'n_val': int(len(t) - s), 'train_first': str(t[0]), 'train_last': str(t[s - 1]),
             'val_first': str(t[s]), 'val_last': str(t[-1]), 'val_level_max': float(yv.max()),
             'val_interp_share': float(ev.interp.mean())}
json.dump(R, open(f'{E}/results.json', 'w'), indent=1)

with open(f'{E}/results.txt', 'w') as f:
    for c, v in R['overall'].items():
        f.write(f"{c:12s} n={v['n_runs']} MAE {v['mae']['mean']:.4f}±{v['mae']['sd']:.4f} RMSE {v['rmse']['mean']:.4f}±{v['rmse']['sd']:.4f} NSE {v['nse']['mean']:.4f}\n")
    f.write(json.dumps(R['tests'], indent=1) + '\n')
print(open(f'{E}/results.txt').read())
