"""Table 9: August 2018 flood withheld from training.
Five runs per learned model, level bands (all, > 8 m, 8-11 m, > 11 m) with sample sizes,
ensemble of the run-averaged LSTM and XGBoost, and paired t-tests against the proposed LSTM
on the run-level MAE above 8 m. Writes results/flood_table.csv and results/flood_table.json."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats

E = '.'
y = np.load(f'{E}/ev_y.npy')                       # (744, 6) observed levels, from ev_base.py
lv = y[:, 0]                                        # level at the first lead hour
BANDS = {'all': np.ones(len(y), bool), 'gt8': lv > 8, '8to11': (lv > 8) & (lv <= 11), 'gt11': lv > 11}

LEARNED = [('Proposed LSTM', 'ev2018_r'), ('(a) Without mask', 'ev2018zf_r'),
           ('(b) Raw instead of routed rainfall', 'ev2018raw_r'), ('(c) Without travel-time shift', 'ev2018noshift_r'),
           ('(d) Without rainfall and discharge inputs', 'ev2018norain_r')]


def score(p):
    out = {}
    for b, m in BANDS.items():
        e = y[m] - p[m]
        out[b] = (float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())))
    return out


def load_runs(key):
    if key == 'xgb':
        fs = sorted(glob.glob(f'{E}/xgb_event_s*.npy'))
    else:
        fs = sorted(glob.glob(f'{E}/runs/{key}_s*/pred_event.npy'))
    return [np.load(f) for f in fs]


rows, J, mae8 = [], {'n': {b: int(m.sum()) for b, m in BANDS.items()}}, {}


def add(name, preds):
    per = [score(p) for p in preds]
    row = {'Model': name, 'runs': len(preds)}
    J[name] = {}
    for b in BANDS:
        mae = np.array([r[b][0] for r in per]); rmse = np.array([r[b][1] for r in per])
        sd = lambda a: float(a.std(ddof=1)) if len(a) > 1 else 0.0
        J[name][b] = {'mae': float(mae.mean()), 'mae_sd': sd(mae), 'rmse': float(rmse.mean()), 'rmse_sd': sd(rmse),
                      'mae_runs': mae.tolist()}
        f = lambda m, s: f'{m:.3f} ± {s:.3f}' if len(preds) > 1 else f'{m:.3f}'
        row[f'MAE {b}'] = f(mae.mean(), sd(mae)); row[f'RMSE {b}'] = f(rmse.mean(), sd(rmse))
    mae8[name] = np.array([r['gt8'][0] for r in per])
    rows.append(row)


add('Persistence', [np.load(f'{E}/ev_persist.npy')])
add('ARIMA(26,1,0), refitted', [np.load(f'{E}/ev_arima.npy')])
xg = load_runs('xgb')
if xg: add('XGBoost (level change)', xg)
lstm = {}
for name, key in LEARNED:
    r = load_runs(key)
    if r:
        lstm[name] = r; add(name, r)
if xg and 'Proposed LSTM' in lstm:
    add('Average of LSTM and XGBoost (run-averaged)', [0.5 * (np.mean(lstm['Proposed LSTM'], 0) + np.mean(xg, 0))])

ref = mae8.get('Proposed LSTM')
for row in rows:
    a = mae8[row['Model']]
    if ref is not None and row['Model'] != 'Proposed LSTM' and len(a) > 1 and len(a) == len(ref):
        p = float(stats.ttest_rel(ref, a).pvalue)
        row['p (MAE > 8 m)'] = f'{p:.3f}'; J[row['Model']]['p_ttest_mae_gt8'] = p
    else:
        row['p (MAE > 8 m)'] = '-'

df = pd.DataFrame(rows)
os.makedirs(f'{E}/results', exist_ok=True)
df.to_csv(f'{E}/results/flood_table.csv', index=False)
json.dump(J, open(f'{E}/results/flood_table.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('Sample sizes:', J['n'])
print(df.to_string(index=False))
