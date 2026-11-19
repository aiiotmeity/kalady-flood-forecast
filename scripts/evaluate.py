import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipe
from scipy import stats

E = '.'
z = np.load(f'{E}/data/win_base.npz'); Y = z['Y'][:, :, 0]; Xp0 = z['Xp'][:, -1, 0]
s = pipe.split(len(Y)); yv = Y[s:]; last = Xp0[s:]
interp = np.load(f'{E}/data/interp_flag.npy')[s:]


def runs(cfg):
    out = []
    for d in sorted(glob.glob(f'{E}/runs/{cfg}_s*')):
        if os.path.exists(d + '/pred.npy'):
            out.append(np.load(d + '/pred.npy'))
    return out


def single(cfg):
    if cfg == 'persistence': return [np.repeat(last[:, None], 6, 1)]
    if cfg == 'arima': return [np.load(f'{E}/runs_arima.npy')]
    return runs(cfg)


def metr(p, sel=None):
    y, q = (yv, p) if sel is None else (yv[sel], p[sel])
    e = y - q
    mae = np.abs(e).mean(0); rmse = np.sqrt((e ** 2).mean(0))
    nse = 1 - (e ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    p95 = np.percentile(np.abs(e), 95, axis=0); p99 = np.percentile(np.abs(e), 99, axis=0)
    return dict(mae=mae, rmse=rmse, nse=nse, p95=p95, p99=p99)


def summary(cfg, sel=None):
    P = single(cfg)
    if not P: return None
    M = [metr(p, sel) for p in P]
    agg = {}
    for k in M[0]:
        a = np.array([m[k] for m in M])          # runs x 6
        agg[k] = dict(h_mean=a.mean(0), h_sd=a.std(0, ddof=1) if len(a) > 1 else np.zeros(6),
                      all_mean=a.mean(1).mean(), all_sd=a.mean(1).std(ddof=1) if len(a) > 1 else 0.0,
                      run_means=a.mean(1))
    agg['n_runs'] = len(P); agg['avgpred'] = np.mean(P, 0)
    return agg


def compare(a, b, sel=None):
    """Wilcoxon on paired per-sample absolute errors (run-averaged preds, mean over horizons); paired t on run MAEs."""
    A, B = summary(a, sel), summary(b, sel)
    y = yv if sel is None else yv[sel]
    pa = A['avgpred'] if sel is None else A['avgpred'][sel]
    pb = B['avgpred'] if sel is None else B['avgpred'][sel]
    ea = np.abs(y - pa).mean(1); eb = np.abs(y - pb).mean(1)
    w = stats.wilcoxon(ea, eb).pvalue
    ra, rb = A['mae']['run_means'], B['mae']['run_means']
    t = stats.ttest_rel(ra, rb).pvalue if len(ra) == len(rb) and len(ra) > 1 else np.nan
    return w, t


if __name__ == '__main__':
    cfgs = sys.argv[1:] or ['persistence', 'arima', 'xgb', 'zerofill_r', 'proposed', 'proposed_r', 'raw_r', 'noshift_r']
    for c in cfgs:
        S = summary(c)
        if S is None: print(c, 'no runs'); continue
        print(f"{c:12s} n={S['n_runs']} MAE {S['mae']['all_mean']:.4f}±{S['mae']['all_sd']:.4f} RMSE {S['rmse']['all_mean']:.4f}±{S['rmse']['all_sd']:.4f} NSE {S['nse']['all_mean']:.4f}")
        print('   H MAE ', np.round(S['mae']['h_mean'], 4), ' RMSE ', np.round(S['rmse']['h_mean'], 4))
