"""Horizon-resolved LIME on a stratified validation subset (and the illustrative 7 Sep 2026 09:00 forecast).
Faithful to the authors' settings: LimeTabularExplainer on unscaled flattened training inputs,
mode=regression, discretize_continuous=False, num_features=25, num_samples=500, random_state=0."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd, pipe, tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer

tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('NT', 2)))
E = '.'; run = sys.argv[1]; part = sys.argv[2]   # part: 'subset' or 'illustrative'
model = tf.keras.models.load_model(f'{E}/runs/{run}/model.keras')
sp, sf, sy = pickle.load(open(f'{E}/runs/{run}/scalers.pkl', 'rb'))
z = np.load(f'{E}/data/win_base.npz'); Xp, Xf, Y, t = z['Xp'], z['Xf'], z['Y'], pd.DatetimeIndex(z['t'])
s = pipe.split(len(Xp)); P, H, nf = pipe.PAST_HOURS, pipe.HORIZON, Xp.shape[2]
flat = lambda a, b: np.hstack([a.reshape(len(a), -1), b.reshape(len(b), -1)])
Xtr = flat(Xp[:s], Xf[:s]); Xva = flat(Xp[s:], Xf[s:])
names = [f'{f}@t-{P - k}' for k in range(P) for f in pipe.PAST_FEATURES] + \
        [f'{n}@h+{h + 1}' for h in range(H) for n in ('future_routed_rain', 'future_mask')]
cat_of = [pipe.CATEGORY[n.split('@')[0]] for n in names[:P * nf]] + ['local'] * (2 * H)
CATS = ['level', 'upstream', 'local', 'discharge', 'temporal']
cache = {}

def predict_all(X):
    key = hash(X.tobytes())
    if key not in cache:
        n = len(X); a = X[:, :P * nf].reshape(n, P, nf); b = X[:, P * nf:].reshape(n, H, 2)
        a_s = sp.transform(a.reshape(-1, nf)).reshape(a.shape); b_s = sf.transform(b.reshape(-1, 2)).reshape(b.shape)
        d = sy.inverse_transform(model.predict([a_s, b_s], batch_size=1024, verbose=0).reshape(-1, 1)).reshape(n, H)
        cache.clear(); cache[key] = d + a[:, -1, 0][:, None]     # level-change model: add last observed level
    return cache[key]

exp = LimeTabularExplainer(Xtr, feature_names=names, mode='regression', discretize_continuous=False, sample_around_instance=True, random_state=0)

def explain(x):
    shares = np.zeros((H, len(CATS))); top = []
    for h in range(H):
        exp.random_state = np.random.RandomState(1000)   # identical perturbations for all horizons
        e = exp.explain_instance(x, lambda X: predict_all(X)[:, h], num_features=25, num_samples=500)
        w = dict(e.as_map()[1] if 1 in e.as_map() else list(e.as_map().values())[0])
        tot = sum(abs(v) for v in w.values()) or 1.0
        for i, v in w.items():
            shares[h, CATS.index(cat_of[i])] += abs(v) / tot
        top.append(sorted([(names[i], float(v)) for i, v in w.items()], key=lambda a: -abs(a[1])))
    return shares, top

if part == 'illustrative':
    j = list(t[s:]).index(pd.Timestamp(os.environ.get('ILL_T', '2026-08-01 07:00')))
    sh, top = explain(Xva[j])
    json.dump({'shares': sh.tolist(), 'top': top, 'cats': CATS}, open(f'{E}/lime_illustrative.json', 'w'))
    print(np.round(sh, 3))
else:
    lv = Y[s:, 0, 0]; rng = np.random.default_rng(0)
    hi = np.where(lv > 5)[0]; mid = np.where((lv > 2) & (lv <= 5))[0]; low = np.where(lv <= 2)[0]
    n_mid = (500 - len(hi)) // 3
    idx = np.concatenate([hi, rng.choice(mid, n_mid, replace=False), rng.choice(low, 500 - len(hi) - n_mid, replace=False)])
    out = f'{E}/lime_subset.npz'
    done = dict(np.load(out)) if os.path.exists(out) else {'idx': idx, 'shares': np.full((len(idx), H, len(CATS)), np.nan)}
    S = done['shares']
    for k, j in enumerate(done['idx']):
        if not np.isnan(S[k, 0, 0]):
            continue
        S[k], _ = explain(Xva[j])
        if k % 20 == 0:
            np.savez(out, idx=done['idx'], shares=S); print(k, flush=True)
    np.savez(out, idx=done['idx'], shares=S)
    json.dump({'n_hi': int(len(hi)), 'n_mid': int(n_mid), 'n_low': int(500 - len(hi) - n_mid)}, open(f'{E}/lime_subset_alloc.json', 'w'))
    print('done')
