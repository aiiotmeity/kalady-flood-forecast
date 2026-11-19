"""Train one configuration with one seed and save validation predictions.
Usage: python train.py CONFIG SEED
CONFIG in: proposed, zerofill, raw, noshift, L2, L4, L8, L10
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pipe

cfg, seed = sys.argv[1], int(sys.argv[2])
max_epochs = int(os.environ.get('MAX_EPOCHS', 100))
out = f'./runs/{cfg}_s{seed}'
os.makedirs(out, exist_ok=True)
if os.path.exists(f'{out}/pred.npy'):
    print('done already'); sys.exit()

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
tf.keras.utils.set_random_seed(seed)
tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('NT', 2)))

kw = {}
resid = cfg.endswith('_r') or cfg.startswith('r_')
base_cfg = cfg[:-2] if cfg.endswith('_r') else cfg
if base_cfg == 'ev2018': kw = {}
if base_cfg == 'raw': kw = dict(routed=False)
if base_cfg == 'noshift': kw = dict(shift=False)
if base_cfg.startswith('L') and base_cfg[1:].isdigit(): kw = dict(L_up=int(base_cfg[1:]))
mask_mode = 'zerofill' if base_cfg == 'zerofill' else 'masked'

cache = f'./data/win_{base_cfg if base_cfg in ("raw","noshift") or (base_cfg.startswith("L") and base_cfg[1:].isdigit()) else "base"}.npz'
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); Xp, Xf, Y = z['Xp'], z['Xf'], z['Y']
else:
    hw = pipe.features(pipe.load_raw('./data/lime.xlsx'), **kw)
    Xp, Xf, Y, t = pipe.windows(hw)
    np.savez(cache, Xp=Xp, Xf=Xf, Y=Y, t=np.array(t, dtype='datetime64[ns]'))
if mask_mode == 'zerofill':
    Xf = Xf[:, :, :1]
s = pipe.split(len(Xp))
import pandas as pd
tt = pd.DatetimeIndex(np.load(cache, allow_pickle=True)['t'])
ev = (tt >= '2018-08-01') & (tt <= '2018-08-31 23:00')
buf = (tt >= '2018-07-30') & (tt <= '2018-09-02')
tr_idx = np.where((np.arange(len(Xp)) < s) & ~(buf if base_cfg == 'ev2018' else np.zeros(len(Xp), bool)))[0]
last = Xp[:, -1, 0].reshape(-1, 1, 1)
if resid:
    Y = Y - last
npf, nff = Xp.shape[2], Xf.shape[2]
sp = StandardScaler().fit(Xp[tr_idx].reshape(-1, npf)); sf = StandardScaler().fit(Xf[tr_idx].reshape(-1, nff))
sy = StandardScaler().fit(Y[tr_idx].reshape(-1, 1))
tr = lambda a, sc: sc.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape).astype('float32')
Xp_s, Xf_s, Y_s = tr(Xp, sp), tr(Xf, sf), tr(Y, sy)

pin = layers.Input((pipe.PAST_HOURS, npf)); x = layers.LSTM(128, return_sequences=True, dropout=0.2)(pin)
x = layers.LSTM(64, dropout=0.1)(x); rep = layers.RepeatVector(pipe.HORIZON)(x)
fin = layers.Input((pipe.HORIZON, nff)); dec = layers.Concatenate()([rep, fin])
d = layers.LSTM(64, return_sequences=True, dropout=0.2)(dec); d = layers.LSTM(32, return_sequences=True)(d)
d = layers.TimeDistributed(layers.Dense(32, activation='relu'))(d); o = layers.TimeDistributed(layers.Dense(1))(d)
m = Model([pin, fin], o)
m.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(1.0), metrics=['mae'])
t0 = time.time()
cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)]
h = m.fit([Xp_s[tr_idx], Xf_s[tr_idx]], Y_s[tr_idx], validation_data=([Xp_s[s:], Xf_s[s:]], Y_s[s:]),
          epochs=max_epochs, batch_size=64, callbacks=cb, verbose=2)
pred = sy.inverse_transform(m.predict([Xp_s[s:], Xf_s[s:]], batch_size=512, verbose=0).reshape(-1, 1)).reshape(-1, pipe.HORIZON)
if resid:
    pred = pred + last[s:, :, 0]
if base_cfg == 'ev2018':
    pe = sy.inverse_transform(m.predict([Xp_s[ev], Xf_s[ev]], batch_size=512, verbose=0).reshape(-1, 1)).reshape(-1, pipe.HORIZON)
    np.save(f'{out}/pred_event.npy', pe + (last[ev, :, 0] if resid else 0))
np.save(f'{out}/pred.npy', pred)
json.dump({'history': {k: [float(v) for v in vals] for k, vals in h.history.items()},
           'best_epoch': int(np.argmin(h.history['val_loss']) + 1), 'epochs': len(h.history['loss']),
           'seconds': time.time() - t0, 'params': int(m.count_params())}, open(f'{out}/info.json', 'w'))
if base_cfg == 'proposed':
    m.save(f'{out}/model.keras')
    pickle.dump((sp, sf, sy), open(f'{out}/scalers.pkl', 'wb'))
print('finished', cfg, seed, time.time() - t0)
