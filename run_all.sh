#!/bin/bash
# Run every experiment for the IJEM revision. Run from this folder:  bash run_all.sh
# Put the Excel file at data/lime.xlsx first. Total time on a GPU: about 1-2 hours.
set -e
mkdir -p runs logs figs
export TF_CPP_MIN_LOG_LEVEL=2
echo "== 1. Build windows (frozen snapshot 2026-11-02 08:00)"
python3 - <<'P'
import sys; sys.path.insert(0, 'scripts'); import pipe, numpy as np, os
hw0 = pipe.load_raw('data/lime.xlsx')
for cfg, kw in [('base', {}), ('raw', dict(routed=False)), ('noshift', dict(shift=False))] + [(f'L{L}', dict(L_up=L)) for L in [2, 4, 8, 10]]:
    p = f'data/win_{cfg}.npz'
    if os.path.exists(p): continue
    hw = pipe.features(hw0, **kw); Xp, Xf, Y, t = pipe.windows(hw)
    np.savez(p, Xp=Xp, Xf=Xf, Y=Y, t=np.array(t, dtype='datetime64[ns]')); print(cfg, Xp.shape)
na = hw0.level_raw.isna().values
flag = np.array([na[i-24:i+6].any() for i in range(24, len(hw0)-6)]); np.save('data/interp_flag.npy', flag)
P
echo "== 2. ARIMA and 2018-event baselines"
python3 scripts/arima.py | tee logs/arima.log
python3 scripts/ev_base.py | tee logs/ev_base.log
echo "== 3. LSTM runs (5 seeds each; kernel-length runs 3 seeds; 2018 hold-out 3 seeds)"
for cfg in proposed_r zerofill_r proposed raw_r noshift_r; do for s in 1 2 3 4 5; do
  echo "  $cfg seed $s"; python3 scripts/train.py $cfg $s > logs/${cfg}_s$s.log 2>&1; done; done
for L in 2 4 8 10; do for s in 1 2 3; do echo "  L$L seed $s"; python3 scripts/train.py L${L}_r $s > logs/L${L}_r_s$s.log 2>&1; done; done
for s in 1 2 3; do echo "  ev2018 seed $s"; python3 scripts/train.py ev2018_r $s > logs/ev2018_r_s$s.log 2>&1; done
echo "== 4. XGBoost (5 seeds). Set XGB_DEVICE=cuda to use the GPU"
for s in 1 2 3 4 5; do python3 scripts/xgb.py $s >> logs/xgb.log 2>&1; done
echo "== 5. LIME on the median run"
python3 - <<'P'
import sys, json, glob, numpy as np; sys.path.insert(0, 'scripts'); import evaluate as ev
r = ev.summary('proposed_r')['mae']['run_means']; runs = sorted(glob.glob('runs/proposed_r_s*'))
json.dump({'run': runs[int(np.argsort(r)[len(r)//2])].split('/')[-1]}, open('rep_run.json', 'w'))
P
R=$(python3 -c "import json;print(json.load(open('rep_run.json'))['run'])")
python3 scripts/lime_run.py $R subset > logs/lime_sub.log 2>&1
echo "== 6. Figures and results"
python3 scripts/fig1.py; python3 scripts/figs.py; python3 scripts/report.py
python3 - <<'P'
import platform, subprocess, tensorflow as tf, json
info = {'platform': platform.platform(), 'python': platform.python_version(), 'tensorflow': tf.__version__,
        'gpus': [d.name for d in tf.config.list_physical_devices('GPU')]}
try: info['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout
except Exception: pass
try: info['cpu'] = [l for l in open('/proc/cpuinfo') if 'model name' in l][0]
except Exception: info['cpu'] = platform.processor()
json.dump(info, open('hardware.json', 'w'), indent=1); print(info)
P
echo "== Done. Zip and send back: results.json results.txt hardware.json figs/ logs/ runs/*/info.json lime_subset.npz"
