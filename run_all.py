"""Run every experiment for the IJEM revision (works on Windows, Linux and Colab).
Usage, from this folder:   python run_all.py
Put the Excel file at data/lime.xlsx first. If interrupted, run again: finished runs are skipped."""
import os, sys, subprocess, json, glob, platform

os.chdir(os.path.dirname(os.path.abspath(__file__)))
for d in ['runs', 'logs', 'figs']:
    os.makedirs(d, exist_ok=True)
env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2', NT=str(os.cpu_count() or 2))
PY = sys.executable


def run(args, log=None):
    print('  ', ' '.join(args), flush=True)
    if log:
        with open(log, 'a') as f:
            r = subprocess.run([PY] + args, stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        r = subprocess.run([PY] + args, env=env)
    if r.returncode != 0:
        print(f'   !! failed (see {log or "output above"})'); sys.exit(1)


if not os.path.exists('data/lime.xlsx'):
    sys.exit('Put the Excel file at data/lime.xlsx first.')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow', tf.__version__, '| GPUs found:', gpus or 'NONE - training will run on the CPU (slow)')
try:
    import xgboost, numpy as np
    xgboost.XGBRegressor(device='cuda', n_estimators=1).fit(np.zeros((4, 2)), np.zeros(4))
    env['XGB_DEVICE'] = 'cuda'; print('XGBoost will use the GPU')
except Exception:
    print('XGBoost will use the CPU')

print('== 1. Build windows (frozen snapshot 2026-11-02 08:00)')
sys.path.insert(0, 'scripts')
import numpy as np, pipe
hw0 = pipe.load_raw('data/lime.xlsx')
for cfg, kw in [('base', {}), ('raw', dict(routed=False)), ('noshift', dict(shift=False))] + \
               [(f'L{L}', dict(L_up=L)) for L in [2, 4, 8, 10]]:
    p = f'data/win_{cfg}.npz'
    if os.path.exists(p):
        continue
    hw = pipe.features(hw0, **kw); Xp, Xf, Y, t = pipe.windows(hw)
    np.savez(p, Xp=Xp, Xf=Xf, Y=Y, t=np.array(t, dtype='datetime64[ns]')); print('  ', cfg, Xp.shape)
na = hw0.level_raw.isna().values
np.save('data/interp_flag.npy', np.array([na[i - 24:i + 6].any() for i in range(24, len(hw0) - 6)]))

print('== 2. ARIMA and 2018-event baselines')
run(['scripts/arima.py'], 'logs/arima.log')
run(['scripts/ev_base.py'], 'logs/ev_base.log')

print('== 3. LSTM runs')
jobs = [(c, s) for c in ['proposed_r', 'zerofill_r', 'proposed', 'raw_r', 'noshift_r'] for s in range(1, 6)]
jobs += [(f'L{L}_r', s) for L in [2, 4, 8, 10] for s in range(1, 4)]
jobs += [(c, s) for c in ['ev2018_r', 'ev2018zf_r', 'ev2018raw_r', 'ev2018noshift_r', 'ev2018norain_r']
         for s in range(1, 6)]   # 2018 flood hold-out: proposed model and ablations, 5 seeds
for i, (c, s) in enumerate(jobs, 1):
    print(f'  [{i}/{len(jobs)}]', end='')
    run(['scripts/train.py', c, str(s)], f'logs/{c}_s{s}.log')

print('== 4. XGBoost, 5 seeds')
for s in range(1, 6):
    run(['scripts/xgb.py', str(s)], 'logs/xgb.log')

print('== 4b. XGBoost on the 2018 flood hold-out, 5 seeds')
for s in range(1, 6):
    run(['scripts/xgb_event.py', str(s)], 'logs/xgb_event.log')

print('== 5. LIME on the median run')
import evaluate as ev
r = ev.summary('proposed_r')['mae']['run_means']; runs = sorted(glob.glob('runs/proposed_r_s*'))
rep = os.path.basename(runs[int(np.argsort(r)[len(r) // 2])])
json.dump({'run': rep}, open('rep_run.json', 'w'))
run(['scripts/lime_run.py', rep, 'subset'], 'logs/lime_sub.log')

print('== 6. Figures and results')
run(['scripts/fig1.py']); run(['scripts/figs.py']); run(['scripts/report.py']); run(['scripts/flood_table.py'])
info = {'platform': platform.platform(), 'python': platform.python_version(), 'tensorflow': tf.__version__,
        'gpus': [d.name for d in gpus], 'processor': platform.processor(), 'cpu_count': os.cpu_count()}
try:
    info['cpu_model'] = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
    info['ram_gb'] = round(int([l for l in open('/proc/meminfo') if l.startswith('MemTotal')][0].split()[1]) / 1024 ** 2, 1)
except Exception:
    pass
try:
    info['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                        capture_output=True, text=True).stdout
except Exception:
    pass
os.makedirs('results', exist_ok=True)
for f in ['hardware.json', 'results/hardware.json']:
    json.dump(info, open(f, 'w'), indent=1)
print(info)
print('== Done. Results: results.json, results.txt, results/flood_table.csv, results/hardware.json')
