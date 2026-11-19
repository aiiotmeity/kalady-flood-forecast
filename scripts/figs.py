import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipe, evaluate as ev
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
E = '.'; F = f'{E}/figs'
rep = json.load(open(f'{E}/rep_run.json'))['run']          # representative (median-MAE) run
z = np.load(f'{E}/data/win_base.npz'); t = pd.DatetimeIndex(z['t']); s = pipe.split(len(t))
tv = t[s:]; yv = ev.yv; p = np.load(f'{E}/runs/{rep}/pred.npy')
W = 6.3
def brk(tt, y):
    tt = pd.DatetimeIndex(tt); y = np.asarray(y, float)
    cut = np.where(np.diff(tt.values).astype('timedelta64[h]').astype(int) > 1)[0] + 1
    return np.insert(tt.values.astype('datetime64[ns]'), cut, np.datetime64('NaT')), np.insert(y, cut, np.nan)

# Fig 3: loss curves
info = json.load(open(f'{E}/runs/{rep}/info.json')); hst = info['history']
fig, ax = plt.subplots(figsize=(W * 0.75, 2.6))
ep = np.arange(1, len(hst['loss']) + 1)
ax.plot(ep, hst['loss'], 'k-o', ms=3, label='Training loss'); ax.plot(ep, hst['val_loss'], 'r--s', ms=3, label='Validation loss')
b = info['best_epoch']; ax.axvline(b, color='grey', ls=':', lw=1); ax.annotate(f'Best epoch ({b})', (b, hst['val_loss'][b - 1]),
                                                                           xytext=(b + 2, (max(hst['val_loss']) + min(hst['val_loss'])) / 2), arrowprops=dict(arrowstyle='->', lw=0.7))
ax.set_xlabel('Epoch'); ax.set_ylabel('Huber loss (scaled units)'); ax.legend(frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{F}/fig3.png', dpi=300); plt.close(fig)

# Fig 4: observed vs forecast, H+1 and H+6
fig, axs = plt.subplots(2, 1, figsize=(W, 3.6), sharex=True)
for ax, h in zip(axs, [0, 5]):
    tt = tv + pd.Timedelta(hours=h)
    ax.plot(*brk(tt, yv[:, h]), color='k', lw=0.6, label='Observed'); ax.plot(*brk(tt, p[:, h]), color='tab:red', lw=0.6, alpha=0.8, label='Forecast')
    rm = np.sqrt(((yv[:, h] - p[:, h]) ** 2).mean())
    ax.set_title(f'({"ab"[h // 5]}) H+{h + 1}, RMSE = {rm:.3f} m', loc='left'); ax.set_ylabel('Water level (m)'); ax.grid(alpha=0.3)
axs[0].legend(frameon=False, ncol=2, loc='upper right'); axs[1].set_xlabel('Date')
fig.tight_layout(); fig.savefig(f'{F}/fig4.png', dpi=300); plt.close(fig)

# Fig 5: residuals
fig, axs = plt.subplots(2, 1, figsize=(W, 3.4), sharex=True)
for ax, h in zip(axs, [0, 5]):
    e = p[:, h] - yv[:, h]; rm = np.sqrt((e ** 2).mean()); tt = tv + pd.Timedelta(hours=h)
    ax.plot(*brk(tt, e), color='tab:blue', lw=0.5); ax.axhspan(-rm, rm, color='grey', alpha=0.25, label=f'±RMSE ({rm:.3f} m)')
    ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('Forecast − observed (m)'); ax.grid(alpha=0.3)
    ax.set_title(f'({"ab"[h // 5]}) H+{h + 1}', loc='left'); ax.legend(frameon=False, loc='upper right')
axs[1].set_xlabel('Date'); fig.tight_layout(); fig.savefig(f'{F}/fig5.png', dpi=300); plt.close(fig)

# Fig 6: scatter and error histogram
fig, axs = plt.subplots(1, 2, figsize=(W, 2.8))
r2 = np.corrcoef(yv[:, 0], p[:, 0])[0, 1] ** 2
axs[0].scatter(yv[:, 0], p[:, 0], s=2, alpha=0.3, color='tab:blue', rasterized=True)
lim = [min(yv.min(), p.min()), max(yv.max(), p.max())]; axs[0].plot(lim, lim, 'k--', lw=0.8, label='1:1 line')
axs[0].set_xlabel('Observed level (m)'); axs[0].set_ylabel('Forecast level (m)'); axs[0].set_title(f'(a) H+1 (R² = {r2:.3f})', loc='left'); axs[0].legend(frameon=False)
e = (p - yv).ravel(); axs[1].hist(e, bins=np.linspace(-1, 1, 81), color='tab:blue')
axs[1].set_xlabel('Forecast error (m)'); axs[1].set_ylabel('Frequency'); axs[1].set_title('(b) Error distribution, all lead times', loc='left')
for a in axs: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{F}/fig6.png', dpi=300); plt.close(fig)

# Fig 7: MAE and RMSE by lead time, proposed (mean ± SD) vs persistence and ARIMA
S = ev.summary('proposed_r'); Pp = ev.summary('persistence'); Ar = ev.summary('arima')
fig, axs = plt.subplots(1, 2, figsize=(W, 2.6)); h = np.arange(1, 7)
for ax, k, lab in zip(axs, ['mae', 'rmse'], ['MAE (m)', 'RMSE (m)']):
    ax.errorbar(h, S[k]['h_mean'], yerr=S[k]['h_sd'], fmt='-o', color='tab:red', ms=3, capsize=2, label='Proposed (mean ± SD)')
    ax.plot(h, Ar[k]['h_mean'], '-s', color='tab:green', ms=3, label='ARIMA'); ax.plot(h, Pp[k]['h_mean'], '--^', color='grey', ms=3, label='Persistence')
    ax.set_xlabel('Forecast lead time (h)'); ax.set_ylabel(lab); ax.grid(alpha=0.3)
axs[0].set_title('(a) MAE', loc='left'); axs[1].set_title('(b) RMSE', loc='left'); axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{F}/fig7.png', dpi=300); plt.close(fig)

# Fig 8: LIME category shares by lead time (validation subset), all samples vs rising-level samples
zz = np.load(f'{E}/lime_subset.npz'); SH = zz['shares']; ix = zz['idx']
Xl = z['Xp'][s:, -1, 0]; rise = z['Y'][s:, 5, 0][ix] - Xl[ix]
cats = ['Level', 'Upstream rain', 'Local rain', 'Discharge', 'Temporal']; cols = ['#4C72B0', '#DD8452', '#55A868', '#8172B3', '#BBBBBB']
fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=False)
for ax, sel, ttl in zip(axs, [np.ones(len(ix), bool), rise > 0.3], ['(a) All subset samples (n = 500)', f'(b) Rising level, > 0.3 m in 6 h (n = {(rise > 0.3).sum()})']):
    m = SH[sel].mean(0); bot = np.zeros(6)
    for k in [1, 2, 3, 4]:
        ax.bar(np.arange(1, 7), m[:, k], bottom=bot, color=cols[k], label=cats[k]); bot += m[:, k]
    ax.set_xlabel('Forecast lead time (h)'); ax.set_ylabel('Share of attribution (non-level)'); ax.set_title(ttl, loc='left'); ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, bot.max() * 1.15)
    for h in range(6): ax.text(h + 1, bot[h] * 1.02, f'{m[h, 0]:.2f}', ha='center', fontsize=6.5)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f'{F}/fig8.png', dpi=300); plt.close(fig)
print('figures written')
