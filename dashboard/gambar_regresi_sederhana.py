from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from gambar_laporan import style_plots, _png


def report_pngs(result):
    style_plots()
    d, m, s = result['observations'], result['model'], result['summary']
    scale = s['plot_scale']
    y_label = s['y_label'] if scale == 1 else 'Gaji tahunan (ribu USD)'
    color = '#38bdf8' if scale != 1 else '#2563eb'
    fig, ax = plt.subplots(figsize=(7.5, 4.1))
    ax.scatter(d.x, d.y/scale, s=32 if len(d)<30 else 13,
               alpha=.85 if len(d)<30 else .38, color=color)
    grid = np.linspace(d.x.min(), d.x.max(), 150)
    ax.plot(grid, (s['b0']+s['b1']*grid)/scale, color='#ba553b', label='Garis OLS')
    ax.set(xlabel=s['x_label'], ylabel=y_label, title='Hubungan X dan Y serta garis regresi sederhana')
    ax.legend(frameon=False)
    scatter = _png(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.3))
    ax=axes[0,0]
    ax.scatter(d.prediksi/scale, d.residual/scale, s=28 if len(d)<30 else 10, alpha=.6, color=color)
    ax.axhline(0,color='#ba553b',lw=1)
    ax.set(xlabel='Nilai prediksi' + (' (ribu USD)' if scale != 1 else ' (menit)'),
           ylabel='Residual' + (' (ribu USD)' if scale != 1 else ' (menit)'), title='Residual terhadap prediksi')
    ax=axes[0,1]
    stats.probplot(d.residual/scale, dist='norm', plot=ax)
    ax.set(title='Q-Q residual', xlabel='Kuantil normal teoretis',
           ylabel='Residual terurut' + (' (ribu USD)' if scale != 1 else ' (menit)'))
    index=np.arange(1,len(d)+1)
    ax=axes[1,0]
    ax.scatter(index,d.residual_baku,s=28 if len(d)<30 else 10,alpha=.6,color=color)
    ax.axhline(2,color='#d97706',ls=':',lw=1); ax.axhline(-2,color='#d97706',ls=':',lw=1)
    ax.axhline(3,color='#ba553b',ls='--',lw=1); ax.axhline(-3,color='#ba553b',ls='--',lw=1)
    ax.set(xlabel='Urutan observasi (bukan waktu)',ylabel='Residual / galat baku estimasi',title='Residual baku sesuai materi')
    ax=axes[1,1]
    ax.scatter(index,d.leverage,s=28 if len(d)<30 else 10,alpha=.6,color=color)
    ax.axhline(s['leverage_threshold'],color='#ba553b',ls='--',label='Ambang 4/n')
    ax.set(xlabel='Urutan observasi (bukan waktu)',ylabel='Leverage',title='Leverage setiap observasi')
    ax.legend(frameon=False,fontsize=8)
    diagnostics = _png(fig)
    return [('01_scatter.png','Diagram pencar dan garis regresi',scatter),
            ('02_diagnostik.png','Diagnostik residual dan leverage',diagnostics)]


def write_all(output, result):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for filename, _, data in report_pngs(result):
        (output/filename).write_bytes(data)
