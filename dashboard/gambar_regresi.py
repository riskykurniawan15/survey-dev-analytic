from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from analisis import MODE_COLORS
from gambar_laporan import style_plots, _png

def report_pngs(result: dict) -> list[tuple[str,str,bytes]]:
    r=result
    style_plots()
    packaged=[]
    d=r['data']; color=MODE_COLORS['Remote']
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.2))
    for ax,col,label in zip(axs,['work_exp','log_gaji','years_code'],['Pengalaman kerja (tahun)','ln(gaji tahunan USD)','Lama menulis kode (tahun)']):
        # Jitter hanya untuk visual; perhitungan memakai skor asli.
        jitter=np.random.default_rng(2026).uniform(-.14,.14,len(d))
        ax.scatter(d[col],d.job_sat+jitter,s=9,alpha=.27,color=color,edgecolors='none')
        slope,intercept=np.polyfit(d[col],d.job_sat,1)
        x=np.array([d[col].min(),d[col].max()]); ax.plot(x,intercept+slope*x,color='#ba553b')
        ax.set(xlabel=label,ylabel='Kepuasan (0-10)',ylim=(-.4,10.5))
    fig.suptitle('Kepuasan kerja dan kandidat prediktor', fontsize=12)
    packaged.append(('diagram_pencar.png','Diagram pencar regresi',_png(fig)))
    cols=['job_sat','work_exp','years_code','log_gaji','hybrid','remote']
    labels=['Kepuasan','Pengalaman','Lama kode','ln(gaji)','Hybrid','Remote']
    corr=d[cols].corr(); fig,ax=plt.subplots(figsize=(7.7,4.5))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.grid(False)
    ax.set_xticks(range(6),labels,rotation=25,ha='right'); ax.set_yticks(range(6),labels)
    for i in range(6):
        for j in range(6):
            v=corr.iloc[i,j]; ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if abs(v)>.65 else 'black')
    fig.colorbar(im,ax=ax,shrink=.8,label='Pearson r'); fig.tight_layout()
    ax.set_title('Korelasi Pearson pada sampel regresi')
    packaged.append(('matriks_korelasi.png','Matriks korelasi',_png(fig)))
    diag=r['diagnostics']; m=r['models'][r['metadata']['chosen']]
    fig,axs=plt.subplots(2,2,figsize=(10.5,6.5))
    ax=axs[0,0]; ax.scatter(diag.fitted,diag.residual,s=9,alpha=.35,color=color); ax.axhline(0,c='#ba553b',lw=1)
    ax.set(xlabel='Nilai prediksi',ylabel='Residual (poin)',title='Residual terhadap prediksi')
    ax=axs[0,1]; stats.probplot(m.resid,dist='norm',plot=ax); ax.set(title='Q-Q residual',xlabel='Kuantil normal teoretis',ylabel='Residual terurut')
    ax=axs[1,0]; ax.scatter(np.arange(len(diag)),diag.residual_baku,s=8,alpha=.4,color=color)
    ax.axhline(3,c='#ba553b',ls='--'); ax.axhline(-3,c='#ba553b',ls='--')
    ax.set(xlabel='Urutan baris analisis (bukan waktu)',ylabel='Residual baku',title='Residual baku per observasi')
    ax=axs[1,1]; ax.scatter(diag.leverage,diag.cook,s=10,alpha=.4,color=color)
    ax.axhline(4/len(d),c='#ba553b',ls='--',label='Cook 4/n'); ax.axvline(2*len(m.params)/len(d),c='#a37721',ls='--',label='Leverage 2p/n')
    ax.set(xlabel='Leverage',ylabel='Jarak Cook',title='Leverage dan pengaruh'); ax.legend(fontsize=8)
    fig.suptitle('Diagnostik regresi kepuasan kerja', fontsize=12)
    packaged.append(('diagnostik.png','Diagnostik residual dan pengaruh',_png(fig)))
    return packaged

def write_all(output, result=None):
    from regresi import run_analysis
    result=run_analysis() if result is None else result
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for filename,label,data in report_pngs(result):
        path=output/filename; path.write_bytes(data); paths.append(path)
    return paths
