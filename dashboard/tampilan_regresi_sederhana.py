"""Penerapan Bagian B Materi 3 pada data dan filter dashboard."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from regresi_sederhana import run_survey
from gambar_regresi_sederhana import report_pngs


@st.cache_data(show_spinner=False)
def compute(people, prediction_x):
    return run_survey(people, prediction_x=prediction_x)


@st.cache_data(show_spinner=False)
def pngs(people, prediction_x):
    return report_pngs(compute(people,prediction_x))


def render_simple_regression(people):
    st.subheader('Regresi sederhana pengalaman dan gaji')
    st.caption('Bagian B Materi 3. Satu prediktor pengalaman kerja dan respons gaji tahunan USD, dengan ambang gaji USD 1.000 seperti tugas sebelumnya.')
    valid=people.loc[(people.comp_yearly>=1000) & people.work_exp.notna() & (people.work_exp>=0)]
    if len(valid)<30 or valid.work_exp.nunique()<2 or valid.comp_yearly.nunique()<2:
        st.info('Diperlukan minimal 30 pasangan lengkap serta variasi pengalaman dan gaji. Sesuaikan filter untuk menjalankan regresi sederhana.')
        return
    xmin,xmax=float(valid.work_exp.min()),float(valid.work_exp.max())
    value=float(np.clip(15,xmin,xmax))
    prediction_x=st.number_input('Pengalaman untuk prediksi (tahun)',min_value=xmin,max_value=xmax,
        value=value,step=1.0,key='prediksi_regresi_sederhana')
    try:
        r=compute(people,prediction_x)
    except (ValueError,np.linalg.LinAlgError) as exc:
        st.info(str(exc)); return
    s=r['summary']
    a,b,c=st.columns(3)
    a.metric('Pasangan lengkap',s['n'])
    b.metric('R kuadrat',f"{s['r2']:.3f}")
    c.metric('Galat estimasi USD',f"{s['se']:,.0f}")
    st.write(f"Gaji tahunan (USD) = {s['b0']:,.2f} + {s['b1']:,.2f} × pengalaman (tahun).")
    st.markdown('**Korelasi Pearson dan Spearman**')
    st.dataframe(r['correlation'],hide_index=True,width='stretch')
    st.caption('Uji dua sisi pada alfa 0,05. Nilai p Spearman memakai pendekatan t, sesuai keluaran SciPy.')
    st.markdown('**Koefisien OLS dan IK 95%**')
    st.dataframe(r['coefficients'].drop(columns='term'),hide_index=True,width='stretch')
    st.write(f"F(1, {s['df_resid']}) = {s['f']:.3f}; p = {s['p_f']:.3g}; adjusted R kuadrat = {s['adj_r2']:.3f}.")
    with st.expander('Alur sampel dan dekomposisi variasi'):
        st.dataframe(r['flow'],hide_index=True)
        st.dataframe(r['anova'],hide_index=True,width='stretch')
    fig=px.scatter(r['data'],x='x',y='y',opacity=.4,
        labels={'x':'Pengalaman kerja (tahun)','y':'Gaji tahunan (USD)'})
    line=pd.DataFrame({'x':[s['xmin'],s['xmax']]})
    fig.add_scatter(x=line.x,y=s['b0']+s['b1']*line.x,mode='lines',name='Garis OLS')
    st.plotly_chart(fig,width='stretch')
    st.markdown('**Prediksi dengan IK rata-rata dan interval individu 95%**')
    st.dataframe(r['prediction'],hide_index=True,width='stretch')
    st.caption('mean_ci_* adalah IK rata-rata; obs_ci_* adalah interval prediksi individu. Interval bergantung pada asumsi regresi klasik. Jika batas bawah gaji negatif atau diagnostik menyimpang, model belum layak untuk menetapkan gaji individu.')
    st.write(f"|Residual baku| > 3: {s['std_above3']} kasus; leverage > 4/n: {s['leverage_count']} kasus. Residual baku dihitung sebagai e/se sesuai materi.")
    if s['sensitivity']:
        sen=s['sensitivity']
        st.caption(f"Pemeriksaan tanpa {sen['dikeluarkan']} kasus dengan |residual baku|>3 menghasilkan kemiringan USD {sen['b1']:,.2f} per tahun dan R kuadrat {sen['r2']:.3f}. Seluruh kasus valid tetap masuk model utama.")
    st.download_button('Unduh koefisien regresi sederhana CSV',r['coefficients'].to_csv(index=False),
        file_name='koefisien_regresi_sederhana.csv',mime='text/csv')
    with st.expander('Gambar laporan dan unduh PNG regresi sederhana'):
        for filename,label,data in pngs(people,prediction_x):
            st.image(data,caption=label,width='stretch')
            st.download_button(f'Unduh {label}',data=data,file_name=filename,
                mime='image/png',key=f'sederhana_{filename}')
