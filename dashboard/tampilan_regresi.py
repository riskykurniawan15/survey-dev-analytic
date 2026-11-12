"""Tab regresi berganda pada data dan filter dashboard yang aktif."""
import numpy as np
import streamlit as st
import plotly.express as px
from regresi import run_analysis
from gambar_regresi import report_pngs


@st.cache_data(show_spinner=False)
def compute(people):
    """OLS dan diagnostik sesuai cakupan tugas materi 4."""
    return run_analysis(people)


@st.cache_data(show_spinner=False)
def pngs(people):
    return report_pngs(compute(people))


def render_regression(people):
    st.subheader('Regresi berganda kepuasan kerja')
    st.caption('Kelanjutan tugas individu 1 dan 2. Model menggunakan kasus lengkap setelah filter aktif.')
    try:
        r = compute(people)
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.info(str(exc)); return
    m = r['metadata']
    a,b,c = st.columns(3)
    a.metric('Kasus lengkap', m['n_complete'])
    b.metric('Model dipilih', m['chosen'])
    c.metric('R kuadrat', f"{r['models'][m['chosen']].rsquared:.3f}")
    st.write('Model A: pengalaman kerja + ln(gaji USD) + dummy Hybrid dan Remote (acuan Kantor). Model B menambah lama menulis kode.')
    st.dataframe(r['comparison'], hide_index=True, width='stretch')
    st.write(f"Uji F parsial B terhadap A: F = {m['partial_f']:.3f}; p = {m['partial_p']:.4f}. Pemilihan mempertimbangkan F parsial, adjusted R kuadrat, AIC dan VIF.")
    st.markdown('**Koefisien OLS, uji t dan interval kepercayaan 95%**')
    st.dataframe(r['coefficients'][m['chosen']], hide_index=True, width='stretch')
    st.caption('Hubungan pada sampel observasional, dengan prediktor lain tetap. Uji F/t dan interval memakai asumsi regresi klasik; baca bersama diagnostik grafik.')
    left,right = st.columns(2)
    with left:
        st.write('VIF Model A'); st.dataframe(r['vif']['A'], hide_index=True)
    with right:
        st.write('VIF Model B'); st.dataframe(r['vif']['B'], hide_index=True)
    diag = r['diagnostics']
    fig = px.scatter(diag, x='fitted', y='residual', opacity=.4,
                     labels={'fitted':'Prediksi kepuasan','residual':'Residual (poin)'})
    fig.add_hline(y=0)
    st.plotly_chart(fig, width='stretch')
    st.write(f"Cook > 4/n: {m['cook_count']} responden; leverage > 2p/n: {m['leverage_count']}; |residual baku| > 3: {m['student_count']}. Periksa pola residual, Q-Q, leverage dan Cook pada gambar laporan. Kasus bertanda tetap masuk model utama.")
    with st.expander('Alur sampel dan analisis sensitivitas'):
        st.dataframe(r['flow'], hide_index=True)
        st.dataframe(r['sensitivity'], hide_index=True)
    st.markdown('**Prediksi pada pengalaman dan gaji median sampel**')
    st.write(f"Pengalaman {m['prediction_work_exp']:g} tahun dan gaji USD {m['prediction_salary']:,.0f} per tahun.")
    st.dataframe(r['prediction'], hide_index=True, width='stretch')
    st.caption('mean_ci_* adalah IK rata-rata; obs_ci_* adalah interval prediksi individu. Keduanya memakai asumsi OLS klasik. Jika grafik menunjukkan penyimpangan asumsi, tafsirkan sebagai ilustrasi bersyarat. Batas dapat melewati skala 0-10; jangan gunakan untuk menilai individu.')
    st.download_button('Unduh koefisien CSV', r['coefficients'][m['chosen']].to_csv(index=False),
                       file_name='koefisien_regresi.csv', mime='text/csv')
    with st.expander('Gambar laporan dan unduh PNG'):
        st.caption('Generator yang sama digunakan oleh builder Word/PDF. Filter aktif mengubah data dan angka pada gambar.')
        for filename, label, data in pngs(people):
            st.image(data, caption=label, width='stretch')
            st.download_button(f'Unduh {label}', data=data, file_name=filename,
                               mime='image/png', key=f'regresi_{filename}')
