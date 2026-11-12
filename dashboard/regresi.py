from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from database import CSV_PATH, build_frames

ROOT = Path(__file__).resolve().parents[1]
TERMS_A = 'work_exp + log_gaji + hybrid + remote'
FORMULAS = {'A': 'job_sat ~ ' + TERMS_A,
            'B': 'job_sat ~ ' + TERMS_A + ' + years_code'}
LABELS = {'Intercept': 'Konstanta', 'work_exp': 'Pengalaman kerja',
          'log_gaji': 'ln(gaji USD)', 'hybrid': 'Hybrid vs Kantor',
          'remote': 'Remote vs Kantor', 'years_code': 'Lama menulis kode'}


def prepare_data(people: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = people.copy()
    def mode(v):
        if v == 'Remote': return 'Remote'
        if v == 'In-person': return 'Kantor'
        if isinstance(v, str) and ('Hybrid' in v or v.startswith('Your choice')): return 'Hybrid'
        return None
    d['mode'] = d.remote_work.map(mode)
    # Ambang identik dengan tugas 1 dan 2, tanpa pemotongan ekor atas.
    d['log_gaji'] = np.log(d.comp_yearly.where(d.comp_yearly >= 1000))
    stages = [('CSV bersih ID', np.ones(len(d), dtype=bool)),
              ('Kepuasan 0 sampai 10', d.job_sat.between(0, 10)),
              ('Mode kerja dikenali', d['mode'].notna()),
              ('Pengalaman kerja terisi', d.work_exp.notna()),
              ('Gaji minimal USD 1000', d.log_gaji.notna()),
              ('Lama menulis kode terisi', d.years_code.notna())]
    keep = pd.Series(True, index=d.index)
    flow = []
    for label, mask in stages:
        before = int(keep.sum())
        keep &= mask
        flow.append({'tahap': label, 'tersisa': int(keep.sum()), 'gugur_tahap': before-int(keep.sum())})
    d['hybrid'] = (d['mode'] == 'Hybrid').astype(int)
    d['remote'] = (d['mode'] == 'Remote').astype(int)
    return d.loc[keep].reset_index(drop=True), pd.DataFrame(flow)


def coef_table(model):
    ci = model.conf_int()
    return pd.DataFrame([{'term': name, 'variabel': LABELS.get(name, name),
        'b': float(model.params[name]), 'se': float(model.bse[name]),
        't': float(model.tvalues[name]), 'p': float(model.pvalues[name]),
        'ci_low': float(ci.loc[name, 0]), 'ci_high': float(ci.loc[name, 1])}
        for name in model.params.index])


def vif_table(model):
    x = model.model.exog
    return pd.DataFrame([{'variabel': LABELS.get(name, name), 'term': name,
        'vif': float(variance_inflation_factor(x, i))}
        for i, name in enumerate(model.model.exog_names) if name != 'Intercept'])


def run_analysis(people=None):
    if people is None: people, _ = build_frames()
    d, flow = prepare_data(people)
    if len(d) < 40 or d['mode'].nunique() < 3:
        raise ValueError('Analisis memerlukan minimal 40 kasus lengkap dan ketiga mode kerja.')
    models = {name: smf.ols(formula, d).fit() for name, formula in FORMULAS.items()}
    comparison = pd.DataFrame([{'model': name, 'n': int(m.nobs), 'k': int(m.df_model),
        'r2': m.rsquared, 'adj_r2': m.rsquared_adj, 'aic': m.aic,
        'se': np.sqrt(m.mse_resid), 'f': m.fvalue, 'p_f': m.f_pvalue,
        'df_resid': int(m.df_resid)} for name, m in models.items()])
    f, p, df = models['B'].compare_f_test(models['A'])
    # YearsCode hanya dipertahankan jika tambahan beralasan secara inferensial dan AIC.
    chosen = 'B' if p < .05 and models['B'].aic < models['A'].aic else 'A'
    m = models[chosen]
    influence = m.get_influence()
    cook, leverage = influence.cooks_distance[0], influence.hat_matrix_diag
    student = influence.resid_studentized_internal
    diag = pd.DataFrame({'response_id': d.response_id, 'fitted': m.fittedvalues,
        'residual': m.resid, 'residual_baku': student, 'leverage': leverage, 'cook': cook})
    flags = cook > 4/len(d)
    sens = smf.ols(FORMULAS[chosen], d.loc[~flags]).fit()
    sens_table = coef_table(m)[['term', 'b', 'p']].merge(
        coef_table(sens)[['term','b','p']], on='term', suffixes=('_utama','_sensitivitas'))
    # Profil representatif dengan nilai kontinu median, sama pada semua mode.
    profiles = pd.DataFrame({'work_exp': [d.work_exp.median()]*3,
        'log_gaji': [np.log(d.comp_yearly.median())]*3,
        'years_code': [d.years_code.median()]*3,
        'hybrid': [0,1,0], 'remote': [0,0,1]})
    prediction = m.get_prediction(profiles).summary_frame(alpha=.05)
    prediction.insert(0,'mode',['Kantor','Hybrid','Remote'])
    coefficients = {n: coef_table(v) for n,v in models.items()}
    for name, model in models.items():
        coefficients[name]['beta_std'] = [np.nan if t=='Intercept' else model.params[t]*d[t].std()/d.job_sat.std() for t in model.params.index]
    metadata = {'n_raw': len(people), 'n_complete': len(d), 'chosen': chosen,
        'partial_f': float(f), 'partial_p': float(p), 'partial_df': int(df),
        'delta_r2': models['B'].rsquared-models['A'].rsquared,
        'cook_threshold': 4/len(d), 'cook_count': int(flags.sum()), 'cook_max': float(cook.max()),
        'leverage_threshold': 2*len(m.params)/len(d),
        'leverage_count': int((leverage > 2*len(m.params)/len(d)).sum()),
        'student_count': int((abs(student)>3).sum()), 'sensitivity_n': int(sens.nobs),
        'prediction_work_exp': float(d.work_exp.median()), 'prediction_salary': float(d.comp_yearly.median()),
        'salary_double_points': float(m.params.log_gaji*np.log(2)),
        'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
        'statsmodels': statsmodels.__version__}
    return dict(data=d, flow=flow, models=models, comparison=comparison, coefficients=coefficients,
        vif={n:vif_table(v) for n,v in models.items()}, diagnostics=diag, sensitivity=sens_table,
        prediction=prediction, metadata=metadata)


def export_result(result, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for key in ['flow','comparison','diagnostics','sensitivity','prediction']:
        result[key].to_csv(output/f'{key}.csv', index=False)
    d = result['data']
    d[['job_sat','work_exp','years_code','comp_yearly','log_gaji']].describe().T.to_csv(output/'deskriptif.csv')
    d[['job_sat','work_exp','years_code','log_gaji','hybrid','remote']].corr().to_csv(output/'korelasi.csv')
    d.groupby('mode').job_sat.agg(['count','mean','std','median']).to_csv(output/'mode.csv')
    for name in result['models']:
        result['coefficients'][name].to_csv(output/f'koefisien_{name}.csv', index=False)
        result['vif'][name].to_csv(output/f'vif_{name}.csv', index=False)
        (output/f'model_{name}.txt').write_text(
            f'Model {name}: {FORMULAS[name]}\n'
            + result['comparison'].loc[result['comparison'].model == name].to_string(index=False)
            + '\n\nKoefisien OLS dan interval kepercayaan 95%\n'
            + result['coefficients'][name].to_string(index=False), encoding='utf-8')
    meta = result['metadata'].copy()
    meta['csv_sha256'] = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    (output/'ringkasan.json').write_text(json.dumps(meta, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Folder tujuan ekspor tabel hasil analisis')
    args = parser.parse_args()
    r = run_analysis(); export_result(r, args.output)
    print(r['comparison'].to_string(index=False))
    print(json.dumps(r['metadata'], indent=2))


if __name__ == '__main__': main()
