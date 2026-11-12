from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def fit_simple(data: pd.DataFrame, *, x_label: str, y_label: str,
               prediction_x: float, minimum_n: int = 3, scale: float = 1.0) -> dict:
    """Kolom masukan: id, x dan y; semua perhitungan memakai satuan asli."""
    d = data[['id', 'x', 'y']].copy()
    d[['x', 'y']] = d[['x', 'y']].apply(pd.to_numeric, errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y']).reset_index(drop=True)
    n = len(d)
    if n < minimum_n or d.x.nunique() < 2 or d.y.nunique() < 2:
        raise ValueError(f'Diperlukan minimal {minimum_n} pasangan lengkap dan variasi X serta Y.')
    design = sm.add_constant(d[['x']], has_constant='add')
    model = sm.OLS(d.y, design).fit()
    xbar, ybar = float(d.x.mean()), float(d.y.mean())
    dx, dy = d.x - xbar, d.y - ybar
    sxx, sxy, syy = float(dx @ dx), float(dx @ dy), float(dy @ dy)
    b1, b0 = sxy / sxx, ybar - (sxy / sxx) * xbar
    fitted, residual = model.fittedvalues, model.resid
    ssr, sse = float(((fitted - ybar)**2).sum()), float((residual**2).sum())
    se = float(np.sqrt(sse / (n - 2)))
    pearson, spearman = stats.pearsonr(d.x, d.y), stats.spearmanr(d.x, d.y)
    def correlation_t(value):
        return float(value * np.sqrt((n - 2) / (1 - value**2))) if abs(value) < 1 else float('inf')
    correlation = pd.DataFrame([
        {'metode': 'Pearson', 'koefisien': float(pearson.statistic),
         't': correlation_t(pearson.statistic), 'df': n - 2, 'p': float(pearson.pvalue)},
        {'metode': 'Spearman', 'koefisien': float(spearman.statistic),
         't': correlation_t(spearman.statistic), 'df': n - 2, 'p': float(spearman.pvalue)}])
    ci = model.conf_int()
    coefficients = pd.DataFrame([
        {'term': term, 'variabel': label, 'b': float(model.params[term]),
         'se': float(model.bse[term]), 't': float(model.tvalues[term]),
         'p': float(model.pvalues[term]), 'ci_low': float(ci.loc[term, 0]),
         'ci_high': float(ci.loc[term, 1])}
        for term, label in [('const', 'Konstanta'), ('x', x_label)]])
    observations = d.assign(dx=dx, dy=dy, dx_dy=dx*dy, dx2=dx**2, dy2=dy**2,
        prediksi=fitted, residual=residual, residual2=residual**2,
        residual_baku=residual/se, leverage=1/n + dx**2/sxx)
    prediction = model.get_prediction(pd.DataFrame({'const': [1.0], 'x': [prediction_x]})).summary_frame(alpha=.05)
    prediction.insert(0, 'x', prediction_x)
    flags = observations.residual_baku.abs() > 3
    sensitivity = None
    if flags.any() and (~flags).sum() >= 3 and d.loc[~flags, 'x'].nunique() >= 2:
        m2 = sm.OLS(d.loc[~flags, 'y'], design.loc[~flags]).fit()
        sensitivity = {'n': int(m2.nobs), 'dikeluarkan': int(flags.sum()),
            'b0': float(m2.params['const']), 'b1': float(m2.params['x']),
            'r2': float(m2.rsquared), 'p_slope': float(m2.pvalues['x'])}
    summary = {'n': n, 'x_label': x_label, 'y_label': y_label, 'plot_scale': scale,
        'sum_x': float(d.x.sum()), 'sum_y': float(d.y.sum()),
        'sum_x2': float((d.x**2).sum()), 'sum_y2': float((d.y**2).sum()),
        'sum_xy': float((d.x*d.y).sum()), 'xbar': xbar, 'ybar': ybar,
        'sxx': sxx, 'sxy': sxy, 'sst': syy, 'ssr': ssr, 'sse': sse,
        'b0': b0, 'b1': b1, 'r2': float(model.rsquared), 'adj_r2': float(model.rsquared_adj),
        'se': se, 'f': float(model.fvalue), 'p_f': float(model.f_pvalue), 'df_resid': n-2,
        't_critical': float(stats.t.ppf(.975, n-2)),
        'xmin': float(d.x.min()), 'xmax': float(d.x.max()), 'prediction_x': prediction_x,
        'residual_min': float(residual.min()), 'residual_max': float(residual.max()),
        'std_abs_max': float(observations.residual_baku.abs().max()),
        'std_above2': int((observations.residual_baku.abs()>2).sum()),
        'std_above3': int(flags.sum()), 'leverage_min': float(observations.leverage.min()),
        'leverage_max': float(observations.leverage.max()), 'leverage_threshold': 4/n,
        'leverage_count': int((observations.leverage>4/n).sum()), 'sensitivity': sensitivity}
    anova = pd.DataFrame([
        {'sumber': 'Regresi', 'SS': ssr, 'df': 1, 'MS': ssr, 'F': float(model.fvalue)},
        {'sumber': 'Residual', 'SS': sse, 'df': n-2, 'MS': sse/(n-2), 'F': np.nan},
        {'sumber': 'Total', 'SS': syy, 'df': n-1, 'MS': np.nan, 'F': np.nan}])
    return dict(data=d, model=model, summary=summary, coefficients=coefficients,
                correlation=correlation, observations=observations, prediction=prediction, anova=anova)


def run_survey(people=None, prediction_x: float = 15.0) -> dict:
    from database import build_frames
    if people is None:
        people, _ = build_frames()
    d = people.copy()
    keep = pd.Series(True, index=d.index)
    flow = []
    for label, mask in [
        ('Responden beridentitas valid', pd.Series(True, index=d.index)),
        ('Pengalaman kerja terisi dan tidak negatif', d.work_exp.notna() & (d.work_exp >= 0)),
        ('Gaji tahunan terisi', d.comp_yearly.notna()),
        ('Gaji minimal USD 1000', d.comp_yearly >= 1000)]:
        before = int(keep.sum()); keep &= mask
        flow.append({'tahap': label, 'tersisa': int(keep.sum()), 'gugur_tahap': before-int(keep.sum())})
    data = d.loc[keep, ['response_id', 'work_exp', 'comp_yearly']].rename(
        columns={'response_id': 'id', 'work_exp': 'x', 'comp_yearly': 'y'})
    result = fit_simple(data, x_label='Pengalaman kerja (tahun)',
        y_label='Gaji tahunan (USD)', prediction_x=prediction_x, minimum_n=30, scale=1000.0)
    result['flow'] = pd.DataFrame(flow)
    result['summary']['n_raw'] = len(people)
    return result


def export_result(result: dict, output, source=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name in ['data', 'coefficients', 'correlation', 'observations', 'prediction', 'anova']:
        result[name].to_csv(output / f'{name}.csv', index=False)
    if 'flow' in result:
        result['flow'].to_csv(output/'flow.csv', index=False)
    result['data'][['x', 'y']].describe().T.to_csv(output/'deskriptif.csv')
    summary = result['summary'].copy()
    if source:
        summary['source_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    (output/'ringkasan.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from database import CSV_PATH
    r = run_survey(); export_result(r, args.output, source=CSV_PATH)
    print(r['coefficients'].to_string(index=False))
    print(r['correlation'].to_string(index=False))


if __name__ == '__main__':
    main()
