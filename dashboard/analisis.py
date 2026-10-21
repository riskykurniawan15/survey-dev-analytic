from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from insights import MIN_SALARY

MODE_ORDER = ["Remote", "Hybrid", "Kantor"]
MODE_COLORS = {"Remote": "#38bdf8", "Hybrid": "#818cf8", "Kantor": "#fbbf24"}
SAT_BENCHMARK = 7.0
ALPHA = 0.05
COLORS = MODE_COLORS
ORDER = MODE_ORDER


@dataclass
class GroupSummary:
    nama: str
    n: int
    mean: float
    sd: float
    var: float
    median: float
    min: float
    max: float
    q1: float
    q3: float
    shapiro_w: float
    shapiro_p: float
    normal: bool
    n_outlier: int
    fence_low: float
    fence_high: float


def classify_mode(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text == "remote":
        return "Remote"
    if text == "in-person":
        return "Kantor"
    if "hybrid" in text or text.startswith("your choice"):
        return "Hybrid"
    return None


def prepare_frame(people: pd.DataFrame) -> pd.DataFrame:
    frame = people.copy()
    frame["mode_kerja"] = frame["remote_work"].map(classify_mode)
    frame["gaji_valid"] = frame["comp_yearly"].notna() & (frame["comp_yearly"] >= MIN_SALARY)
    frame["log_gaji"] = np.where(frame["gaji_valid"], np.log1p(frame["comp_yearly"]), np.nan)
    return frame


def satisfaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["job_sat"]).copy()


def mode_satisfaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    return satisfaction_frame(df).dropna(subset=["mode_kerja"]).copy()


def salary_mode_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["gaji_valid"] & df["mode_kerja"].notna()].copy()


def fmt_num(value: float, digits: int = 3) -> str:
    text = f"{float(value):,.{digits}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "< 0,001"
    return fmt_num(value, 3)


def interpret_effect_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "sangat kecil"
    if ad < 0.5:
        return "kecil"
    if ad < 0.8:
        return "sedang"
    return "besar"


def interpret_eta2(eta2: float) -> str:
    if eta2 < 0.01:
        return "sangat kecil"
    if eta2 < 0.06:
        return "kecil"
    if eta2 < 0.14:
        return "sedang"
    return "besar"


def _outliers_iqr(values: np.ndarray) -> tuple[int, float, float]:
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((values < low) | (values > high)).sum())
    return n_out, low, high


def _shapiro_safe(values: np.ndarray, seed: int = 7) -> tuple[float, float]:
    clean = values[~np.isnan(values)]
    if len(clean) < 3:
        return float("nan"), float("nan")
    sample = clean
    if len(clean) > 100:
        rng = np.random.default_rng(seed)
        sample = rng.choice(clean, size=100, replace=False)
    result = stats.shapiro(sample)
    return float(result.statistic), float(result.pvalue)


def summarize_group(name: str, values: np.ndarray) -> GroupSummary:
    clean = values[~np.isnan(values)].astype(float)
    sw_w, sw_p = _shapiro_safe(clean)
    n_out, low, high = _outliers_iqr(clean)
    return GroupSummary(
        nama=name,
        n=int(len(clean)),
        mean=float(clean.mean()),
        sd=float(clean.std(ddof=1)),
        var=float(clean.var(ddof=1)),
        median=float(np.median(clean)),
        min=float(clean.min()),
        max=float(clean.max()),
        q1=float(np.percentile(clean, 25)),
        q3=float(np.percentile(clean, 75)),
        shapiro_w=sw_w,
        shapiro_p=sw_p,
        normal=bool(sw_p >= ALPHA) if not np.isnan(sw_p) else False,
        n_outlier=n_out,
        fence_low=low,
        fence_high=high,
    )


def cohens_d_one_sample(values: np.ndarray, mu0: float) -> float:
    return float((values.mean() - mu0) / values.std(ddof=1))


def cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled)


def eta_squared_anova(groups: list[np.ndarray]) -> float:
    all_values = np.concatenate(groups)
    grand = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((all_values - grand) ** 2).sum())
    return float(ss_between / ss_total) if ss_total else 0.0


def qq_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_vals = np.sort(values[~np.isnan(values)])
    n = len(sorted_vals)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return theoretical, sorted_vals


def descriptive_by_mode(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for name in ORDER:
        values = frame.loc[frame["mode_kerja"] == name, column].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        rows.append(asdict(summarize_group(name, values)))
    return pd.DataFrame(rows)


def one_sample_satisfaction(df: pd.DataFrame) -> dict | None:
    values = satisfaction_frame(df)["job_sat"].to_numpy(dtype=float)
    if len(values) < 3:
        return None
    result = stats.ttest_1samp(values, SAT_BENCHMARK)
    mean = float(values.mean())
    n = len(values)
    se = values.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(1 - ALPHA, df=n - 1, loc=mean, scale=se)
    d = cohens_d_one_sample(values, SAT_BENCHMARK)
    summary = summarize_group("Semua", values)
    return {
        "uji": "One-sample t-test (dua sisi)",
        "variabel": "Kepuasan kerja (0–10)",
        "h0": f"μ = {SAT_BENCHMARK:g}",
        "h1": f"μ ≠ {SAT_BENCHMARK:g}",
        "benchmark": SAT_BENCHMARK,
        "t": float(result.statistic),
        "df": int(result.df) if hasattr(result, "df") and result.df is not None else n - 1,
        "p": float(result.pvalue),
        "mean": mean,
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)),
        "n": n,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "cohen_d": d,
        "tolak_h0": bool(result.pvalue < ALPHA),
        "keputusan": "Tolak H0" if result.pvalue < ALPHA else "Gagal menolak H0",
        "ringkas": asdict(summary),
    }


def independent_remote_kantor(df: pd.DataFrame) -> dict | None:
    frame = mode_satisfaction_frame(df)
    remote = frame.loc[frame["mode_kerja"] == "Remote", "job_sat"].to_numpy(dtype=float)
    kantor = frame.loc[frame["mode_kerja"] == "Kantor", "job_sat"].to_numpy(dtype=float)
    if len(remote) < 3 or len(kantor) < 3:
        return None
    levene = stats.levene(remote, kantor, center="median")
    equal_var = bool(levene.pvalue >= ALPHA)
    student = stats.ttest_ind(remote, kantor, equal_var=True)
    welch = stats.ttest_ind(remote, kantor, equal_var=False)
    chosen = student if equal_var else welch
    d = cohens_d_independent(remote, kantor)
    return {
        "uji": "Independent t-test (dua sisi)" if equal_var else "Welch t-test (dua sisi)",
        "variabel": "Kepuasan kerja",
        "h0": "μ_Remote = μ_Kantor",
        "h1": "μ_Remote ≠ μ_Kantor",
        "levene_stat": float(levene.statistic),
        "levene_p": float(levene.pvalue),
        "varian_homogen": equal_var,
        "t_student": float(student.statistic),
        "p_student": float(student.pvalue),
        "t_welch": float(welch.statistic),
        "p_welch": float(welch.pvalue),
        "t": float(chosen.statistic),
        "df": float(chosen.df) if hasattr(chosen, "df") and chosen.df is not None else float(len(remote) + len(kantor) - 2),
        "p": float(chosen.pvalue),
        "mean_remote": float(remote.mean()),
        "mean_kantor": float(kantor.mean()),
        "n_remote": int(len(remote)),
        "n_kantor": int(len(kantor)),
        "selisih": float(remote.mean() - kantor.mean()),
        "cohen_d": d,
        "tolak_h0": bool(chosen.pvalue < ALPHA),
        "keputusan": "Tolak H0" if chosen.pvalue < ALPHA else "Gagal menolak H0",
    }


def anova_mode_satisfaction(df: pd.DataFrame) -> dict | None:
    frame = mode_satisfaction_frame(df)
    groups = [frame.loc[frame["mode_kerja"] == name, "job_sat"].to_numpy(dtype=float) for name in ORDER]
    if any(len(group) < 3 for group in groups):
        return None
    levene = stats.levene(*groups, center="median")
    equal_var = bool(levene.pvalue >= ALPHA)
    anova = stats.f_oneway(*groups)
    welch = stats.alexandergovern(*groups)
    kw = stats.kruskal(*groups)
    eta2 = eta_squared_anova(groups)
    ratios = {
        f"{ORDER[i]}/{ORDER[j]}": float(groups[i].var(ddof=1) / groups[j].var(ddof=1))
        for i, j in [(0, 1), (0, 2), (1, 2)]
    }
    primary_p = float(anova.pvalue if equal_var else welch.pvalue)
    return {
        "uji": "One-way ANOVA" if equal_var else "Welch ANOVA (Alexander-Govern)",
        "variabel": "Kepuasan kerja",
        "h0": "μ_Remote = μ_Hybrid = μ_Kantor",
        "h1": "Minimal satu rata-rata berbeda",
        "levene_stat": float(levene.statistic),
        "levene_p": float(levene.pvalue),
        "varian_homogen": equal_var,
        "rasio_varian": ratios,
        "F": float(anova.statistic),
        "p_anova": float(anova.pvalue),
        "welch_stat": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "kruskal_h": float(kw.statistic),
        "kruskal_p": float(kw.pvalue),
        "p": primary_p,
        "eta2": eta2,
        "n_total": int(len(frame)),
        "tolak_h0": bool(primary_p < ALPHA),
        "keputusan": "Tolak H0" if primary_p < ALPHA else "Gagal menolak H0",
    }


def tukey_mode_satisfaction(df: pd.DataFrame) -> pd.DataFrame | None:
    frame = mode_satisfaction_frame(df)
    if frame["mode_kerja"].nunique() < 2:
        return None
    counts = frame["mode_kerja"].value_counts()
    if (counts < 3).any():
        return None
    result = pairwise_tukeyhsd(frame["job_sat"], frame["mode_kerja"], alpha=ALPHA)
    table = pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0])
    table.columns = ["kelompok1", "kelompok2", "selisih", "p_adj", "batas_bawah", "batas_atas", "tolak"]
    for col in ["selisih", "p_adj", "batas_bawah", "batas_atas"]:
        table[col] = table[col].astype(float)
    table["tolak"] = table["tolak"].astype(bool)
    return table


def salary_anova_log(df: pd.DataFrame) -> dict | None:
    frame = salary_mode_frame(df)
    groups = [frame.loc[frame["mode_kerja"] == name, "log_gaji"].to_numpy(dtype=float) for name in ORDER]
    if any(len(group) < 3 for group in groups):
        return None
    levene = stats.levene(*groups, center="median")
    equal_var = bool(levene.pvalue >= ALPHA)
    anova = stats.f_oneway(*groups)
    welch = stats.alexandergovern(*groups)
    eta2 = eta_squared_anova(groups)
    medians = {name: float(frame.loc[frame["mode_kerja"] == name, "comp_yearly"].median()) for name in ORDER}
    means = {name: float(frame.loc[frame["mode_kerja"] == name, "comp_yearly"].mean()) for name in ORDER}
    counts = {name: int((frame["mode_kerja"] == name).sum()) for name in ORDER}
    primary_p = float(anova.pvalue if equal_var else welch.pvalue)
    tukey = pairwise_tukeyhsd(frame["log_gaji"], frame["mode_kerja"], alpha=ALPHA)
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tukey_df.columns = ["kelompok1", "kelompok2", "selisih", "p_adj", "batas_bawah", "batas_atas", "tolak"]
    for col in ["selisih", "p_adj", "batas_bawah", "batas_atas"]:
        tukey_df[col] = tukey_df[col].astype(float)
    tukey_df["tolak"] = tukey_df["tolak"].astype(bool)
    return {
        "uji": "One-way ANOVA pada log(gaji+1)" if equal_var else "Welch ANOVA pada log(gaji+1)",
        "levene_p": float(levene.pvalue),
        "varian_homogen": equal_var,
        "F": float(anova.statistic),
        "p_anova": float(anova.pvalue),
        "welch_p": float(welch.pvalue),
        "p": primary_p,
        "eta2": eta2,
        "median_usd": medians,
        "mean_usd": means,
        "n": counts,
        "tukey": tukey_df,
        "tolak_h0": bool(primary_p < ALPHA),
        "keputusan": "Tolak H0" if primary_p < ALPHA else "Gagal menolak H0",
    }


def run_all(people: pd.DataFrame) -> dict:
    base = prepare_frame(people)
    sat_mode = mode_satisfaction_frame(base)
    one = one_sample_satisfaction(base)
    two = independent_remote_kantor(base)
    three = anova_mode_satisfaction(base)
    tukey = tukey_mode_satisfaction(base)
    desc = descriptive_by_mode(sat_mode, "job_sat")
    salary = salary_anova_log(base)
    return {
        "data": base,
        "data_mode_sat": sat_mode,
        "deskriptif": desc,
        "satu_sampel": one,
        "dua_kelompok": two,
        "anova": three,
        "tukey": tukey,
        "gaji": salary,
        "alpha": ALPHA,
        "benchmark": SAT_BENCHMARK,
        "asumsi_catatan": (
            "n besar sehingga CLT menopang uji-t/ANOVA; Shapiro dihitung pada sampel acak 100. "
            "Jika Levene menolak homogenitas, uji utama memakai Welch."
        ),
        "n_responden": int(len(base)),
        "n_kepuasan": int(base["job_sat"].notna().sum()),
        "n_mode_sat": int(len(sat_mode)),
    }
