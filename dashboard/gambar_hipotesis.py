from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analisis import MODE_COLORS, ORDER, SAT_BENCHMARK, qq_points

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#0f172a",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": "#0f172a",
        "grid.color": "#cbd5e1",
        "grid.alpha": 0.55,
    }
)


def _save(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    data = buf.getvalue()
    plt.close(fig)
    return data


def chart_boxplot(result: dict) -> bytes | None:
    frame = result["data_mode_sat"]
    present = [name for name in ORDER if (frame["mode_kerja"] == name).any()]
    if not present:
        return None
    data = [frame.loc[frame["mode_kerja"] == name, "job_sat"].to_numpy() for name in present]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bp = ax.boxplot(
        data,
        tick_labels=present,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="#0f172a", linewidth=1.6),
        whiskerprops=dict(color="#475569"),
        capprops=dict(color="#475569"),
        flierprops=dict(
            marker="o",
            markerfacecolor="#fb7185",
            markersize=4,
            alpha=0.45,
            markeredgecolor="#fb7185",
        ),
    )
    for patch, name in zip(bp["boxes"], present):
        patch.set_facecolor(MODE_COLORS[name])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#1e293b")
    ax.axhline(SAT_BENCHMARK, color="#f43f5e", linestyle="--", linewidth=1.4, label=f"Benchmark {SAT_BENCHMARK:g}")
    ax.set_title("Distribusi kepuasan kerja menurut mode kerja")
    ax.set_ylabel("Kepuasan kerja (0–10)")
    ax.set_xlabel("Mode kerja")
    ax.grid(axis="y", linestyle=":")
    ax.legend(loc="lower right", frameon=False)
    return _save(fig)


def chart_means(result: dict) -> bytes | None:
    desc = result["deskriptif"]
    if desc.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(desc))
    bars = ax.bar(
        x,
        desc["mean"],
        yerr=desc["sd"],
        capsize=5,
        color=[MODE_COLORS[n] for n in desc["nama"]],
        edgecolor="#1e293b",
        alpha=0.85,
    )
    ax.axhline(SAT_BENCHMARK, color="#f43f5e", linestyle="--", linewidth=1.4, label=f"Benchmark {SAT_BENCHMARK:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(desc["nama"])
    ax.set_ylabel("Rata-rata kepuasan kerja")
    ax.set_xlabel("Mode kerja")
    ax.set_title("Rata-rata dan simpangan baku kepuasan kerja")
    for bar, mean in zip(bars, desc["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 10.5)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle=":")
    return _save(fig)


def chart_qq(result: dict) -> bytes | None:
    frame = result["data_mode_sat"]
    present = [name for name in ORDER if (frame["mode_kerja"] == name).sum() >= 3]
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present), figsize=(9.5, 3.4), squeeze=False)
    for ax, name in zip(axes[0], present):
        values = frame.loc[frame["mode_kerja"] == name, "job_sat"].to_numpy(dtype=float)
        if len(values) > 120:
            values = np.random.default_rng(11).choice(values, size=120, replace=False)
        theo, obs = qq_points(values)
        ax.scatter(theo, obs, color=MODE_COLORS[name], s=18, alpha=0.75, zorder=3)
        slope, intercept, _, _, _ = stats.linregress(theo, obs)
        line_x = np.array([theo.min(), theo.max()])
        ax.plot(line_x, intercept + slope * line_x, color="#64748b", linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel("Kuantil normal")
        ax.set_ylabel("Sampel")
        ax.grid(True, linestyle=":")
    fig.suptitle("Plot Q–Q kepuasan kerja per mode", y=1.02)
    fig.tight_layout()
    return _save(fig)


def chart_tukey(result: dict) -> bytes | None:
    tukey = result["tukey"]
    if tukey is None or tukey.empty:
        return None
    labels = [f"{a} vs {b}" for a, b in zip(tukey["kelompok1"], tukey["kelompok2"])]
    means = tukey["selisih"].to_numpy()
    lows = tukey["batas_bawah"].to_numpy()
    highs = tukey["batas_atas"].to_numpy()
    reject = tukey["tolak"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = np.arange(len(labels))
    colors = ["#34d399" if flag else "#94a3b8" for flag in reject]
    ax.errorbar(
        means,
        y,
        xerr=[means - lows, highs - means],
        fmt="o",
        color="#0f172a",
        ecolor="#64748b",
        elinewidth=1.4,
        capsize=4,
        markersize=6,
    )
    for yi, mean, color in zip(y, means, colors):
        ax.scatter([mean], [yi], color=color, s=55, zorder=4)
    ax.axvline(0, color="#f43f5e", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Selisih rata-rata kepuasan")
    ax.set_title("Selisih rata-rata kepuasan antar mode kerja")
    ax.grid(axis="x", linestyle=":")
    return _save(fig)


def chart_salary_medians(result: dict) -> bytes | None:
    gaji = result["gaji"]
    if not gaji:
        return None
    medians = [gaji["median_usd"][name] for name in ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(ORDER, medians, color=[MODE_COLORS[n] for n in ORDER], edgecolor="#1e293b", alpha=0.85)
    ax.set_ylabel("Median gaji tahunan (USD)")
    ax.set_xlabel("Mode kerja")
    ax.set_title("Median gaji tahunan menurut mode kerja")
    for bar, value in zip(bars, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1200,
            f"{int(value):,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", linestyle=":")
    return _save(fig)


def report_pngs(result: dict) -> list[tuple[str, str, bytes | None]]:
    return [
        ("01_boxplot.png", "Boxplot kepuasan", chart_boxplot(result)),
        ("02_mean_sd.png", "Mean ± SD kepuasan", chart_means(result)),
        ("03_qqplot.png", "Q-Q plot", chart_qq(result)),
        ("04_tukey.png", "Tukey HSD", chart_tukey(result)),
        ("05_salary.png", "Median gaji", chart_salary_medians(result)),
    ]
