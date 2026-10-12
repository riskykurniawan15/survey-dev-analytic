from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from insights import valid_salary

AGE_ORDER = [
    "18-24 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    "65 years or older",
    "Prefer not to say",
]
REMOTE_ORDER = [
    ("Remote", "Remote"),
    ("Hybrid (some in-person, leans heavy to flexibility)", "Hybrid, lebih fleksibel"),
    ("Hybrid (some remote, leans heavy to in-person)", "Hybrid, lebih kantor"),
    ("Your choice (very flexible, you can come in when you want or just as needed)", "Bebas pilih"),
    ("In-person", "Kantor"),
]

REPORT_CHARTS = [
    ("01_usia.png", "Gambar 1. Usia"),
    ("02_histogram_pengalaman.png", "Gambar 2. Pengalaman"),
    ("03_boxplot_gaji.png", "Gambar 3. Gaji"),
    ("04_scatter_pengalaman_gaji.png", "Gambar 4. Sebaran"),
    ("05_pengalaman_peran.png", "Gambar 5. Peran"),
]


def fmt_int(value) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def label_age(value) -> str:
    if not isinstance(value, str):
        return "Tidak diisi"
    if value.startswith("Prefer"):
        return "Tidak menjawab"
    return value.replace(" years old", "").replace(" years or older", "+")


def label_role(value) -> str:
    if not isinstance(value, str):
        return "Tidak diisi"
    role = value.split(", ", 1)[1] if value.lower().startswith("developer, ") else value
    names = {
        "front-end": "Front-end",
        "back-end": "Back-end",
        "full-stack": "Full-stack",
        "mobile": "Mobile",
        "desktop or enterprise applications": "Aplikasi desktop",
        "embedded applications or devices": "Aplikasi tertanam",
        "Student": "Mahasiswa",
        "Engineering manager": "Manajer rekayasa",
        "Architect, software or solutions": "Arsitek perangkat lunak",
        "Academic researcher": "Peneliti akademik",
        "Other (please specify):": "Lainnya",
    }
    return names.get(role, role[:1].upper() + role[1:] if role else role)


def style_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "Segoe UI",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_age(people: pd.DataFrame):
    style_plots()
    counts = people["age"].dropna().value_counts()
    labels, values = [], []
    for key in AGE_ORDER:
        if key in counts.index:
            labels.append(label_age(key))
            values.append(int(counts[key]))
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    bars = ax.bar(labels, values, color="#1d4ed8")
    ax.set_title("Distribusi kelompok usia responden")
    ax.set_xlabel("Kelompok usia")
    ax.set_ylabel("Jumlah responden")
    gap = 8 if max(values) >= 80 else max(values) * 0.08
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + gap, fmt_int(value), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(max(values) * 1.14, max(values) + gap * 1.4))
    return fig


def figure_hist(exp: pd.Series):
    style_plots()
    clean = exp.dropna()
    if clean.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.hist(clean, bins=range(0, 66, 5), color="#0f766e", edgecolor="white")
    ax.set_title("Distribusi pengalaman kerja")
    ax.set_xlabel("Pengalaman kerja (tahun)")
    ax.set_ylabel("Jumlah responden")
    return fig


def figure_box(paid: pd.DataFrame):
    style_plots()
    labels, data = [], []
    above = 0
    for raw, label in REMOTE_ORDER:
        part = paid.loc[paid["remote_work"] == raw, "comp_yearly"]
        if part.empty:
            continue
        above += int((part > 400_000).sum())
        labels.append(label)
        data.append(part)
    if not data:
        return None, above
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#bfdbfe")
        patch.set_edgecolor("#1e3a8a")
    ax.set_title("Sebaran gaji tahunan menurut mode kerja")
    ax.set_xlabel("Mode kerja")
    ax.set_ylabel("Gaji tahunan (USD)")
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylim(0, 400_000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: fmt_int(value)))
    return fig, above


def figure_scatter(paid: pd.DataFrame):
    style_plots()
    pair = paid.dropna(subset=["work_exp", "comp_yearly"])
    hidden = int((pair["comp_yearly"] > 400_000).sum()) if not pair.empty else 0
    shown = pair[pair["comp_yearly"] <= 400_000]
    if shown.empty:
        return None, hidden
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.scatter(shown["work_exp"], shown["comp_yearly"], s=16, alpha=0.45, color="#0369a1", edgecolors="none")
    ax.set_title("Pengalaman kerja dan gaji tahunan")
    ax.set_xlabel("Pengalaman kerja (tahun)")
    ax.set_ylabel("Gaji tahunan (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: fmt_int(value)))
    return fig, hidden


def figure_role(people: pd.DataFrame):
    style_plots()
    subset = people.dropna(subset=["dev_type", "work_exp"])
    if subset.empty:
        return None
    counts = subset.groupby("dev_type").size()
    keep = counts[counts >= 40].index
    med = (
        subset[subset["dev_type"].isin(keep)]
        .groupby("dev_type")["work_exp"]
        .median()
        .sort_values(ascending=True)
    )
    if med.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.barh([label_role(name) for name in med.index], med.values, color="#047857")
    ax.set_title("Median pengalaman kerja menurut peran")
    ax.set_xlabel("Median pengalaman kerja (tahun)")
    ax.set_ylabel("Peran")
    return fig


def chart_age(people: pd.DataFrame, path: Path) -> Path | None:
    fig = figure_age(people)
    return None if fig is None else save_figure(fig, path)


def chart_hist(exp: pd.Series, path: Path) -> Path | None:
    fig = figure_hist(exp)
    return None if fig is None else save_figure(fig, path)


def chart_box(paid: pd.DataFrame, path: Path) -> tuple[Path | None, int]:
    fig, above = figure_box(paid)
    if fig is None:
        return None, above
    return save_figure(fig, path), above


def chart_scatter(paid: pd.DataFrame, path: Path) -> tuple[Path | None, int]:
    fig, hidden = figure_scatter(paid)
    if fig is None:
        return None, hidden
    return save_figure(fig, path), hidden


def chart_role(people: pd.DataFrame, path: Path) -> Path | None:
    fig = figure_role(people)
    return None if fig is None else save_figure(fig, path)


def report_pngs(people: pd.DataFrame) -> list[tuple[str, str, bytes | None]]:
    paid = valid_salary(people)
    figures = [
        figure_age(people),
        figure_hist(people["work_exp"]),
        figure_box(paid)[0],
        figure_scatter(paid)[0],
        figure_role(people),
    ]
    packaged = []
    for (filename, label), fig in zip(REPORT_CHARTS, figures):
        packaged.append((filename, label, None if fig is None else _png(fig)))
    return packaged
