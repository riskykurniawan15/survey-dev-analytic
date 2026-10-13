from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import CSV_PATH, DB_PATH, ensure_database, load_tables
from gambar_laporan import label_role, report_pngs
from insights import (
    MIN_SALARY,
    build_brief,
    eta_squared,
    experience_by_role,
    role_experience_note,
    skill_ranking,
    valid_salary,
)

AGE_ORDER = [
    "18-24 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    "65 years or older",
    "Prefer not to say",
]
AI_USE_ORDER = [
    "Yes, I use AI tools daily",
    "Yes, I use AI tools weekly",
    "Yes, I use AI tools monthly or infrequently",
    "No, but I plan to soon",
    "No, and I don't plan to",
]
SENTIMENT_ORDER = [
    "Very favorable",
    "Favorable",
    "Indifferent",
    "Unsure",
    "Unfavorable",
    "Very unfavorable",
]
TRUST_ORDER = [
    "Highly trust",
    "Somewhat trust",
    "Neither trust nor distrust",
    "Somewhat distrust",
    "Highly distrust",
]
REMOTE_ORDER = [
    "Remote",
    "Hybrid (some in-person, leans heavy to flexibility)",
    "Hybrid (some remote, leans heavy to in-person)",
    "Your choice (very flexible, you can come in when you want or just as needed)",
    "In-person",
]

ACCENT = "#38bdf8"
PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#fb7185", "#a78bfa", "#22d3ee"]


def as_text(value) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(round(value)):,}".replace(",", ".")


def years(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1f}".replace(".", ",")


def label_age(value: str | None) -> str:
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    if value.startswith("Prefer"):
        return "Tidak menjawab"
    return value.replace(" years old", "").replace(" years or older", "+")


def label_ed(value: str | None) -> str:
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    text = value.lower()
    if text.startswith("bachelor"):
        return "Sarjana (S1)"
    if text.startswith("master"):
        return "Magister (S2)"
    if text.startswith("professional"):
        return "Gelar profesional / S3"
    if text.startswith("associate"):
        return "Diploma"
    if text.startswith("some college"):
        return "Kuliah tanpa gelar"
    if text.startswith("secondary"):
        return "Sekolah menengah"
    if text.startswith("primary"):
        return "Sekolah dasar"
    return "Lainnya"


def label_employment(value: str | None) -> str:
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    text = value.lower()
    if text == "employed":
        return "Karyawan"
    if text.startswith("independent"):
        return "Freelancer"
    if text == "student":
        return "Mahasiswa"
    if text.startswith("not employed"):
        return "Tidak bekerja"
    if text == "retired":
        return "Pensiun"
    return "Tidak menjawab"


def label_remote(value: str | None) -> str:
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    text = value.lower()
    if text == "remote":
        return "Remote"
    if text == "in-person":
        return "Kantor"
    if "flexibility" in text:
        return "Hybrid, lebih fleksibel"
    if "in-person" in text:
        return "Hybrid, lebih kantor"
    if text.startswith("your choice"):
        return "Bebas pilih"
    return value


def label_ai_use(value: str | None) -> str:
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    text = value.lower()
    if "daily" in text:
        return "Setiap hari"
    if "weekly" in text:
        return "Setiap minggu"
    if "monthly" in text:
        return "Jarang"
    if "plan to" in text:
        return "Belum, berencana"
    if "don't plan" in text or "do not plan" in text:
        return "Tidak berencana"
    return value


def label_sentiment(value: str | None) -> str:
    mapping = {
        "Very favorable": "Sangat positif",
        "Favorable": "Positif",
        "Indifferent": "Netral",
        "Unsure": "Tidak yakin",
        "Unfavorable": "Negatif",
        "Very unfavorable": "Sangat negatif",
    }
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    return mapping.get(value, value)


def label_trust(value: str | None) -> str:
    mapping = {
        "Highly trust": "Sangat percaya",
        "Somewhat trust": "Cukup percaya",
        "Neither trust nor distrust": "Netral",
        "Somewhat distrust": "Cukup tidak percaya",
        "Highly distrust": "Sangat tidak percaya",
    }
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    return mapping.get(value, value)


def label_threat(value: str | None) -> str:
    mapping = {"Yes": "Ya", "No": "Tidak", "I'm not sure": "Tidak yakin"}
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    return mapping.get(value, value)


def label_new_role(value: str | None) -> str:
    value = as_text(value)
    if not value:
        return "Tidak diisi"
    text = value.lower()
    if "neither" in text:
        return "Belum mempertimbangkan"
    if "somewhat" in text:
        return "Sedikit mempertimbangkan"
    if "strongly" in text:
        return "Sangat mempertimbangkan"
    if "voluntarily" in text:
        return "Sudah pindah, sukarela"
    if "involuntarily" in text:
        return "Sudah pindah, terpaksa"
    return value


def ordered_counts(series: pd.Series, order: list[str], labeler) -> pd.DataFrame:
    clean = series.dropna()
    counts = clean.value_counts()
    rows = []
    seen = set()
    for key in order:
        if key in counts.index:
            rows.append({"label": labeler(key), "jumlah": int(counts[key])})
            seen.add(key)
    for key, jumlah in counts.items():
        if key not in seen:
            rows.append({"label": labeler(key), "jumlah": int(jumlah)})
    return pd.DataFrame(rows)


def style_figure(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Segoe UI, sans-serif", size=13),
        margin=dict(l=8, r=72, t=52, b=8),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.28,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.16)", zeroline=False, title=None, automargin=True)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", zeroline=False, title=None, automargin=True)
    return fig


def bar_chart(frame: pd.DataFrame, title: str, color: str = ACCENT, horizontal: bool = True) -> go.Figure:
    labels = frame["label"].tolist()
    values = frame["jumlah"].tolist()
    height = max(340, 36 * len(labels) + 90) if horizontal else 380
    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=color),
                text=values,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>%{x} responden<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=color),
                text=values,
                textposition="auto",
                hovertemplate="%{x}<br>%{y} responden<extra></extra>",
            )
        )
    return style_figure(fig, title, height=height)


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1240px;}
          h1 {font-weight: 650; letter-spacing: -0.03em; margin-bottom: 0.2rem;}
          .kpis {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
            gap: 0.75rem;
            margin: 0.2rem 0 0.4rem;
          }
          .kpi {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.8rem 0.9rem 0.75rem;
            min-width: 0;
          }
          .kpi span {display: block; color: #94a3b8; font-size: 0.8rem; line-height: 1.3; white-space: nowrap;}
          .kpi strong {
            display: block;
            margin: 0.2rem 0 0.15rem;
            font-size: 1.4rem;
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
            color: #f8fafc;
            white-space: nowrap;
          }
          .kpi em {display: block; color: #7dd3fc; font-style: normal; font-size: 0.75rem;}
          .panel {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.95rem 1rem 1rem;
            margin-bottom: 0.7rem;
            height: 100%;
          }
          .panel h3 {margin: 0 0 0.4rem; font-size: 1rem; color: #7dd3fc;}
          .panel p {margin: 0; color: #e2e8f0; line-height: 1.5; font-size: 0.94rem;}
          div[data-testid="stTabs"] button {font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def cards(items: list[dict[str, str]]) -> None:
    if not items:
        st.info("Belum cukup data untuk bagian ini.")
        return
    for start in range(0, len(items), 2):
        columns = st.columns(2)
        for column, item in zip(columns, items[start : start + 2]):
            with column:
                st.markdown(
                    f"<div class='panel'><h3>{item['title']}</h3><p>{item['body']}</p></div>",
                    unsafe_allow_html=True,
                )


@st.cache_data(show_spinner=False)
def load_data(csv_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    del csv_mtime
    ensure_database()
    return load_tables()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filter")
    st.sidebar.caption("Kosong berarti semua nilai pada kolom itu ikut dihitung.")

    countries = sorted(df["country"].dropna().unique())
    ages = [age for age in AGE_ORDER if age in set(df["age"].dropna())]
    jobs = sorted(df["employment"].dropna().unique(), key=label_employment)
    remotes = [item for item in REMOTE_ORDER if item in set(df["remote_work"].dropna())]

    picked_country = st.sidebar.multiselect("Negara", countries, format_func=lambda v: v, placeholder="Semua negara")
    picked_age = st.sidebar.multiselect("Usia", ages, format_func=label_age, placeholder="Semua usia")
    picked_job = st.sidebar.multiselect(
        "Status kerja", jobs, format_func=label_employment, placeholder="Semua status"
    )
    picked_remote = st.sidebar.multiselect(
        "Mode kerja", remotes, format_func=label_remote, placeholder="Semua mode"
    )

    filtered = df
    if picked_country:
        filtered = filtered[filtered["country"].isin(picked_country)]
    if picked_age:
        filtered = filtered[filtered["age"].isin(picked_age)]
    if picked_job:
        filtered = filtered[filtered["employment"].isin(picked_job)]
    if picked_remote:
        filtered = filtered[filtered["remote_work"].isin(picked_remote)]
    return filtered


def show_metrics(filtered: pd.DataFrame, total: int) -> None:
    paid = valid_salary(filtered)
    salary = paid["comp_yearly"].median() if not paid.empty else None
    experience = filtered["work_exp"].median()
    satisfaction = filtered["job_sat"].mean()
    ai = filtered["ai_select"].dropna()
    daily = 100 * (ai == "Yes, I use AI tools daily").mean() if not ai.empty else None

    share = 100 * len(filtered) / total if total else 0
    sat_text = f"{satisfaction:.2f}".replace(".", ",") if pd.notna(satisfaction) else "—"
    ai_text = f"{daily:.1f}%".replace(".", ",") if daily is not None else "—"
    items = [
        ("Responden", f"{len(filtered):,}".replace(",", "."), f"{share:.0f}% dari total"),
        ("Gaji median", money(salary), "USD per tahun"),
        ("Pengalaman", years(experience) if pd.notna(experience) else "—", "tahun, nilai median"),
        ("Kepuasan", sat_text, "rata-rata, skala 0–10"),
        ("AI tiap hari", ai_text, "dari yang menjawab"),
    ]
    cards_html = "".join(
        f"<div class='kpi'><span>{label}</span><strong>{value}</strong><em>{note}</em></div>"
        for label, value, note in items
    )
    st.markdown(f"<div class='kpis'>{cards_html}</div>", unsafe_allow_html=True)


def chart_salary_by(df: pd.DataFrame, column: str, labeler, title: str, limit: int = 12) -> go.Figure | None:
    paid = valid_salary(df)
    if paid.empty:
        return None
    grouped = (
        paid.dropna(subset=[column])
        .groupby(column)["comp_yearly"]
        .agg(median="median", jumlah="size")
        .reset_index()
    )
    grouped = grouped[grouped["jumlah"] >= 8].sort_values("median", ascending=False).head(limit)
    if grouped.empty:
        return None
    grouped["label"] = grouped[column].map(labeler)
    fig = go.Figure(
        go.Bar(
            x=grouped["median"],
            y=grouped["label"],
            orientation="h",
            marker=dict(color="#818cf8"),
            text=[money(v) for v in grouped["median"]],
            textposition="outside",
            cliponaxis=False,
            customdata=grouped["jumlah"],
            hovertemplate="%{y}<br>Median USD %{x:,.0f}<br>%{customdata} responden<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, title, height=max(340, 36 * len(grouped) + 90))


def chart_experience_by_role(profile: pd.DataFrame) -> go.Figure:
    labeled = profile.copy()
    labeled["label"] = labeled["dev_type"].map(label_role)
    fig = go.Figure(
        go.Bar(
            x=labeled["median_pengalaman"],
            y=labeled["label"],
            orientation="h",
            marker=dict(color="#34d399"),
            text=[years(value) for value in labeled["median_pengalaman"]],
            textposition="outside",
            cliponaxis=False,
            customdata=labeled["responden"],
            hovertemplate="%{y}<br>Median %{x:.1f} tahun<br>%{customdata} responden<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, "Median pengalaman menurut peran", height=max(340, 36 * len(labeled) + 90))


def render_role_link(filtered: pd.DataFrame) -> None:
    min_people = 40
    profile = experience_by_role(filtered, min_people=min_people)
    if profile.empty:
        st.info("Peran dengan jumlah responden cukup belum tersedia pada filter ini.")
        return

    top = profile.iloc[0]
    st.caption(
        f"Batang teratas pada filter ini: {label_role(top['dev_type'])}, "
        f"median {years(top['median_pengalaman'])} tahun ({int(top['responden'])} orang). "
        f"Grafik ini memakai ambang yang sama dengan gambar laporan, yaitu minimal {min_people} responden."
    )
    association, sample_size = eta_squared(filtered, "dev_type", "work_exp")
    st.plotly_chart(chart_experience_by_role(profile), width="stretch", config={"displayModeBar": False})
    note = role_experience_note(filtered, min_people=min_people)
    if note:
        st.caption(note)
    if association is None:
        st.caption("Data peran dan pengalaman pada filter ini belum cukup untuk mengukur kekuatannya.")
    else:
        strength = "lemah" if association < 0.06 else "sedang" if association < 0.14 else "kuat"
        eta_text = f"{association:.2f}".replace(".", ",")
        st.caption(
            f"Kaitan peran dengan pengalaman {strength} (η² = {eta_text}, n = {sample_size}). "
            "Nilai mendekati 0 berarti lama bekerja mirip di semua peran. "
            "Nilai mendekati 1 berarti peran sangat membedakan pengalaman."
        )

    table = pd.DataFrame(
        {
            "Peran": profile["dev_type"].map(label_role),
            "Responden": profile["responden"],
            "Median pengalaman (tahun)": profile["median_pengalaman"].map(years),
            "Median gaji (USD)": profile["median_gaji"].map(money),
            "Korelasi pengalaman–gaji": profile["korelasi"].map(
                lambda value: "—" if pd.isna(value) else f"{value:.2f}".replace(".", ",")
            ),
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Korelasi dihitung dengan Spearman di dalam tiap peran, hanya jika ada minimal 20 responden bergaji valid. "
        "Nilai positif berarti peran itu cenderung dibayar lebih tinggi saat pengalamannya lebih panjang."
    )


def render_report_downloads(people: pd.DataFrame) -> None:
    st.subheader("Unduh grafik laporan")
    st.caption(
        "PNG ini memakai judul, label, satuan, dan warna yang sama dengan gambar di laporan Word. "
        "Tanpa filter, berkasnya sama dengan dokumen. Filter di samping mengubah angka di dalam gambar."
    )
    names = ["Usia", "Pengalaman", "Gaji", "Sebaran", "Peran"]
    columns = st.columns(5)
    for column, name, (filename, label, png) in zip(columns, names, report_pngs(people)):
        with column:
            if png is None:
                st.caption(f"{name}: data belum cukup.")
            else:
                st.download_button(
                    name,
                    data=png,
                    file_name=filename,
                    mime="image/png",
                    key=f"unduh-{filename}",
                    help=f"{label}. Berkas {filename}",
                    width="stretch",
                )


def render_overview(filtered: pd.DataFrame, skills: pd.DataFrame) -> None:
    render_report_downloads(filtered)
    brief = build_brief(filtered, skills)
    st.subheader("Temuan")
    cards(brief["findings"])
    st.subheader("Keputusan yang bisa diambil")
    cards(brief["decisions"])

    left, right = st.columns(2)
    with left:
        ages = ordered_counts(filtered["age"], AGE_ORDER, label_age)
        if not ages.empty:
            st.plotly_chart(bar_chart(ages, "Kelompok usia", horizontal=False), width="stretch", config={"displayModeBar": False})
    with right:
        remote = ordered_counts(filtered["remote_work"], REMOTE_ORDER, label_remote)
        if not remote.empty:
            st.plotly_chart(bar_chart(remote, "Mode kerja", color="#fbbf24", horizontal=False), width="stretch", config={"displayModeBar": False})

    education = (
        filtered["ed_level"].map(label_ed).value_counts().rename_axis("label").reset_index(name="jumlah")
    )
    if not education.empty:
        st.plotly_chart(bar_chart(education, "Pendidikan"), width="stretch", config={"displayModeBar": False})

    countries = (
        filtered["country"].dropna().value_counts().head(12).rename_axis("label").reset_index(name="jumlah")
    )
    if not countries.empty:
        st.plotly_chart(bar_chart(countries, "12 negara terbanyak", color="#34d399"), width="stretch", config={"displayModeBar": False})


def render_pay(filtered: pd.DataFrame) -> None:
    st.caption(
        "Gaji memakai ConvertedCompYearly dalam USD. Nilai di bawah "
        f"{MIN_SALARY:,} USD tidak diikutkan karena tidak masuk akal sebagai pendapatan tahunan.".replace(",", ".")
    )
    by_role = chart_salary_by(filtered, "dev_type", label_role, "Median gaji menurut peran")
    if by_role:
        st.plotly_chart(by_role, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Peran dengan jumlah responden cukup belum tersedia pada filter ini.")
    render_role_link(filtered)
    by_remote = chart_salary_by(filtered, "remote_work", label_remote, "Median gaji menurut mode kerja")
    if by_remote:
        st.plotly_chart(by_remote, width="stretch", config={"displayModeBar": False})

    paid = valid_salary(filtered).dropna(subset=["work_exp", "comp_yearly"])
    if not paid.empty:
        sample = paid.sample(n=min(len(paid), 1200), random_state=7)
        fig = go.Figure(
            go.Scatter(
                x=sample["work_exp"],
                y=sample["comp_yearly"],
                mode="markers",
                marker=dict(size=8, color=ACCENT, opacity=0.55),
                hovertemplate="Pengalaman %{x} tahun<br>USD %{y:,.0f}<extra></extra>",
            )
        )
        style_figure(fig, "Pengalaman kerja dan gaji tahunan")
        fig.update_yaxes(title="USD / tahun")
        fig.update_xaxes(title="Tahun pengalaman")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    paid_for_corr = valid_salary(filtered)
    if len(paid_for_corr) >= 20:
        numeric = paid_for_corr[["work_exp", "years_code", "comp_yearly", "job_sat"]].rank().corr()
        numeric.index = ["Pengalaman kerja", "Tahun menulis kode", "Gaji tahunan", "Kepuasan kerja"]
        numeric.columns = numeric.index
        st.subheader("Korelasi peringkat")
        st.dataframe(numeric.round(2), width="stretch")
        st.caption("Spearman: 1 berarti naik bersama, -1 berarti berlawanan, 0 berarti tidak berkaitan secara monoton.")


def render_tech(filtered: pd.DataFrame, skills: pd.DataFrame) -> None:
    specs = [
        ("language", "Bahasa yang pernah dipakai", ACCENT),
        ("database", "Basis data yang pernah dipakai", "#34d399"),
        ("webframe", "Framework web yang pernah dipakai", "#fbbf24"),
    ]
    for kind, title, color in specs:
        ranking = skill_ranking(skills, filtered["response_id"], kind, limit=12)
        if ranking.empty:
            st.info(f"Tidak ada data untuk {title.lower()}.")
            continue
        ranking = ranking.rename(columns={"name": "label", "pengguna": "jumlah"})
        st.plotly_chart(bar_chart(ranking, title, color=color), width="stretch", config={"displayModeBar": False})
    st.caption("Satu responden bisa memilih lebih dari satu teknologi, jadi jumlahnya tidak dijumlahkan menjadi 100%.")


def render_ai(filtered: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        usage = ordered_counts(filtered["ai_select"], AI_USE_ORDER, label_ai_use)
        if not usage.empty:
            st.plotly_chart(bar_chart(usage, "Seberapa sering memakai AI", horizontal=False), width="stretch", config={"displayModeBar": False})
    with right:
        sentiment = ordered_counts(filtered["ai_sent"], SENTIMENT_ORDER, label_sentiment)
        if not sentiment.empty:
            st.plotly_chart(
                bar_chart(sentiment, "Sikap terhadap AI", color="#34d399", horizontal=False),
                width="stretch", config={"displayModeBar": False},
            )

    left, right = st.columns(2)
    with left:
        trust = ordered_counts(filtered["ai_acc"], TRUST_ORDER, label_trust)
        if not trust.empty:
            st.plotly_chart(bar_chart(trust, "Kepercayaan pada akurasi AI", color="#fb7185"), width="stretch", config={"displayModeBar": False})
    with right:
        threat = (
            filtered["ai_threat"]
            .map(label_threat)
            .value_counts()
            .rename_axis("label")
            .reset_index(name="jumlah")
        )
        threat = threat[threat["label"] != "Tidak diisi"]
        if not threat.empty:
            st.plotly_chart(bar_chart(threat, "Apakah AI mengancam pekerjaan?", color="#fbbf24", horizontal=False), width="stretch", config={"displayModeBar": False})

    satisfaction = filtered["job_sat"].dropna()
    if not satisfaction.empty:
        counts = satisfaction.round().astype(int).value_counts().sort_index()
        fig = go.Figure(
            go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                marker=dict(color="#a78bfa"),
                hovertemplate="Skor %{x}<br>%{y} responden<extra></extra>",
            )
        )
        style_figure(fig, "Distribusi kepuasan kerja")
        fig.update_xaxes(title="Skor kepuasan")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    career = (
        filtered["new_role"]
        .map(label_new_role)
        .value_counts()
        .rename_axis("label")
        .reset_index(name="jumlah")
    )
    career = career[career["label"] != "Tidak diisi"]
    if not career.empty:
        st.plotly_chart(bar_chart(career, "Pertimbangan pindah karier atau industri"), width="stretch", config={"displayModeBar": False})


def render_table(filtered: pd.DataFrame) -> None:
    view = filtered.copy()
    view["Usia"] = view["age"].map(label_age)
    view["Pendidikan"] = view["ed_level"].map(label_ed)
    view["Status kerja"] = view["employment"].map(label_employment)
    view["Peran"] = view["dev_type"].map(label_role)
    view["Mode kerja"] = view["remote_work"].map(label_remote)
    view["Negara"] = view["country"].fillna("—")
    view["Gaji USD"] = view["comp_yearly"]
    view["Kepuasan"] = view["job_sat"]
    view["Pengalaman"] = view["work_exp"]
    view["Pakai AI"] = view["ai_select"].map(label_ai_use)
    columns = ["Usia", "Pendidikan", "Status kerja", "Peran", "Mode kerja", "Negara", "Gaji USD", "Kepuasan", "Pengalaman", "Pakai AI"]
    st.dataframe(view[columns], width="stretch", hide_index=True, height=520)
    st.download_button(
        "Unduh cuplikan CSV",
        data=view[columns].to_csv(index=False).encode("utf-8-sig"),
        file_name="cuplikan_survei.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Dashboard Survei Pengembang",
        page_icon=":bar_chart:",
        layout="wide",
    )
    inject_css()

    if not CSV_PATH.exists():
        st.error(f"Berkas data tidak ditemukan: {CSV_PATH}")
        return

    respondents, skills = load_data(CSV_PATH.stat().st_mtime)
    filtered = apply_filters(respondents)

    st.title("Dashboard Survei Pengembang")
    st.markdown(
        "Analisis responden, kompensasi, teknologi, dan adopsi AI. "
        f"Sumber CSV dimuat ke SQLite (`{Path(DB_PATH).name}`) supaya jawaban ganda bisa dihitung per teknologi."
    )

    show_metrics(filtered, len(respondents))
    st.write("")

    overview, pay, tech, ai, table = st.tabs(
        ["Ringkasan", "Kompensasi", "Teknologi", "Adopsi AI", "Data"]
    )
    with overview:
        render_overview(filtered, skills)
    with pay:
        render_pay(filtered)
    with tech:
        render_tech(filtered, skills)
    with ai:
        render_ai(filtered)
    with table:
        render_table(filtered)

    with st.sidebar:
        st.divider()
        st.caption(f"{len(respondents):,} responden di database.".replace(",", "."))
        st.caption("Label grafik diterjemahkan. Nilai asli survei tetap tersimpan di SQLite.")


if __name__ == "__main__":
    main()
