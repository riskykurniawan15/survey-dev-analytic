from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from analisis import (
    COLORS,
    ORDER,
    SAT_BENCHMARK,
    fmt_num,
    fmt_p,
    interpret_effect_d,
    interpret_eta2,
    qq_points,
    run_all,
)


def style_figure(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Segoe UI, sans-serif", size=13),
        margin=dict(l=8, r=24, t=52, b=8),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.16)", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", zeroline=False, automargin=True)
    return fig


def cards(items: list[dict[str, str]]) -> None:
    if not items:
        return
    for start in range(0, len(items), 2):
        columns = st.columns(2)
        for column, item in zip(columns, items[start : start + 2]):
            with column:
                st.markdown(
                    f"<div class='panel'><h3>{item['title']}</h3><p>{item['body']}</p></div>",
                    unsafe_allow_html=True,
                )


def show_kpis(result: dict) -> None:
    one, anova = result["satu_sampel"], result["anova"]
    desc = result["deskriptif"]
    by_name = desc.set_index("nama") if not desc.empty else desc

    def mode_kpi(name: str) -> tuple[str, str]:
        if name not in by_name.index:
            return "—", "n=0"
        return fmt_num(by_name.loc[name, "mean"], 2), f"n={int(by_name.loc[name, 'n'])}"

    remote_mean, remote_n = mode_kpi("Remote")
    hybrid_mean, hybrid_n = mode_kpi("Hybrid")
    kantor_mean, kantor_n = mode_kpi("Kantor")
    mean_sat = fmt_num(one["mean"], 2) if one else "—"
    bench = f"benchmark {fmt_num(result['benchmark'], 0)}" if one else "kepuasan belum cukup"
    anova_text = anova["keputusan"] if anova else "—"
    anova_p = f"p = {fmt_p(anova['p'])}" if anova else "mode belum lengkap"
    items = [
        ("Responden", f"{result['n_responden']:,}".replace(",", "."), f"kepuasan n={result['n_kepuasan']}"),
        ("Mean kepuasan", mean_sat, bench),
        ("Remote", remote_mean, remote_n),
        ("Hybrid", hybrid_mean, hybrid_n),
        ("Kantor", kantor_mean, kantor_n),
        ("ANOVA", anova_text, anova_p),
    ]
    html = "".join(
        f"<div class='kpi'><span>{label}</span><strong>{value}</strong><em>{note}</em></div>"
        for label, value, note in items
    )
    st.markdown(f"<div class='kpis'>{html}</div>", unsafe_allow_html=True)


def chart_box(frame) -> go.Figure:
    fig = go.Figure()
    for name in ORDER:
        values = frame.loc[frame["mode_kerja"] == name, "job_sat"]
        if values.empty:
            continue
        fig.add_trace(
            go.Box(
                y=values,
                name=name,
                marker_color=COLORS[name],
                boxmean=True,
            )
        )
    fig.add_hline(y=SAT_BENCHMARK, line_dash="dash", line_color="#fb7185", annotation_text="Benchmark 7")
    return style_figure(fig, "Kepuasan kerja menurut mode kerja")


def chart_means(desc) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=desc["nama"],
            y=desc["mean"],
            error_y=dict(type="data", array=desc["sd"]),
            marker_color=[COLORS[n] for n in desc["nama"]],
            text=[fmt_num(v, 2) for v in desc["mean"]],
            textposition="outside",
        )
    )
    fig.add_hline(y=SAT_BENCHMARK, line_dash="dash", line_color="#fb7185")
    fig.update_yaxes(range=[0, 10.5], title="Kepuasan (0–10)")
    return style_figure(fig, "Rata-rata ± SD kepuasan")


def chart_qq(frame) -> go.Figure:
    present = [name for name in ORDER if (frame["mode_kerja"] == name).any()]
    fig = make_subplots(rows=1, cols=max(len(present), 1), subplot_titles=present or ["—"])
    for i, name in enumerate(present, start=1):
        values = frame.loc[frame["mode_kerja"] == name, "job_sat"].to_numpy(dtype=float)
        if len(values) > 120:
            values = np.random.default_rng(11).choice(values, size=120, replace=False)
        if len(values) < 3:
            continue
        theo, obs = qq_points(values)
        fig.add_trace(
            go.Scatter(x=theo, y=obs, mode="markers", marker=dict(color=COLORS[name], size=7), showlegend=False),
            row=1,
            col=i,
        )
        slope, intercept, _, _, _ = stats.linregress(theo, obs)
        fig.add_trace(
            go.Scatter(
                x=[float(theo.min()), float(theo.max())],
                y=[intercept + slope * float(theo.min()), intercept + slope * float(theo.max())],
                mode="lines",
                line=dict(color="#94a3b8", width=1.4),
                showlegend=False,
            ),
            row=1,
            col=i,
        )
    return style_figure(fig, "Plot Q-Q (sampel per mode)", height=380)


def chart_tukey(tukey) -> go.Figure:
    labels = [f"{a} vs {b}" for a, b in zip(tukey["kelompok1"], tukey["kelompok2"])]
    fig = go.Figure()
    for i, row in tukey.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["batas_bawah"], row["batas_atas"]],
                y=[labels[i], labels[i]],
                mode="lines",
                line=dict(color="#64748b", width=3),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["selisih"]],
                y=[labels[i]],
                mode="markers",
                marker=dict(color="#34d399" if row["tolak"] else "#94a3b8", size=11),
                showlegend=False,
                hovertemplate=f"{labels[i]}<br>selisih %{{x:.3f}}<br>p_adj={fmt_p(row['p_adj'])}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#fb7185")
    return style_figure(fig, "Interval Tukey HSD", height=360)


def render_hypothesis(people) -> None:
    st.caption(
        "Filter di samping ikut dipakai di sini. Uji: kepuasan versus 7, Remote versus Kantor, "
        "lalu ANOVA tiga mode. Pelengkap: ANOVA pada log-gaji."
    )
    result = run_all(people)
    frame = result["data_mode_sat"]
    one, two, anova = result["satu_sampel"], result["dua_kelompok"], result["anova"]
    gaji = result["gaji"]
    show_kpis(result)

    if frame.empty or one is None:
        st.info("Filter ini belum cukup untuk uji kepuasan. Longgarkan mode kerja atau negara.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ringkasan", "Asumsi", "Hasil uji", "Gaji (pelengkap)", "Rekomendasi"])

    with tab1:
        items = [
            {
                "title": "Kepuasan vs benchmark 7",
                "body": (
                    f"{one['keputusan']} (t = {fmt_num(one['t'], 3)}, p = {fmt_p(one['p'])}). "
                    f"Mean {fmt_num(one['mean'], 2)}, d = {fmt_num(one['cohen_d'], 2)} "
                    f"({interpret_effect_d(one['cohen_d'])})."
                ),
            }
        ]
        if two:
            items.append(
                {
                    "title": "Remote vs Kantor",
                    "body": (
                        f"{two['keputusan']} via {two['uji']} "
                        f"(t = {fmt_num(two['t'], 3)}, p = {fmt_p(two['p'])}). "
                        f"Selisih mean {fmt_num(two['selisih'], 2)}, d = {fmt_num(two['cohen_d'], 2)}."
                    ),
                }
            )
        if anova:
            items.append(
                {
                    "title": "ANOVA tiga mode",
                    "body": (
                        f"{anova['keputusan']} ({anova['uji']}, p = {fmt_p(anova['p'])}). "
                        f"η² = {fmt_num(anova['eta2'], 3)} ({interpret_eta2(anova['eta2'])}). "
                        f"Kruskal–Wallis p = {fmt_p(anova['kruskal_p'])} sebagai cek ketahanan."
                    ),
                }
            )
        items.append(
            {
                "title": "Cara baca",
                "body": (
                    "Ada variabel numerik (kepuasan, gaji) dan faktor mode kerja dengan tiga taraf. "
                    "Uji ini memakai cuplikan yang sama dengan tab lain, termasuk filter di samping."
                ),
            }
        )
        cards(items)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(chart_box(frame), width="stretch", config={"displayModeBar": False})
        with right:
            if not result["deskriptif"].empty:
                st.plotly_chart(chart_means(result["deskriptif"]), width="stretch", config={"displayModeBar": False})

    with tab2:
        st.caption(result["asumsi_catatan"])
        desc = result["deskriptif"].copy()
        if desc.empty:
            st.info("Deskripsi per mode belum tersedia pada filter ini.")
        else:
            show = desc[["nama", "n", "mean", "sd", "shapiro_w", "shapiro_p", "normal", "n_outlier"]].copy()
            show.columns = ["Mode", "n", "Mean", "SD", "W", "p Shapiro*", "Normal?*", "Pencilan IQR"]
            for column in ["Mean", "SD", "W", "p Shapiro*"]:
                digits = 3 if column in {"W", "p Shapiro*"} else 2
                show[column] = show[column].map(lambda v, d=digits: fmt_num(v, d))
            show["Normal?*"] = show["Normal?*"].map(lambda v: "Ya" if v else "Tidak")
            st.dataframe(show, width="stretch", hide_index=True)
        if two:
            st.markdown(
                f"- Levene Remote–Kantor: p = {fmt_p(two['levene_p'])} → "
                f"{'homogen' if two['varian_homogen'] else 'tidak homogen'} → uji {two['uji']}."
            )
        if anova:
            st.markdown(
                f"- Levene tiga mode: p = {fmt_p(anova['levene_p'])} → "
                f"{'homogen' if anova['varian_homogen'] else 'tidak homogen'} → {anova['uji']}."
            )
        st.plotly_chart(chart_qq(frame), width="stretch", config={"displayModeBar": False})

    with tab3:
        st.subheader("a) One-sample t-test")
        st.markdown(
            f"**H0:** {one['h0']} | **H1:** {one['h1']}  \n"
            f"t({one['df']}) = {fmt_num(one['t'], 3)}, p = {fmt_p(one['p'])}, "
            f"CI 95% = [{fmt_num(one['ci_low'], 2)}, {fmt_num(one['ci_high'], 2)}], "
            f"d = {fmt_num(one['cohen_d'], 2)}. **{one['keputusan']}.**"
        )
        st.subheader("b) Remote vs Kantor")
        if two:
            st.markdown(
                f"**H0:** {two['h0']} | **H1:** {two['h1']}  \n"
                f"{two['uji']}: t = {fmt_num(two['t'], 3)}, p = {fmt_p(two['p'])}, "
                f"d = {fmt_num(two['cohen_d'], 2)}. **{two['keputusan']}.**"
            )
        else:
            st.info("Remote atau Kantor belum cukup pada filter ini.")
        st.subheader("c) ANOVA + Tukey")
        if anova:
            st.markdown(
                f"**H0:** {anova['h0']}  \n"
                f"{anova['uji']}: F = {fmt_num(anova['F'], 3)}, p(ANOVA) = {fmt_p(anova['p_anova'])}, "
                f"p(Welch) = {fmt_p(anova['welch_p'])}, η² = {fmt_num(anova['eta2'], 3)}. "
                f"**{anova['keputusan']}.**"
            )
        else:
            st.info("Ketiga mode kerja belum lengkap pada filter ini.")
        tukey = result["tukey"]
        if tukey is not None and not tukey.empty:
            show_t = tukey.assign(
                pasangan=lambda d: d["kelompok1"] + " vs " + d["kelompok2"],
                selisih=lambda d: d["selisih"].map(lambda v: fmt_num(v, 3)),
                p_adj=lambda d: d["p_adj"].map(fmt_p),
                signifikan=lambda d: d["tolak"].map(lambda v: "Ya" if v else "Tidak"),
            )[["pasangan", "selisih", "p_adj", "signifikan"]]
            show_t.columns = ["Pasangan", "Selisih", "p adj", "Signifikan"]
            st.dataframe(show_t, width="stretch", hide_index=True)
            st.plotly_chart(chart_tukey(tukey), width="stretch", config={"displayModeBar": False})

    with tab4:
        st.caption(
            "Gaji miring ke kanan, jadi ANOVA dijalankan pada log(gaji+1). "
            "Median USD ditampilkan untuk interpretasi praktis."
        )
        if not gaji:
            st.info("Gaji valid per mode belum cukup pada filter ini.")
        else:
            rows = [
                {
                    "Mode": name,
                    "n": gaji["n"][name],
                    "Median USD": f"{int(gaji['median_usd'][name]):,}".replace(",", "."),
                    "Mean USD": f"{int(gaji['mean_usd'][name]):,}".replace(",", "."),
                }
                for name in ORDER
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
            st.markdown(
                f"{gaji['uji']}: F = {fmt_num(gaji['F'], 3)}, p = {fmt_p(gaji['p'])}, "
                f"η² = {fmt_num(gaji['eta2'], 3)}. **{gaji['keputusan']}.**"
            )
            gt = gaji["tukey"].assign(
                pasangan=lambda d: d["kelompok1"] + " vs " + d["kelompok2"],
                signifikan=lambda d: d["tolak"].map(lambda v: "Ya" if v else "Tidak"),
                p_adj=lambda d: d["p_adj"].map(fmt_p),
            )[["pasangan", "p_adj", "signifikan"]]
            gt.columns = ["Pasangan (log gaji)", "p adj", "Signifikan"]
            st.dataframe(gt, width="stretch", hide_index=True)

    with tab5:
        recs = [
            {
                "title": "Keputusan kepuasan",
                "body": (
                    "Rata-rata kepuasan sedikit di atas benchmark 7, tetapi efeknya kecil. "
                    "Remote lebih tinggi daripada Kantor; perbedaan antar tiga mode signifikan meski η² kecil."
                    if two and anova
                    else "Baca hasil uji pada filter ini sebelum menyamakan kepuasan antar mode kerja."
                ),
            },
            {
                "title": "Keputusan kompensasi",
                "body": (
                    (
                        f"Median gaji Remote USD {int(gaji['median_usd']['Remote']):,}, "
                        f"Hybrid {int(gaji['median_usd']['Hybrid']):,}, "
                        f"Kantor {int(gaji['median_usd']['Kantor']):,} (pemisah ribuan titik). "
                        "ANOVA log-gaji menolak kesamaan mean transformasi."
                    ).replace(",", ".")
                    if gaji
                    else "Gaji valid pada filter ini belum cukup untuk membandingkan ketiga mode."
                ),
            },
            {
                "title": "Rekomendasi",
                "body": (
                    "Kebijakan hybrid/remote perlu menimbang kepuasan dan kompensasi bersama. "
                    "Jangan menyamakan paket gaji Remote dengan Kantor tanpa penyesuaian pasar."
                ),
            },
            {
                "title": "Keterbatasan",
                "body": (
                    "Observasional (bukan eksperimen acak), negara/peran belum dikontrol, "
                    "gaji sangat miring, dan n antar mode tidak seimbang (Kantor lebih kecil)."
                ),
            },
        ]
        cards(recs)
