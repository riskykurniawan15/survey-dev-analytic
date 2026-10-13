from __future__ import annotations

import pandas as pd

MIN_SALARY = 1_000
MIN_GROUP = 20


def valid_salary(df: pd.DataFrame) -> pd.DataFrame:
    salary = df["comp_yearly"]
    return df[salary.notna() & (salary >= MIN_SALARY)].copy()


def median_of(df: pd.DataFrame, column: str) -> float | None:
    series = df[column].dropna()
    if series.empty:
        return None
    return float(series.median())


def spearman(df: pd.DataFrame, left: str, right: str) -> tuple[float | None, int]:
    pair = df[[left, right]].dropna()
    if len(pair) < MIN_GROUP:
        return None, len(pair)
    correlation = pair[left].rank().corr(pair[right].rank())
    if pd.isna(correlation):
        return None, len(pair)
    return float(correlation), len(pair)


def eta_squared(df: pd.DataFrame, group: str, value: str) -> tuple[float | None, int]:
    clean = df.dropna(subset=[group, value])
    if len(clean) < MIN_GROUP or clean[group].nunique() < 2:
        return None, len(clean)
    overall = clean[value].mean()
    total = ((clean[value] - overall) ** 2).sum()
    if total == 0:
        return None, len(clean)
    means = clean.groupby(group)[value].transform("mean")
    between = ((means - overall) ** 2).sum()
    return float(between / total), len(clean)


def experience_by_role(df: pd.DataFrame, min_people: int = 8) -> pd.DataFrame:
    people = df.dropna(subset=["dev_type"])
    rows = []
    for role, group in people.groupby("dev_type"):
        experience = group["work_exp"].dropna()
        if len(experience) < min_people:
            continue
        paid = valid_salary(group)
        correlation, pairs = spearman(paid, "work_exp", "comp_yearly")
        rows.append(
            {
                "dev_type": role,
                "responden": int(len(group)),
                "median_pengalaman": float(experience.median()),
                "median_gaji": float(paid["comp_yearly"].median()) if len(paid) >= min_people else None,
                "korelasi": correlation,
                "n_korelasi": pairs,
            }
        )
    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    return profile.sort_values(["median_pengalaman", "responden"], ascending=[False, False]).reset_index(drop=True)


LEADER_ROLES = (
    ("Architect, software or solutions", "arsitek perangkat lunak"),
    ("Engineering manager", "manajer rekayasa"),
    ("Senior executive (C-suite, VP, etc.)", "eksekutif senior"),
    ("Project manager", "manajer proyek"),
    ("Product manager", "manajer produk"),
)
OLDER_AGES = {"55-64 years old", "65 years or older"}


def _years(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def _role_snapshot(df: pd.DataFrame, role: str) -> dict | None:
    group = df[df["dev_type"] == role]
    experience = group["work_exp"].dropna()
    if experience.empty:
        return None
    aged = group["age"].dropna()
    return {
        "n": int(len(experience)),
        "median": float(experience.median()),
        "q1": float(experience.quantile(0.25)),
        "long_share": float((experience >= 25).mean()),
        "older_share": float(aged.isin(OLDER_AGES).mean()) if not aged.empty else None,
    }


def role_experience_note(df: pd.DataFrame, min_people: int = 15) -> str | None:
    if df.empty or "dev_type" not in df.columns:
        return None
    snapshots = {role: _role_snapshot(df, role) for role, _label in LEADER_ROLES}
    architect = snapshots["Architect, software or solutions"]
    manager = snapshots["Engineering manager"]
    sentences: list[str] = []

    def piece(label: str, snap: dict) -> str:
        return f"{label} {_years(snap['median'])} tahun ({_count(snap['n'])} orang)"

    if architect and manager:
        gap = architect["median"] - manager["median"]
        if gap > 0.05:
            compare = f"Arsitek lebih tinggi {_years(gap)} tahun."
        elif gap < -0.05:
            compare = f"Manajer rekayasa lebih tinggi {_years(abs(gap))} tahun."
        else:
            compare = "Median keduanya sama."
        sentences.append(
            "Pada filter ini, median "
            + piece("arsitek perangkat lunak", architect)
            + " dan "
            + piece("manajer rekayasa", manager)
            + f". {compare}"
        )
        if architect["n"] >= 8 and manager["n"] >= 8:
            sentences.append(
                f"Kuartil bawah {_years(architect['q1'])} tahun pada arsitek dan "
                f"{_years(manager['q1'])} tahun pada manajer rekayasa. "
                f"Pengalaman 25 tahun atau lebih: {_pct(100 * architect['long_share'])} arsitek dan "
                f"{_pct(100 * manager['long_share'])} manajer rekayasa."
            )
            if (
                architect["older_share"] is not None
                and manager["older_share"] is not None
                and (architect["older_share"] > 0 or manager["older_share"] > 0)
            ):
                sentences.append(
                    f"Usia 55 tahun ke atas: {_pct(100 * architect['older_share'])} arsitek dan "
                    f"{_pct(100 * manager['older_share'])} manajer rekayasa."
                )
    elif architect or manager:
        label, snap = ("arsitek perangkat lunak", architect) if architect else ("manajer rekayasa", manager)
        status = "ikut tampil" if snap["n"] >= min_people else f"berada di bawah ambang {min_people} responden"
        sentences.append(f"Pada filter ini, {piece(label, snap)} dan {status}.")

    shown, hidden = [], []
    for role, label in LEADER_ROLES[2:]:
        snap = snapshots[role]
        if snap is None:
            continue
        bit = piece(label, snap)
        if snap["n"] >= min_people:
            shown.append(bit)
        else:
            hidden.append(bit)
    if shown:
        sentences.append("Peran pimpinan lain yang ikut tampil: " + ", ".join(shown) + ".")
    if hidden:
        sentences.append(
            f"Peran pimpinan di bawah ambang {min_people} responden: " + ", ".join(hidden) + "."
        )
    if not sentences:
        return None
    return " ".join(sentences)


def top_share(series: pd.Series) -> tuple[str, float, int] | None:
    clean = series.dropna()
    if clean.empty:
        return None
    counts = clean.value_counts()
    label = str(counts.index[0])
    count = int(counts.iloc[0])
    return label, 100 * count / len(clean), count


def skill_ranking(skills: pd.DataFrame, response_ids: pd.Series, kind: str, limit: int = 10) -> pd.DataFrame:
    subset = skills[(skills["kind"] == kind) & (skills["response_id"].isin(response_ids))]
    if subset.empty:
        return pd.DataFrame(columns=["name", "pengguna"])
    counts = (
        subset.groupby("name")["response_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(limit)
        .rename("pengguna")
        .reset_index()
    )
    return counts


def _count(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _money(value: float | None) -> str:
    if value is None:
        return "tidak cukup data"
    return f"USD {_count(int(round(value)))}"


def _pct(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + "%"


def _age(value: str) -> str:
    if value.startswith("Prefer"):
        return "yang tidak menyebut usia"
    return value.replace(" years old", "").replace(" years or older", "+")


def build_brief(df: pd.DataFrame, skills: pd.DataFrame) -> dict:
    findings: list[dict[str, str]] = []
    decisions: list[dict[str, str]] = []
    n = len(df)
    if n == 0:
        empty = {"title": "Filter kosong", "body": "Tidak ada responden pada kombinasi filter ini."}
        return {"findings": [empty], "decisions": []}

    age = top_share(df["age"])
    if age:
        findings.append(
            {
                "title": "Profil responden",
                "body": (
                    f"Dari {_count(n)} responden pada filter ini, kelompok usia terbesar adalah "
                    f"{_age(age[0])} ({_pct(age[1])})."
                ),
            }
        )

    paid = valid_salary(df)
    remote = paid[paid["remote_work"] == "Remote"]
    office = paid[paid["remote_work"] == "In-person"]
    remote_med = median_of(remote, "comp_yearly") if len(remote) >= MIN_GROUP else None
    office_med = median_of(office, "comp_yearly") if len(office) >= MIN_GROUP else None
    if remote_med and office_med:
        gap = 100 * (remote_med - office_med) / office_med
        direction = "lebih tinggi" if gap >= 0 else "lebih rendah"
        findings.append(
            {
                "title": "Gaji remote dan kantor",
                "body": (
                    f"Median gaji remote {_money(remote_med)} ({len(remote)} orang), "
                    f"kantor {_money(office_med)} ({len(office)} orang). "
                    f"Remote {direction} {_pct(abs(gap))} dibanding kerja di kantor. "
                    "Selisih ini belum memisahkan negara atau peran."
                ),
            }
        )
        higher = "remote" if remote_med >= office_med else "kerja di kantor"
        decisions.append(
            {
                "title": "Kebijakan mode kerja",
                "body": (
                    f"Selisih median mengarah ke {higher}. Angka ini belum mengendalikan negara, "
                    f"peran, atau senioritas, jadi jangan dipakai sebagai satu-satunya alasan "
                    f"mengubah kebijakan. Bandingkan di dalam negara dan peran yang sama sebelum memutuskan."
                ),
            }
        )

    languages = skill_ranking(skills, df["response_id"], "language", limit=3)
    if not languages.empty:
        top_name = languages.iloc[0]["name"]
        top_users = int(languages.iloc[0]["pengguna"])
        share = 100 * top_users / n
        second = languages.iloc[1]["name"] if len(languages) > 1 else None
        follow = f" Berikutnya {second}." if second else ""
        findings.append(
            {
                "title": "Keterampilan yang paling dipakai",
                "body": (
                    f"{top_name} dipakai {_count(top_users)} responden ({_pct(share)} dari filter ini)."
                    + follow
                ),
            }
        )
        decisions.append(
            {
                "title": "Prioritas pembelajaran",
                "body": (
                    f"Untuk pelatihan atau rekrutmen pada populasi yang sedang difilter, "
                    f"dahulukan {top_name} karena basis penggunanya paling lebar."
                    + (f" {second} layak jadi prioritas kedua." if second else "")
                ),
            }
        )

    ai = df["ai_select"].dropna()
    if len(ai) >= MIN_GROUP:
        daily = 100 * (ai == "Yes, I use AI tools daily").mean()
        active = 100 * ai.isin(
            ["Yes, I use AI tools daily", "Yes, I use AI tools weekly"]
        ).mean()
        trust = df["ai_acc"].dropna()
        distrust = 0.0
        if not trust.empty:
            distrust = 100 * trust.isin(["Somewhat distrust", "Highly distrust"]).mean()
        findings.append(
            {
                "title": "Adopsi AI",
                "body": (
                    f"{_pct(daily)} memakai AI setiap hari dan {_pct(active)} memakai AI "
                    f"minimal setiap minggu. {_pct(distrust)} dari yang menjawab menyatakan "
                    f"tidak percaya pada akurasi AI."
                ),
            }
        )
        decisions.append(
            {
                "title": "Aturan pemakaian AI",
                "body": (
                    "Pemakaian sudah cukup luas untuk dijadikan kebiasaan kerja, tetapi ketidakpercayaan "
                    "masih besar. Keputusan yang seimbang: bolehkan AI untuk draf dan pencarian, "
                    "wajibkan tinjauan manusia sebelum kode atau analisis dipakai, dan batasi data sensitif."
                ),
            }
        )

    threat = df["ai_threat"].dropna()
    if len(threat) >= MIN_GROUP:
        worried = 100 * (threat == "Yes").mean()
        if worried >= 25:
            decisions.append(
                {
                    "title": "Risiko peran kerja",
                    "body": (
                        f"{_pct(worried)} responden merasa AI mengancam pekerjaannya. "
                        "Respons yang lebih berguna daripada melarang alat adalah memperjelas "
                        "tugas yang tetap menjadi tanggung jawab manusia: desain, penilaian, dan akuntabilitas."
                    ),
                }
            )

    correlation, pairs = spearman(valid_salary(df), "work_exp", "comp_yearly")
    if correlation is not None:
        strength = "lemah" if abs(correlation) < 0.3 else "sedang" if abs(correlation) < 0.6 else "kuat"
        findings.append(
            {
                "title": "Pengalaman dan gaji",
                "body": (
                    f"Korelasi peringkat pengalaman kerja dan gaji tahunan {strength} "
                    f"(Spearman r = {correlation:.2f}, n = {pairs}). "
                    "Lama bekerja tidak otomatis menjelaskan kompensasi."
                ),
            }
        )

    if len(findings) > 4:
        findings = findings[:4]
    if len(decisions) > 3:
        decisions = decisions[:3]
    return {"findings": findings, "decisions": decisions}
