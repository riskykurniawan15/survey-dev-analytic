from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT.parent / "so_dev_survey.csv"
DB_PATH = ROOT / "survey.db"

MISSING = {"", "NA", "N/A", "nan", "None", "null"}

SKILL_COLUMNS = {
    "language": "LanguageHaveWorkedWith",
    "database": "DatabaseHaveWorkedWith",
    "webframe": "WebframeHaveWorkedWith",
}


def clean_text(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in MISSING:
        return None
    return text


def to_number(value) -> float | None:
    text = clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered.startswith("less than"):
        return 0.5
    if lowered.startswith("more than"):
        return 51.0
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def split_items(value) -> list[str]:
    text = clean_text(value)
    if text is None:
        return []
    items = []
    for part in text.split(";"):
        name = part.strip()
        if name and name not in MISSING:
            items.append(name)
    return items


def build_frames(csv_path: Path = CSV_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)

    respondents = pd.DataFrame(
        {
            "response_id": pd.to_numeric(raw["ResponseId"], errors="coerce").astype("Int64"),
            "main_branch": raw["MainBranch"].map(clean_text),
            "age": raw["Age"].map(clean_text),
            "ed_level": raw["EdLevel"].map(clean_text),
            "employment": raw["Employment"].map(clean_text),
            "work_exp": raw["WorkExp"].map(to_number),
            "years_code": raw["YearsCode"].map(to_number),
            "dev_type": raw["DevType"].map(clean_text),
            "org_size": raw["OrgSize"].map(clean_text),
            "remote_work": raw["RemoteWork"].map(clean_text),
            "country": raw["Country"].map(clean_text),
            "industry": raw["Industry"].map(clean_text),
            "comp_yearly": raw["ConvertedCompYearly"].map(to_number),
            "job_sat": raw["JobSat"].map(to_number),
            "ai_select": raw["AISelect"].map(clean_text),
            "ai_sent": raw["AISent"].map(clean_text),
            "ai_acc": raw["AIAcc"].map(clean_text),
            "ai_complex": raw["AIComplex"].map(clean_text),
            "ai_threat": raw["AIThreat"].map(clean_text),
            "new_role": raw["NewRole"].map(clean_text),
        }
    )
    respondents = respondents.dropna(subset=["response_id"]).copy()
    respondents["response_id"] = respondents["response_id"].astype(int)

    skill_rows = []
    for kind, column in SKILL_COLUMNS.items():
        for response_id, cell in zip(respondents["response_id"], raw.loc[respondents.index, column]):
            for name in split_items(cell):
                skill_rows.append({"response_id": int(response_id), "kind": kind, "name": name})

    skills = pd.DataFrame(skill_rows, columns=["response_id", "kind", "name"])
    return respondents, skills


def ensure_database(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")

    csv_mtime = csv_path.stat().st_mtime
    if db_path.exists() and db_path.stat().st_mtime >= csv_mtime:
        return db_path

    respondents, skills = build_frames(csv_path)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        respondents.to_sql("respondents", conn, index=False)
        skills.to_sql("skills", conn, index=False)
        conn.execute("CREATE INDEX idx_skills_kind_name ON skills(kind, name)")
        conn.execute("CREATE INDEX idx_skills_response ON skills(response_id)")
        conn.execute("CREATE INDEX idx_respondents_country ON respondents(country)")
    return db_path


def load_tables(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_database(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        respondents = pd.read_sql("SELECT * FROM respondents", conn)
        skills = pd.read_sql("SELECT response_id, kind, name FROM skills", conn)
    return respondents, skills


if __name__ == "__main__":
    database = ensure_database()
    people, skills = load_tables(database)
    print(f"database: {database}")
    print(f"respondents: {len(people)}")
    print(f"skill rows: {len(skills)}")
    print(people[["comp_yearly", "job_sat", "work_exp"]].describe().round(1))
