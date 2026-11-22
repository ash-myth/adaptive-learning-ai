"""
data_preprocessing.py  (v2 — 7 sources)
========================================
Adaptive AI Learning Intelligence Platform — Phase 1
-----------------------------------------------------
Merges 7 course datasets from 6 platforms into one unified courses.csv.

Sources
-------
1. coursea_data.csv               — original Coursera (891 rows)
2. Udemy_Courses.csv              — original Udemy   (3,682 rows)
3. combined_simplilearn_coursera  — Coursera + Udacity + FutureLearn + Simplilearn (8,092)
4. comined_edx_udacity_coursera   — EdX + Udacity + Coursera (2,377)
5. coursera_1000_Courses.csv      — Coursera top-1000 (1,000)
6. Coursera_courses_catalog.csv   — Coursera full catalog (5,164)
7. udemy_courses.csv              — Udemy extended (3,678)

Output
------
data/courses.csv          — unified catalogue
data/preprocessing_report.txt
"""

import pandas as pd
import numpy as np
import re, os
from collections import defaultdict

os.makedirs("data", exist_ok=True)

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(str(msg))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Shared parsers
# ════════════════════════════════════════════════════════════════════════════

def parse_rating(value) -> float:
    """
    Handles:  4.7  |  '4.7stars'  |  '4.7 stars'  |  'not-mentioned'  |  NaN
    Returns float in [0, 5] or NaN.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower().replace("stars", "").replace("star", "").strip()
    if s in ("not-mentioned", "not found", "n/a", ""):
        return np.nan
    try:
        v = float(s)
        return v if 0 <= v <= 5 else np.nan
    except ValueError:
        return np.nan


def parse_enrollment(value) -> int:
    """
    Handles:  '5.3k'  |  '130k'  |  '10,438 '  |  98175  |  NaN
    """
    if pd.isna(value):
        return 0
    s = str(value).strip().lower().replace(",", "").replace(" ", "")
    for suffix, mult in [("k", 1_000), ("m", 1_000_000)]:
        if s.endswith(suffix):
            try:
                return int(float(s[:-1]) * mult)
            except ValueError:
                return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_duration_minutes(value) -> float:
    """
    Universal duration parser covering all 7 dataset formats:

      "1.5 hours"               → 90
      "37 mins"                 → 37
      "Approx. 14 hours to complete" → 840
      "one hour"                → 60
      "1 hour"                  → 60
      "Approximately 3 months to complete" → 540  (3 × 3hrs/week × 4 weeks... ≈ 360-720, use 180/month)
      "3 - 6 Months"            → 810  (4.5 months × 180 mins/month)
      "1 - 4 Weeks"             → 300  (2.5 weeks × 120 mins/week)
      "Less Than 2 Hours"       → 90
      "Not Found" / NaN         → NaN
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()

    if s in ("not found", "not-mentioned", "n/a", ""):
        return np.nan

    # "one hour" / "two hours" etc.
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8}
    for word, num in word_map.items():
        if s.startswith(word):
            if "hour" in s:
                return float(num * 60)
            if "week" in s:
                return float(num * 7 * 60)

    # "Less than X hours"
    lt_match = re.search(r"less\s+than\s+([\d.]+)\s+hour", s)
    if lt_match:
        return float(lt_match.group(1)) * 60 * 0.75   # assume 75% of upper bound

    # Range: "3 - 6 Months" / "1 - 4 Weeks"
    range_match = re.search(r"([\d.]+)\s*[-–to]+\s*([\d.]+)\s*(month|week|hour|day)", s)
    if range_match:
        lo, hi   = float(range_match.group(1)), float(range_match.group(2))
        mid      = (lo + hi) / 2
        unit     = range_match.group(3)
        if "month" in unit:   return round(mid * 4 * 3 * 60, 1)   # ~3 hrs/week
        if "week"  in unit:   return round(mid * 3 * 60, 1)        # ~3 hrs/week
        if "hour"  in unit:   return round(mid * 60, 1)
        if "day"   in unit:   return round(mid * 8 * 60, 1)

    # "Approximately X months to complete"
    approx_month = re.search(r"([\d.]+)\s*month", s)
    if approx_month:
        return round(float(approx_month.group(1)) * 4 * 3 * 60, 1)

    # "Approximately X weeks"
    approx_week = re.search(r"([\d.]+)\s*week", s)
    if approx_week:
        return round(float(approx_week.group(1)) * 3 * 60, 1)

    # "Approx. 14 hours to complete" / "14 hours"
    hours_match = re.search(r"([\d.]+)\s*hour", s)
    mins_match  = re.search(r"([\d.]+)\s*min",  s)
    total = 0.0
    if hours_match:
        total += float(hours_match.group(1)) * 60
    if mins_match:
        total += float(mins_match.group(1))
    if total > 0:
        return round(total, 1)

    # Bare float (e.g. Udemy new: content_duration stored as hours)
    try:
        return round(float(s) * 60, 1)
    except ValueError:
        return np.nan


# Unified difficulty map → integer scale 0-3
DIFFICULTY_MAP = {
    # raw string → (int, label)
    "beginner":          (1, "Beginner"),
    "beginner level":    (1, "Beginner"),
    "intermediate":      (2, "Intermediate"),
    "intermediate level":(2, "Intermediate"),
    "advanced":          (3, "Advanced"),
    "advanced level":    (3, "Advanced"),
    "expert level":      (3, "Advanced"),
    "mixed":             (0, "Mixed"),
    "mixed/not-mentioned":(0,"Mixed"),
    "not-mentioned":     (0, "Mixed"),
    "not found":         (0, "Mixed"),
    "all levels":        (0, "Mixed"),
    # numeric (edx/udacity)
    "0":                 (1, "Beginner"),
    "1":                 (2, "Intermediate"),
    "2":                 (3, "Advanced"),
}

def parse_difficulty(value):
    if pd.isna(value):
        return 0, "Mixed"
    key = str(value).strip().lower()
    # If it starts with 'subtitles:' it's a misplaced field — treat as Mixed
    if key.startswith("subtitles"):
        return 0, "Mixed"
    return DIFFICULTY_MAP.get(key, (0, "Mixed"))


CERT_DURATION_ESTIMATES = {
    "SPECIALIZATION":           480,
    "COURSE":                   120,
    "PROFESSIONAL CERTIFICATE": 240,
    "Professional Certificate": 240,
}

DIFFICULTY_LABELS = {0: "Mixed", 1: "Beginner", 2: "Intermediate", 3: "Advanced"}

def normalise_score(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NLP Subject Tagger (same domain seeds, applied to all sources)
# ════════════════════════════════════════════════════════════════════════════

DOMAIN_SEEDS = {
    "Machine Learning & AI":       ["machine learning","deep learning","neural network",
                                    "tensorflow","pytorch","keras","nlp","computer vision",
                                    "reinforcement","artificial intelligence"," ai ","scikit",
                                    "llm","transformer","generative","chatgpt","bert"],
    "Data Science & Analytics":    ["data science","data analyst","analytics","statistics",
                                    "pandas","excel","tableau","power bi","sql","database",
                                    "visualization","big data","data engineering","spark"],
    "Programming & Software Dev":  ["python","javascript","java"," c++ ","c#","programming",
                                    "software","algorithm","data structure","coding","developer",
                                    "react","node","backend","frontend","api","django","flask",
                                    "swift","kotlin","ruby","scala"],
    "Web Development":             ["web development","html","css","web design","wordpress",
                                    "bootstrap","angular","vue","full stack","web app"],
    "Cloud & DevOps":              ["cloud","aws","azure","google cloud","devops","docker",
                                    "kubernetes","terraform","linux","networking","devsecops",
                                    "ci/cd","ansible","jenkins"],
    "Cybersecurity":               ["security","cybersecurity","hacking","ethical","penetration",
                                    "cryptography","firewall","soc","comptia","cissp","sscp"],
    "Business & Finance":          ["business","finance","accounting","economics","management",
                                    "marketing","strategy","investment","trading","startup",
                                    "entrepreneurship","leadership","mba","project management"],
    "Graphic Design & Media":      ["design","photoshop","illustrator","graphic","ui","ux",
                                    "animation","video","photography","creative","branding",
                                    "figma","canva","adobe","after effects"],
    "Health & Medicine":           ["health","medicine","medical","biology","anatomy","nursing",
                                    "clinical","nutrition","fitness","psychology","pharmacology"],
    "Personal Development":        ["happiness","productivity","mindfulness","career",
                                    "communication","public speaking","emotional","wellbeing",
                                    "life","habits","self","motivation"],
    "Science & Engineering":       ["physics","chemistry","engineering","mathematics","calculus",
                                    "algebra","science","research","quantum","robotics",
                                    "mechanical","electrical","civil","biomedical"],
    "Law & Social Sciences":       ["law","legal","rights","policy","social","humanities",
                                    "history","philosophy","ethics","political","sociology"],
    "Music & Arts":                ["music","piano","guitar","singing","instrument",
                                    "art","drawing","painting","composition","theory"],
}

def tag_subject(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Other"
    t = " " + text.lower() + " "
    scores = defaultdict(int)
    for domain, seeds in DOMAIN_SEEDS.items():
        for seed in seeds:
            if seed in t:
                scores[domain] += 1
    return max(scores, key=scores.get) if scores else "Other"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Per-source parsers → unified schema
# ════════════════════════════════════════════════════════════════════════════

UNIFIED_COLS = [
    "course_id","title","subject","provider","source",
    "difficulty_level","difficulty_label","rating",
    "enrollment_count","num_reviews","duration_mins",
    "price","is_free","cert_type","skills_raw",
    "duration_estimated","popularity_score","course_url",
]

def make_slug(title: str) -> str:
    s = str(title).lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

def construct_url(source: str, title: str) -> str:
    slug = make_slug(title)
    if "coursera"     in source: return f"https://www.coursera.org/learn/{slug}"
    if "udemy"        in source: return f"https://www.udemy.com/course/{slug}/"
    if "edx"          in source: return f"https://www.edx.org/learn/{slug}"
    if "udacity"      in source: return f"https://www.udacity.com/course/{slug}"
    if "future_learn" in source: return f"https://www.futurelearn.com/courses/{slug}"
    if "simplilearn"  in source: return f"https://www.simplilearn.com/courses/{slug}"
    return ""

def _make_id(prefix, index):
    return f"{prefix}_{index}"


# ── Source 1: Original Coursera ───────────────────────────────────────────────
def parse_original_coursera(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    rows = []
    for i, r in df.iterrows():
        dl, dlabel = parse_difficulty(r.get("course_difficulty",""))
        enrolled   = parse_enrollment(r.get("course_students_enrolled", 0))
        cert       = str(r.get("course_Certificate_type","")).strip()
        # Duration estimated from cert type (no real duration in this dataset)
        dur   = CERT_DURATION_ESTIMATES.get(cert, 120)
        est   = True
        rows.append({
            "course_id":        _make_id("OC", i),
            "title":            str(r.get("course_title","")).strip(),
            "subject":          np.nan,
            "provider":         str(r.get("course_organization","")).strip(),
            "source":           "coursera",
            "difficulty_level": dl,
            "difficulty_label": dlabel,
            "rating":           parse_rating(r.get("course_rating", np.nan)),
            "enrollment_count": enrolled,
            "num_reviews":      np.nan,
            "duration_mins":    dur,
            "price":            np.nan,
            "is_free":          False,
            "cert_type":        cert,
            "skills_raw":       np.nan,
            "duration_estimated": est,
            "course_url":       construct_url("coursera", str(r.get("course_title",""))),
        })
    return pd.DataFrame(rows)


# ── Source 2: Original Udemy ──────────────────────────────────────────────────
def parse_original_udemy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        dl, dlabel = parse_difficulty(r.get("level",""))
        price_raw  = r.get("price", 0)
        try:
            price = float(str(price_raw).replace("Free","0").strip())
        except (ValueError, TypeError):
            price = 0.0
        dur = parse_duration_minutes(r.get("content_duration",""))
        rows.append({
            "course_id":        _make_id("OU", i),
            "title":            str(r.get("course_title","")).strip(),
            "subject":          str(r.get("subject","")).strip(),
            "provider":         "Udemy",
            "source":           "udemy",
            "difficulty_level": dl,
            "difficulty_label": dlabel,
            "rating":           np.nan,
            "enrollment_count": parse_enrollment(r.get("num_subscribers", 0)),
            "num_reviews":      pd.to_numeric(r.get("num_reviews", np.nan), errors="coerce"),
            "duration_mins":    dur,
            "price":            price,
            "is_free":          price == 0.0,
            "cert_type":        np.nan,
            "skills_raw":       np.nan,
            "duration_estimated": False,
            "course_url":       construct_url("udemy", str(r.get("course_title",""))),
        })
    return pd.DataFrame(rows)


# ── Source 3: Combined Simplilearn/Coursera/Udacity/FutureLearn ───────────────
def parse_simplilearn_combined(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        site = str(r.get("Site","")).strip()
        dl, dlabel = parse_difficulty(r.get("Level",""))
        rating_raw = r.get("Rating", np.nan)
        rating     = parse_rating(rating_raw)
        dur        = parse_duration_minutes(r.get("Duration",""))
        enrolled   = parse_enrollment(r.get("Number of viewers", 0))
        skills     = str(r.get("Skills","")).strip() if pd.notna(r.get("Skills")) else np.nan

        # Derive price from monthly_access fields (rough indicator)
        monthly = r.get("Monthly access", np.nan)
        try:
            price = float(str(monthly).replace("$","").replace(",","").strip())
        except (ValueError, TypeError):
            price = np.nan

        real_url = str(r.get("URL","")).strip()
        if not real_url or real_url == "nan":
            real_url = construct_url(site.lower(), str(r.get("Title","")))
        rows.append({
            "course_id":        _make_id("SC", i),
            "title":            str(r.get("Title","")).strip(),
            "subject":          str(r.get("Category","")).strip(),
            "provider":         str(r.get("Instructors","Coursera")).strip() or site,
            "source":           site.lower().replace(" ","_") if site else "coursera",
            "difficulty_level": dl,
            "difficulty_label": dlabel,
            "rating":           rating,
            "enrollment_count": enrolled,
            "num_reviews":      parse_enrollment(r.get("Number of Reviews", 0)),
            "duration_mins":    dur,
            "price":            price,
            "is_free":          pd.isna(price) or price == 0.0,
            "cert_type":        str(r.get("Program Type","")).strip() or np.nan,
            "skills_raw":       skills,
            "duration_estimated": pd.isna(dur),
            "course_url":       real_url,
        })
    return pd.DataFrame(rows)


# ── Source 4: Combined EdX / Udacity / Coursera ───────────────────────────────
def parse_edx_udacity(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        dl, dlabel = parse_difficulty(str(r.get("Level", "0")))
        platform   = str(r.get("Platform","edx")).strip()
        skills     = str(r.get("Skills","")).strip() if pd.notna(r.get("Skills")) else np.nan
        rows.append({
            "course_id":        _make_id("EU", i),
            "title":            str(r.get("Course_Name","")).strip(),
            "subject":          np.nan,
            "provider":         platform,
            "source":           platform.lower(),
            "difficulty_level": dl,
            "difficulty_label": dlabel,
            "rating":           parse_rating(r.get("Rating", np.nan)),
            "enrollment_count": parse_enrollment(r.get("Number of students", 0)),
            "num_reviews":      np.nan,
            "duration_mins":    np.nan,
            "price":            np.nan,
            "is_free":          False,
            "cert_type":        np.nan,
            "skills_raw":       skills,
            "duration_estimated": True,
            "course_url":       construct_url(platform.lower(), str(r.get("Course_Name",""))),
        })
    return pd.DataFrame(rows)


# ── Source 5: Coursera 1000 ────────────────────────────────────────────────────
def parse_coursera_1000(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        dl, dlabel = parse_difficulty(r.get("Difficulty",""))
        dur        = parse_duration_minutes(r.get("Duration",""))
        cert       = str(r.get("Type_Of_Certificate","")).strip()
        # Use cert duration estimate if parsed duration is still NaN
        if pd.isna(dur):
            dur = CERT_DURATION_ESTIMATES.get(cert, 120)
            est = True
        else:
            est = False
        skills = str(r.get("Skills","")).strip() if "Skills" in df.columns and pd.notna(r.get("Skills")) else np.nan
        rows.append({
            "course_id":        _make_id("C1", i),
            "title":            str(r.get("Course_Name","")).strip(),
            "subject":          np.nan,
            "provider":         str(r.get("Company_Name","Coursera")).strip(),
            "source":           "coursera",
            "difficulty_level": dl,
            "difficulty_label": dlabel,
            "rating":           parse_rating(r.get("Ratings", np.nan)),
            "enrollment_count": 0,
            "num_reviews":      np.nan,
            "duration_mins":    dur,
            "price":            np.nan,
            "is_free":          False,
            "cert_type":        cert,
            "skills_raw":       skills,
            "duration_estimated": est,
            "course_url":       construct_url("coursera", str(r.get("Course_Name",""))),
        })
    return pd.DataFrame(rows)


# ── Source 6: Coursera Catalog ────────────────────────────────────────────────
def parse_coursera_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        level_raw  = str(r.get("course_level","")).strip()
        dl, dlabel = parse_difficulty(level_raw)
        dur        = parse_duration_minutes(r.get("time_required",""))
        skills     = str(r.get("course_skills","")).strip() if pd.notna(r.get("course_skills","")) else np.nan
        cat        = str(r.get("category","")).strip()
        sub_cat    = str(r.get("sub_category","")).strip() if "sub_category" in df.columns else ""
        subject    = sub_cat if sub_cat and sub_cat != "nan" else cat
        real_link = str(r.get("course_link","")).strip()
        if not real_link or real_link == "nan":
            real_link = construct_url("coursera", str(r.get("course_name","")))
        rows.append({
            "course_id":        _make_id("CC", i),
            "title":            str(r.get("course_name","")).strip(),
            "subject":          subject if subject != "nan" else np.nan,
            "provider":         str(r.get("university_name","Coursera")).strip(),
            "source":           "coursera",
            "difficulty_level": dl,
            "difficulty_label": dlabel,
            "rating":           parse_rating(r.get("course_rating", np.nan)),
            "enrollment_count": 0,
            "num_reviews":      np.nan,
            "duration_mins":    dur,
            "price":            np.nan,
            "is_free":          False,
            "cert_type":        str(r.get("course_type","")).strip() or np.nan,
            "skills_raw":       skills,
            "duration_estimated": pd.isna(dur),
            "course_url":       real_link,
        })
    return pd.DataFrame(rows)


# ── Source 7: Udemy Extended ──────────────────────────────────────────────────
def parse_udemy_new(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        dl, dlabel = parse_difficulty(r.get("level",""))
        # content_duration already in hours (float)
        dur_hrs    = r.get("content_duration", np.nan)
        dur        = float(dur_hrs) * 60 if pd.notna(dur_hrs) else np.nan
        price_raw  = r.get("price", 0)
        try:
            price = float(str(price_raw).replace("Free","0").strip())
        except (ValueError, TypeError):
            price = 0.0
        real_url_u = str(r.get("url","")).strip()
        if not real_url_u or real_url_u == "nan":
            real_url_u = construct_url("udemy", str(r.get("course_title","")))
        rows.append({
            "course_id":        _make_id("UN", i),
            "title":            str(r.get("course_title","")).strip(),
            "subject":          str(r.get("subject","")).strip(),
            "provider":         "Udemy",
            "source":           "udemy",
            "difficulty_level": dl,
            "difficulty_label": dlabel,
            "rating":           np.nan,
            "enrollment_count": parse_enrollment(r.get("num_subscribers", 0)),
            "num_reviews":      pd.to_numeric(r.get("num_reviews", np.nan), errors="coerce"),
            "duration_mins":    dur,
            "price":            price,
            "is_free":          price == 0.0,
            "cert_type":        np.nan,
            "skills_raw":       np.nan,
            "duration_estimated": False,
            "course_url":       real_url_u,
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Main pipeline
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    log("\n" + "="*65)
    log("ADAPTIVE LEARNING PLATFORM — DATA PREPROCESSING v2 (7 sources)")
    log("="*65)

    # ── Load all sources ──────────────────────────────────────────────────
    sources = [
        ("Original Coursera",        parse_original_coursera,  "coursea_data.csv"),
        ("Original Udemy",           parse_original_udemy,     "Udemy_Courses.csv"),
        ("Simplilearn/Coursera/etc", parse_simplilearn_combined,"combined_simplilearn_coursera.csv"),
        ("EdX/Udacity/Coursera",     parse_edx_udacity,        "comined_edx_udacity_coursera.csv"),
        ("Coursera Top-1000",        parse_coursera_1000,      "coursera_1000_Courses.csv"),
        ("Coursera Catalog",         parse_coursera_catalog,   "Coursera_courses_catalog.csv"),
        ("Udemy Extended",           parse_udemy_new,          "udemy_courses.csv"),
    ]

    frames = []
    for name, parser, filename in sources:
        try:
            df = parser(filename)
            log(f"[LOAD] {name:<30} {len(df):>6,} rows")
            frames.append(df)
        except FileNotFoundError:
            log(f"[SKIP] {name} — {filename} not found")

    # ── Merge ─────────────────────────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)
    log(f"\n[MERGE] Total before dedup: {len(df):,}")

    # ── Deduplicate ───────────────────────────────────────────────────────
    # Normalise titles for comparison: lowercase, strip punctuation
    df["_title_norm"] = (
        df["title"]
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )

    # Keep the row with the richest data for each duplicate title
    # Priority: has rating > has enrollment > has duration
    df["_richness"] = (
        df["rating"].notna().astype(int)
        + df["enrollment_count"].gt(0).astype(int)
        + df["duration_mins"].notna().astype(int)
    )
    df = df.sort_values("_richness", ascending=False)
    before = len(df)
    df = df.drop_duplicates(subset=["_title_norm"])
    df = df.drop(columns=["_title_norm", "_richness"])
    log(f"[DEDUP] Removed {before - len(df):,} duplicates — remaining: {len(df):,}")

    # ── Remove empty / garbage titles ─────────────────────────────────────
    df = df[df["title"].str.len() > 3]
    df = df[~df["title"].str.lower().isin(["nan", "none", ""])]
    log(f"[CLEAN] After title filter: {len(df):,}")

    # ── NLP subject tagger ────────────────────────────────────────────────
    log("\n[NLP] Tagging subjects from titles + skills ...")
    def _tag(row):
        # Combine title + skills_raw for richer signal
        text = str(row["title"]) + " " + str(row.get("skills_raw",""))
        return tag_subject(text)

    df["subject_nlp"] = df.apply(_tag, axis=1)

    # Use existing subject when available and not generic
    generic = {"nan", "none", "", "other", "data science",
               "business finance", "musical instruments",
               "graphic design", "web development"}
    df["subject"] = df.apply(
        lambda r: (
            r["subject_nlp"]
            if pd.isna(r["subject"]) or str(r["subject"]).strip().lower() in generic
            else str(r["subject"]).strip()
        ),
        axis=1
    )
    df = df.drop(columns=["subject_nlp"])

    subject_counts = df["subject"].value_counts()
    log("[SUBJECTS] Distribution:")
    for subj, cnt in subject_counts.items():
        log(f"  {subj:<40} {cnt:>5}")

    # ── Consolidate messy subject labels into 13 standard categories ──────
    log("\n[SUBJECTS] Consolidating into standard taxonomy ...")
    SUBJECT_CONSOLIDATION = {
        # Business
        "Business":                     "Business & Finance",
        "Business Finance":             "Business & Finance",
        "Business Strategy":            "Business & Finance",
        "Business Essentials":          "Business & Finance",
        "Entrepreneurship":             "Business & Finance",
        "Finance":                      "Business & Finance",
        "Marketing":                    "Business & Finance",
        "Economics":                    "Business & Finance",
        "Leadership and Management":    "Business & Finance",
        "Negocios":                     "Business & Finance",
        # Data / ML
        "Data Science":                 "Data Science & Analytics",
        "Data Analysis":                "Data Science & Analytics",
        "Data Management":              "Data Science & Analytics",
        "Probability and Statistics":   "Data Science & Analytics",
        "Math and Logic":               "Science & Engineering",
        "Algorithms":                   "Programming & Software Dev",
        "Machine Learning":             "Machine Learning & AI",
        # Tech
        "Computer Science":             "Programming & Software Dev",
        "Software Development":         "Programming & Software Dev",
        "Mobile and Web Development":   "Web Development",
        "Information Technology":       "Cloud & DevOps",
        "Cloud Computing":              "Cloud & DevOps",
        "Computer Security and Networks":"Cybersecurity",
        "Security":                     "Cybersecurity",
        "Networking":                   "Cloud & DevOps",
        "Support and Operations":       "Cloud & DevOps",
        # Science
        "Physical Science and Engineering": "Science & Engineering",
        "Electrical Engineering":       "Science & Engineering",
        "Mechanical Engineering":       "Science & Engineering",
        "Basic Science":                "Science & Engineering",
        "Chemistry":                    "Science & Engineering",
        "Physics and Astronomy":        "Science & Engineering",
        "Environmental Science and Sustainability": "Science & Engineering",
        "Research Methods":             "Science & Engineering",
        "Research":                     "Science & Engineering",
        # Health
        "Health":                       "Health & Medicine",
        "Public Health":                "Health & Medicine",
        "Patient Care":                 "Health & Medicine",
        "Healthcare Management":        "Health & Medicine",
        "Health Informatics":           "Health & Medicine",
        "Nutrition":                    "Health & Medicine",
        "Psychology":                   "Health & Medicine",
        "Animal Health":                "Health & Medicine",
        # Arts / Design
        "Design and Product":           "Graphic Design & Media",
        "Music and Art":                "Music & Arts",
        "Arts and Humanities":          "Music & Arts",
        # Social / Law
        "Social Sciences":              "Law & Social Sciences",
        "Governance and Society":       "Law & Social Sciences",
        "Philosophy":                   "Law & Social Sciences",
        "History":                      "Law & Social Sciences",
        "Law":                          "Law & Social Sciences",
        # Personal dev
        "Education":                    "Personal Development",
        "Language Learning":            "Personal Development",
        "Learning English":             "Personal Development",
        "Other Languages":              "Personal Development",
        # Other → keep as-is or remap
        "Musical Instruments":          "Music & Arts",
        "Graphic Design":               "Graphic Design & Media",
        "Web Development":              "Web Development",
        "データサイエンス":               "Data Science & Analytics",
        "计算机科学":                    "Programming & Software Dev",
        "Ciencias de la Computación":   "Programming & Software Dev",
        "Tecnologia da informação":     "Cloud & DevOps",
        "not-mentioned":                "Other",
    }

    df["subject"] = df["subject"].replace(SUBJECT_CONSOLIDATION)

    # Anything still not in our 13 standard categories → tag via NLP again
    STANDARD_SUBJECTS = set(DOMAIN_SEEDS.keys()) | {"Other"}
    mask_other = ~df["subject"].isin(STANDARD_SUBJECTS)
    log(f"[SUBJECTS] Re-tagging {mask_other.sum()} rows with non-standard subjects ...")
    df.loc[mask_other, "subject"] = df.loc[mask_other, "title"].apply(tag_subject)

    subject_counts_final = df["subject"].value_counts()
    log("[SUBJECTS] Final distribution:")
    for subj, cnt in subject_counts_final.items():
        log(f"  {subj:<40} {cnt:>5}")

    # ── Cap extreme duration outliers ──────────────────────────────────────
    # 99th percentile cap — keeps realistic courses, drops data entry errors
    dur_p99 = df["duration_mins"].quantile(0.99)
    n_capped = (df["duration_mins"] > dur_p99).sum()
    df["duration_mins"] = df["duration_mins"].clip(upper=dur_p99)
    log(f"\n[DURATION] Capped {n_capped} outliers at 99th percentile ({dur_p99:.0f} mins)")

    # ── Fill remaining duration nulls ─────────────────────────────────────
    log("\n[DURATION] Filling remaining nulls ...")
    known_median = df["duration_mins"].dropna().median()
    log(f"[DURATION] Median of known durations: {known_median:.0f} mins")
    df["duration_mins"] = df["duration_mins"].fillna(known_median)

    # ── Fill rating nulls with source-level median ────────────────────────
    source_rating_median = df.groupby("source")["rating"].transform(
        lambda x: x.fillna(x.median())
    )
    global_rating_median = df["rating"].median()
    df["rating"] = source_rating_median.fillna(global_rating_median)

    # ── Popularity score ──────────────────────────────────────────────────
    df["enrollment_count"] = df["enrollment_count"].fillna(0)
    df["popularity_score"] = (
        0.7 * normalise_score(df["enrollment_count"]) +
        0.3 * normalise_score(df["rating"])
    ).round(4)

    # ── Final schema ──────────────────────────────────────────────────────
    final_cols = [
        "course_id","title","subject","provider","source",
        "difficulty_level","difficulty_label","rating",
        "enrollment_count","num_reviews","duration_mins",
        "price","is_free","cert_type","skills_raw",
        "duration_estimated","popularity_score","course_url",
    ]
    df_final = df[final_cols].reset_index(drop=True)
    df_final.to_csv("data/courses.csv", index=False)

    log(f"\n{'='*65}")
    log(f"OUTPUT → data/courses.csv")
    log(f"Total courses : {len(df_final):,}")
    log(f"Sources breakdown:")
    for src, cnt in df_final["source"].value_counts().items():
        log(f"  {src:<25} {cnt:>5,}")
    log(f"\nDifficulty distribution:")
    for diff, cnt in df_final["difficulty_label"].value_counts().items():
        log(f"  {diff:<15} {cnt:>5,}")
    log(f"\nRating range  : {df_final['rating'].min():.1f} – {df_final['rating'].max():.1f}")
    log(f"Duration range: {df_final['duration_mins'].min():.0f} – {df_final['duration_mins'].max():.0f} mins")
    log(f"Null ratings  : {df_final['rating'].isna().sum()}")
    log(f"Null duration : {df_final['duration_mins'].isna().sum()}")

    log(f"\nSample rows:")
    log(df_final[["course_id","title","subject","difficulty_label",
                  "rating","popularity_score","source"]].head(5).to_string())
    log("="*65)

    with open("data/preprocessing_report.txt","w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n[DONE] Report saved → data/preprocessing_report.txt")
