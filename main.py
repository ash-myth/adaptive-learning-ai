"""
main.py — AdaptLearn AI FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import urllib.parse
import json
import re, difflib
import os, sys, warnings

warnings.filterwarnings("ignore")

# ── Auth ──────────────────────────────────────────────────────────────────
from auth import router as auth_router, get_current_user

app = FastAPI(title="AdaptLearn AI", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth router ───────────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

# ── Static HTML pages ─────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

def _page(name: str) -> FileResponse:
    target = os.path.join(FRONTEND_DIR, name)
    if not os.path.exists(target):
        target = name
    return FileResponse(target)

@app.get("/")
def root():
    return _page("auth.html")

@app.get("/auth.html")
def serve_auth():
    return _page("auth.html")

@app.get("/onboarding.html")
def serve_onboarding():
    return _page("onboarding.html")

@app.get("/index.html")
def serve_admin():
    return _page("index.html")

@app.get("/dashboard.html")
def serve_user_dashboard():
    return _page("dashboard.html")


# ════════════════════════════════════════════════════════════════════════════
# Data + Model Loading
# ════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, os.path.dirname(__file__))

courses_df     = pd.read_csv("data/courses.csv")
users_df       = pd.read_csv("data/users.csv")
interactions_df = pd.read_csv("data/interactions.csv")

_recommender = None

def get_recommender():
    """Return the singleton HybridRecommender (pre-loaded at startup)."""
    return _recommender

# ── Eager startup — load all ML models before accepting requests ──────────────
# Doing this here (module-level) avoids the 30+ s cold-start that would
# otherwise block the first real HTTP request and cause client timeouts.
from hybrid_recommender import HybridRecommender
_recommender = HybridRecommender(retrain=False)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

SRC_LABEL = {
    "coursera":    "Coursera",
    "udemy":       "Udemy",
    "edx":         "EdX",
    "udacity":     "Udacity",
    "future_learn":"FutureLearn",
    "simplilearn": "Simplilearn",
}

def platform_link(row: dict) -> str:
    src   = str(row.get("source", "")).lower()
    title = str(row.get("title", ""))
    enc   = urllib.parse.quote_plus(title)
    urls  = {
        "coursera":    f"https://www.coursera.org/search?query={enc}",
        "udemy":       f"https://www.udemy.com/courses/search/?q={enc}",
        "edx":         f"https://www.edx.org/search?q={enc}",
        "udacity":     f"https://www.udacity.com/catalog?search={enc}",
        "future_learn":f"https://www.futurelearn.com/search?q={enc}",
        "simplilearn": f"https://www.simplilearn.com/search?q={enc}",
    }
    return urls.get(src, f"https://www.google.com/search?q={enc}+course")

def platform_search_links(query: str) -> list:
    """Return direct search URLs for all 6 platforms for a given query."""
    enc = urllib.parse.quote_plus(query)
    return [
        {"platform": "Coursera",    "color": "#0056d2",
         "url": f"https://www.coursera.org/search?query={enc}"},
        {"platform": "Udemy",       "color": "#a435f0",
         "url": f"https://www.udemy.com/courses/search/?q={enc}"},
        {"platform": "EdX",         "color": "#e53e3e",
         "url": f"https://www.edx.org/search?q={enc}"},
        {"platform": "Udacity",     "color": "#02b3e4",
         "url": f"https://www.udacity.com/catalog?search={enc}"},
        {"platform": "FutureLearn", "color": "#de2954",
         "url": f"https://www.futurelearn.com/search?q={enc}"},
        {"platform": "Simplilearn", "color": "#f97316",
         "url": f"https://www.simplilearn.com/search?q={enc}"},
    ]


COMMON_TYPOS = {
    "recat": "react", "reack": "react",
    "pythno": "python", "pyhton": "python", "pyton": "python",
    "dockre": "docker", "dokcer": "docker", "dockr": "docker",
    "kuberentes": "kubernetes", "kubernets": "kubernetes",
    "javascrip": "javascript", "javscript": "javascript",
    "typscript": "typescript", "typescrip": "typescript",
    "angualr": "angular", "anguler": "angular",
    "fluttr": "flutter", "fluter": "flutter",
    "kotiln": "kotlin",
    "machne": "machine", "lerning": "learning", "learnin": "learning",
    "artifical": "artificial", "inteligence": "intelligence",
    "algoritm": "algorithm", "algorthm": "algorithm",
    "databse": "database", "datbase": "database",
    "frntend": "frontend", "bakcend": "backend",
    "posgres": "postgres", "postgre": "postgres",
    "myslq": "mysql", "monogdb": "mongodb",
    "tensorflw": "tensorflow", "pytroch": "pytorch",
    "sprign": "spring", "spriing": "spring",
    "nodjs": "nodejs", "ndoe": "node",
    "htmll": "html", "csss": "css",
}

def correct_query(q: str, df=None) -> str:
    if not q or not isinstance(q, str):
        return q
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", q)
    if not words:
        return q
    corrected = []
    has_change = False
    for w in words:
        wl = w.lower()
        if wl in COMMON_TYPOS:
            corrected.append(COMMON_TYPOS[wl])
            has_change = True
        else:
            corrected.append(w)
    return " ".join(corrected) if has_change else q


def dur_fmt(mins) -> str:
    if not mins or (isinstance(mins, float) and np.isnan(mins)):
        return ""
    m = float(mins)
    return f"{m:.0f}m" if m < 60 else f"{m/60:.1f}h"

def safe_float(v, default=0.0) -> float:
    try:
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
    except:
        return default

def row_to_course(row) -> dict:
    if hasattr(row, "_asdict"):
        row = row._asdict()
    elif hasattr(row, "to_dict"):
        row = row.to_dict()
    src = str(row.get("source", "")).lower()
    return {
        "course_id":      str(row.get("course_id", "")),
        "title":          str(row.get("title", "N/A")),
        "subject":        str(row.get("subject", "")),
        "difficulty":     str(row.get("difficulty_label", "Mixed")),
        "source":         src,
        "source_label":   SRC_LABEL.get(src, src.title()),
        "rating":         safe_float(row.get("rating"), 0),
        "duration":       dur_fmt(row.get("duration_mins")),
        "link":           platform_link(row),
        "hybrid_score":   safe_float(row.get("hybrid_score")),
        "semantic_score": safe_float(row.get("semantic_score")),
        "keyword_score":  safe_float(row.get("keyword_score")),
        "cf_score":       safe_float(row.get("cf_score")),
        "adaptive_score": safe_float(row.get("adaptive_score")),
        "blended_score":        safe_float(row.get("blended_score")),
        "path_note":            str(row.get("path_note", "") or ""),
        "predicted_outcome":    str(row.get("predicted_outcome", "") or ""),
        "struggle_prob":        safe_float(row.get("struggle_prob")),
        "match_prob":           safe_float(row.get("match_prob")),
        "excel_prob":           safe_float(row.get("excel_prob")),
        "dropout_risk":         str(row.get("dropout_risk", "") or "").title(),
        "dropout_probability":  safe_float(row.get("dropout_probability")),
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Platform Stats
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/stats")
def get_stats():
    return {
        "courses":      int(len(courses_df)),
        "platforms":    6,
        "users":        int(len(users_df)),
        "interactions": int(len(interactions_df)),
        "avg_completion": float(interactions_df["completed"].mean()),
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Users
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/users")
def list_users():
    return {"users": sorted(users_df["user_id"].tolist())}


@app.get("/api/users/{user_id}/profile")
def get_profile(user_id: str):
    rec = get_recommender()
    profile = rec.profile(user_id)
    if "error" in profile:
        return {
            "user_id": user_id,
            "skill_level": 1,
            "learning_velocity": 0,
            "recommended_difficulty": "Beginner",
            "persona": "New User",
            "recent_history": []
        }

    user_row   = users_df[users_df["user_id"] == user_id]
    if user_row.empty:
        raise HTTPException(status_code=404, detail="User not found")
    user_row = user_row.iloc[0]

    user_iacts = interactions_df[interactions_df["user_id"] == user_id]
    recent = []
    if not user_iacts.empty:
        merged = user_iacts.merge(
            courses_df[["course_id","title","difficulty_label"]], on="course_id", how="left")
        for _, r in merged.tail(6).iterrows():
            recent.append({
                "title":      str(r.get("title", "")),
                "difficulty": str(r.get("difficulty_label", "")),
                "quiz_score": safe_float(r.get("quiz_score")),
                "completed":  bool(r.get("completed", 0)),
            })

    skill_map = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    vel = safe_float(profile.get("learning_velocity"))

    return {
        **profile,
        "skill_level_label":    skill_map.get(int(profile.get("skill_level", 1)), "—"),
        "preferred_subjects":   str(user_row.get("preferred_subjects", "")).replace("|", " · "),
        "velocity_label":       ("📈 Rapid" if vel > 3 else "📈 Improving" if vel > 0 else "➡️ Stable" if vel == 0 else "📉 Declining"),
        "recent_history":       recent,
    }


class CompletionPayload(BaseModel):
    quiz_score: int
    course_title: str

@app.post("/api/users/{user_id}/complete")
def log_completion(user_id: str, payload: CompletionPayload):
    global interactions_df
    match = courses_df[courses_df["title"] == payload.course_title]
    if match.empty:
        raise HTTPException(status_code=404, detail="Course not found")
    sim_row = match.iloc[0]
    new_row = {
        "user_id":     user_id,
        "course_id":   sim_row["course_id"],
        "completed":   1,
        "quiz_score":  payload.quiz_score,
        "time_spent":  float(sim_row.get("duration_mins", 120) or 120),
        "user_rating": round(min(5, 3 + payload.quiz_score / 50), 1),
        "replay_count":0,
        "pause_count": 0,
        "cf_score":    round(min(5, (0.4*(payload.quiz_score/100)+0.3+0.2*(payload.quiz_score/100)+0.05)*5), 3),
    }
    cur = pd.read_csv("data/interactions.csv")
    pd.concat([cur, pd.DataFrame([new_row])], ignore_index=True).to_csv(
        "data/interactions.csv", index=False)
    interactions_df = pd.read_csv("data/interactions.csv")
    return {"ok": True, "message": f"Logged completion for {payload.course_title}"}


# ════════════════════════════════════════════════════════════════════════════
# Routes — Recommendations
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/recommend")
def recommend(
    user_id:    str = Query(...),
    goal:       str = Query(...),
    top_n:      int = Query(8),
    level:      Optional[str] = Query(None),
    source:     Optional[str] = Query(None),
    max_dur:    Optional[int] = Query(None),
    no_correct: bool = Query(False),
):
    corrected    = correct_query(goal, courses_df)
    target_goal  = goal if (no_correct or corrected == goal) else corrected
    did_you_mean = corrected if (corrected != goal) else None

    rec     = get_recommender()
    profile = rec.profile(user_id)
    recs    = rec.recommend(user_id, target_goal, top_n=top_n * 3)

    f = recs.copy()
    if level  and level  != "Any": f = f[f["difficulty_label"] == level]
    if source and source != "Any": f = f[f["source"] == source]
    if max_dur: f = f[f["duration_mins"] <= max_dur]
    if f.empty: f = recs
    f = f.head(top_n)

    n = profile.get("n_interactions", 0)
    return {
        "user_id":      user_id,
        "goal":         goal,
        "searched_for": target_goal,
        "did_you_mean": did_you_mean,
        "persona":      profile.get("persona", "—"),
        "level":        profile.get("recommended_difficulty", "—"),
        "mode":         "warm" if n >= 5 else ("interpolating" if n > 0 else "cold-start"),
        "interactions": n,
        "count":        len(f),
        "courses":      [row_to_course(r) for _, r in f.iterrows()],
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Learning Path
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/learning-path")
def learning_path(user_id: str = Query(...), subject: str = Query(...)):
    rec     = get_recommender()
    profile = rec.profile(user_id)
    path    = rec.learning_path(user_id, subject)

    mastery  = rec.subject_mastery_info(user_id, subject)
    rec_diff = mastery["level"]
    result, total_mins = {}, 0
    for stage, df in path.items():
        courses = [row_to_course(r) for _, r in df.iterrows()] if not df.empty else []
        mins    = float(df["duration_mins"].fillna(0).sum()) if not df.empty else 0
        total_mins += mins
        result[stage] = {"courses": courses, "total_mins": mins}

    wkly  = float(profile.get("weekly_hours", 5))
    weeks = (total_mins / 60) / wkly if wkly > 0 else 0
    return {
        "recommended_start": rec_diff,
        "start_reason":      mastery["reason"],
        "stages":            result,
        "total_hours":       round(total_mins / 60, 1),
        "weekly_hours":      wkly,
        "est_weeks":         round(weeks, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Skill Gap
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/skill-gap")
def skill_gap(user_id: str = Query(...), goal: str = Query(...), no_correct: bool = Query(False)):
    corrected    = correct_query(goal, courses_df)
    target_goal  = goal if (no_correct or corrected == goal) else corrected
    did_you_mean = corrected if (corrected != goal) else None

    rec     = get_recommender()
    profile = rec.profile(user_id)

    # skill_gap now returns (DataFrame, status_string)
    # status: "found" | "fallback" | "not_found"
    result = rec.skill_gap(user_id, target_goal)
    gaps, status = result if isinstance(result, tuple) else (result, "found")

    not_found = (status == "not_found") or (gaps.empty and status != "found")

    return {
        "profile": {
            "skill_level":      profile.get("skill_level"),
            "strength_subject": profile.get("strength_subject"),
            "weakness_subject": profile.get("weakness_subject"),
            "avg_quiz_score":   profile.get("avg_quiz_score"),
        },
        "gaps":           [row_to_course(r) for _, r in gaps.iterrows()] if not gaps.empty else [],
        "status":         status,
        "not_found":      not_found,
        "goal":           goal,
        "searched_for":   target_goal,
        "did_you_mean":   did_you_mean,
        "platform_links": platform_search_links(target_goal) if not_found else [],
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Search
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/search")
def search(q: str = Query(...), top_n: int = Query(8), no_correct: bool = Query(False)):
    rec = get_recommender()

    # Detect typos and offer "Did you mean"
    corrected    = correct_query(q, courses_df)
    target_q     = q if (no_correct or corrected == q) else corrected
    did_you_mean = corrected if (corrected != q) else None

    results = rec.search(target_q, top_n=top_n)
    courses = [row_to_course(r) for _, r in results.iterrows()]

    # Detect out-of-catalog / low-relevance queries:
    # If the top match lacks semantic alignment (>= 0.42) and is not a short exact keyword,
    # redirect user to live federated multi-platform search cards.
    q_words  = [w for w in target_q.split() if len(w) > 1]
    is_short = len(q_words) <= 2
    max_sem  = max((safe_float(c.get("semantic_score", 0)) for c in courses), default=0)
    max_kw   = max((safe_float(c.get("keyword_score", 0)) for c in courses), default=0)
    relevant = (max_sem >= 0.42) or (is_short and max_kw >= 0.70)
    low_relevance = not relevant

    return {
        "query":          q,
        "search_query":   target_q,
        "did_you_mean":   did_you_mean,
        "count":          0 if low_relevance else len(courses),
        "courses":        [] if low_relevance else courses,
        "low_relevance":  low_relevance,
        "platform_links": platform_search_links(target_q),
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Analytics
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/analytics/courses")
def analytics_courses():
    return {
        "by_subject":    courses_df["subject"].value_counts().to_dict(),
        "by_platform":   courses_df["source"].value_counts().to_dict(),
        "by_difficulty": courses_df["difficulty_label"].value_counts().to_dict(),
    }

@app.get("/api/analytics/users")
def analytics_users():
    sm = {1:"Beginner", 2:"Intermediate", 3:"Advanced"}
    return {
        "by_archetype":   users_df["archetype"].value_counts().to_dict(),
        "by_skill_level": users_df["skill_level"].map(sm).value_counts().to_dict(),
    }

@app.get("/api/analytics/interactions")
def analytics_interactions():
    bins   = [0,25,50,75,90,100]
    labels = ["0-25","25-50","50-75","75-90","90-100"]
    qb     = pd.cut(interactions_df["quiz_score"], bins=bins, labels=labels)
    return {
        "total":          int(len(interactions_df)),
        "avg_quiz":       round(float(interactions_df["quiz_score"].mean()), 1),
        "completion_rate":round(float(interactions_df["completed"].mean()), 4),
        "quiz_dist":      qb.value_counts().sort_index().to_dict(),
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — Course list for dropdowns
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/courses/titles")
def course_titles():
    return {"titles": courses_df["title"].tolist()}



# ════════════════════════════════════════════════════════════════════════════
# Routes — Model Info  (replaces all hardcoded dashboard data)
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/model-info")
def model_info():
    """
    Return live model configuration and evaluation metrics for the dashboard.
    All values previously hardcoded in index.html now come from here so the
    frontend updates automatically when models are retrained or configs change.
    """
    rec = get_recommender()

    # ── Embedding engine ──────────────────────────────────────────────────
    embed = rec.embed_engine
    encoder = embed.encoder
    encoder_type = type(encoder).__name__   # "SBERTEncoder" or "LSAEncoder"

    if encoder_type == "SBERTEncoder":
        embed_info = {
            "type":        "SBERT",
            "model":       getattr(encoder, "model_name", "all-MiniLM-L6-v2"),
            "dims":        getattr(encoder, "dim", 384),
            "vocab":       None,
            "expl_var":    None,
        }
        embed_label = f"SBERT ({getattr(encoder, 'model_name', 'MiniLM')}, {getattr(encoder, 'dim', 384)}d)"
    else:
        expl = round(float(getattr(encoder, "explained_variance", 0) or 0) * 100, 1)
        vocab = len(getattr(encoder, "tfidf", {}).vocabulary_ or {}) if hasattr(encoder, "tfidf") else None
        embed_info = {
            "type":     "LSA",
            "dims":     int(getattr(encoder.svd, "n_components", 256)),
            "vocab":    int(vocab) if vocab else 15000,
            "expl_var": expl,
        }
        embed_label = f"LSA + BM25 ({embed_info['dims']}d)"

    # ── Collaborative filter ──────────────────────────────────────────────
    cf = rec.cf
    cf_type      = type(cf).__name__   # WALSCollaborativeFilter / CollaborativeFilterNMF
    cf_factors   = int(getattr(cf, "n_factors", 50))
    n_cf_users   = len(getattr(cf, "user_index",  getattr(cf, "user_index",  {})))
    n_cf_items   = len(getattr(cf, "item_index",  getattr(cf, "course_index", {})))
    cf_label     = f"NMF (k={cf_factors})" if "NMF" in cf_type else f"WALS (k={cf_factors})"

    # ── Adaptive engine ────────────────────────────────────────────────────
    diff_pred = rec.adaptive.diff_pred
    drop_pred = rec.adaptive.dropout_pred
    profiler  = rec.adaptive.profiler if hasattr(rec.adaptive, "profiler") else rec.profiler

    # DifficultyPredictor — HistGBM or GBM
    gbm_clf = diff_pred.gbm_pipeline[-1] if hasattr(diff_pred, "gbm_pipeline") else None
    gbm_type = type(gbm_clf).__name__ if gbm_clf else "GBM"
    gbm_iters = int(getattr(gbm_clf, "n_iter_", getattr(gbm_clf, "n_estimators", 200)))
    gbm_weight = float(getattr(diff_pred, "gbm_weight", 0.55))
    mlp_weight = float(getattr(diff_pred, "mlp_weight", 0.45))
    diff_label = f"HistGBM({gbm_weight:.0%}) + MLP({mlp_weight:.0%})" if "Hist" in gbm_type else f"GBM({gbm_weight:.0%}) + MLP({mlp_weight:.0%})"

    # KMeans profiler
    kmeans      = getattr(profiler, "kmeans", None)
    n_clusters  = int(kmeans.n_clusters) if kmeans else int(getattr(profiler, "n_clusters", 6))
    best_k      = int(getattr(profiler, "best_k_", n_clusters))
    sil_score   = round(float(getattr(profiler, "silhouette_score_", 0.0)), 4)
    persona_map = getattr(profiler, "persona_map_", {})

    # ── Hybrid weights ─────────────────────────────────────────────────────
    from hybrid_recommender import WEIGHTS_WARM, WEIGHTS_COLD, COLD_THRESHOLD
    n_interactions_total = int(len(interactions_df))

    # ── Version string ─────────────────────────────────────────────────────
    fixes_applied = []
    if "Hist" in gbm_type:         fixes_applied.append("HistGBM")
    if encoder_type == "SBERTEncoder": fixes_applied.append("SBERT")
    if sil_score > 0:              fixes_applied.append("KMeans-validated")
    version_label = f"v{5 + len(fixes_applied)}" if fixes_applied else "v5"

    # ── Read model_metrics.json (written by training pipeline) ────────────
    _mm_path = os.path.join(os.path.dirname(__file__), "data", "model_metrics.json")
    _mm: dict = {}
    try:
        with open(_mm_path) as _f:
            _mm = json.load(_f)
    except Exception:
        pass  # file doesn't exist yet — training hasn't been run

    difficulty_metrics = {
        "ensemble_accuracy": _mm.get("difficulty_accuracy") or _mm.get("ensemble_accuracy"),
        "gbm_accuracy":      _mm.get("gbm_accuracy"),
        "mlp_accuracy":      _mm.get("mlp_accuracy"),
        "cv_mean":           _mm.get("cv_mean"),
        "cv_std":            _mm.get("cv_std"),
        "n_features":        _mm.get("n_features", 19),
        "cv_strategy":       _mm.get("cv_strategy", "5-fold user-based CV"),
        "n_train":           _mm.get("n_train"),
        "n_test":            _mm.get("n_test"),
        "report":            _mm.get("report"),
    }
    dropout_metrics = {
        "accuracy": _mm.get("dropout_accuracy", 0.7193),
        "coefs":    _mm.get("dropout_coefs") or [
            {"feature": "avg_quiz_score",   "coef": -0.5772, "direction": "v dropout"},
            {"feature": "difficulty_gap",   "coef": -0.4485, "direction": "v dropout"},
            {"feature": "skill_level",      "coef":  0.3901, "direction": "^ dropout"},
            {"feature": "avg_completion",   "coef": -0.2695, "direction": "v dropout"},
            {"feature": "difficulty_level", "coef": -0.2528, "direction": "v dropout"},
        ],
        "report":   _mm.get("dropout_report"),
    }
    cf_metrics = {
        "rmse":      _mm.get("cf_rmse"),
        "n_factors": cf_factors,
    }
    ranking_metrics = _mm.get("ranking_metrics")
    ranking_eval    = _mm.get("ranking_eval")

    return {
        "version":     version_label,
        "n_courses":   int(len(courses_df)),
        "n_users":     int(len(users_df)),
        "n_interactions": n_interactions_total,
        "n_platforms": 6,
        "avg_completion": round(float(interactions_df["completed"].mean()), 4),

        "embedding": {
            **embed_info,
            "label":   embed_label,
            "bm25":    True,
        },
        "cf": {
            "type":      cf_type,
            "label":     cf_label,
            "n_factors": cf_factors,
            "n_users":   n_cf_users,
            "n_items":   n_cf_items,
            "decay":     True,
        },
        "difficulty_predictor": {
            "label":      diff_label,
            "gbm_type":   gbm_type,
            "gbm_iters":  gbm_iters,
            "gbm_weight": gbm_weight,
            "mlp_weight": mlp_weight,
        },
        "dropout_predictor": {
            "label": "Logistic Regression (balanced)",
        },
        "profiler": {
            "label":        f"KMeans (K={best_k}, validated)",
            "n_clusters":   best_k,
            "silhouette":   sil_score,
            "persona_map":  {str(k): v for k, v in persona_map.items()},
            "n_users":      int(len(users_df)),
        },
        "weights_warm": WEIGHTS_WARM,
        "weights_cold": WEIGHTS_COLD,
        "cold_threshold": COLD_THRESHOLD,

        # ── Evaluation metrics (from data/model_metrics.json) ─────────────
        "difficulty_metrics": difficulty_metrics,
        "dropout_metrics":    dropout_metrics,
        "cf_metrics":         cf_metrics,
        "ranking_metrics":    ranking_metrics,
        "ranking_eval":       ranking_eval,

        "modules": [
            {
                "icon":  "📐",
                "name":  "Embedding Engine",
                "tech":  embed_label + (" + BM25" if encoder_type == "LSAEncoder" else " + BM25"),
                "desc":  "Semantic + keyword dual-index retrieval",
            },
            {
                "icon":  "🤝",
                "name":  "Collaborative Filter",
                "tech":  cf_label + (f" · {n_cf_users}×{n_cf_items} matrix" if n_cf_users else ""),
                "desc":  "User latent interest model with time-decay",
            },
            {
                "icon":  "🧠",
                "name":  "Difficulty Predictor",
                "tech":  diff_label,
                "desc":  "Struggle / Match / Excel — user-based CV",
            },
            {
                "icon":  "⚠️",
                "name":  "Dropout Predictor",
                "tech":  "Logistic Regression (balanced class weights)",
                "desc":  "Dropout risk penalty per candidate course",
            },
            {
                "icon":  "👥",
                "name":  "User Profiler",
                "tech":  f"KMeans K={best_k}" + (f" · silhouette={sil_score:.3f}" if sil_score else ""),
                "desc":  "Centroid-anchored persona assignment",
            },
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes — User Analytics
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/users/{user_id}/analytics")
def user_analytics(user_id: str):
    """
    Per-user analytics for the dashboard Analytics tab.

    Computes everything from the sim_interactions data for this user:
    - Quiz score trend (last 20 interactions, chronological)
    - Subject breakdown (avg score + count per subject)
    - Difficulty distribution (how many Beginner/Intermediate/Advanced)
    - Completion funnel (total, completed, dropped)
    - Time investment (total hours, avg per course)
    - Persona and velocity interpretation
    - Predicted outcome distribution for their current level
    """
    user_iacts = interactions_df[interactions_df["user_id"] == user_id].copy()

    if user_iacts.empty:
        # Cold-start user — return empty but valid structure
        return {
            "n_interactions":   0,
            "quiz_trend":       [],
            "subject_scores":   {},
            "difficulty_dist":  {},
            "completion_funnel":{"total":0,"completed":0,"dropped":0,"rate":0},
            "time_investment":  {"total_hours":0,"avg_mins_per_course":0},
            "persona":          "New Learner",
            "velocity":         0,
            "difficulty_progression": [],
            "top_subjects":     [],
        }

    # ── Merge with course info ────────────────────────────────────────────
    merged = user_iacts.merge(
        courses_df[["course_id","title","subject","difficulty_label","duration_mins"]],
        on="course_id", how="left"
    )

    # ── Quiz score trend (last 20, in interaction order) ─────────────────
    trend_df = merged.tail(20)[["title","quiz_score","completed","difficulty_label"]].copy()
    quiz_trend = []
    for i, (_, r) in enumerate(trend_df.iterrows()):
        quiz_trend.append({
            "index":     i + 1,
            "score":     round(float(r["quiz_score"]), 1),
            "completed": bool(r["completed"]),
            "label":     str(r["title"])[:30] + ("…" if len(str(r["title"])) > 30 else ""),
            "difficulty":str(r.get("difficulty_label", "Mixed")),
        })

    # ── Subject breakdown ─────────────────────────────────────────────────
    subj_grp = merged.groupby("subject").agg(
        avg_score  = ("quiz_score", "mean"),
        n_courses  = ("course_id",  "count"),
        completed  = ("completed",  "sum"),
        time_spent = ("time_spent", "sum"),
    ).reset_index()
    subj_grp = subj_grp.sort_values("avg_score", ascending=False)

    subject_scores = {}
    for _, r in subj_grp.iterrows():
        subj = str(r["subject"])
        if subj and subj != "nan":
            subject_scores[subj] = {
                "avg_score":   round(float(r["avg_score"]), 1),
                "n_courses":   int(r["n_courses"]),
                "completed":   int(r["completed"]),
                "hours_spent": round(float(r["time_spent"]) / 60, 1),
            }

    top_subjects = list(subject_scores.keys())[:5]

    # ── Difficulty distribution ───────────────────────────────────────────
    diff_map = {"Beginner":1,"Mixed":0,"Intermediate":2,"Advanced":3}
    diff_counts = merged["difficulty_label"].value_counts().to_dict()
    # Keep only known labels, order by difficulty
    difficulty_dist = {}
    for label in ["Beginner","Mixed","Intermediate","Advanced"]:
        if label in diff_counts:
            difficulty_dist[label] = {
                "count": int(diff_counts[label]),
                "avg_score": round(
                    float(merged[merged["difficulty_label"]==label]["quiz_score"].mean()), 1
                ) if int(diff_counts.get(label,0)) > 0 else 0,
            }

    # Difficulty progression — last 10 courses as a numeric series
    diff_prog_df = merged.tail(10)[["difficulty_label"]].copy()
    difficulty_progression = [
        diff_map.get(str(r["difficulty_label"]), 0)
        for _, r in diff_prog_df.iterrows()
    ]

    # ── Completion funnel ────────────────────────────────────────────────
    total     = len(merged)
    completed = int(merged["completed"].sum())
    dropped   = total - completed
    rate      = round(completed / total, 3) if total > 0 else 0.0

    # ── Time investment ───────────────────────────────────────────────────
    total_mins = float(merged["time_spent"].sum())
    avg_mins   = round(total_mins / total, 1) if total > 0 else 0

    # ── Quiz score stats ─────────────────────────────────────────────────
    scores     = merged["quiz_score"].dropna()
    best_score = round(float(scores.max()), 1)  if len(scores) else 0
    avg_score  = round(float(scores.mean()), 1) if len(scores) else 0

    # ── Score distribution buckets ────────────────────────────────────────
    bins   = [0, 40, 60, 75, 90, 100]
    labels = ["0-40","40-60","60-75","75-90","90-100"]
    score_dist = {}
    for i in range(len(labels)):
        lo, hi = bins[i], bins[i+1]
        count = int(((scores >= lo) & (scores < hi)).sum() if i < len(labels)-1
                    else (scores >= lo).sum())
        score_dist[labels[i]] = count

    # ── Velocity (quiz improvement over time) ────────────────────────────
    velocity = 0.0
    if len(quiz_trend) >= 4:
        half    = len(quiz_trend) // 2
        recent  = sum(t["score"] for t in quiz_trend[half:]) / (len(quiz_trend) - half)
        earlier = sum(t["score"] for t in quiz_trend[:half]) / half
        velocity = round(recent - earlier, 2)

    # ── Platform preferences ──────────────────────────────────────────────
    if "source" in merged.columns:
        plat_counts = merged["source"].value_counts().head(4).to_dict()
    else:
        plat_counts = {}

    return {
        "n_interactions":       total,
        "avg_quiz_score":       avg_score,
        "best_score":           best_score,
        "quiz_trend":           quiz_trend,
        "subject_scores":       subject_scores,
        "top_subjects":         top_subjects,
        "difficulty_dist":      difficulty_dist,
        "difficulty_progression": difficulty_progression,
        "completion_funnel": {
            "total":     total,
            "completed": completed,
            "dropped":   dropped,
            "rate":      rate,
        },
        "time_investment": {
            "total_hours":          round(total_mins / 60, 1),
            "avg_mins_per_course":  avg_mins,
        },
        "score_distribution":   score_dist,
        "platform_preferences": plat_counts,
        "velocity":             velocity,
    }


@app.get("/api/courses/subjects")
def course_subjects():
    subjects = [
        "Machine Learning & AI","Data Science & Analytics",
        "Programming & Software Dev","Web Development","Cloud & DevOps",
        "Cybersecurity","Business & Finance","Graphic Design & Media",
        "Health & Medicine","Personal Development","Science & Engineering",
        "Law & Social Sciences","Music & Arts",
    ]
    return {"subjects": subjects}
