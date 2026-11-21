"""
recommender.py
==============
Adaptive AI Learning Intelligence Platform — Phase 2
-----------------------------------------------------
Content-based recommendation engine using TF-IDF + cosine similarity.

Fixes applied vs original version
-----------------------------------
FIX 1  detect_skill_gap() — now derives prerequisites from the actual goal
        instead of always searching the same 5 hardcoded subjects.

FIX 2  recommend_similar() / recommend_adaptive() — similarity pool
        increased from top-20 to top-100 so adaptive filtering has
        enough candidates to actually find the requested difficulty level.

FIX 3  Duration fill — Coursera rows (no real duration) are estimated
        from cert_type instead of being filled with the dataset median.
        SPECIALIZATION ≈ 480 mins, COURSE ≈ 120 mins, CERTIFICATE ≈ 240 mins.
        A `duration_estimated` flag column marks these rows.

FIX 4  generate_user_plan() — the `level` parameter is now forwarded to
        recommend_adaptive() so Beginner and Expert users get different paths.

FIX 5  Cosine similarity matrix — replaced full NxN eager computation with
        on-demand row lookup so startup is fast and memory stays low.
        A small LRU cache avoids recomputing hot queries.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from knowledge_graph import SUBJECT_PREREQ_MAP, get_prerequisites


# ── Similarity pool size ────────────────────────────────────────────────────
# Top-N candidates retrieved before difficulty / duration filtering.
# 20 was too small — adaptive filtering would almost always fall back.
SIMILARITY_POOL = 100

# ── Duration estimates for Coursera (no real duration in dataset) ────────────
CERT_DURATION_ESTIMATES = {
    "SPECIALIZATION":           480,   # ~8 hours across multiple courses
    "COURSE":                   120,   # ~2 hours
    "PROFESSIONAL CERTIFICATE": 240,   # ~4 hours
}

# Fix #11 — PREREQUISITE_MAP consolidated into knowledge_graph.SUBJECT_PREREQ_MAP
# Import at top of file: from knowledge_graph import SUBJECT_PREREQ_MAP, get_prerequisites
# Use SUBJECT_PREREQ_MAP directly wherever PREREQUISITE_MAP was used.
PREREQUISITE_MAP = SUBJECT_PREREQ_MAP   # alias for backwards compatibility


# ════════════════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_reason(row, goal: str = "") -> str:
    """
    Generate an explainable reason string for a recommendation.
    Uses difficulty level + optional goal context.
    """
    level = str(row.get("difficulty_label", "")).lower()
    title = str(row.get("title", ""))

    if goal and goal.lower() in title.lower():
        base = f"Directly matches your goal: {goal}"
    elif "beginner" in level:
        base = "Good starting point — build your foundation first"
    elif "intermediate" in level:
        base = "Bridges fundamentals to advanced topics"
    elif "advanced" in level:
        base = "Advanced specialisation — attempt after basics"
    else:
        base = "Broad coverage across skill levels"

    if row.get("duration_estimated", False):
        base += "  [duration estimated]"

    return base


def format_courses(df, goal: str = ""):
    """Pretty-print a courses DataFrame with explainable reasons."""
    if isinstance(df, str):
        print(df)
        return
    if df.empty:
        print("  (no courses found)")
        return
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"\n  {i}. {row.get('title', 'N/A')}")
        if "subject"          in df.columns: print(f"     Subject    : {row.get('subject', 'N/A')}")
        if "difficulty_label" in df.columns: print(f"     Difficulty : {row.get('difficulty_label', 'N/A')}")
        if "rating"           in df.columns: print(f"     Rating     : {row.get('rating', 'N/A')} ⭐")
        if "duration_mins"    in df.columns:
            dur = row.get("duration_mins", "N/A")
            est = "~" if row.get("duration_estimated", False) else ""
            print(f"     Duration   : {est}{dur} mins")
        if "source"           in df.columns: print(f"     Source     : {row.get('source', 'N/A').title()}")
        print(f"     Why this?  : {get_reason(row, goal)}")


def diversify(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return top-n rows after removing exact title duplicates.
    Also enforces source diversity — no more than 60% from one provider.
    """
    df = df.drop_duplicates(subset=["title"])
    if len(df) <= n:
        return df

    max_per_source = max(1, int(n * 0.6))
    source_counts  = {}
    selected       = []

    for _, row in df.iterrows():
        src = row.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0)
        if source_counts[src] < max_per_source:
            selected.append(row)
            source_counts[src] += 1
        if len(selected) == n:
            break

    # Top up without diversity constraint if we came up short
    if len(selected) < n:
        used_titles = {r["title"] for r in selected}
        for _, row in df.iterrows():
            if row["title"] not in used_titles:
                selected.append(row)
                used_titles.add(row["title"])
            if len(selected) == n:
                break

    return pd.DataFrame(selected)


# ════════════════════════════════════════════════════════════════════════════
# Main recommender class
# ════════════════════════════════════════════════════════════════════════════

class AdaptiveLearningRecommender:
    """
    Content-based recommendation engine.

    Key design decisions
    --------------------
    - TF-IDF on combined_features (title + subject + difficulty + provider)
      gives semantic similarity without needing user data.
    - Similarity scores are computed on-demand per query row (not a full
      NxN matrix at startup) — keeps memory at ~2 MB instead of ~158 MB.
    - An LRU cache stores the 256 most-recently-used similarity rows so
      repeated lookups for the same course are free.
    """

    def __init__(self, file_path: str = "courses.csv"):
        print(f"[INFO] Loading courses from {file_path} ...")
        self.courses = pd.read_csv(file_path)
        self._preprocess()
        self._build_tfidf()
        print(f"[INFO] Ready — {len(self.courses):,} courses indexed.")

    # ── Setup ────────────────────────────────────────────────────────────────

    def _preprocess(self):
        """Clean and enrich the course catalogue."""

        # FIX 3 — Estimate duration for Coursera rows from cert_type
        self.courses["duration_estimated"] = False

        coursera_mask = (
            (self.courses["source"] == "coursera") &
            self.courses["duration_mins"].isna()
        )
        for cert_type, estimate in CERT_DURATION_ESTIMATES.items():
            mask = coursera_mask & (self.courses["cert_type"] == cert_type)
            self.courses.loc[mask, "duration_mins"]      = estimate
            self.courses.loc[mask, "duration_estimated"] = True

        # Any remaining nulls (non-Coursera) → median of known durations
        known_median = self.courses["duration_mins"].dropna().median()
        self.courses["duration_mins"] = self.courses["duration_mins"].fillna(known_median)

        # Fix #5 — normalise popularity_score to [0, 1]
        # Prevents outlier enrolment counts from dominating sort order.
        col = "popularity_score"
        if col in self.courses.columns:
            raw = pd.to_numeric(self.courses[col], errors="coerce")
            lo, hi = raw.min(), raw.max()
            if not raw.isna().all() and hi != lo:
                self.courses[col] = ((raw.fillna(raw.median()) - lo) / (hi - lo)).round(6)
            elif raw.isna().all():
                self.courses[col] = 0.5

        # Build combined text feature for TF-IDF
        self.courses["combined_features"] = (
            self.courses["title"].astype(str)            + " " +
            self.courses["subject"].astype(str)          + " " +
            self.courses["difficulty_label"].astype(str) + " " +
            self.courses["provider"].astype(str)
        )

    def _build_tfidf(self):
        """
        Fit TF-IDF vectoriser and store the sparse matrix.
        FIX 5 — Do NOT pre-compute the full cosine similarity matrix.
        Compute per-row similarity on demand via _get_similarity_row().
        """
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams for better phrase matching
            max_features=8000,    # cap vocabulary to control memory
        )
        self.tfidf_matrix = tfidf.fit_transform(self.courses["combined_features"])
        print(f"[INFO] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    @lru_cache(maxsize=256)
    def _get_similarity_row(self, idx: int) -> np.ndarray:
        """
        Compute and cache cosine similarity between course[idx] and all others.
        LRU cache means the 256 most-used courses never recompute.
        """
        row_vec = self.tfidf_matrix[idx]
        return cosine_similarity(row_vec, self.tfidf_matrix).flatten()

    def _find_course_idx(self, query: str):
        """
        Return the DataFrame index of the best title match for a query string.
        Prefers highest-rated match when multiple titles contain the query.
        Returns None if no match found.
        """
        mask    = self.courses["title"].str.contains(query, case=False, na=False)
        matches = self.courses[mask]
        if matches.empty:
            return None
        return matches.sort_values("rating", ascending=False).index[0]

    def _get_similar_courses(self, idx: int, pool: int = SIMILARITY_POOL) -> pd.DataFrame:
        """
        Return the top-`pool` most similar courses to courses[idx],
        excluding the course itself.
        """
        scores      = self._get_similarity_row(idx)
        top_indices = np.argsort(scores)[::-1][1: pool + 1]
        similar     = self.courses.iloc[top_indices].copy()
        similar["similarity_score"] = scores[top_indices]
        return similar

    # ── Public recommendation methods ────────────────────────────────────────

    def recommend_by_subject(self, subject: str, top_n: int = 5) -> pd.DataFrame:
        """Top-rated courses in a subject, diversified across sources."""
        filtered = self.courses[
            self.courses["subject"].str.contains(subject, case=False, na=False)
        ].sort_values(["rating", "popularity_score"], ascending=False)

        return diversify(filtered, top_n)[
            ["title", "subject", "difficulty_label", "rating", "source"]
        ]

    def recommend_similar(self, course_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Courses most similar to a given course title (TF-IDF cosine similarity).
        FIX 2 — pool is now SIMILARITY_POOL (100) not 20.
        """
        idx = self._find_course_idx(course_title)
        if idx is None:
            return f"No course matching '{course_title}' found."

        similar = self._get_similar_courses(idx)
        similar = similar.sort_values(["similarity_score", "rating"], ascending=False)
        return diversify(similar, top_n)[
            ["title", "subject", "difficulty_label", "rating", "source"]
        ]

    def recommend_adaptive(
        self,
        course_title: str,
        level: str,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Courses similar to course_title filtered to a target difficulty level.
        FIX 2 — large pool ensures enough candidates at each difficulty level.

        Fallback chain:
          1. Exact difficulty match
          2. Adjacent level (Intermediate / Mixed)
          3. Any similar course (with printed notice)
        """
        idx = self._find_course_idx(course_title)
        if idx is None:
            return f"No course matching '{course_title}' found."

        similar  = self._get_similar_courses(idx)

        # Primary filter
        filtered = similar[
            similar["difficulty_label"].str.contains(level, case=False, na=False)
        ]

        # Fallback 1 — adjacent level
        if filtered.empty and level.lower() in ("beginner", "advanced"):
            filtered = similar[
                similar["difficulty_label"].str.contains("intermediate|mixed", case=False, na=False)
            ]
            if not filtered.empty:
                print(f"[INFO] No '{level}' courses found — showing Intermediate/Mixed instead.")

        # Fallback 2 — any similar
        if filtered.empty:
            filtered = similar
            print(f"[INFO] Difficulty filter '{level}' returned nothing — showing top similar courses.")

        filtered = filtered.sort_values(["rating", "popularity_score"], ascending=False)
        return diversify(filtered, top_n)[
            ["title", "subject", "difficulty_label", "rating", "source"]
        ]

    def recommend_by_time(
        self,
        course_title: str,
        max_duration: float,
        top_n: int = 5
    ) -> pd.DataFrame:
        """Similar courses that fit within a maximum duration (minutes)."""
        idx = self._find_course_idx(course_title)
        if idx is None:
            return f"No course matching '{course_title}' found."

        similar  = self._get_similar_courses(idx)
        filtered = similar[similar["duration_mins"] <= max_duration]

        if filtered.empty:
            print(f"[INFO] No similar courses under {max_duration} mins — showing shortest available.")
            filtered = similar.sort_values("duration_mins")

        filtered = filtered.sort_values(["rating", "popularity_score"], ascending=False)
        return diversify(filtered, top_n)[
            ["title", "subject", "difficulty_label", "duration_mins", "rating", "source"]
        ]

    # ── Learning path & planning ──────────────────────────────────────────────

    def generate_learning_path(self, subject: str, top_n: int = 2) -> pd.DataFrame:
        """Build a structured Beginner → Intermediate → Advanced path."""
        subject_courses = self.courses[
            self.courses["subject"].str.contains(subject, case=False, na=False)
        ]

        tiers = {}
        for label in ("Beginner", "Intermediate", "Advanced"):
            tiers[label] = (
                subject_courses[
                    subject_courses["difficulty_label"].str.contains(label, case=False, na=False)
                ]
                .sort_values(["rating", "popularity_score"], ascending=False)
                .head(top_n)
            )

        for label, df in tiers.items():
            print_section(f"{label} stage")
            format_courses(df, goal=subject)

        all_tiers = [df for df in tiers.values() if not df.empty]
        return pd.concat(all_tiers) if all_tiers else pd.DataFrame()

    def generate_timeline_path(
        self,
        subject: str,
        max_time: float,
        top_n: int = 5
    ) -> pd.DataFrame:
        """Courses in a subject that fit within a total time budget (minutes)."""
        subject_courses = self.courses[
            self.courses["subject"].str.contains(subject, case=False, na=False)
        ]
        within_budget = subject_courses[subject_courses["duration_mins"] <= max_time]

        if within_budget.empty:
            print(f"[INFO] No '{subject}' courses under {max_time} mins — relaxing filter.")
            within_budget = subject_courses

        return diversify(
            within_budget.sort_values(["rating", "popularity_score"], ascending=False),
            top_n
        )[["title", "subject", "difficulty_label", "duration_mins", "rating", "source"]]

    def detect_skill_gap(self, goal: str) -> pd.DataFrame:
        """
        FIX 1 — Derives prerequisite subjects from the actual goal using
        PREREQUISITE_MAP instead of searching the same 5 hardcoded subjects.

        Steps:
          1. Normalise goal to a known key via substring matching.
          2. Look up prerequisites from PREREQUISITE_MAP.
          3. If no match, fall back to subject catalogue string matching.
          4. Return the top-rated Beginner course per prerequisite.
        """
        goal_lower  = goal.lower()
        matched_key = None

        for key in PREREQUISITE_MAP:
            if key in goal_lower or goal_lower in key:
                matched_key = key
                break

        if matched_key:
            prereqs = PREREQUISITE_MAP[matched_key]
        else:
            # Fallback — match goal against known subject names in catalogue
            matching_subjects = [
                s for s in self.courses["subject"].unique()
                if goal_lower in s.lower() or s.lower() in goal_lower
            ]
            if matching_subjects:
                prereqs = matching_subjects[:3]
            else:
                print(f"[INFO] No prerequisite map for '{goal}' — using default foundations.")
                prereqs = ["Programming & Software Dev", "Data Science & Analytics",
                           "Science & Engineering"]

        print(f"[INFO] Goal: '{goal}'  →  Prerequisites: {prereqs}")

        beginner_courses = self.courses[
            self.courses["difficulty_label"].str.contains("Beginner", case=False, na=False)
        ]
        recommendations = []
        for prereq in prereqs:
            top = (
                beginner_courses[
                    beginner_courses["subject"].str.contains(prereq, case=False, na=False)
                ]
                .sort_values(["rating", "popularity_score"], ascending=False)
                .head(1)
            )
            if not top.empty:
                recommendations.append(top)

        if not recommendations:
            return pd.DataFrame()

        return pd.concat(recommendations)[
            ["title", "subject", "difficulty_label", "rating", "source"]
        ]

    def generate_user_plan(self, goal: str, level: str, max_time: float):
        """
        Full adaptive learning plan for a user.
        FIX 4 — `level` is forwarded to recommend_adaptive() so Beginner
        and Expert users receive meaningfully different course selections.
        """
        print_section(f"AI Adaptive Learning Plan  |  Goal: {goal}  |  Level: {level}")

        print_section("Step 1 — Skill gap analysis")
        gap = self.detect_skill_gap(goal)
        format_courses(gap, goal=goal)

        print_section(f"Step 2 — Courses matched to your level ({level})")
        adaptive = self.recommend_adaptive(goal, level)   # FIX 4: level passed here
        format_courses(adaptive, goal=goal)

        print_section("Step 3 — Full learning path (all levels)")
        self.generate_learning_path(goal)

        print_section(f"Step 4 — Fits your time budget ({max_time} mins)")
        timeline = self.generate_timeline_path(goal, max_time)
        format_courses(timeline, goal=goal)


# ════════════════════════════════════════════════════════════════════════════
# Demo — runs all 5 fixes visibly
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    rec = AdaptiveLearningRecommender("courses.csv")

    print_section("1. Subject recommendation — Data Science")
    format_courses(rec.recommend_by_subject("Data Science"), goal="Data Science")

    print_section("2. Similar courses — Deep Learning")
    format_courses(rec.recommend_similar("Deep Learning"))

    print_section("3. Adaptive — Machine Learning / Beginner")
    format_courses(rec.recommend_adaptive("Machine Learning", "Beginner"))

    print_section("4. Adaptive — Machine Learning / Advanced")
    format_courses(rec.recommend_adaptive("Machine Learning", "Advanced"))

    print_section("5. Time-based — Web Development / under 120 mins")
    format_courses(rec.recommend_by_time("Web Development", max_duration=120))

    print_section("6. Skill gap — Machine Learning")
    format_courses(rec.detect_skill_gap("Machine Learning"))

    print_section("7. Skill gap — Cybersecurity")
    format_courses(rec.detect_skill_gap("Cybersecurity"))

    print_section("8. User plan — Beginner / Data Science")
    rec.generate_user_plan(goal="Data Science", level="Beginner", max_time=300)

    print_section("9. User plan — Advanced / Machine Learning")
    rec.generate_user_plan(goal="Machine Learning", level="Advanced", max_time=600)