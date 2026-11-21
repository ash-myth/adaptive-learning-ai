"""
hybrid_recommender.py  (v2)
============================
Adaptive AI Learning Intelligence Platform — Full Hybrid Recommender

Changes from v1
---------------
- BM25 keyword score added as 5th signal (prevents topic drift)
- Candidate pool: semantic search + BM25 top matches merged
  (v1 used semantic-only, missing rare-term courses completely)
- Subject-locked candidate filtering for technical queries
- Adaptive blend weights auto-tune per query based on BM25 hit count

Hybrid score formula
--------------------
  final = 0.30 × semantic   (LSA cosine sim)
        + 0.25 × keyword     (BM25 score)      ← NEW
        + 0.25 × cf          (NMF dot product)
        + 0.15 × adaptive    (MLP + LogReg)
        + 0.05 × popularity  (enrolment × rating)

  Cold-start: semantic 0.40, keyword 0.30, cf 0.05, adaptive 0.15, pop 0.10
  Warm user:  as above
"""

import numpy as np
import pandas as pd
import pickle, os, re
from collections import deque
from threading import Lock
from embeddings           import EmbeddingEngine
from collaborative_filter import CollaborativeFilter
from adaptive_engine      import (
    AdaptiveEngine, DifficultyPredictor, DropoutRiskPredictor,
    UserProfiler, build_features
)
from knowledge_graph      import PREREQ_MAP, get_prerequisites

WEIGHTS_COLD = dict(semantic=0.40, keyword=0.30, cf=0.05, adaptive=0.15, pop=0.10)
WEIGHTS_WARM = dict(semantic=0.30, keyword=0.25, cf=0.25, adaptive=0.15, pop=0.05)
WEIGHTS_COLD = dict(semantic=0.70, keyword=0.20, cf=0.00, adaptive=0.00, pop=0.10)
WEIGHTS_WARM = dict(semantic=0.00, keyword=0.00, cf=1.00, adaptive=0.00, pop=0.00)
COLD_THRESHOLD = 5


def _sep():
    print("\n" + "=" * 62)

def print_section(title):
    _sep(); print(f"  {title}"); print("=" * 62)


SESSION_WINDOW = int(os.environ.get("SESSION_WINDOW", "5"))  # last N interactions


class SessionContextBuffer:
    """
    Per-user short-term session context buffer (fix #15).

    Motivation
    ----------
    The CF and adaptive models reflect long-term interaction history — they
    cannot react to what a user is doing *right now*.  A user who started a
    Python course 20 minutes ago and is now searching "backend development"
    should see Python-backend recommendations, not whatever the long-term CF
    profile predicts.

    How it works
    ------------
    log_interaction() adds the latest course interaction to a per-user deque
    of length SESSION_WINDOW (default 5).  get_context_query() returns the
    subjects and keywords of those recent courses as a string that is appended
    to the user's search goal before retrieval — biasing the candidate pool
    toward the current session's topics without modifying any model weights.

    Thread-safety
    -------------
    All reads and writes are protected by a single threading.Lock so the
    buffer is safe to use from FastAPI's async thread pool.

    Persistence
    -----------
    The buffer is in-memory only — it resets on server restart.  This is
    intentional: session context should be short-lived.  For cross-session
    continuity, the full interaction log (retraining_pipeline) handles it.
    """

    def __init__(self, window: int = SESSION_WINDOW):
        self.window    = window
        self._sessions: dict[str, deque] = {}
        self._lock     = Lock()

    def log_interaction(
        self,
        user_id:   str,
        course_id: str,
        subject:   str = "",
        title:     str = "",
    ) -> None:
        """Record a course interaction for a user in the session buffer."""
        entry = {"course_id": course_id, "subject": subject, "title": title}
        with self._lock:
            if user_id not in self._sessions:
                self._sessions[user_id] = deque(maxlen=self.window)
            self._sessions[user_id].appendleft(entry)   # most recent first

    def get_context_query(self, user_id: str) -> str:
        """
        Return a short context string derived from recent interactions.

        The string contains subject names and title keywords from the last
        SESSION_WINDOW courses the user interacted with.  Empty string if
        no session context exists for this user.
        """
        with self._lock:
            session = list(self._sessions.get(user_id, []))

        if not session:
            return ""

        terms: list[str] = []
        seen:  set[str]  = set()
        for entry in session:
            for field in ("subject", "title"):
                for word in str(entry.get(field, "")).lower().split():
                    if len(word) > 3 and word not in seen:
                        terms.append(word)
                        seen.add(word)
        return " ".join(terms[:20])  # cap at 20 terms to avoid query bloat

    def clear(self, user_id: str) -> None:
        """Clear session context for a user (e.g. on explicit logout)."""
        with self._lock:
            self._sessions.pop(user_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


class HybridRecommender:
    def __init__(self, courses_path="courses.csv", users_path="data/users.csv",
                 interactions_path="data/interactions.csv", retrain=False):
        print_section("Loading Adaptive Learning Platform")
        # Robust path resolution: check local root and data/ directory
        for p_attr, p_val in [("courses_path", courses_path), ("users_path", users_path), ("interactions_path", interactions_path)]:
            if not os.path.exists(p_val):
                alt = os.path.join("data", os.path.basename(p_val))
                if os.path.exists(alt):
                    if p_attr == "courses_path": courses_path = alt
                    elif p_attr == "users_path": users_path = alt
                    elif p_attr == "interactions_path": interactions_path = alt

        self.courses_df      = pd.read_csv(courses_path)
        self.users_df        = pd.read_csv(users_path)
        self.interactions_df = pd.read_csv(interactions_path)
        print(f"  Courses: {len(self.courses_df):,}  Users: {len(self.users_df):,}  "
              f"Interactions: {len(self.interactions_df):,}")

        # Fix #5 — normalise popularity_score to [0, 1] at load time
        # ---------------------------------------------------------------
        # popularity_score may be a raw enrolment count, a composite index,
        # or already in [0,1] depending on the data pipeline.  Using raw
        # values directly in the hybrid formula means a single viral course
        # with enrolment=500,000 would dwarf every other popularity signal.
        # Min-max normalisation is applied once here so every downstream
        # consumer — _hybrid_score, _hybrid_scores_batch, rank_courses,
        # skill_gap — all see clean [0, 1] values with no extra work.
        self._normalise_popularity()

        print_section("Module 1 — Semantic + Keyword Embedding Engine")
        self.embed_engine = EmbeddingEngine(self.courses_df, force_rebuild=retrain)

        print_section("Module 2 — Collaborative Filter (NMF)")
        cf_path = "data/cf_model.pkl"
        if not retrain and os.path.exists(cf_path):
            self.cf = CollaborativeFilter.load(cf_path)
        else:
            self.cf = CollaborativeFilter()
            self.cf.fit(self.interactions_df)
            self.cf.save(cf_path)
        import sklearn as _sk
        _SK_VER = _sk.__version__

        print_section("Module 3 — Adaptive Engine (MLP + LogReg + KMeans)")
        mp = "data/adaptive_models.pkl"
        _loaded_ok = False
        if not retrain and os.path.exists(mp):
            try:
                with open(mp, "rb") as f:
                    m = pickle.load(f)
                saved_sk = m.get("sklearn_version", "")
                # Allow major.minor cross-compatibility (e.g. 1.7.x / 1.8.x)
                sk_compat = (
                    saved_sk == _SK_VER or
                    (saved_sk and saved_sk.split(".")[0] == _SK_VER.split(".")[0])
                )
                if not sk_compat:
                    print(f"  [Models] sklearn version mismatch "
                          f"(saved={saved_sk}, running={_SK_VER}) — retraining.")
                elif "diff_state" in m:
                    diff_pred = DifficultyPredictor()
                    diff_pred.__dict__.update(m["diff_state"])
                    drop_pred = DropoutRiskPredictor()
                    drop_pred.__dict__.update(m["drop_state"])
                    profiler  = UserProfiler()
                    profiler.__dict__.update(m["profiler_state"])

                    # Backward compatibility patch for scikit-learn internal attribute changes
                    for obj in (diff_pred, drop_pred):
                        for sub in getattr(obj, "__dict__", {}).values():
                            if hasattr(sub, "named_steps"):
                                for step in sub.named_steps.values():
                                    if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
                                        step._fill_dtype = step._fit_dtype
                                    elif hasattr(step, "_fill_dtype") and not hasattr(step, "_fit_dtype"):
                                        step._fit_dtype = step._fill_dtype

                    print("  [Models] Loaded from cache.")
                    _loaded_ok = True
                else:
                    print("  [Models] Cache format unrecognised — retraining.")
            except Exception as _e:
                print(f"  [Models] Cache load failed ({_e}) — retraining.")

        if not _loaded_ok:
            features_df = build_features(
                self.interactions_df, self.users_df, self.courses_df)
            diff_pred = DifficultyPredictor(); diff_pred.fit(features_df)
            drop_pred = DropoutRiskPredictor(); drop_pred.fit(features_df)
            profiler  = UserProfiler(); profiler.fit(self.interactions_df)

            def _extract_state(obj):
                st = {k: v for k, v in obj.__dict__.items()}
                if isinstance(st.get("user_clusters_"), pd.DataFrame):
                    st["user_clusters_"] = dict(zip(
                        st["user_clusters_"]["user_id"].astype(str),
                        st["user_clusters_"]["persona_cluster"].astype(int)
                    ))
                return st

            with open(mp, "wb") as f:
                pickle.dump({
                    "sklearn_version": _SK_VER,
                    "diff_state":      _extract_state(diff_pred),
                    "drop_state":      _extract_state(drop_pred),
                    "profiler_state":  _extract_state(profiler),
                }, f)
            print(f"  [Models] Saved (sklearn {_SK_VER}) -> {mp}")

        self.adaptive = AdaptiveEngine(diff_pred, drop_pred, profiler)
        self.profiler = profiler
        # Fix #15 — session context buffer
        # Tracks the last SESSION_WINDOW course interactions per user.
        # inject_session_context() uses this to bias recommendations
        # toward the user's current session topics.
        self.session_buffer = SessionContextBuffer(window=SESSION_WINDOW)

        print_section("Platform Ready")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalise_popularity(self):
        """
        Min-max normalise courses_df["popularity_score"] to [0, 1] in-place.

        Safe against all edge cases:
        - All-NaN column      → filled with 0.5 (neutral mid-point)
        - Constant column     → filled with 1.0 (all equally popular)
        - Mixed NaN + values  → NaN filled with column median before scaling
        - Already in [0, 1]   → no-op (min=0, max=1 → identity transform)

        Logs the pre-normalisation range so anomalies are visible at startup.
        """
        col = "popularity_score"
        if col not in self.courses_df.columns:
            self.courses_df[col] = 0.5
            return

        raw = pd.to_numeric(self.courses_df[col], errors="coerce")
        lo, hi = raw.min(), raw.max()
        print(f"  [PopNorm] {col} raw range: [{lo:.4g}, {hi:.4g}]")

        if raw.isna().all():
            self.courses_df[col] = 0.5
            print(f"  [PopNorm] All NaN — filled with 0.5")
            return

        # Fill NaN with median of known values before scaling
        median = float(raw.median())
        raw    = raw.fillna(median)

        if hi == lo:
            # Constant column — every course equally popular
            self.courses_df[col] = 1.0
            print(f"  [PopNorm] Constant value — filled with 1.0")
            return

        self.courses_df[col] = ((raw - lo) / (hi - lo)).round(6)
        print(f"  [PopNorm] Normalised to [0, 1]  (median was {median:.4g})")

    def inject_session_context(self, user_id: str, goal: str) -> str:
        """
        Append recent session context to the goal query (fix #15).

        If the user has recent session activity, their last N course
        subjects and title keywords are appended to the goal string
        before candidate retrieval.  This biases the embedding and
        BM25 search toward the current session's topic cluster.

        Example
        -------
        goal = "backend development"
        session context = "python programming flask api"
        → enriched goal = "backend development python programming flask api"
        """
        ctx = self.session_buffer.get_context_query(user_id)
        if not ctx:
            return goal
        enriched = f"{goal} {ctx}"
        return enriched

    @staticmethod
    def compute_weights_for_n(n: int) -> dict[str, float]:
        """
        Compute hybrid scoring weights via linear interpolation between
        WEIGHTS_COLD and WEIGHTS_WARM over interactions 0 to COLD_THRESHOLD.

        Avoids discontinuous jumps when a user moves from cold-start to warm.
        At n=0: 100% WEIGHTS_COLD (pure semantic/keyword retrieval).
        At n >= COLD_THRESHOLD (5): 100% WEIGHTS_WARM (CF ranking).
        For 0 < n < COLD_THRESHOLD: linear blend:
            alpha = n / COLD_THRESHOLD
            w = (1 - alpha) * WEIGHTS_COLD + alpha * WEIGHTS_WARM
        """
        if n <= 0:
            return dict(WEIGHTS_COLD)
        if n >= COLD_THRESHOLD:
            return dict(WEIGHTS_WARM)
        alpha = float(n) / float(COLD_THRESHOLD)
        w = {}
        for k in WEIGHTS_COLD:
            w[k] = round((1.0 - alpha) * WEIGHTS_COLD[k] + alpha * WEIGHTS_WARM[k], 4)
        return w

    def _weights(self, user_id: str) -> dict[str, float]:
        n = len(self.interactions_df[self.interactions_df["user_id"] == user_id])
        return self.compute_weights_for_n(n)

    def _get_profile(self, user_id):
        # Pass the already-loaded courses_df so get_profile() never
        # hits the disk — fix #3 eliminates pd.read_csv() per request.
        return self.profiler.get_profile(
            user_id, self.interactions_df, self.users_df,
            courses_df=self.courses_df)

    def _candidate_pool(self, query, size=300):
        """
        Merge semantic top-N with BM25 top-N to ensure rare-term courses
        are included even when LSA misses them.
        """
        # Semantic candidates
        expanded  = self.embed_engine._expand(query)
        qv        = self.embed_engine.encoder.transform([expanded])[0].reshape(1, -1)
        from sklearn.metrics.pairwise import cosine_similarity
        s_sims    = cosine_similarity(qv, self.embed_engine.embeddings).flatten()
        s_top     = set(np.argsort(s_sims)[::-1][:size].tolist())

        # BM25 candidates
        bm25_raw  = self.embed_engine.bm25.score(query)
        b_top     = set(np.argsort(bm25_raw)[::-1][:size // 2].tolist())

        merged    = list(s_top | b_top)
        candidates = self.courses_df.iloc[merged].copy()
        candidates["_sem_sim"]  = s_sims[merged]
        candidates["_bm25_raw"] = bm25_raw[merged]
        return candidates

    # ── Single-row helper (kept for learning_path / explain paths) ──────────

    def _hybrid_score(self, course_row, profile, sem_sim, bm25_raw, weights):
        """Single-row scoring — used by learning_path() and explain()."""
        cid      = course_row["course_id"]
        bm25_max = float(profile.get("_bm25_max", 1.0) or 1.0)
        bm25_n   = float(np.clip(float(bm25_raw) / (bm25_max + 1e-9), 0.0, 1.0))
        cf_raw   = self.cf.predict_score(profile.get("user_id", ""), cid)
        cf_n     = float(np.clip(cf_raw, 0.0, 1.0))
        adap_df  = self.adaptive.rank_courses(
            profile, pd.DataFrame([course_row]), top_n=1)
        raw_adap = float(adap_df.iloc[0]["adaptive_score"]) if not adap_df.empty else 0.0
        adap_n   = float(np.clip((raw_adap + 0.20) / 0.80, 0.0, 1.0))
        pop_n    = float(np.clip(course_row.get("popularity_score", 0.5) or 0.5, 0.0, 1.0))
        sem_n    = float(np.clip(sem_sim, 0.0, 1.0))
        # Zero-CF fallback: if CF was weighted, but predicted 0 for this item and semantic is 0
        zero_cf = (weights.get("cf", 0.0) > 0.0) and (cf_n == 0.0) and (weights.get("semantic", 0.0) == 0.0)
        # Zero-CF fallback: if CF was weighted, but predicted < 1e-6 for this item and semantic is 0
        zero_cf = (weights.get("cf", 0.0) > 0.0) and (cf_n < 1e-6) and (weights.get("semantic", 0.0) == 0.0)
        if zero_cf:
            final = (WEIGHTS_COLD["semantic"]  * sem_n
                   + WEIGHTS_COLD["keyword"]   * bm25_n
                   + WEIGHTS_COLD["pop"]       * pop_n)
        else:
            final = (weights["semantic"]  * sem_n
                   + weights["keyword"]   * bm25_n
                   + weights["cf"]        * cf_n
                   + weights["adaptive"]  * adap_n
                   + weights["pop"]       * pop_n)

        res = {
            "hybrid_score":    round(final,  4),
            "semantic_score":  round(sem_n,  4),
            "keyword_score":   round(bm25_n, 4),
            "cf_score":        round(cf_n,   4),
            "adaptive_score":  round(adap_n, 4),
            "popularity_score":round(pop_n,  4),
        }
        if not adap_df.empty:
            r0 = adap_df.iloc[0]
            res["predicted_outcome"]   = str(r0.get("predicted_outcome", ""))
            res["struggle_prob"]       = round(float(r0.get("struggle_prob", 0.0)), 4)
            res["match_prob"]          = round(float(r0.get("match_prob", 0.0)), 4)
            res["excel_prob"]          = round(float(r0.get("excel_prob", 0.0)), 4)
            res["dropout_risk"]        = str(r0.get("dropout_risk", ""))
            res["dropout_probability"] = round(float(r0.get("dropout_probability", 0.0)), 4)
        return res

    # ── Vectorised batch scorer — used by recommend() ─────────────────────

    def _hybrid_scores_batch(self, candidates: pd.DataFrame,
                             profile: dict, weights: dict) -> pd.DataFrame:
        """
        Score all N candidates in three model calls instead of 3N.

        Fix #2 — vectorised adaptive scoring
        -------------------------------------
        Old path (recommend):
          for each of 120 candidates:
              _hybrid_score()          # 1 CF lookup + 1 GBM + 1 MLP + 1 LogReg
          → 120 × 4 = 480 model calls per request

        New path:
          CF scores     — vectorised dict comprehension (no model, O(1) lookup)
          Adaptive      — AdaptiveEngine.rank_courses() calls predict_batch()
                          and predict_risk_batch() → 3 total model calls for all N
          Semantic/BM25 — already pre-computed as columns in candidates DataFrame
          → 3 model calls per request, independent of N
        """
        n        = len(candidates)
        user_id  = profile.get("user_id", "")

        # ── Semantic scores (pre-computed, guaranteed [0, 1], no NaNs) ─────
        if "_sem_sim" in candidates.columns:
            sem_raw = pd.to_numeric(candidates["_sem_sim"], errors="coerce").fillna(0.0).values.astype(float)
        else:
            sem_raw = np.zeros(n, dtype=float)
        sem = np.clip(sem_raw, 0.0, 1.0)

        # ── BM25 keyword scores (normalise by max in candidate pool, no NaNs) ─
        if "_bm25_raw" in candidates.columns:
            bm_raw = pd.to_numeric(candidates["_bm25_raw"], errors="coerce").fillna(0.0).values.astype(float)
        else:
            bm_raw = np.zeros(n, dtype=float)
        c_bm_max = float(np.nanmax(bm_raw)) if len(bm_raw) > 0 and np.nanmax(bm_raw) > 0 else 1.0
        bm25_max = max(float(profile.get("_bm25_max", 1.0) or 1.0), c_bm_max)
        kw  = np.clip(bm_raw / (bm25_max + 1e-9), 0.0, 1.0)

        # ── CF scores (dict lookup, guaranteed [0, 1]) ─────────────────────
        cf_raw = np.array([
            self.cf.predict_score(user_id, cid)
            for cid in candidates["course_id"]
        ], dtype=float)
        cf = np.clip(cf_raw, 0.0, 1.0)

        # ── Adaptive scores (guaranteed [0, 1] via min-max on candidate pool) ─
        ranked = self.adaptive.rank_courses(profile, candidates, top_n=n)
        adap_map = dict(zip(ranked["course_id"], ranked["adaptive_score"]))
        adap_raw = np.array([adap_map.get(cid, 0.0)
                         for cid in candidates["course_id"]], dtype=float)
        a_min, a_max = float(adap_raw.min()), float(adap_raw.max())
        if a_max > a_min:
            adap = (adap_raw - a_min) / (a_max - a_min)
        else:
            adap = np.clip(adap_raw, 0.0, 1.0)

        # ── Popularity (guaranteed [0, 1] via min-max on candidate pool) ───
        pop_raw = candidates["popularity_score"].fillna(0.5).values.astype(float)
        p_min, p_max = float(pop_raw.min()), float(pop_raw.max())
        if p_max > p_min:
            pop = (pop_raw - p_min) / (p_max - p_min)
        else:
            pop = np.clip(pop_raw, 0.0, 1.0)

        # ── Weighted sum with degenerate zero-CF fallback ─────────────────
        zero_cf_degenerate = (weights.get("cf", 0.0) > 0.0) and (float(cf.max()) == 0.0)
        zero_cf_degenerate = (weights.get("cf", 0.0) > 0.0) and (float(cf.max()) < 1e-6)
        if zero_cf_degenerate:
            fb = WEIGHTS_COLD
            hybrid = (fb["semantic"]  * sem
                    + fb["keyword"]   * kw
                    + fb["cf"]        * cf
                    + fb["adaptive"]  * adap
                    + fb["pop"]       * pop)
        else:
            hybrid = (weights["semantic"]  * sem
                    + weights["keyword"]   * kw
                    + weights["cf"]        * cf
                    + weights["adaptive"]  * adap
                    + weights["pop"]       * pop)

        out = candidates.copy()
        out["hybrid_score"]    = hybrid.round(4)
        out["semantic_score"]  = sem.round(4)
        out["keyword_score"]   = kw.round(4)
        out["cf_score"]        = cf.round(4)
        out["adaptive_score"]  = adap.round(4)
        out["popularity_score"]= pop.round(4)

        # Carry per-candidate ML detail from ranked (outcome, risk, etc.)
        detail_cols = ["predicted_outcome", "struggle_prob", "match_prob",
                       "excel_prob", "dropout_risk", "dropout_probability"]
        for col in detail_cols:
            if col in ranked.columns:
                detail_map = dict(zip(ranked["course_id"], ranked[col]))
                out[col] = out["course_id"].map(detail_map)

        return out

    # ── Diversity ─────────────────────────────────────────────────────────────

    @staticmethod
    def _diversify(scored: pd.DataFrame, top_n: int,
                   max_per_subject: int = 2) -> pd.DataFrame:
        """
        Return top_n results with at most max_per_subject courses per subject.

        Fix #7 — subject-diversity cap
        --------------------------------
        Without this, the top-8 results for "machine learning" can be 7 ML
        courses and 1 other — the user sees no breadth and the list feels
        repetitive.  Maximal Marginal Relevance (MMR) is the academically
        correct solution, but it requires pairwise embedding comparisons for
        every candidate.  This greedy subject cap is a practical alternative:
        it guarantees diversity in O(N) time with no extra model calls.

        Algorithm
        ---------
        Iterate candidates in descending hybrid_score order (already sorted).
        Accept a course if its subject has been selected fewer than
        max_per_subject times.  Once top_n courses are accepted, stop.
        If the pool is exhausted before top_n, fill remaining slots without
        the subject cap so we always return exactly top_n results (or all
        available if fewer than top_n remain).

        Parameters
        ----------
        scored          : DataFrame sorted descending by hybrid_score,
                          title-deduplicated.
        top_n           : target result count.
        max_per_subject : maximum courses allowed per subject (default 2).
        """
        subject_counts: dict[str, int] = {}
        selected: list[int]            = []
        overflow: list[int]            = []   # candidates that hit the cap

        for idx, row in scored.iterrows():
            subj  = str(row.get("subject", "Unknown"))
            count = subject_counts.get(subj, 0)
            if count < max_per_subject:
                selected.append(idx)
                subject_counts[subj] = count + 1
            else:
                overflow.append(idx)
            if len(selected) == top_n:
                break

        # Top-up without the subject cap if we came up short
        if len(selected) < top_n:
            needed = top_n - len(selected)
            selected.extend(overflow[:needed])

        return scored.loc[selected].reset_index(drop=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(self, user_id, goal, top_n=8):
        profile  = self._get_profile(user_id)
        weights  = self._weights(user_id)
        # Fix #15 — enrich goal with session context before retrieval
        # Appends recent subject/title keywords so the candidate pool
        # reflects what the user is currently working on.
        goal     = self.inject_session_context(user_id, goal)
        candidates = self._candidate_pool(goal, size=300)

        # Store BM25 max for normalisation
        bm25_max = float(candidates["_bm25_raw"].max()) or 1.0
        profile["_bm25_max"] = bm25_max

        # Filter seen courses
        seen = set(self.interactions_df[
            self.interactions_df["user_id"] == user_id]["course_id"])
        candidates = candidates[~candidates["course_id"].isin(seen)]

        # Vectorised scoring — replaces the old per-candidate for-loop.
        # All 120 candidates are scored in 3 model calls total (fix #2).
        scored = self._hybrid_scores_batch(
            candidates.head(120).reset_index(drop=True), profile, weights
        )
        # Sort and deduplicate titles first, then apply subject-diversity cap.
        # _diversify() enforces at most 2 courses per subject (fix #7),
        # preventing 7-of-8 results from the same subject.
        deduped = (scored
                   .sort_values(
                       by=["hybrid_score", "semantic_score", "keyword_score", "popularity_score"],
                       ascending=[False, False, False, False]
                   )
                   .drop_duplicates("title"))
        result = self._diversify(deduped, top_n=top_n, max_per_subject=2)
        return result

    def subject_mastery_info(self, user_id: str, subject: str) -> dict:
        """
        Determine learner's mastery level specifically for `subject` based on
        completed courses and quiz performance in that topic domain.
        """
        user_iacts = self.interactions_df[self.interactions_df["user_id"] == user_id]
        if user_iacts.empty:
            return {"level": "Beginner", "reason": "No previous course interactions"}

        merged = user_iacts.merge(
            self.courses_df[["course_id", "subject", "difficulty_label"]],
            on="course_id",
            how="inner"
        )
        subj_iacts = merged[merged["subject"].str.contains(subject, case=False, na=False)]
        if subj_iacts.empty:
            return {"level": "Beginner", "reason": f"New to {subject} — starting with fundamentals"}

        n_subj = len(subj_iacts)
        avg_quiz = float(subj_iacts["quiz_score"].mean()) if "quiz_score" in subj_iacts else 0.0
        n_completed = int(subj_iacts["completed"].sum()) if "completed" in subj_iacts else 0

        if n_completed >= 3 and avg_quiz >= 75.0:
            return {
                "level": "Advanced",
                "reason": f"Demonstrated domain mastery ({n_completed} completed in {subject}, {avg_quiz:.0f}% avg quiz)"
            }
        elif (n_completed >= 1 and avg_quiz >= 60.0) or (n_subj >= 2 and avg_quiz >= 70.0):
            return {
                "level": "Intermediate",
                "reason": f"Prior domain experience ({n_completed} completed in {subject}, {avg_quiz:.0f}% avg quiz)"
            }
        else:
            return {
                "level": "Beginner",
                "reason": f"Foundations recommended ({avg_quiz:.0f}% avg quiz in {subject})"
            }

    def subject_mastery_level(self, user_id: str, subject: str) -> str:
        return self.subject_mastery_info(user_id, subject)["level"]

    def learning_path(self, user_id, subject, top_n=3):
        profile  = self._get_profile(user_id)
        rec_diff = profile.get("recommended_difficulty", "Beginner")
        rec_diff = self.subject_mastery_level(user_id, subject)
        weights  = self._weights(user_id)
        subj_courses = self.courses_df[
            self.courses_df["subject"].str.contains(subject, case=False, na=False)]

        path = {}
        # Pre-encode subject query once for all tiers (avoids encoding per course)
        q_vec = self.embed_engine.encode_query(subject)

        for label in ("Beginner","Intermediate","Advanced"):
            tier = subj_courses[
                subj_courses["difficulty_label"].str.contains(label, case=False, na=False)]
            if tier.empty:
                path[label] = pd.DataFrame(); continue

            bm25_raw = self.embed_engine.bm25.score(subject)
            bm25_max = float(bm25_raw.max()) or 1.0
            profile["_bm25_max"] = bm25_max

            scored = []
            for _, c in tier.iterrows():
                idx = self.courses_df[self.courses_df["course_id"]==c["course_id"]].index
                sem = float(np.dot(
                    self.embed_engine.embeddings[idx[0]],
                    q_vec
                )) if len(idx) > 0 else 0.0
                bm  = bm25_raw[idx[0]] if len(idx) > 0 else 0.0
                sc  = self._hybrid_score(c, profile, sem, bm, weights)
                note = ("★ Your current level — start here" if label == rec_diff
                        else "Optional review" if label == "Beginner" and rec_diff in ("Intermediate","Advanced")
                        else "Future goal" if label == "Advanced" and rec_diff == "Beginner"
                        else "Next step")
                scored.append({**c.to_dict(), **sc, "path_note": note})

            path[label] = (pd.DataFrame(scored)
                           .sort_values(
                               by=["hybrid_score", "semantic_score", "keyword_score", "popularity_score"],
                               ascending=[False, False, False, False]
                           )
                           .head(top_n))
        return path

    def skill_gap(self, user_id, goal):
        """
        Prerequisite analysis for a learning goal.

        v1 problem: used LSA centroid distances between broad subjects.
        This made "data structures and algorithms" return Cloud & DevOps
        courses because subject centroids are too coarse and centroid
        distance ≠ prerequisite relationship.

        v2 fix: three-layer approach
        ----------------------------
        1. Explicit prerequisite map — covers 40+ common tech goals.
           "data structures and algorithms" → ["python basics",
           "programming fundamentals", "discrete mathematics"]
        2. BM25 direct search — for each prerequisite phrase, BM25
           finds beginner courses whose titles/skills contain those
           exact terms. No centroid confusion.
        3. Smart fallback for unknown goals:
           - If prereq_override is supplied (from Claude API in main.py),
             use those queries directly — most accurate path.
           - Otherwise, build prereq queries from the goal tokens themselves
             (e.g. "rag model" → ["retrieval augmented generation",
             "machine learning", "nlp fundamentals"]).
             This prevents the old subject-mode bug that returned
             "Health & Medicine" courses for technical queries.
        """
        # Fix #11 — prerequisite map imported from knowledge_graph.py
        # The inline PREREQ_MAP (40+ entries) has been consolidated into
        # knowledge_graph.PREREQ_MAP — single source of truth for both
        # HybridRecommender and AdaptiveLearningRecommender.
        # PREREQ_MAP is already imported at the top of this module.

        # Normalise goal for matching
        goal_lower = goal.lower().strip()

        # Find best matching key (longest substring match wins)
        matched_key   = None
        matched_score = 0
        for key in PREREQ_MAP:
            if key in goal_lower or goal_lower in key:
                score = len(key)   # prefer longer / more specific matches
                if score > matched_score:
                    matched_key   = key
                    matched_score = score

        # Words so common in course titles that matching them proves nothing.
        # A query "cricket fundamentals" must NOT match "Programming Fundamentals"
        # just because both share the word "fundamentals".
        FILLER_WORDS = {
            # Generic course title words
            "fundamentals", "introduction", "basics", "beginner", "advanced",
            "intermediate", "course", "learning", "learn", "guide", "complete",
            "tutorial", "overview", "essential", "essentials", "master",
            "mastering", "programming", "development", "developer", "science",
            # Generic CS/data meta-words that appear in thousands of unrelated courses
            "structure", "structures", "data", "system", "systems", "theory",
            "concept", "concepts", "method", "methods", "technique", "techniques",
            "model", "models", "analysis", "design", "language", "languages",
            "technology", "technologies", "tool", "tools", "type", "types",
            "using", "with", "for", "and", "the",
        }

        if matched_key:
            prereq_queries = PREREQ_MAP[matched_key]
        else:
            # Layer 3b — title-presence gate (the only reliable signal).
            #
            # Every previous approach (BM25 threshold, top-N title check,
            # per-result token intersection) failed because:
            #   - BM25 scores generic words like "fundamentals" or "introduction"
            #     even when the goal word is absent from every course.
            #   - Token intersection includes those same filler words.
            #
            # The only reliable check: does the goal word(s) literally appear
            # in at least one course title in the catalogue?
            # If yes → topic exists, build prereq queries from goal tokens.
            # If no  → topic is not in the catalogue → not_found.
            #
            # This correctly handles:
            #   "cricket"  → no course title contains "cricket" → not_found ✓
            #   "football" → no course title contains "football" → not_found ✓
            #   "rag model" → titles contain "rag" or "retrieval" → found ✓
            #   "machine learning" → titles contain these words → found ✓

            goal_tokens_meaningful = [
                t for t in goal_lower.split()
                if len(t) >= 3 and t not in FILLER_WORDS
            ]
            if not goal_tokens_meaningful:
                return pd.DataFrame(), "not_found"

            all_titles_lower  = self.courses_df["title"].str.lower().fillna("")
            all_subj_lower    = self.courses_df["subject"].str.lower().fillna("")
            all_skills_lower  = self.courses_df["skills_raw"].str.lower().fillna("") \
                                if "skills_raw" in self.courses_df.columns else pd.Series([""] * len(self.courses_df))

            # Check titles, subjects AND skills_raw so concepts like
            # "priority queue" that appear in skills but not titles still pass.
            def _tok_in_corpus(tok):
                return (
                    all_titles_lower.str.contains(re.escape(tok), regex=True).any()
                    or all_subj_lower.str.contains(re.escape(tok), regex=True).any()
                    or all_skills_lower.str.contains(re.escape(tok), regex=True).any()
                )

            if not any(_tok_in_corpus(tok) for tok in goal_tokens_meaningful):
                return pd.DataFrame(), "not_found"

            # Topic confirmed in catalogue — build prereq queries from goal tokens
            prereq_queries = [goal_lower] + goal_tokens_meaningful[:3]

        # For each prerequisite query, BM25-search for top Beginner course.
        # The title token check uses only NON-FILLER goal tokens so that
        # "cricket fundamentals" cannot match "Programming Fundamentals".
        beg = self.courses_df[
            self.courses_df["difficulty_label"].str.contains("Beginner", na=False)
        ]

        seen_ids = set()
        recs     = []
        for q in prereq_queries[:4]:
            bm25_scores = self.embed_engine.bm25.score(q)

            # Non-filler tokens from this specific query — the only ones
            # allowed to drive a title match.
            q_meaningful = {
                t.lower() for t in q.split()
                if len(t) >= 3 and t.lower() not in FILLER_WORDS
            }

            beg_indices = beg.index.tolist()
            # No BM25_MIN_SCORE threshold — single-word queries like "linux"
            # score low on BM25 and were being filtered out entirely.
            # Title token check below is what prevents false matches.
            beg_scores  = sorted(
                [(i, bm25_scores[i]) for i in beg_indices],
                key=lambda x: -x[1]
            )

            for idx, score in beg_scores[:30]:
                row         = self.courses_df.iloc[idx]
                cid         = row["course_id"]
                title_lower = str(row["title"]).lower()
                skills_lower = str(row.get("skills_raw", "")).lower()

                # Accept if any meaningful query token appears in title OR skills_raw.
                # This lets "priority queue" match "Data Structures" courses
                # whose skills_raw contains "priority queue" even if the title doesn't.
                if q_meaningful and not any(
                    tok in title_lower or tok in skills_lower
                    for tok in q_meaningful
                ):
                    continue

                if cid not in seen_ids:
                    seen_ids.add(cid)
                    recs.append(row)
                    break

        if not recs:
            return pd.DataFrame(), "not_found"

        result = pd.DataFrame(recs)
        return (result[["course_id","title","subject","difficulty_label",
                          "rating","duration_mins","source","course_url",
                          "popularity_score"]].reset_index(drop=True), "found")

    def search(self, query, top_n=8):
        return self.embed_engine.search(query, top_k=top_n)

    def profile(self, user_id):
        return self._get_profile(user_id)

    def explain(self, user_id, course_id):
        profile    = self._get_profile(user_id)
        course_row = self.courses_df[self.courses_df["course_id"] == course_id]
        if course_row.empty:
            return "Course not found."
        return self.adaptive.explain(profile, course_row.iloc[0])


if __name__ == "__main__":
    rec = HybridRecommender()
    for goal in ["backend development","docker kubernetes","machine learning",
                 "personal finance","graphic design"]:
        print(f"\n=== {goal} ===")
        recs = rec.recommend("U0005", goal, top_n=4)
        for _, r in recs.iterrows():
            print(f"  [{r['hybrid_score']:.4f}] {r['title'][:55]}  |  {r['subject']}")
