"""
retraining_pipeline.py
======================
Adaptive AI Learning Intelligence Platform — Feedback Loop & Retraining Pipeline

Fix #9 — closing the feedback loop
------------------------------------
The system was previously frozen at training time.  User completions, quiz
scores, and dropout events were never fed back, so the CF and adaptive models
could not improve with usage.

This module provides three things:

1. InteractionLogger
   Thread-safe event recorder.  Call log_event() from any API endpoint
   whenever a user interacts with a course (start, progress, complete,
   quiz, dropout).  Events are accumulated in memory and flushed to
   data/interactions.csv in batches.  All writes are atomic (write to
   a temp file, then rename) so a crash never corrupts the main CSV.

2. RetrainingPipeline
   Orchestrates scheduled model updates:
     - run_cf_retrain()       — retrain CollaborativeFilter on the full
                                updated interactions log.  Designed to
                                run weekly (cron / APScheduler).
     - run_adaptive_retrain() — retrain DifficultyPredictor,
                                DropoutRiskPredictor, UserProfiler on
                                the updated interactions log.  Run after
                                CF retrain or independently monthly.
     - run_nightly_cache_invalidation() — delete the embeddings cache
                                so EmbeddingEngine rebuilds on next
                                startup.  Run nightly only if new courses
                                were added to courses.csv.

3. Scheduler (optional)
   A lightweight APScheduler wrapper.  Call start_scheduler() to run all
   three jobs on a background thread alongside your FastAPI server.
   Set ENABLE_SCHEDULER=False to manage cron yourself instead.

Schema expected in interactions.csv
-------------------------------------
user_id    : str
course_id  : str
cf_score   : float   — composite engagement signal (quiz × completion × replay)
quiz_score : float   — 0–100
completed  : int     — 0 or 1
replay_count: int    — number of re-watches
time_spent : float   — minutes spent on course
timestamp  : str     — ISO-8601, added automatically by log_event()

Usage
-----
    # In your FastAPI app startup:
    from retraining_pipeline import InteractionLogger, start_scheduler
    logger = InteractionLogger()
    start_scheduler(logger, interactions_path="data/interactions.csv")

    # In your endpoint handlers:
    logger.log_event(
        user_id="U0005",
        course_id="C1234",
        event_type="complete",
        quiz_score=82.0,
        time_spent_mins=94.0,
    )
"""

from __future__ import annotations

import csv
import logging
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)
import pickle
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Paths (override via environment variables if needed) ──────────────────────
INTERACTIONS_PATH   = os.environ.get("INTERACTIONS_PATH",   "data/interactions.csv")
CF_MODEL_PATH       = os.environ.get("CF_MODEL_PATH",       "data/cf_model.pkl")
ADAPTIVE_MODEL_PATH = os.environ.get("ADAPTIVE_MODEL_PATH", "data/adaptive_models.pkl")
EMBEDDINGS_CACHE    = os.environ.get("EMBEDDINGS_CACHE",    "data/embeddings.pkl")
COURSES_PATH        = os.environ.get("COURSES_PATH",        "courses.csv")
USERS_PATH          = os.environ.get("USERS_PATH",          "data/users.csv")

# ── Interaction schema ────────────────────────────────────────────────────────
INTERACTION_COLS = [
    "user_id", "course_id", "cf_score", "quiz_score",
    "completed", "replay_count", "time_spent", "timestamp",
]

# ── cf_score computation weights ─────────────────────────────────────────────
# cf_score is a composite engagement signal used by the CF model.
# Weights are tunable — these defaults match the original dataset's semantics.
CF_SCORE_QUIZ_WEIGHT       = 0.50   # quiz performance (normalised 0-1)
CF_SCORE_COMPLETION_WEIGHT = 0.35   # binary completion signal
CF_SCORE_REPLAY_WEIGHT     = 0.15   # replay engagement (log-scaled, capped)
CF_SCORE_MAX_REPLAY        = 5      # replays beyond this give diminishing returns

EventType = Literal["start", "progress", "complete", "quiz", "dropout"]


# ════════════════════════════════════════════════════════════════════════════
# 1. InteractionLogger
# ════════════════════════════════════════════════════════════════════════════

class InteractionLogger:
    """
    Thread-safe event recorder with batched atomic CSV writes.

    Design decisions
    ----------------
    - In-memory buffer: avoids a disk write on every API call.  Events are
      accumulated and flushed in batches (default: every 50 events or every
      60 seconds, whichever comes first).
    - Atomic writes: events are written to a temp file in the same directory
      then renamed over the target.  rename() is atomic on POSIX; on Windows
      it uses shutil.move() as a fallback.  A crash mid-write never leaves
      a half-written CSV.
    - cf_score derivation: computed from quiz_score, completed, and
      replay_count so callers don't need to calculate it themselves.
    - Deduplication: if the same (user_id, course_id) pair is logged twice
      in a session, the second event updates the first (higher cf_score wins)
      rather than creating a duplicate row.
    """

    def __init__(
        self,
        interactions_path: str = INTERACTIONS_PATH,
        flush_every_n:     int = 50,
        flush_every_secs:  int = 60,
    ):
        self.path            = Path(interactions_path)
        self.flush_every_n   = flush_every_n
        self.flush_every_secs = flush_every_secs

        self._buffer:  list[dict] = []
        self._lock     = threading.Lock()
        self._last_flush = time.monotonic()

        # Ensure the data directory and file exist
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            pd.DataFrame(columns=INTERACTION_COLS).to_csv(self.path, index=False)
            logger.info("[Logger] Created interactions file: %s", self.path)

        # Start background flush thread
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._periodic_flush, daemon=True, name="interaction-flush"
        )
        self._flush_thread.start()
        logger.info("[Logger] Started (flush_every_n=%d, flush_every_secs=%ds)",
                    flush_every_n, flush_every_secs)

    # ── Public API ────────────────────────────────────────────────────────────

    def log_event(
        self,
        user_id:        str,
        course_id:      str,
        event_type:     EventType = "complete",
        quiz_score:     float = 0.0,
        time_spent_mins: float = 0.0,
        replay_count:   int   = 0,
        completed:      int   = 0,
    ) -> None:
        """
        Record a user-course interaction event.

        Parameters
        ----------
        user_id         : user identifier (must match users.csv)
        course_id       : course identifier (must match courses.csv)
        event_type      : one of start / progress / complete / quiz / dropout
        quiz_score      : raw quiz score 0–100
        time_spent_mins : minutes spent on the course in this session
        replay_count    : cumulative number of replays for this course
        completed       : 1 if the user finished the course, else 0
        """
        # Infer completed from event_type if not explicitly set
        if event_type == "complete" and completed == 0:
            completed = 1
        if event_type == "dropout":
            completed = 0

        cf_score = self._compute_cf_score(quiz_score, completed, replay_count)

        event: dict = {
            "user_id":      user_id,
            "course_id":    course_id,
            "cf_score":     round(cf_score, 4),
            "quiz_score":   round(float(quiz_score), 2),
            "completed":    int(completed),
            "replay_count": int(replay_count),
            "time_spent":   round(float(time_spent_mins), 2),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            # Deduplication: update existing buffer entry if same user+course
            key = (user_id, course_id)
            for i, existing in enumerate(self._buffer):
                if (existing["user_id"], existing["course_id"]) == key:
                    if cf_score >= existing["cf_score"]:
                        self._buffer[i] = event  # higher engagement wins
                    return

            self._buffer.append(event)
            should_flush = len(self._buffer) >= self.flush_every_n

        if should_flush:
            self.flush()

    def flush(self) -> int:
        """
        Flush the in-memory buffer to disk atomically.

        Returns the number of new rows appended.
        """
        with self._lock:
            if not self._buffer:
                return 0
            to_write      = self._buffer.copy()
            self._buffer  = []
            self._last_flush = time.monotonic()

        n = self._append_to_csv(to_write)
        logger.info("[Logger] Flushed %d event(s) → %s", n, self.path)
        return n

    def stop(self) -> None:
        """Flush remaining buffer and stop the background thread."""
        self._stop_event.set()
        self.flush()
        self._flush_thread.join(timeout=5)
        logger.info("[Logger] Stopped.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_cf_score(
        quiz_score: float,
        completed:  int,
        replay_count: int,
    ) -> float:
        """
        Derive a composite engagement confidence signal for the CF model.

        cf_score = 0.50 × (quiz_score / 100)
                 + 0.35 × completed
                 + 0.15 × log1p(min(replay_count, 5)) / log1p(5)

        All components are in [0, 1] so cf_score ∈ [0, 1].
        The CF model treats this as a *confidence* weight (Hu et al. 2008):
        higher cf_score → model is more confident the user benefited.
        """
        quiz_n    = float(np.clip(quiz_score, 0, 100)) / 100.0
        comp_n    = float(np.clip(completed, 0, 1))
        replay_n  = (
            float(np.log1p(min(replay_count, CF_SCORE_MAX_REPLAY)))
            / float(np.log1p(CF_SCORE_MAX_REPLAY))
        )
        return (
            CF_SCORE_QUIZ_WEIGHT       * quiz_n
            + CF_SCORE_COMPLETION_WEIGHT * comp_n
            + CF_SCORE_REPLAY_WEIGHT     * replay_n
        )

    def _append_to_csv(self, rows: list[dict]) -> int:
        """
        Atomically append rows to the interactions CSV.

        Strategy:
          1. Read the current CSV into a DataFrame.
          2. Merge with new rows (new rows override existing same user+course
             if cf_score is higher).
          3. Write to a temp file in the same directory.
          4. Rename temp file over the original (atomic on POSIX).
        """
        if not rows:
            return 0

        new_df = pd.DataFrame(rows, columns=INTERACTION_COLS)

        try:
            existing = pd.read_csv(self.path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            existing = pd.DataFrame(columns=INTERACTION_COLS)

        # For duplicate (user_id, course_id) pairs, keep the row with the
        # higher cf_score — prefer richer engagement signals over stale ones.
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = (
            combined
            .sort_values("cf_score", ascending=False)
            .drop_duplicates(subset=["user_id", "course_id"], keep="first")
            .sort_values(["user_id", "timestamp"])
            .reset_index(drop=True)
        )

        # Atomic write: temp file → rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, suffix=".tmp", prefix="interactions_"
        )
        try:
            os.close(tmp_fd)
            combined.to_csv(tmp_path, index=False)
            try:
                os.replace(tmp_path, self.path)          # atomic on POSIX
            except OSError:
                shutil.move(tmp_path, self.path)          # Windows fallback
        except Exception:
            os.unlink(tmp_path)
            raise

        return len(new_df)

    def _periodic_flush(self) -> None:
        """Background thread: flush every flush_every_secs seconds."""
        while not self._stop_event.is_set():
            time.sleep(5)
            elapsed = time.monotonic() - self._last_flush
            if elapsed >= self.flush_every_secs:
                with self._lock:
                    has_data = bool(self._buffer)
                if has_data:
                    self.flush()


# ════════════════════════════════════════════════════════════════════════════
# 2. RetrainingPipeline
# ════════════════════════════════════════════════════════════════════════════

class RetrainingPipeline:
    """
    Orchestrates scheduled model retraining from the updated interactions log.

    Three jobs:
    ───────────
    run_cf_retrain()
        Reload interactions.csv, retrain CollaborativeFilter (NMF), save.
        Intended cadence: weekly.  NMF on 116k+ rows takes < 60s.

    run_adaptive_retrain()
        Retrain DifficultyPredictor (HistGBM+MLP), DropoutRiskPredictor
        (LogReg), and UserProfiler (KMeans) from scratch on the full
        updated interactions log.
        Intended cadence: monthly, or whenever CF RMSE degrades noticeably.

    run_nightly_cache_invalidation()
        Delete the embeddings cache file so EmbeddingEngine rebuilds on
        the next server startup.  Only triggered if courses.csv has been
        modified since the cache was written.
    """

    def __init__(
        self,
        interactions_path:   str = INTERACTIONS_PATH,
        cf_model_path:       str = CF_MODEL_PATH,
        adaptive_model_path: str = ADAPTIVE_MODEL_PATH,
        embeddings_cache:    str = EMBEDDINGS_CACHE,
        courses_path:        str = COURSES_PATH,
        users_path:          str = USERS_PATH,
    ):
        self.interactions_path   = Path(interactions_path)
        self.cf_model_path       = Path(cf_model_path)
        self.adaptive_model_path = Path(adaptive_model_path)
        self.embeddings_cache    = Path(embeddings_cache)
        self.courses_path        = Path(courses_path)
        self.users_path          = Path(users_path)

    # ── Job 1: weekly CF retrain ──────────────────────────────────────────────

    def run_cf_retrain(self) -> dict:
        """
        Retrain the Collaborative Filter on the latest interactions log.

        Returns a summary dict with timing and row count.
        """
        logger.info("[Pipeline] Starting CF retrain...")
        t0 = time.time()

        interactions_df = self._load_interactions()
        if interactions_df.empty:
            logger.warning("[Pipeline] No interactions found — skipping CF retrain.")
            return {"status": "skipped", "reason": "empty interactions"}

        n_rows = len(interactions_df)
        logger.info("[Pipeline] Loaded %d interactions for CF retrain.", n_rows)

        # Import here to avoid circular imports at module level
        from collaborative_filter import CollaborativeFilter

        cf = CollaborativeFilter()
        cf.fit(interactions_df)

        self.cf_model_path.parent.mkdir(parents=True, exist_ok=True)
        cf.save(str(self.cf_model_path))

        elapsed = round(time.time() - t0, 1)
        logger.info("[Pipeline] CF retrain complete in %.1fs — saved → %s",
                    elapsed, self.cf_model_path)
        return {
            "status":       "success",
            "model":        "CollaborativeFilter",
            "n_interactions": n_rows,
            "elapsed_secs": elapsed,
            "saved_to":     str(self.cf_model_path),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

    # ── Job 2: adaptive model retrain ────────────────────────────────────────

    def run_adaptive_retrain(self) -> dict:
        """
        Retrain DifficultyPredictor, DropoutRiskPredictor, and UserProfiler.

        Slower than CF retrain (~5–15 min on 116k rows) — run monthly or
        on-demand after significant interaction volume growth.
        """
        logger.info("[Pipeline] Starting adaptive model retrain...")
        t0 = time.time()

        interactions_df = self._load_interactions()
        courses_df      = pd.read_csv(self.courses_path)
        users_df        = pd.read_csv(self.users_path)

        if interactions_df.empty:
            logger.warning("[Pipeline] No interactions — skipping adaptive retrain.")
            return {"status": "skipped", "reason": "empty interactions"}

        from adaptive_engine import (
            DifficultyPredictor, DropoutRiskPredictor,
            UserProfiler, build_features,
        )

        logger.info("[Pipeline] Building feature matrix...")
        features_df = build_features(interactions_df, users_df, courses_df)

        logger.info("[Pipeline] Training DifficultyPredictor (HistGBM + MLP)...")
        diff_pred = DifficultyPredictor()
        diff_pred.fit(features_df)

        logger.info("[Pipeline] Training DropoutRiskPredictor (LogReg)...")
        drop_pred = DropoutRiskPredictor()
        drop_pred.fit(features_df)

        logger.info("[Pipeline] Fitting UserProfiler (KMeans)...")
        profiler = UserProfiler()
        profiler.fit(interactions_df)

        # Save as state-dict (avoids __main__ pickle issues — same pattern
        # as HybridRecommender.__init__)
        def _state(obj: object) -> dict:
            return {k: v for k, v in obj.__dict__.items()}

        self.adaptive_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.adaptive_model_path, "wb") as f:
            pickle.dump({
                "diff_state":     _state(diff_pred),
                "drop_state":     _state(drop_pred),
                "profiler_state": _state(profiler),
            }, f)

        elapsed = round(time.time() - t0, 1)
        logger.info("[Pipeline] Adaptive retrain complete in %.1fs — saved → %s",
                    elapsed, self.adaptive_model_path)
        return {
            "status":       "success",
            "models":       ["DifficultyPredictor", "DropoutRiskPredictor", "UserProfiler"],
            "n_interactions": len(interactions_df),
            "feature_rows": len(features_df),
            "elapsed_secs": elapsed,
            "saved_to":     str(self.adaptive_model_path),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

    # ── Job 3: nightly embedding cache invalidation ───────────────────────────

    def run_nightly_cache_invalidation(self) -> dict:
        """
        Delete the embeddings cache if courses.csv has changed since it was built.

        EmbeddingEngine will rebuild automatically on the next server start.
        This keeps semantic search current when new courses are added to the
        catalogue without requiring a manual cache flush.
        """
        logger.info("[Pipeline] Checking embedding cache freshness...")

        if not self.embeddings_cache.exists():
            return {"status": "skipped", "reason": "cache does not exist"}

        if not self.courses_path.exists():
            return {"status": "skipped", "reason": "courses.csv not found"}

        cache_mtime   = self.embeddings_cache.stat().st_mtime
        courses_mtime = self.courses_path.stat().st_mtime

        if courses_mtime <= cache_mtime:
            logger.info("[Pipeline] Cache is current — no invalidation needed.")
            return {
                "status":    "skipped",
                "reason":    "courses.csv not modified since cache was built",
                "cache_age_hours": round((time.time() - cache_mtime) / 3600, 1),
            }

        self.embeddings_cache.unlink()
        logger.info("[Pipeline] Embedding cache invalidated — will rebuild on next start.")
        return {
            "status":    "invalidated",
            "cache":     str(self.embeddings_cache),
            "reason":    "courses.csv modified after cache was written",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_interactions(self) -> pd.DataFrame:
        """Load and validate the interactions CSV."""
        if not self.interactions_path.exists():
            return pd.DataFrame(columns=INTERACTION_COLS)
        try:
            df = pd.read_csv(self.interactions_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=INTERACTION_COLS)

        missing = [c for c in ["user_id", "course_id", "cf_score"] if c not in df.columns]
        if missing:
            logger.error("[Pipeline] interactions.csv missing columns: %s", missing)
            return pd.DataFrame(columns=INTERACTION_COLS)

        df["cf_score"] = pd.to_numeric(df["cf_score"], errors="coerce").fillna(0.0)
        return df


# ════════════════════════════════════════════════════════════════════════════
# 3. Scheduler (APScheduler wrapper — optional)
# ════════════════════════════════════════════════════════════════════════════

ENABLE_SCHEDULER = os.environ.get("ENABLE_SCHEDULER", "true").lower() == "true"


def start_scheduler(
    logger_instance: InteractionLogger,
    interactions_path: str = INTERACTIONS_PATH,
    cf_retrain_hour:   int = 3,     # 3 AM UTC weekly (Sunday)
    adaptive_retrain_day: str = "1",# 1st of month
    invalidation_hour: int = 2,     # 2 AM UTC nightly
) -> object | None:
    """
    Start background APScheduler jobs for automated retraining.

    Jobs
    ----
    Weekly   (Sun 03:00 UTC) — CF retrain
    Monthly  (1st  03:30 UTC) — Adaptive model retrain
    Nightly  (02:00 UTC)     — Embedding cache invalidation check

    Returns the scheduler object so callers can call .shutdown() on exit.
    Returns None if APScheduler is not installed or ENABLE_SCHEDULER=false.

    Usage in FastAPI
    ----------------
        from contextlib import asynccontextmanager
        from retraining_pipeline import InteractionLogger, start_scheduler

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            interaction_logger = InteractionLogger()
            scheduler = start_scheduler(interaction_logger)
            yield
            if scheduler:
                scheduler.shutdown()
            interaction_logger.stop()

        app = FastAPI(lifespan=lifespan)
    """
    if not ENABLE_SCHEDULER:
        logger.info("[Scheduler] Disabled via ENABLE_SCHEDULER env var.")
        return None

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning(
            "[Scheduler] APScheduler not installed — scheduled retraining disabled. "
            "Install with: pip install apscheduler"
        )
        return None

    pipeline = RetrainingPipeline(interactions_path=interactions_path)
    scheduler = BackgroundScheduler(timezone="UTC")

    # ── Flush logger before each retrain so models see the latest events ──
    def _cf_job():
        logger_instance.flush()
        result = pipeline.run_cf_retrain()
        logger.info("[Scheduler] CF retrain result: %s", result)

    def _adaptive_job():
        logger_instance.flush()
        result = pipeline.run_adaptive_retrain()
        logger.info("[Scheduler] Adaptive retrain result: %s", result)

    def _cache_job():
        result = pipeline.run_nightly_cache_invalidation()
        logger.info("[Scheduler] Cache invalidation result: %s", result)

    # Weekly CF retrain — Sunday 03:00 UTC
    scheduler.add_job(
        _cf_job,
        CronTrigger(day_of_week="sun", hour=cf_retrain_hour, minute=0),
        id="cf_weekly_retrain",
        name="Weekly CF retrain",
        misfire_grace_time=3600,   # if server was down, run within 1hr of window
    )

    # Monthly adaptive retrain — 1st of month 03:30 UTC
    scheduler.add_job(
        _adaptive_job,
        CronTrigger(day=adaptive_retrain_day, hour=cf_retrain_hour, minute=30),
        id="adaptive_monthly_retrain",
        name="Monthly adaptive retrain",
        misfire_grace_time=7200,
    )

    # Nightly cache invalidation — 02:00 UTC
    scheduler.add_job(
        _cache_job,
        CronTrigger(hour=invalidation_hour, minute=0),
        id="nightly_cache_invalidation",
        name="Nightly embedding cache check",
        misfire_grace_time=1800,
    )

    scheduler.start()
    logger.info(
        "[Scheduler] Started — CF weekly (Sun %02d:00 UTC), "
        "Adaptive monthly (1st %02d:30 UTC), "
        "Cache nightly (%02d:00 UTC)",
        cf_retrain_hour, cf_retrain_hour, invalidation_hour,
    )
    return scheduler


# ════════════════════════════════════════════════════════════════════════════
# CLI runner — invoke directly to trigger any job manually
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Retraining pipeline CLI")
    parser.add_argument(
        "job",
        choices=["cf", "adaptive", "cache", "all"],
        help=(
            "cf       — retrain CollaborativeFilter\n"
            "adaptive — retrain DifficultyPredictor + DropoutRisk + UserProfiler\n"
            "cache    — invalidate embedding cache if courses.csv changed\n"
            "all      — run all three in order"
        ),
    )
    parser.add_argument("--interactions", default=INTERACTIONS_PATH)
    parser.add_argument("--cf-model",     default=CF_MODEL_PATH)
    parser.add_argument("--adaptive-model", default=ADAPTIVE_MODEL_PATH)
    parser.add_argument("--embeddings",   default=EMBEDDINGS_CACHE)
    parser.add_argument("--courses",      default=COURSES_PATH)
    parser.add_argument("--users",        default=USERS_PATH)
    args = parser.parse_args()

    pipeline = RetrainingPipeline(
        interactions_path   = args.interactions,
        cf_model_path       = args.cf_model,
        adaptive_model_path = args.adaptive_model,
        embeddings_cache    = args.embeddings,
        courses_path        = args.courses,
        users_path          = args.users,
    )

    results = {}
    if args.job in ("cf", "all"):
        results["cf"] = pipeline.run_cf_retrain()
    if args.job in ("adaptive", "all"):
        results["adaptive"] = pipeline.run_adaptive_retrain()
    if args.job in ("cache", "all"):
        results["cache"] = pipeline.run_nightly_cache_invalidation()

    print(json.dumps(results, indent=2))
