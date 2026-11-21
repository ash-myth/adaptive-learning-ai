"""
evaluation_harness.py
=====================
Adaptive AI Learning Intelligence Platform — Offline Evaluation Harness

Fix #14 — offline evaluation harness
--------------------------------------
Previously there was no way to compare recommendation quality across model
versions.  Changes to weights, models, or retrieval strategy were deployed
blind — the only signal was live user behaviour (which takes weeks to
accumulate).

This module implements a leave-one-out replay evaluation:
  For each held-out user, replay their interaction history one step at a
  time.  At each step, ask the recommender for top-K courses given the
  user's goal, then check whether the course they actually took next is
  in the recommended list.  Aggregate hit-rate@K, NDCG@K, and MRR@K.

This gives an offline proxy for click-through rate without needing live
A/B traffic.  It's not perfect (position bias, novelty, etc.), but it
lets you answer "is v2 better than v1?" before deploying.

Usage
-----
    # Basic evaluation
    from evaluation_harness import EvaluationHarness
    harness = EvaluationHarness("courses.csv", "data/users.csv",
                                "data/interactions.csv")
    results = harness.evaluate(top_k=10, n_users=200)
    harness.print_report(results)

    # Compare two model versions
    results_v1 = harness.evaluate(model_path_cf="data/cf_v1.pkl", top_k=10)
    results_v2 = harness.evaluate(model_path_cf="data/cf_v2.pkl", top_k=10)
    harness.compare(results_v1, results_v2, label_a="v1", label_b="v2")

    # CLI
    python evaluation_harness.py --top-k 10 --n-users 200
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_COURSES_PATH      = os.environ.get("COURSES_PATH",      "courses.csv")
DEFAULT_USERS_PATH        = os.environ.get("USERS_PATH",        "data/users.csv")
DEFAULT_INTERACTIONS_PATH = os.environ.get("INTERACTIONS_PATH", "data/interactions.csv")
DEFAULT_TOP_K             = 10
DEFAULT_N_USERS           = 200    # sample size — full eval can take minutes
MIN_INTERACTIONS_PER_USER = 5      # skip users with too few interactions


# ════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════════════════════

def _hit_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """1 if any of the top-k recommended items is relevant, else 0."""
    return float(any(r in relevant for r in recommended[:k]))


def _ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    Rewards hitting relevant items in higher positions.
    Binary relevance: 1 if course is in the held-out set, 0 otherwise.
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, r in enumerate(recommended[:k])
        if r in relevant
    )
    # Ideal DCG: all relevant items at top positions
    n_rel  = min(len(relevant), k)
    idcg   = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Mean Reciprocal Rank @ K.

    Returns 1/rank of the first relevant item in the list, 0 if none found.
    """
    for i, r in enumerate(recommended[:k]):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k recommendations that are relevant."""
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / k


# ════════════════════════════════════════════════════════════════════════════
# EvaluationHarness
# ════════════════════════════════════════════════════════════════════════════

class EvaluationHarness:
    """
    Offline leave-one-out replay evaluation for the hybrid recommender.

    Evaluation protocol
    -------------------
    For each sampled user (with ≥ MIN_INTERACTIONS_PER_USER interactions):
      1. Sort interactions chronologically.
      2. Use the first 80% as "history" (what the model can see).
      3. Treat the remaining 20% as "future" (held-out ground truth).
      4. For each user, call recommender.recommend(user_id, goal, top_k=K).
      5. Check how many "future" courses appear in the top-K list.
      6. Compute hit@K, NDCG@K, MRR@K, precision@K per user.
      7. Average across all evaluated users.

    This is a leave-future-out split (not leave-one-out in the strict sense)
    because it respects temporal ordering — the model never "sees" future
    interactions during recommendation.
    """

    def __init__(
        self,
        courses_path:      str = DEFAULT_COURSES_PATH,
        users_path:        str = DEFAULT_USERS_PATH,
        interactions_path: str = DEFAULT_INTERACTIONS_PATH,
    ):
        self.courses_path      = courses_path
        self.users_path        = users_path
        self.interactions_path = interactions_path

        logger.info("[Harness] Loading data...")
        self.courses_df      = pd.read_csv(courses_path)
        self.users_df        = pd.read_csv(users_path)
        self.interactions_df = self._load_and_sort_interactions()
        logger.info(
            "[Harness] Loaded — courses: %d, users: %d, interactions: %d",
            len(self.courses_df), len(self.users_df), len(self.interactions_df),
        )

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_and_sort_interactions(self) -> pd.DataFrame:
        """Load interactions, sort chronologically per user."""
        df = pd.read_csv(self.interactions_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        else:
            # No timestamp — use row order as proxy for chronological order
            df = df.sort_values("user_id").reset_index(drop=True)
        return df

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        top_k:            int  = DEFAULT_TOP_K,
        n_users:          int  = DEFAULT_N_USERS,
        history_fraction: float = 0.8,
        random_seed:      int  = 42,
        model_path_cf:    str | None = None,
        model_path_adaptive: str | None = None,
        verbose:          bool = True,
    ) -> dict[str, Any]:
        """
        Run leave-future-out replay evaluation.

        Parameters
        ----------
        top_k            : number of recommendations to evaluate at.
        n_users          : number of users to sample (for speed).
        history_fraction : fraction of each user's history used as context.
        random_seed      : reproducibility seed for user sampling.
        model_path_cf    : override path to CF model .pkl (for A/B comparison).
        model_path_adaptive : override path to adaptive models .pkl.
        verbose          : print per-user progress every 20 users.

        Returns
        -------
        dict with keys: hit_rate, ndcg, mrr, precision, n_users_evaluated,
                        top_k, elapsed_secs, per_user_results
        """
        # ── Load recommender ──────────────────────────────────────────────
        recommender = self._load_recommender(model_path_cf, model_path_adaptive)

        # ── Sample eligible users ─────────────────────────────────────────
        user_counts = self.interactions_df.groupby("user_id").size()
        eligible    = user_counts[user_counts >= MIN_INTERACTIONS_PER_USER].index.tolist()

        rng = np.random.default_rng(random_seed)
        rng.shuffle(eligible)
        sample_users = eligible[:n_users]

        logger.info(
            "[Harness] Evaluating %d users (eligible: %d, top_k=%d)...",
            len(sample_users), len(eligible), top_k,
        )

        # ── Replay loop ───────────────────────────────────────────────────
        per_user: list[dict] = []
        t0 = time.time()

        for i, user_id in enumerate(sample_users):
            result = self._evaluate_user(
                user_id, recommender, top_k, history_fraction
            )
            if result is not None:
                per_user.append(result)

            if verbose and (i + 1) % 20 == 0:
                avg_hit = np.mean([r["hit_at_k"] for r in per_user]) if per_user else 0
                logger.info(
                    "[Harness] %d/%d users  hit@%d so far: %.3f",
                    i + 1, len(sample_users), top_k, avg_hit,
                )

        elapsed = round(time.time() - t0, 1)

        if not per_user:
            logger.warning("[Harness] No users produced valid evaluation results.")
            return {"status": "no_results", "n_users_evaluated": 0}

        return {
            "hit_rate":          round(float(np.mean([r["hit_at_k"]     for r in per_user])), 4),
            "ndcg":              round(float(np.mean([r["ndcg_at_k"]    for r in per_user])), 4),
            "mrr":               round(float(np.mean([r["mrr_at_k"]     for r in per_user])), 4),
            "precision":         round(float(np.mean([r["precision_at_k"] for r in per_user])), 4),
            "n_users_evaluated": len(per_user),
            "top_k":             top_k,
            "history_fraction":  history_fraction,
            "elapsed_secs":      elapsed,
            "per_user_results":  per_user,
        }

    def _evaluate_user(
        self,
        user_id:          str,
        recommender:      Any,
        top_k:            int,
        history_fraction: float,
    ) -> dict | None:
        """Evaluate one user — returns None if the user is ineligible."""
        user_iacts = self.interactions_df[
            self.interactions_df["user_id"] == user_id
        ].reset_index(drop=True)

        n = len(user_iacts)
        if n < MIN_INTERACTIONS_PER_USER:
            return None

        split     = max(1, int(n * history_fraction))
        history   = user_iacts.iloc[:split]
        future    = user_iacts.iloc[split:]
        future_ids = set(future["course_id"].tolist())

        if not future_ids:
            return None

        # Determine the user's goal for the recommendation call
        user_row = self.users_df[self.users_df["user_id"] == user_id]
        goal     = str(user_row.iloc[0]["goal"]) if not user_row.empty else ""

        # Get recommendations using only the history interactions
        # We temporarily mask the future interactions from the recommender
        # by patching its interactions_df for this call.
        original_df = recommender.interactions_df
        try:
            recommender.interactions_df = history
            recs = recommender.recommend(user_id, goal, top_n=top_k)
            rec_ids = recs["course_id"].tolist() if not recs.empty else []
        except Exception as e:
            logger.debug("[Harness] recommend() failed for %s: %s", user_id, e)
            rec_ids = []
        finally:
            recommender.interactions_df = original_df

        return {
            "user_id":       user_id,
            "n_history":     split,
            "n_future":      len(future_ids),
            "hit_at_k":      _hit_at_k(rec_ids,      future_ids, top_k),
            "ndcg_at_k":     _ndcg_at_k(rec_ids,     future_ids, top_k),
            "mrr_at_k":      _mrr_at_k(rec_ids,      future_ids, top_k),
            "precision_at_k":_precision_at_k(rec_ids, future_ids, top_k),
            "rec_ids":       rec_ids,
            "future_ids":    list(future_ids),
        }

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, results: dict, label: str = "Model") -> None:
        """Pretty-print evaluation results."""
        if results.get("status") == "no_results":
            print("[Harness] No results to report.")
            return
        k   = results["top_k"]
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  Evaluation report — {label}")
        print(sep)
        print(f"  Users evaluated : {results['n_users_evaluated']}")
        print(f"  Top-K           : {k}")
        print(f"  History split   : {results['history_fraction']:.0%} / {1-results['history_fraction']:.0%}")
        print(f"  Elapsed         : {results['elapsed_secs']}s")
        print(f"  {'-'*48}")
        print(f"  Hit rate  @{k:<3}  : {results['hit_rate']:.4f}")
        print(f"  NDCG      @{k:<3}  : {results['ndcg']:.4f}")
        print(f"  MRR       @{k:<3}  : {results['mrr']:.4f}")
        print(f"  Precision @{k:<3}  : {results['precision']:.4f}")
        print(sep)

    def compare(
        self,
        results_a: dict,
        results_b: dict,
        label_a:   str = "Version A",
        label_b:   str = "Version B",
    ) -> None:
        """Side-by-side comparison of two evaluation result dicts."""
        metrics = ["hit_rate", "ndcg", "mrr", "precision"]
        k       = results_a.get("top_k", "?")
        sep     = "=" * 60
        print(f"\n{sep}")
        print(f"  Comparison @ top-{k}")
        print(f"  {'Metric':<18} {label_a:>14} {label_b:>14} {'Delta':>10}")
        print(f"  {'-'*56}")
        for m in metrics:
            a     = results_a.get(m, 0.0)
            b     = results_b.get(m, 0.0)
            delta = b - a
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
            print(f"  {m:<18} {a:>14.4f} {b:>14.4f} {arrow}{abs(delta):>9.4f}")
        print(sep)

    def save_results(self, results: dict, path: str) -> None:
        """Save results dict to JSON (omitting per_user_results for size)."""
        slim = {k: v for k, v in results.items() if k != "per_user_results"}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(slim, f, indent=2)
        logger.info("[Harness] Results saved → %s", path)

    # ── Recommender loading ───────────────────────────────────────────────────

    def _load_recommender(
        self,
        cf_path:       str | None,
        adaptive_path: str | None,
    ) -> Any:
        """
        Load HybridRecommender.  Override model paths for A/B comparison.
        Uses retrain=False (load from cache) by default.
        """
        from hybrid_recommender import HybridRecommender

        rec = HybridRecommender(
            courses_path      = self.courses_path,
            users_path        = self.users_path,
            interactions_path = self.interactions_path,
            retrain           = False,
        )

        # Override CF model if a specific path is provided
        if cf_path and Path(cf_path).exists():
            from collaborative_filter import CollaborativeFilter
            rec.cf = CollaborativeFilter.load(cf_path)
            logger.info("[Harness] CF model overridden from: %s", cf_path)

        # Override adaptive models if provided
        if adaptive_path and Path(adaptive_path).exists():
            import pickle
            from adaptive_engine import (
                DifficultyPredictor, DropoutRiskPredictor, UserProfiler, AdaptiveEngine,
            )
            with open(adaptive_path, "rb") as f:
                m = pickle.load(f)
            if "diff_state" in m:
                diff_pred = DifficultyPredictor()
                diff_pred.__dict__.update(m["diff_state"])
                drop_pred = DropoutRiskPredictor()
                drop_pred.__dict__.update(m["drop_state"])
                profiler  = UserProfiler()
                profiler.__dict__.update(m["profiler_state"])
                rec.adaptive = AdaptiveEngine(diff_pred, drop_pred, profiler)
            logger.info("[Harness] Adaptive models overridden from: %s", adaptive_path)

        return rec


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline evaluation harness")
    parser.add_argument("--courses",      default=DEFAULT_COURSES_PATH)
    parser.add_argument("--users",        default=DEFAULT_USERS_PATH)
    parser.add_argument("--interactions", default=DEFAULT_INTERACTIONS_PATH)
    parser.add_argument("--top-k",        type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--n-users",      type=int, default=DEFAULT_N_USERS)
    parser.add_argument("--history",      type=float, default=0.8,
                        help="Fraction of each user's history to use as context (default 0.8)")
    parser.add_argument("--cf-model",     default=None,
                        help="Override CF model path (for A/B)")
    parser.add_argument("--adaptive-model", default=None,
                        help="Override adaptive model path (for A/B)")
    parser.add_argument("--save",         default=None,
                        help="Save results to this JSON path")
    parser.add_argument("--compare",      default=None,
                        help="Compare against results in this JSON file")
    args = parser.parse_args()

    harness = EvaluationHarness(
        courses_path      = args.courses,
        users_path        = args.users,
        interactions_path = args.interactions,
    )

    results = harness.evaluate(
        top_k            = args.top_k,
        n_users          = args.n_users,
        history_fraction = args.history,
        model_path_cf    = args.cf_model,
        model_path_adaptive = args.adaptive_model,
    )
    harness.print_report(results, label="Current model")

    if args.save:
        harness.save_results(results, args.save)

    if args.compare and Path(args.compare).exists():
        with open(args.compare) as f:
            baseline = json.load(f)
        harness.compare(baseline, results, label_a="Baseline", label_b="Current")
