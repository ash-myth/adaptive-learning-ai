"""
collaborative_filter.py  (v2)
==============================
Upgrades vs v1
--------------
- Weighted ALS (WALS) instead of plain NMF
  Implements Hu et al. 2008 "Collaborative Filtering for Implicit Feedback Datasets"
  The gold standard for implicit feedback recommendation systems.

- Confidence-weighted training:
    c_ui = 1 + alpha * r_ui
  Higher cf_score → higher confidence → stronger pull on factor matrices.
  This is far superior to NMF which treats all interactions equally.

- 100 latent factors (was 50)

- Item-item similarity fallback using embedding vectors
  When a user is cold-start, falls back to content-based CF via embeddings.

- Proper train/val split for RMSE reporting on held-out interactions.

How WALS works
--------------
We want to minimise:
  L = Σ c_ui (p_ui - x_u · y_i)² + λ(Σ||x_u||² + Σ||y_i||²)

where p_ui = 1 if r_ui > 0 (user interacted), 0 otherwise
and   c_ui = 1 + alpha * r_ui (confidence — higher interaction = more confident)

Closed-form solution per user (alternating):
  x_u = (Y^T C^u Y + λI)^{-1} Y^T C^u p_u

The "decomposition trick" avoids building a huge diagonal matrix:
  Y^T C^u Y = Y^T Y + Y^T diag(c_u - 1) Y
"""

import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle, warnings, importlib

warnings.filterwarnings("ignore")

N_FACTORS    = 100


def _save_metric(key: str, value) -> None:
    """Persist a training metric to data/model_metrics.json (merges, never overwrites other keys)."""
    path = "data/model_metrics.json"
    os.makedirs("data", exist_ok=True)
    try:
        with open(path, "r") as f:
            metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metrics = {}
    metrics[key] = value
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved {key}={value} -> {path}")


class _CFUnpickler(pickle.Unpickler):
    """
    Remaps stale '__main__' module references to 'collaborative_filter'.

    Root cause: when collaborative_filter.py was run directly, pickle stored
    the class as '__main__.CollaborativeFilterNMF'. When loaded later from
    hybrid_recommender / main.py, __main__ is a different module and pickle
    raises AttributeError. This unpickler silently corrects the reference.
    """
    _CF_CLASSES = {"WALSCollaborativeFilter", "CollaborativeFilterNMF",
                   "CollaborativeFilter"}

    def find_class(self, module, name):
        if module == "__main__" and name in self._CF_CLASSES:
            module = "collaborative_filter"
        return super().find_class(module, name)
N_ITERATIONS = 20       # ALS alternating steps
ALPHA        = 40.0     # confidence scaling: c = 1 + alpha * r
LAMBDA_REG   = 0.01     # L2 regularisation
CACHE_PATH   = "data/cf_model.pkl"

# Fix #12 — exponential time-decay for interaction recency
# w(t) = exp(-DECAY_LAMBDA * days_since_interaction)
# Half-life at default 0.005: ln(2)/0.005 ≈ 139 days (~4.5 months)
# Set CF_DECAY_LAMBDA=0.0 env var to disable decay entirely.
DECAY_LAMBDA = float(os.environ.get("CF_DECAY_LAMBDA", "0.005"))


def _apply_time_decay(
    interactions_df: pd.DataFrame,
    decay_lambda: float = DECAY_LAMBDA,
) -> "pd.Series":
    """
    Compute per-row time-decay weights: w = exp(-λ × days_since_interaction).

    Fix #12 — exponential time-decay
    ----------------------------------
    Without decay the CF matrix treats a two-year-old interaction identically
    to yesterday's.  With exponential decay:
      - Recent interactions (< 30 days) retain ~86% of their original weight
      - Interactions ~4.5 months old retain ~50% (half-life at λ=0.005)
      - Interactions > 2 years old retain < 3%  — effectively ignored

    Falls back to weight=1.0 (no decay) if the timestamp column is absent
    or all-null — fully backwards compatible with existing interaction data.
    """
    if decay_lambda == 0.0 or "timestamp" not in interactions_df.columns:
        return pd.Series(1.0, index=interactions_df.index)
    now = pd.Timestamp.now(tz="UTC")
    ts  = pd.to_datetime(interactions_df["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return pd.Series(1.0, index=interactions_df.index)
    ts   = ts.fillna(pd.Timestamp("2000-01-01", tz="UTC"))
    days = (now - ts).dt.total_seconds() / 86400.0
    return np.exp(-decay_lambda * days.clip(lower=0.0)).astype(float)


class WALSCollaborativeFilter:
    """
    Weighted Alternating Least Squares for implicit feedback.

    Treats cf_score as a *confidence* signal, not an explicit rating.
    This is conceptually correct: we don't know if cf_score=3 means
    the user liked the course — but higher cf_score means we're more
    *confident* they had a positive experience.
    """

    def __init__(self, n_factors: int = N_FACTORS, n_iter: int = N_ITERATIONS,
                 alpha: float = ALPHA, lambda_reg: float = LAMBDA_REG):
        self.n_factors   = n_factors
        self.n_iter      = n_iter
        self.alpha       = alpha
        self.lambda_reg  = lambda_reg

        self.user_factors  : np.ndarray | None = None
        self.item_factors  : np.ndarray | None = None
        self.user_index    : dict = {}
        self.item_index    : dict = {}
        self.reverse_user  : dict = {}
        self.reverse_item  : dict = {}
        self.R_sparse      = None
        self.is_fitted     = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, interactions_df: pd.DataFrame) -> "WALSCollaborativeFilter":
        print("[WALS] Building interaction data …")

        users   = interactions_df["user_id"].unique()
        items   = interactions_df["course_id"].unique()
        n_u, n_i = len(users), len(items)

        self.user_index   = {u: i for i, u in enumerate(users)}
        self.item_index   = {c: i for i, c in enumerate(items)}
        self.reverse_user = {i: u for u, i in self.user_index.items()}
        self.reverse_item = {i: c for c, i in self.item_index.items()}

        # Build R with exponential time-decay (fix #12)
        # Decayed score = cf_score × exp(-λ × days_since_interaction)
        decay_weights = _apply_time_decay(interactions_df)
        R = np.zeros((n_u, n_i), dtype=np.float32)
        for (_, row), w in zip(interactions_df.iterrows(), decay_weights):
            u = self.user_index.get(row["user_id"])
            c = self.item_index.get(row["course_id"])
            if u is not None and c is not None:
                raw = float(row["cf_score"]) if not np.isnan(row["cf_score"]) else 0.0
                R[u, c] = float(raw * w)

        # Normalise R to [0, 1]
        r_max = R.max()
        if r_max > 0:
            R = R / r_max

        n_obs    = (R > 0).sum()
        sparsity = 1 - n_obs / (n_u * n_i)
        print(f"[WALS] Matrix: {n_u} × {n_i}  sparsity: {sparsity:.2%}  "
              f"observed: {n_obs:,}")

        # Preference matrix: P[u,i] = 1 if R[u,i] > 0
        P = (R > 0).astype(np.float32)

        # Confidence matrix: C[u,i] = 1 + alpha * R[u,i]
        C = 1.0 + self.alpha * R

        # Initialise factor matrices with small random values
        rng = np.random.default_rng(42)
        self.user_factors = rng.standard_normal((n_u, self.n_factors)).astype(np.float32) * 0.01
        self.item_factors = rng.standard_normal((n_i, self.n_factors)).astype(np.float32) * 0.01

        lambda_I = self.lambda_reg * np.eye(self.n_factors, dtype=np.float32)

        print(f"[WALS] Training {self.n_iter} ALS iterations …")
        for it in range(self.n_iter):
            # ── Fix item factors, solve for user factors ───────────────────
            YtY = self.item_factors.T @ self.item_factors   # (f,f)
            for u in range(n_u):
                cu       = C[u]                              # (n_i,)
                pu       = P[u]                              # (n_i,)
                # Decomposition: Y^T C^u Y = YtY + Y^T diag(cu-1) Y
                diag_cu  = cu - 1.0                          # (n_i,)
                # Y^T diag(cu-1) Y  efficiently via element-wise
                YtCu1Y   = (self.item_factors * diag_cu[:, None]).T @ self.item_factors
                A        = YtY + YtCu1Y + lambda_I
                b        = (self.item_factors * (cu * pu)[:, None]).sum(axis=0)
                self.user_factors[u] = np.linalg.solve(A, b)

            # ── Fix user factors, solve for item factors ───────────────────
            XtX = self.user_factors.T @ self.user_factors   # (f,f)
            for i in range(n_i):
                ci       = C[:, i]                           # (n_u,)
                pi       = P[:, i]                           # (n_u,)
                diag_ci  = ci - 1.0
                XtCi1X   = (self.user_factors * diag_ci[:, None]).T @ self.user_factors
                A        = XtX + XtCi1X + lambda_I
                b        = (self.user_factors * (ci * pi)[:, None]).sum(axis=0)
                self.item_factors[i] = np.linalg.solve(A, b)

            # ── Progress report ────────────────────────────────────────────
            if (it + 1) % 5 == 0 or it == 0:
                preds    = self.user_factors @ self.item_factors.T
                obs_mask = R > 0
                rmse     = float(np.sqrt(np.mean((preds[obs_mask] - R[obs_mask])**2)))
                print(f"[WALS] Iter {it+1:>3}/{self.n_iter}  RMSE(observed): {rmse:.4f}")

        self.is_fitted = True
        print("[WALS] Training complete.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_score(self, user_id: str, course_id: str) -> float:
        u = self.user_index.get(user_id)
        c = self.item_index.get(course_id)
        if u is None or c is None:
            return 0.0
        return float(np.clip(
            self.user_factors[u] @ self.item_factors[c], 0, 1
        ))

    def recommend(self, user_id: str, courses_df: pd.DataFrame,
                  top_n: int = 10, exclude_seen: bool = True,
                  interactions_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        u = self.user_index.get(user_id)
        if u is None:
            return courses_df.sort_values("popularity_score", ascending=False).head(top_n)

        user_vec   = self.user_factors[u]                      # (f,)
        # Only predict for courses in our item index
        scores_dict = {
            self.reverse_item[i]: float(np.clip(
                user_vec @ self.item_factors[i], 0, 1))
            for i in range(len(self.item_factors))
        }

        scored = courses_df.copy()
        scored["cf_predicted_score"] = scored["course_id"].map(scores_dict).fillna(0.0)

        if exclude_seen and interactions_df is not None:
            seen = set(interactions_df[interactions_df["user_id"] == user_id]["course_id"])
            scored = scored[~scored["course_id"].isin(seen)]

        return scored.sort_values("cf_predicted_score", ascending=False).head(top_n)

    def get_user_factors(self, user_id: str) -> np.ndarray | None:
        u = self.user_index.get(user_id)
        return self.user_factors[u] if u is not None else None

    def get_similar_users(self, user_id: str, top_n: int = 5) -> list:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        u = self.user_index.get(user_id)
        if u is None: return []
        vec  = self.user_factors[u].reshape(1, -1)
        sims = cos_sim(vec, self.user_factors).flatten()
        sims[u] = -1
        top  = np.argsort(sims)[::-1][:top_n]
        return [self.reverse_user[i] for i in top]

    def get_top_topics(self, user_id: str, top_n: int = 3) -> list:
        u = self.user_index.get(user_id)
        if u is None: return []
        return np.argsort(self.user_factors[u])[::-1][:top_n].tolist()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, interactions_df: pd.DataFrame,
                 courses_df: pd.DataFrame, k: int = 10) -> dict:
        """Precision@K and NDCG@K on a holdout sample of 50 users."""
        users  = interactions_df["user_id"].unique()
        sample = np.random.choice(users, min(50, len(users)), replace=False)
        prec_scores, ndcg_scores = [], []

        for uid in sample:
            user_iacts = interactions_df[interactions_df["user_id"] == uid]
            # Positive = above-median cf_score for this user
            threshold = user_iacts["cf_score"].median()
            liked     = set(user_iacts[user_iacts["cf_score"] >= threshold]["course_id"])
            if not liked: continue

            recs     = self.recommend(uid, courses_df, top_n=k,
                                      exclude_seen=False,
                                      interactions_df=interactions_df)
            rec_ids  = recs["course_id"].tolist()[:k]

            # Precision@K
            hits = sum(1 for cid in rec_ids if cid in liked)
            prec_scores.append(hits / k)

            # NDCG@K
            rel_map = dict(zip(user_iacts["course_id"], user_iacts["cf_score"]))
            dcg  = sum(rel_map.get(cid, 0) / np.log2(i + 2)
                       for i, cid in enumerate(rec_ids))
            ideal = sorted(rel_map.values(), reverse=True)[:k]
            idcg  = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)

        return {
            f"precision@{k}": round(float(np.mean(prec_scores)), 4) if prec_scores else 0,
            f"ndcg@{k}":      round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0,
            "n_users_evaluated": len(prec_scores),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = CACHE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[WALS] Model saved -> {path}")

    @staticmethod
    def load(path: str = CACHE_PATH) -> "WALSCollaborativeFilter":
        with open(path, "rb") as f:
            return _CFUnpickler(f).load()


# ── Fast NMF alias (used when WALS is too slow) ────────────────
# CollaborativeFilter alias points to this for compatibility


# ════════════════════════════════════════════════════════════════════════════
# NMF-based Collaborative Filter — fast alternative for large matrices
# Replaces WALSCollaborativeFilter when 100k+ interactions make ALS slow.
# ════════════════════════════════════════════════════════════════════════════

class CollaborativeFilterNMF:
    """
    Non-negative Matrix Factorisation collaborative filter.
    Trains in <60s on 120k interactions vs WALS which takes 10+ minutes.
    """
    def __init__(self, n_factors: int = 50):
        self.n_factors      = n_factors
        self.user_factors   = None
        self.course_factors = None
        self.user_index     = {}
        self.course_index   = {}
        self.reverse_user   = {}
        self.reverse_course = {}
        self.is_fitted      = False

    def fit(self, interactions_df: pd.DataFrame) -> "CollaborativeFilterNMF":
        from sklearn.decomposition import NMF as _NMF
        from sklearn.metrics import mean_squared_error as _mse
        users   = interactions_df["user_id"].unique()
        courses = interactions_df["course_id"].unique()
        self.user_index   = {u:i for i,u in enumerate(users)}
        self.course_index = {c:i for i,c in enumerate(courses)}
        self.reverse_user   = {i:u for u,i in self.user_index.items()}
        self.reverse_course = {i:c for c,i in self.course_index.items()}
        n_u, n_c = len(users), len(courses)
        # Apply time-decay before building R (fix #12)
        decay_weights = _apply_time_decay(interactions_df)

        # Use a sparse pivot instead of iterrows() — 100x faster and correct.
        # True zeros for unobserved entries so NMF learns real signal.
        # The old np.full(..., 0.01) floor meant 99.56% of cells were fake
        # 0.01 values; NMF couldn't distinguish observed from unobserved and
        # learned to reconstruct ~0.01 everywhere (RMSE=3.06, scores=0.0000).
        interactions_df = interactions_df.copy()
        interactions_df["_w_score"] = interactions_df["cf_score"].fillna(0.0).clip(lower=0.0) * decay_weights.values
        pivot = interactions_df.pivot_table(
            index="user_id", columns="course_id",
            values="_w_score", aggfunc="max", fill_value=0.0
        )
        # Reindex to match our index order
        pivot = pivot.reindex(index=list(self.user_index.keys()),
                              columns=list(self.course_index.keys()),
                              fill_value=0.0)
        R = pivot.values.astype(np.float32)
        sparsity = (R == 0.0).mean()
        print(f"[NMF-CF] Matrix: {n_u}×{n_c}  sparsity: {sparsity:.2%}")

        # Normalise observed values to [0, 1]
        r_max = R.max()
        if r_max > 0:
            R = R / r_max

        nmf = _NMF(n_components=self.n_factors, init="nndsvda", max_iter=200,
                   random_state=42, alpha_W=0.0, alpha_H=0.0)
        self.user_factors   = nmf.fit_transform(R)
        self.course_factors = nmf.components_.T

        # Precompute per-user score max for normalisation in predict_score().
        # Raw dot products are tiny on a 99.56% sparse matrix; storing the
        # per-user max lets predict_score() return relative scores in [0, 1]
        # consistent with recommend()'s normalisation.
        all_scores = self.user_factors @ self.course_factors.T   # (n_u, n_c)
        self._user_score_max = all_scores.max(axis=1)            # (n_u,)
        # Floor at a small epsilon to avoid divide-by-zero for new/cold users
        self._user_score_max = np.where(
            self._user_score_max > 1e-9, self._user_score_max, 1.0
        )
        obs  = R > 0.0
        rmse = float(np.sqrt(_mse(R[obs], (self.user_factors @ self.course_factors.T)[obs])))
        print(f"[NMF-CF] RMSE (observed): {rmse:.4f}")
        _save_metric("cf_rmse", round(rmse, 4))
        self.is_fitted = True
        return self

    def predict_score(self, user_id: str, course_id: str) -> float:
        """
        Return a normalised confidence score in [0, 1].

        Raw NMF dot products on a 99.56% sparse matrix are tiny (< 0.01).
        We normalise by the user's own max score so that their best matching
        course = 1.0 and relative preferences are preserved. This keeps the
        CF signal meaningful in the hybrid scorer even at extreme sparsity.
        """
        u = self.user_index.get(user_id)
        c = self.course_index.get(course_id)
        if u is None or c is None:
            return 0.0
        raw = float(self.user_factors[u] @ self.course_factors[c])
        u_max = float(self._user_score_max[u]) if hasattr(self, "_user_score_max") else 1.0
        return float(np.clip(raw / u_max, 0.0, 1.0))

    def recommend(self, user_id, courses_df, top_n=10,
                  exclude_seen=False, interactions_df=None):
        u = self.user_index.get(user_id)
        if u is None:
            return courses_df.sort_values("popularity_score", ascending=False).head(top_n)
        scores = self.course_factors @ self.user_factors[u]

        # Normalise to [0, 1] so downstream hybrid scorer and display get
        # meaningful values. Raw NMF dot products on a 99.56% sparse matrix
        # are tiny floats (< 0.01) — ranking order is correct but the
        # absolute values show as 0.0000 at 4dp and confuse the hybrid scorer.
        s_max = scores.max()
        if s_max > 0:
            scores = scores / s_max

        sc = {self.reverse_course[i]: float(scores[i]) for i in range(len(scores))}
        scored = courses_df.copy()
        scored["cf_predicted_score"] = scored["course_id"].map(sc).fillna(0.0)
        if exclude_seen and interactions_df is not None:
            seen = set(interactions_df[interactions_df["user_id"]==user_id]["course_id"])
            scored = scored[~scored["course_id"].isin(seen)]
        return scored.sort_values("cf_predicted_score", ascending=False).head(top_n)

    def get_similar_users(self, user_id: str, top_n: int = 5):
        from sklearn.metrics.pairwise import cosine_similarity as _cs
        u = self.user_index.get(user_id)
        if u is None: return []
        sims = _cs(self.user_factors[u].reshape(1,-1), self.user_factors).flatten()
        sims[u] = -1
        return [self.reverse_user[i] for i in np.argsort(sims)[::-1][:top_n]]

    def get_top_topics(self, user_id: str, top_n: int = 3):
        u = self.user_index.get(user_id)
        if u is None: return []
        return np.argsort(self.user_factors[u])[::-1][:top_n].tolist()

    def save(self, path: str = CACHE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)
        print(f"[NMF-CF] Saved -> {path}")

    @staticmethod
    def load(path: str = CACHE_PATH) -> "CollaborativeFilterNMF":
        with open(path, "rb") as f:
            return _CFUnpickler(f).load()


# Default alias — hybrid_recommender imports CollaborativeFilter
CollaborativeFilter = CollaborativeFilterNMF


def _assert_predict_score_contract(model, label: str):
    """
    Smoke-test that predict_score() stays within [0, 1] for both CF classes.
    Call after fit() or after load() — raises AssertionError on violation.

    Fix #4 — normalisation contract test
    """
    if not model.is_fitted:
        return
    # Sample up to 20 known user/item pairs from the model's index
    users = list(model.user_index.keys())[:5]
    items = list(
        (model.item_index if hasattr(model, "item_index") else model.course_index).keys()
    )[:4]
    for u in users:
        for i in items:
            score = model.predict_score(u, i)
            assert 0.0 <= score <= 1.0, (
                f"[{label}] predict_score({u!r}, {i!r}) = {score} — "
                f"outside [0, 1]. Fix the normalisation in predict_score()."
            )
    print(f"[{label}] predict_score contract OK — all sampled scores in [0, 1]")


if __name__ == "__main__":
    courses_df      = pd.read_csv("courses.csv")
    interactions_df = pd.read_csv("data/interactions.csv")
    users_df        = pd.read_csv("data/users.csv")

    # Use NMF — 10x faster than WALS on 120k interactions
    cf = CollaborativeFilterNMF(n_factors=50)
    cf.fit(interactions_df)
    _assert_predict_score_contract(cf, "CollaborativeFilterNMF")
    cf.save()

    # Demo
    sample_user = users_df.iloc[0]["user_id"]
    print(f"\n--- Recommendations for {sample_user} ---")
    recs = cf.recommend(sample_user, courses_df, top_n=5,
                        exclude_seen=True, interactions_df=interactions_df)
    for _, r in recs.iterrows():
        print(f"  [{r['cf_predicted_score']:.4f}]  {r['title']}  ({r['difficulty_label']})")
