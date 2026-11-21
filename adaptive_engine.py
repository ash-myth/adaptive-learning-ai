"""
adaptive_engine.py
==================
Adaptive AI Learning Intelligence Platform — Adaptive Intelligence Engine
-------------------------------------------------------------------------
This is the 'brain' that makes the system genuinely adaptive.

Three ML models inside
-----------------------

1. DifficultyPredictor (MLPClassifier — Neural Network)
   Predicts whether a user will STRUGGLE, MATCH, or EXCEL at a course
   based on their profile and interaction history features.
   This replaces the original if/else label check.

2. DropoutRiskPredictor (LogisticRegression)
   Predicts probability of course dropout before completion.
   Used to proactively suggest easier prerequisites.

3. UserProfiler
   Maintains a dynamic user profile updated after each interaction.
   Uses KMeans clustering to discover user segments from interaction data.
   Computes a learning_velocity metric that adapts the learning path.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.cluster       import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics       import silhouette_score
from sklearn.metrics       import classification_report
from sklearn.pipeline      import Pipeline
from sklearn.impute        import SimpleImputer
import pickle, os, warnings, json

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)


def _save_metric(key: str, value) -> None:
    """Persist a single training metric to data/model_metrics.json.
    Merges into the existing file so different models don't overwrite each other."""
    path = "data/model_metrics.json"
    try:
        with open(path, "r") as f:
            metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metrics = {}
    metrics[key] = value
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved {key}={value} -> {path}")

DIFFICULTY_MAP   = {"Mixed": 0, "Beginner": 1, "Intermediate": 2, "Advanced": 3}
OUTCOME_LABELS   = ["Struggle", "Match", "Excel"]
N_USER_CLUSTERS  = 6


# ════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ════════════════════════════════════════════════════════════════════════════

def build_features(
    interactions_df: pd.DataFrame,
    users_df:        pd.DataFrame,
    courses_df:      pd.DataFrame,
) -> pd.DataFrame:
    
    df = interactions_df.copy()
    df = df.merge(
        users_df[["user_id","skill_level","weekly_hours","archetype"]],
        on="user_id", how="left"
    )
    df = df.merge(
        courses_df[["course_id","difficulty_level","difficulty_label",
                    "rating","popularity_score","duration_mins","subject","provider"]],
        on="course_id", how="left"
    )

    le = LabelEncoder()
    df["archetype_enc"]    = le.fit_transform(df["archetype"].fillna("Unknown"))
    df["subject_enc"]      = LabelEncoder().fit_transform(df["subject"].fillna("Other"))
    df["difficulty_level"] = pd.to_numeric(df["difficulty_level"], errors="coerce").fillna(0).astype(int)
    df["difficulty_gap"]   = df["difficulty_level"] - df["skill_level"].fillna(1)
    df = df.sort_values(["user_id"]).reset_index(drop=True)

    # Rolling stats (causal — only past interactions visible)
    df["avg_quiz_score"] = df.groupby("user_id")["quiz_score"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(df["quiz_score"].mean())
    df["avg_completion"] = df.groupby("user_id")["completed"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0.5)
    df["n_completed"]    = df.groupby("user_id")["completed"].transform(
        lambda x: x.expanding().sum().shift(1)).fillna(0)
    df["avg_replay"]     = df.groupby("user_id")["replay_count"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0)

    # New feature 1: n_interactions (experience counter)
    df["n_interactions"] = df.groupby("user_id").cumcount()

    # New feature 2: causal time efficiency (only past interactions visible, avoiding target leakage)
    raw_eff = (df["time_spent"] / df["duration_mins"].clip(lower=1)).clip(0, 3)
    df["_raw_eff_temp"] = raw_eff
    df["time_efficiency"] = df.groupby("user_id")["_raw_eff_temp"].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(1.0)
    df.drop(columns=["_raw_eff_temp"], inplace=True)

    # New feature 3: replay rate (confusion signal)
    df["replay_rate"] = df["replay_count"] / (df["n_interactions"].clip(lower=1) + 1)

    # New feature 4: subject affinity
    df["subject_affinity"] = df.groupby(["user_id","subject_enc"])["quiz_score"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(df["quiz_score"].mean())

    # New feature 5: difficulty history
    df["diff_x_comp"]       = df["difficulty_level"] * df["completed"]
    df["difficulty_history"] = (
        df.groupby("user_id")["diff_x_comp"].transform(lambda x: x.expanding().sum().shift(1)) /
        df.groupby("user_id")["completed"].transform(lambda x: x.expanding().sum().shift(1).clip(lower=0.01))
    ).fillna(0)

    # New feature 6: completion trend
    df["comp_recent5"]     = df.groupby("user_id")["completed"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)).fillna(0.5)
    df["completion_trend"] = df["comp_recent5"] - df["avg_completion"]

    # New feature 7: provider familiarity
    df["provider_familiarity"] = df.groupby(["user_id","provider"]).cumcount()

    # Outcome and dropout labels
    df["outcome"] = np.where(df["quiz_score"] < 50, 0,
                    np.where(df["quiz_score"] < 75, 1, 2))
    df["dropout"] = 1 - df["completed"]

    # Fill nulls
    df["rating"]           = pd.to_numeric(df["rating"],errors="coerce").fillna(df["rating"].median() if df["rating"].notna().any() else 4.0)
    df["popularity_score"] = df["popularity_score"].fillna(0.5)
    df["duration_mins"]    = df["duration_mins"].fillna(120.0)

    return df.fillna(0)


FEATURE_COLS = [
    # Original 11
    "skill_level",
    "weekly_hours",
    "archetype_enc",
    "avg_quiz_score",
    "avg_completion",
    "n_completed",
    "difficulty_level",
    "difficulty_gap",
    "rating",
    "popularity_score",
    "duration_mins",
    # New 8 behavioural features
    "time_efficiency",
    "replay_rate",
    "subject_affinity",
    "difficulty_history",
    "completion_trend",
    "n_interactions",
    "provider_familiarity",
    "avg_replay",
]


# ════════════════════════════════════════════════════════════════════════════
# Model 1 — Difficulty outcome predictor (MLP Neural Network)
# ════════════════════════════════════════════════════════════════════════════

class DifficultyPredictor:
    """
    Soft-voting ensemble: HistGradientBoosting + MLP.

    Fix #8 — GradientBoostingClassifier → HistGradientBoostingClassifier
    -----------------------------------------------------------------------
    GradientBoostingClassifier builds trees with exact greedy split search:
    O(N × F) per tree — slow on 116k rows.

    HistGradientBoostingClassifier bins each feature into 255 buckets and
    operates on histograms: O(bins × F) per tree, independent of N.
    On 116k rows this is 5–10× faster, enabling:
      - 200 → 300 estimators at the same wall-clock cost
      - Full 5-fold CV on all training users (was 3-fold on a 20k sample)
      - Native NaN handling — no SimpleImputer needed in the GBM pipeline

    Ensemble design:
    - HistGBM excels at tabular data with mixed feature types (our case)
    - MLP captures non-linear feature interactions
    - Soft voting averages probability distributions → more calibrated
    """

    def __init__(self):
        # HistGradientBoostingClassifier (fix #8):
        #   - Histogram-based splits → 5-10× faster than GradientBoosting
        #   - Native NaN handling: SimpleImputer not needed in this pipeline
        #   - max_iter=300 (was 200) at the same wall-clock cost
        #   - l2_regularization for better generalisation on 116k rows
        #   - early_stopping prevents overfitting without a fixed n_estimators
        self.gbm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    HistGradientBoostingClassifier(
                max_iter            = 300,
                max_depth           = 4,
                learning_rate       = 0.08,
                min_samples_leaf    = 20,
                l2_regularization   = 0.1,
                random_state        = 42,
                early_stopping      = True,
                validation_fraction = 0.1,
                n_iter_no_change    = 20,
            )),
        ])
        self.mlp_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("clf",     MLPClassifier(
                hidden_layer_sizes  = (256, 128, 64),
                activation          = "relu",
                solver              = "adam",
                max_iter            = 600,
                random_state        = 42,
                early_stopping      = True,
                validation_fraction = 0.1,
                n_iter_no_change    = 25,
                learning_rate_init  = 0.001,
            )),
        ])
        self.is_fitted = False
        self.classes_  = OUTCOME_LABELS
        # Ensemble weights: GBM slightly trusted more on tabular
        self.gbm_weight = 0.55
        self.mlp_weight = 0.45

    def fit(self, features_df: pd.DataFrame) -> "DifficultyPredictor":
        X = features_df[FEATURE_COLS].values
        y = features_df["outcome"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"[DifficultyPredictor] Training GBM ({len(X_train):,} samples, {len(FEATURE_COLS)} features)...")
        self.gbm_pipeline.fit(X_train, y_train)
        print(f"[DifficultyPredictor] Training MLP...")
        self.mlp_pipeline.fit(X_train, y_train)

        # Ensemble evaluation
        gbm_proba = self.gbm_pipeline.predict_proba(X_test)
        mlp_proba = self.mlp_pipeline.predict_proba(X_test)
        ens_proba = self.gbm_weight * gbm_proba + self.mlp_weight * mlp_proba
        y_pred    = np.argmax(ens_proba, axis=1)
        accuracy  = (y_pred == y_test).mean()

        gbm_acc   = (self.gbm_pipeline.predict(X_test) == y_test).mean()
        mlp_acc   = (self.mlp_pipeline.predict(X_test) == y_test).mean()

        print(f"[DifficultyPredictor] GBM accuracy:      {gbm_acc:.3f}")
        print(f"[DifficultyPredictor] MLP accuracy:      {mlp_acc:.3f}")
        print(f"[DifficultyPredictor] Ensemble accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred,
                                    target_names=OUTCOME_LABELS,
                                    zero_division=0))

        # ── Persist metrics so the dashboard can display them ─────────────
        _save_metric("difficulty_accuracy", round(float(accuracy), 4))
        _save_metric("gbm_accuracy",        round(float(gbm_acc), 4))
        _save_metric("mlp_accuracy",        round(float(mlp_acc), 4))

        # Feature importances from GBM
        # HistGradientBoostingClassifier exposes feature_importances_ after fit
        clf = self.gbm_pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_") and clf.feature_importances_ is not None:
            fi = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
            print("[DifficultyPredictor] Top 5 features:")
            for feat, imp in fi[:5]:
                print(f"  {feat:<30} {imp:.4f}")
        else:
            print("[DifficultyPredictor] Feature importances not available (skipping)")

        # Fix #8 — full 5-fold CV on all training users
        # HistGBM is 5-10× faster than GBM, making this affordable on 116k rows.
        # CV runs on the complete training split (no sampling), giving a
        # statistically robust estimate instead of a noisy 20k-row proxy.
        print(f"[DifficultyPredictor] Running 5-fold CV on all {len(X_train):,} "
              f"training samples (HistGBM — fast)...")
        cv_scores = cross_val_score(
            self.gbm_pipeline, X_train, y_train,
            cv=5, scoring="accuracy", n_jobs=-1
        )
        print(f"[DifficultyPredictor] HistGBM 5-fold CV (full train set): "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        self.is_fitted = True
        return self

    def predict(self, user_features: dict) -> dict:
        """Ensemble probability prediction for a single user-course pair."""
        x = np.array([[user_features.get(col, 0) for col in FEATURE_COLS]])
        gbm_p = self.gbm_pipeline.predict_proba(x)[0]
        mlp_p = self.mlp_pipeline.predict_proba(x)[0]
        proba = self.gbm_weight * gbm_p + self.mlp_weight * mlp_p
        pred  = int(np.argmax(proba))
        return {
            "predicted_outcome": OUTCOME_LABELS[pred],
            "struggle_prob":  round(float(proba[0]), 3),
            "match_prob":     round(float(proba[1]), 3),
            "excel_prob":     round(float(proba[2]), 3),
            "confidence":     round(float(proba[pred]), 3),
        }

    def predict_batch(self, X: np.ndarray) -> dict:
        """
        Vectorised ensemble prediction for a 2-D feature matrix.

        Replaces calling predict() in a Python loop over N candidates.
        Instead of N × 2 model calls we make exactly 2 model calls total
        (one GBM, one MLP) regardless of N — roughly 40× faster at N=120.

        Parameters
        ----------
        X : np.ndarray, shape (N, len(FEATURE_COLS))

        Returns
        -------
        dict with arrays of length N:
            predicted_outcome, struggle_prob, match_prob, excel_prob
        """
        gbm_proba = self.gbm_pipeline.predict_proba(X)          # (N, 3)
        mlp_proba = self.mlp_pipeline.predict_proba(X)          # (N, 3)
        proba     = self.gbm_weight * gbm_proba + self.mlp_weight * mlp_proba
        preds     = np.argmax(proba, axis=1)                     # (N,)
        return {
            "predicted_outcome": np.array([OUTCOME_LABELS[p] for p in preds]),
            "struggle_prob":     proba[:, 0].round(3),
            "match_prob":        proba[:, 1].round(3),
            "excel_prob":        proba[:, 2].round(3),
        }


# ════════════════════════════════════════════════════════════════════════════
# Model 2 — Dropout risk predictor (Logistic Regression)
# ════════════════════════════════════════════════════════════════════════════

class DropoutRiskPredictor:
    """
    Logistic Regression model predicting probability of course dropout.

    Why Logistic Regression here instead of MLP?
    - Dropout is a binary outcome — LogReg is calibrated for this
    - Coefficients are directly interpretable (good for Explainable AI)
    - Faster inference — can run on every candidate course during ranking
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C             = 1.0,
                solver        = "lbfgs",
                max_iter      = 1000,
                random_state  = 42,
                class_weight  = "balanced",  # handle class imbalance
            )),
        ])
        self.is_fitted    = False
        self.feature_cols = FEATURE_COLS

    def fit(self, features_df: pd.DataFrame) -> "DropoutRiskPredictor":
        X = features_df[self.feature_cols].values
        y = features_df["dropout"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.pipeline.fit(X_train, y_train)
        y_pred   = self.pipeline.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"[DropoutPredictor] Test accuracy: {accuracy:.3f}")

        # ── Persist metric so the dashboard can display it ────────────────
        _save_metric("dropout_accuracy", round(float(accuracy), 4))

        # Feature importance via logistic regression coefficients
        coefs     = self.pipeline.named_steps["clf"].coef_[0]
        imp_pairs = sorted(zip(self.feature_cols, coefs),
                           key=lambda x: abs(x[1]), reverse=True)
        print("[DropoutPredictor] Top feature importances:")
        for feat, coef in imp_pairs[:5]:
            direction = "^ dropout" if coef > 0 else "v dropout"
            print(f"  {feat:<25} {coef:+.4f}  {direction}")

        self.is_fitted = True
        return self

    def _fix_imputer_compat(self):
        """Ensure cross-version sklearn compatibility for SimpleImputer."""
        if hasattr(self, "pipeline") and hasattr(self.pipeline, "named_steps"):
            if "imputer" in self.pipeline.named_steps:
                imp = self.pipeline.named_steps["imputer"]
                if not hasattr(imp, "_fill_dtype"):
                    imp._fill_dtype = getattr(imp, "_fit_dtype", np.dtype("float64"))
                if not hasattr(imp, "_fit_dtype"):
                    imp._fit_dtype = getattr(imp, "_fill_dtype", np.dtype("float64"))

    def predict_risk(self, user_features: dict) -> dict:
        """Return dropout probability and risk level for a user-course pair."""
        self._fix_imputer_compat()
        x     = np.array([[user_features.get(col, 0) for col in self.feature_cols]])
        proba = self.pipeline.predict_proba(x)[0]
        risk  = float(proba[1])

        if risk > 0.70:
            level = "HIGH"
        elif risk > 0.40:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "dropout_probability": round(risk, 3),
            "risk_level":          level,
        }

    def predict_risk_batch(self, X: np.ndarray) -> dict:
        """
        Vectorised dropout prediction for a 2-D feature matrix.

        One model call for all N candidates instead of N calls.

        Parameters
        ----------
        X : np.ndarray, shape (N, len(FEATURE_COLS))

        Returns
        -------
        dict with arrays of length N:
            dropout_probability, risk_level
        """
        self._fix_imputer_compat()
        proba = self.pipeline.predict_proba(X)[:, 1]            # (N,)
        levels = np.where(proba > 0.70, "HIGH",
                 np.where(proba > 0.40, "MEDIUM", "LOW"))
        return {
            "dropout_probability": proba.round(3),
            "risk_level":          levels,
        }


# ════════════════════════════════════════════════════════════════════════════
# Model 3 — User Profiler with KMeans clustering
# ════════════════════════════════════════════════════════════════════════════

class UserProfiler:
    """
    Dynamic user profiler that:
    1. Clusters users into learning personas using KMeans
    2. Maintains a live profile updated from interaction signals
    3. Computes learning_velocity — how quickly the user is progressing

    Fix #10 — KMeans validation + centroid-anchored stable persona names
    -----------------------------------------------------------------------
    Old problems:
    a) n_clusters=6 was hardcoded with no quality validation.  A bad K
       produces meaningless personas that hurt recommendation diversity.
    b) Persona names were hardcoded to cluster indices (0="High Achiever",
       1="Steady Climber", …).  KMeans label assignment is non-deterministic
       after a retrain — cluster 0 could become "Struggling Learner" and the
       system would silently misname every user's persona.

    Fixes:
    a) Silhouette analysis at fit time: tries K in [4, 5, 6, 7, 8] and picks
       the K with the highest silhouette score on a 5,000-user sample.
       Falls back to N_USER_CLUSTERS if fewer users exist or scores are flat.
    b) Persona names are assigned by matching each cluster's centroid to a
       named archetype definition — NOT by index.  The archetype with the
       closest centroid distance wins.  Labels are stable across retrains
       because they're anchored to data geometry, not random label order.
    """

    # Archetype definitions: each is a vector over CLUSTER_FEATURE_ORDER.
    # Values are conceptual directions (high/mid/low), not literal scores.
    # _assign_personas() finds the closest centroid to each archetype.
    _ARCHETYPE_VECTORS: dict[str, list[float]] = {
        "High Achiever":      [1.0, 1.0, 1.0, 0.5, 1.0, 0.2, 0.1],  # many, complete, high quiz
        "Steady Climber":     [0.6, 0.7, 0.6, 0.5, 0.6, 0.3, 0.3],  # moderate everything
        "Quick Sampler":      [1.0, 0.3, 0.5, 0.2, 0.5, 0.1, 0.2],  # many interactions, low completion
        "Deep Diver":         [0.4, 0.8, 0.7, 1.0, 0.7, 1.0, 0.5],  # high time, high replay
        "Struggling Learner": [0.5, 0.4, 0.3, 0.6, 0.3, 0.5, 0.8],  # low quiz, high std
        "Curious Explorer":   [0.8, 0.5, 0.6, 0.4, 0.5, 0.3, 0.4],  # broad, moderate scores
    }
    # Feature order must match _user_agg_features() column order (excl user_id)
    _CLUSTER_FEATURE_ORDER = [
        "n_interactions", "completion_rate", "avg_quiz_score",
        "avg_time_spent", "avg_cf_score", "avg_replay_count", "std_quiz_score",
    ]

    def __init__(self, n_clusters: int = N_USER_CLUSTERS):
        self.n_clusters      = n_clusters
        self.kmeans          = None          # built after K validation
        self.scaler          = StandardScaler()
        self.is_fitted       = False
        self.persona_map_:   dict[int, str] = {}   # cluster_id → persona name
        self.silhouette_score_: float        = 0.0
        self.best_k_:           int          = n_clusters

    def _user_agg_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-user statistics for clustering."""
        agg = interactions_df.groupby("user_id").agg(
            n_interactions    = ("course_id",   "count"),
            completion_rate   = ("completed",   "mean"),
            avg_quiz_score    = ("quiz_score",  "mean"),
            avg_time_spent    = ("time_spent",  "mean"),
            avg_cf_score      = ("cf_score",    "mean"),
            avg_replay_count  = ("replay_count","mean"),
            std_quiz_score    = ("quiz_score",  "std"),
        ).reset_index().fillna(0)
        return agg

    def _validate_k(self, X: np.ndarray, k_range: range) -> int:
        """
        Pick K via silhouette analysis on a sample of up to 5,000 users.

        Silhouette score measures how similar a point is to its own cluster
        vs other clusters — ranges from -1 (wrong cluster) to +1 (dense,
        well-separated).  We try each K and pick the one with the highest
        mean silhouette score.

        Falls back to self.n_clusters if there are too few users to
        evaluate multiple K values meaningfully.
        """
        n = len(X)
        if n < max(k_range) * 2:
            print(f"[UserProfiler] Too few users ({n}) for K validation "
                  f"— using K={self.n_clusters}")
            return self.n_clusters

        # Sample for speed: silhouette on 5k points is representative
        sample_size = min(5000, n)
        rng  = np.random.default_rng(42)
        idx  = rng.choice(n, sample_size, replace=False)
        Xs   = X[idx]

        best_k, best_score = self.n_clusters, -1.0
        print("[UserProfiler] Silhouette analysis:")
        for k in k_range:
            km    = KMeans(n_clusters=k, random_state=42, n_init=5)
            labs  = km.fit_predict(Xs)
            score = silhouette_score(Xs, labs, sample_size=min(2000, sample_size))
            print(f"  K={k}  silhouette={score:.4f}")
            if score > best_score:
                best_k, best_score = k, score

        print(f"[UserProfiler] Best K={best_k}  silhouette={best_score:.4f}")
        self.silhouette_score_ = best_score
        return best_k

    def _assign_personas(self, centroids: np.ndarray) -> dict[int, str]:
        """
        Map cluster indices to named archetypes by centroid proximity.

        For each archetype, find the cluster whose (scaled) centroid is
        closest to the archetype's conceptual vector.  Uses greedy
        assignment — each cluster gets exactly one persona name.

        This is stable across retrains: if cluster 2's centroid best
        matches "High Achiever" after retrain, it gets that label
        regardless of its index — unlike the old hardcoded index mapping.
        """
        from sklearn.metrics.pairwise import euclidean_distances

        # Build archetype matrix in the same feature space as the centroids
        archetype_names = list(self._ARCHETYPE_VECTORS.keys())
        A = np.array([self._ARCHETYPE_VECTORS[n] for n in archetype_names])

        # Centroids are in scaled space; scale the archetypes too
        # (archetypes are unit-range [0,1], centroids are z-scored)
        # We compare directions, not magnitudes, so cosine distance is used.
        def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
            bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            return 1.0 - an @ bn.T    # (n_archetypes, n_clusters)

        # Pad or trim archetype vectors to match centroid dimensionality
        n_feat = centroids.shape[1]
        n_arch = A.shape[1]
        if n_feat > n_arch:
            A = np.pad(A, ((0, 0), (0, n_feat - n_arch)))
        elif n_feat < n_arch:
            A = A[:, :n_feat]

        dist = _cosine_dist(A, centroids)   # (n_archetypes, n_clusters)

        # Greedy assignment: pick the archetype-cluster pair with lowest
        # distance, assign it, then remove both from consideration.
        n_clusters  = centroids.shape[0]
        assigned_clusters  = set()
        assigned_archetypes = set()
        persona_map: dict[int, str] = {}

        # Flatten distances and sort ascending
        pairs = sorted(
            [(dist[ai, ci], ai, ci)
             for ai in range(len(archetype_names))
             for ci in range(n_clusters)],
            key=lambda x: x[0],
        )
        for _, ai, ci in pairs:
            if ai in assigned_archetypes or ci in assigned_clusters:
                continue
            persona_map[ci] = archetype_names[ai]
            assigned_clusters.add(ci)
            assigned_archetypes.add(ai)
            if len(persona_map) == n_clusters:
                break

        # Any unmatched clusters (if n_clusters > n_archetypes) → generic label
        for ci in range(n_clusters):
            if ci not in persona_map:
                persona_map[ci] = f"Learner Type {ci}"

        return persona_map

    def fit(self, interactions_df: pd.DataFrame) -> "UserProfiler":
        agg          = self._user_agg_features(interactions_df)
        cluster_cols = [c for c in agg.columns if c != "user_id"]
        X            = self.scaler.fit_transform(agg[cluster_cols].values)

        # Fix #10a — validate K via silhouette analysis
        best_k   = self._validate_k(X, k_range=range(4, 9))
        self.best_k_ = best_k

        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.kmeans.fit(X)

        agg["persona_cluster"] = self.kmeans.labels_

        # Fix #10b — assign persona names by centroid proximity (not by index)
        self.persona_map_ = self._assign_personas(self.kmeans.cluster_centers_)
        self.user_clusters_ = dict(zip(agg["user_id"].astype(str), agg["persona_cluster"].astype(int)))
        self.is_fitted      = True

        print("[UserProfiler] Cluster -> persona mapping (centroid-anchored):")
        for cid, name in sorted(self.persona_map_.items()):
            count = (agg["persona_cluster"] == cid).sum()
            print(f"  Cluster {cid} -> {name:<22} : {count} users")
        return self

    def get_profile(
        self,
        user_id:         str,
        interactions_df: pd.DataFrame,
        users_df:        pd.DataFrame,
        courses_df:      pd.DataFrame | None = None,
    ) -> dict:
        """
        Build a comprehensive live profile for a user.
        This is what feeds into the adaptive recommendations.

        courses_df — fix #3
        -------------------
        Previously this method called pd.read_csv("courses.csv") on every
        invocation to join subject data for strength/weakness analysis.
        That means every API recommendation request triggered a full CSV
        read from disk (~13k rows).  courses_df is now passed in from the
        caller (HybridRecommender already holds self.courses_df in memory)
        and falls back to the disk read only if omitted, preserving
        backwards compatibility with standalone usage.
        """
        user_row   = users_df[users_df["user_id"] == user_id]
        user_iacts = interactions_df[interactions_df["user_id"] == user_id]

        if user_row.empty:
            return {"error": f"User {user_id} not found"}

        user       = user_row.iloc[0]
        n_interact = len(user_iacts)

        if n_interact == 0:
            return {
                "user_id":       user_id,
                "skill_level":   int(user["skill_level"]),
                "archetype":     user["archetype"],
                "persona":       "New User",
                "learning_velocity": 0.0,
                "recommended_difficulty": "Beginner",
                "avg_quiz_score": 0.0,
                "completion_rate": 0.0,
                "warning": "No interaction history yet — cold start",
            }

        avg_quiz    = float(user_iacts["quiz_score"].mean())
        comp_rate   = float(user_iacts["completed"].mean())
        avg_time    = float(user_iacts["time_spent"].mean())

        # Learning velocity: improvement in quiz score over last 10 interactions
        if n_interact >= 5:
            recent_half  = user_iacts.tail(n_interact // 2)["quiz_score"].mean()
            earlier_half = user_iacts.head(n_interact // 2)["quiz_score"].mean()
            velocity     = float(recent_half - earlier_half)
        else:
            velocity = 0.0

        # Recommended difficulty based on quiz performance + velocity
        if avg_quiz > 75 and velocity >= 0:
            recommended_diff = "Advanced"
        elif avg_quiz > 60:
            recommended_diff = "Intermediate"
        elif avg_quiz > 45:
            recommended_diff = "Beginner"
        else:
            recommended_diff = "Beginner"

        # Persona from cluster — handles both dict and legacy DataFrame format
        persona = "Unclustered"
        if self.is_fitted:
            if isinstance(self.user_clusters_, dict):
                # New format: dict[user_id_str -> cluster_int]
                cid = self.user_clusters_.get(str(user_id))
                if cid is not None:
                    persona = self.persona_map_.get(int(cid), "Unknown")
            else:
                # Legacy format: DataFrame with columns ['user_id', 'persona_cluster']
                try:
                    cluster_row = self.user_clusters_[
                        self.user_clusters_["user_id"] == user_id
                    ]
                    if not cluster_row.empty:
                        persona = self.persona_map_.get(
                            int(cluster_row.iloc[0]["persona_cluster"]), "Unknown"
                        )
                except (KeyError, TypeError):
                    pass

        # Strengths and weaknesses
        # Use the in-memory courses_df passed by the caller (fix #3).
        # Fall back to disk only when called standalone (courses_df is None).
        _courses = (courses_df if courses_df is not None
                    else pd.read_csv("courses.csv"))
        by_subject = (
            user_iacts
            .merge(_courses[["course_id", "subject"]], on="course_id", how="left")
            .groupby("subject")["quiz_score"].mean()
            .sort_values(ascending=False)
        )
        strength    = by_subject.index[0]  if len(by_subject) > 0 else "N/A"
        weakness    = by_subject.index[-1] if len(by_subject) > 1 else "N/A"

        return {
            "user_id":                user_id,
            "archetype":              user["archetype"],
            "persona":                persona,
            "skill_level":            int(user["skill_level"]),
            "weekly_hours":           float(user["weekly_hours"]),
            "n_interactions":         n_interact,
            "avg_quiz_score":         round(avg_quiz, 1),
            "completion_rate":        round(comp_rate, 3),
            "avg_time_spent_mins":    round(avg_time, 1),
            "learning_velocity":      round(velocity, 2),
            "recommended_difficulty": recommended_diff,
            "strength_subject":       strength,
            "weakness_subject":       weakness,
            "goal":                   user["goal"],
        }


# ════════════════════════════════════════════════════════════════════════════
# Adaptive Engine — puts the three models together
# ════════════════════════════════════════════════════════════════════════════

class AdaptiveEngine:
    """
    Orchestrates all three ML models to produce adaptive recommendations.

    Given a user profile + a set of candidate courses:
    1. Predicts difficulty outcome for each candidate (MLP)
    2. Predicts dropout risk for each candidate (LogReg)
    3. Adjusts recommendation ranking based on both predictions
    4. Flags courses with high dropout risk and suggests prerequisites
    5. Returns structured recommendations with full explainability
    """

    def __init__(
        self,
        difficulty_predictor: DifficultyPredictor,
        dropout_predictor:    DropoutRiskPredictor,
        user_profiler:        UserProfiler,
    ):
        self.diff_pred    = difficulty_predictor
        self.dropout_pred = dropout_predictor
        self.profiler     = user_profiler

    def _build_user_features(
        self,
        profile:    dict,
        course_row: pd.Series,
        archetype_enc: int = 0,
    ) -> dict:
        """Assemble the feature dict for model inference."""
        diff_level = DIFFICULTY_MAP.get(
            str(course_row.get("difficulty_label", "Mixed")), 0
        )
        return {
            "skill_level":       profile.get("skill_level", 1),
            "weekly_hours":      profile.get("weekly_hours", 5),
            "archetype_enc":     archetype_enc,
            "avg_quiz_score":    profile.get("avg_quiz_score", 50),
            "avg_completion":    profile.get("completion_rate", 0.5),
            "n_completed":       profile.get("n_interactions", 0),
            "difficulty_level":  diff_level,
            "difficulty_gap":    diff_level - profile.get("skill_level", 1),
            "rating":            float(course_row.get("rating", 4.0)),
            "popularity_score":  float(course_row.get("popularity_score", 0.5)),
            "duration_mins":     float(course_row.get("duration_mins", 120)),
        }

    def _build_feature_matrix(self, profile: dict,
                               candidates: pd.DataFrame) -> np.ndarray:
        """
        Build a 2-D feature matrix (N × F) for all candidate courses at once.

        This is the vectorised counterpart of _build_user_features().
        User-level features (skill_level, weekly_hours, …) are the same for
        every row and are broadcast across all N candidates.  Course-level
        features (difficulty_level, rating, …) vary per row.

        Called by rank_courses() so that both ML models receive the full
        matrix in a single call rather than one dict at a time.
        """
        n = len(candidates)
        # User-level scalars — same for all candidates
        skill_level   = float(profile.get("skill_level", 1))
        weekly_hours  = float(profile.get("weekly_hours", 5))
        avg_quiz      = float(profile.get("avg_quiz_score", 50))
        avg_comp      = float(profile.get("completion_rate", 0.5))
        n_completed   = float(profile.get("n_interactions", 0))

        diff_levels = candidates["difficulty_label"].map(
            lambda v: DIFFICULTY_MAP.get(str(v), 0)).values.astype(float)

        X = np.column_stack([
            np.full(n, skill_level),                             # skill_level
            np.full(n, weekly_hours),                            # weekly_hours
            np.zeros(n),                                         # archetype_enc (default 0)
            np.full(n, avg_quiz),                                # avg_quiz_score
            np.full(n, avg_comp),                                # avg_completion
            np.full(n, n_completed),                             # n_completed
            diff_levels,                                         # difficulty_level
            diff_levels - skill_level,                           # difficulty_gap
            pd.to_numeric(candidates["rating"],
                          errors="coerce").fillna(4.0).values,   # rating
            candidates["popularity_score"].fillna(0.5).values,   # popularity_score
            candidates["duration_mins"].fillna(120.0).values,    # duration_mins
            np.full(n, float(profile.get("avg_time_efficiency", 1.0) or 1.0)), # time_efficiency (causal user baseline)
            np.zeros(n),   # replay_rate
            np.full(n, avg_quiz),                                # subject_affinity
            np.zeros(n),   # difficulty_history
            np.zeros(n),   # completion_trend
            np.full(n, n_completed),                             # n_interactions
            np.zeros(n),   # provider_familiarity
            np.zeros(n),   # avg_replay
        ])                                                       # shape (N, 19)
        return X

    def rank_courses(
        self,
        profile:     dict,
        candidates:  pd.DataFrame,
        top_n:       int = 10,
    ) -> pd.DataFrame:
        """
        Rank candidate courses for a user using all three ML signals.

        Adaptive scoring formula:
          adaptive_score = (
              0.35 * match_prob        (GBM+MLP ensemble)
            + 0.25 * excel_prob        (GBM+MLP ensemble)
            - 0.20 * dropout_prob      (LogReg penalty)
            + 0.20 * popularity_score  (catalogue signal)
          )

        Vectorised — fix #2
        -------------------
        Old: N iterations × 3 model.predict_proba() calls = 3N calls total.
        New: _build_feature_matrix() assembles all N rows at once, then
             predict_batch() and predict_risk_batch() each make exactly
             1 model call regardless of N.  Total: 3 calls for any N.
        At N=120 this is ~40× faster per request.
        """
        if candidates.empty:
            return pd.DataFrame()

        # ── Build full feature matrix in one pass ─────────────────────────
        X = self._build_feature_matrix(profile, candidates)

        # ── Single batch call per model ───────────────────────────────────
        diff_results = self.diff_pred.predict_batch(X)
        drop_results = self.dropout_pred.predict_risk_batch(X)

        pop_scores = candidates["popularity_score"].fillna(0.5).values

        adaptive_scores = (
            0.35 * diff_results["match_prob"]
            + 0.25 * diff_results["excel_prob"]
            - 0.20 * drop_results["dropout_probability"]
            + 0.20 * pop_scores
        )

        result = candidates.copy().reset_index(drop=True)
        result["adaptive_score"]      = adaptive_scores.round(4)
        result["predicted_outcome"]   = diff_results["predicted_outcome"]
        result["struggle_prob"]       = diff_results["struggle_prob"]
        result["match_prob"]          = diff_results["match_prob"]
        result["excel_prob"]          = diff_results["excel_prob"]
        result["dropout_risk"]        = drop_results["risk_level"]
        result["dropout_probability"] = drop_results["dropout_probability"]

        return result.sort_values("adaptive_score", ascending=False).head(top_n)

    def explain(self, profile: dict, course_row: pd.Series) -> str:
        """
        Generate a human-readable explanation for a recommendation.
        This is the Explainable AI layer.
        """
        feats     = self._build_user_features(profile, course_row)
        diff_pred = self.diff_pred.predict(feats)
        drop_pred = self.dropout_pred.predict_risk(feats)
        outcome   = diff_pred["predicted_outcome"]
        diff      = feats["difficulty_gap"]

        reasons = []
        if outcome == "Match":
            reasons.append(f"difficulty matches your level (match probability: {diff_pred['match_prob']:.0%})")
        elif outcome == "Excel":
            reasons.append("this course may be slightly easy — good for confidence building")
        elif outcome == "Struggle":
            reasons.append(f"⚠ this may be too hard — consider prerequisites first")

        if diff > 1:
            reasons.append("consider starting with Beginner content in this subject")
        elif diff < -1:
            reasons.append("you're ready to move up to a harder course")

        if drop_pred["risk_level"] == "HIGH":
            reasons.append(f"⚠ high dropout risk ({drop_pred['dropout_probability']:.0%}) — break into shorter sessions")

        quiz = profile.get("avg_quiz_score", 50)
        if quiz > 75:
            reasons.append(f"your quiz average ({quiz:.0f}%) suggests you're ready for this level")

        return " · ".join(reasons) if reasons else "recommended based on your learning profile"


if __name__ == "__main__":
    courses_df      = pd.read_csv("courses.csv")
    interactions_df = pd.read_csv("data/interactions.csv")
    users_df        = pd.read_csv("data/users.csv")

    # Build features
    features_df = build_features(interactions_df, users_df, courses_df)
    print(f"Feature matrix: {features_df.shape}")
    print("Outcome distribution:")
    print(features_df["outcome"].value_counts().to_string())

    # Train models
    print("\n=== Training DifficultyPredictor (MLP Neural Network) ===")
    diff_pred = DifficultyPredictor()
    diff_pred.fit(features_df)

    print("\n=== Training DropoutRiskPredictor (Logistic Regression) ===")
    drop_pred = DropoutRiskPredictor()
    drop_pred.fit(features_df)

    print("\n=== Fitting UserProfiler (KMeans) ===")
    profiler = UserProfiler()
    profiler.fit(interactions_df)

    # Save all models — store raw state dicts to avoid __main__ pickle issue
    def _extract_state(obj):
        return {k: v for k, v in obj.__dict__.items()}

    with open("data/adaptive_models.pkl", "wb") as f:
        pickle.dump({
            "diff_state":     _extract_state(diff_pred),
            "drop_state":     _extract_state(drop_pred),
            "profiler_state": _extract_state(profiler),
        }, f)
    print("\nModels saved → data/adaptive_models.pkl")

    # Demo: profile a user and rank courses
    engine     = AdaptiveEngine(diff_pred, drop_pred, profiler)
    sample_uid = users_df.iloc[10]["user_id"]
    profile    = profiler.get_profile(sample_uid, interactions_df, users_df,
                                       courses_df=courses_df)

    print(f"\n=== Profile for {sample_uid} ===")
    for k, v in profile.items():
        print(f"  {k:<30} {v}")

    print(f"\n=== Adaptive Course Ranking ===")
    candidates = courses_df[
        courses_df["subject"].str.contains("Machine Learning", case=False, na=False)
    ].head(20)
    ranked = engine.rank_courses(profile, candidates, top_n=5)
    for _, r in ranked.iterrows():
        explanation = engine.explain(profile, r)
        print(f"\n  [{r['adaptive_score']:.3f}] {r['title']}")
        print(f"    Outcome: {r['predicted_outcome']}  Dropout risk: {r['dropout_risk']}")
        print(f"    Why: {explanation}")
