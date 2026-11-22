"""
user_simulator.py
=================
Adaptive AI Learning Intelligence Platform — User Behaviour Simulator
----------------------------------------------------------------------
Generates a realistic synthetic user-interaction dataset used by the
collaborative filtering and adaptive difficulty modules.

Why simulate users?
-------------------
We have no real users yet. This is standard practice in academic ML
projects — you declare it clearly in your report as "synthetic data
generation" and it is methodologically accepted.

The simulator is NOT random noise. Each user has:
  - A learner archetype  (Fast / Steady / Struggling / Curious)
  - A subject preference distribution
  - A difficulty comfort zone
  - A completion probability that depends on difficulty vs skill match
  - Quiz score distribution conditioned on difficulty match
  - Realistic dropout patterns (more drops on hard courses)

This makes the collaborative filter learn meaningful latent factors
rather than fitting random noise.

Output
------
data/users.csv          — 2000 user profiles
data/interactions.csv   — ~118,000 course interaction records
"""

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field


np.random.seed(42)
os.makedirs("data", exist_ok=True)

N_USERS         = 2000
INTERACTIONS_PER_USER_MEAN = 60
INTERACTIONS_PER_USER_STD  = 12


# ── Learner archetypes ────────────────────────────────────────────────────────

@dataclass
class LearnerArchetype:
    name: str
    completion_rate_base: float   # base probability of finishing a course
    quiz_score_mean:      float   # mean quiz performance 0-100
    quiz_score_std:       float
    difficulty_tolerance: float   # 0=avoids hard, 1=seeks hard
    engagement_factor:    float   # affects time_spent and replay rate
    description:          str     = ""

ARCHETYPES = [
    LearnerArchetype("Fast Learner",   0.80, 82, 10, 0.70, 1.20,
                     "High completion, good scores, tolerates hard content"),
    LearnerArchetype("Steady Learner", 0.65, 70, 12, 0.50, 1.00,
                     "Average completion, consistent scores"),
    LearnerArchetype("Struggling",     0.40, 52, 18, 0.25, 0.75,
                     "Low completion, struggles with intermediate+"),
    LearnerArchetype("Curious Sampler",0.30, 65, 15, 0.60, 0.90,
                     "Samples many courses, rarely finishes, decent scores"),
    LearnerArchetype("Specialist",     0.75, 78, 11, 0.65, 1.10,
                     "High completion in preferred subject, ignores others"),
]

ARCHETYPE_WEIGHTS = [0.20, 0.35, 0.25, 0.12, 0.08]

# Subject list (from our catalogue)
SUBJECTS = [
    "Machine Learning & AI",
    "Data Science & Analytics",
    "Programming & Software Dev",
    "Web Development",
    "Cloud & DevOps",
    "Cybersecurity",
    "Business & Finance",
    "Graphic Design & Media",
    "Health & Medicine",
    "Personal Development",
    "Science & Engineering",
    "Law & Social Sciences",
    "Music & Arts",
]

DIFFICULTY_LEVELS = {
    "Beginner":     1,
    "Mixed":        0,
    "Intermediate": 2,
    "Advanced":     3,
}


def generate_users(n: int = N_USERS) -> pd.DataFrame:
    """Generate synthetic user profiles."""
    users = []
    for uid in range(n):
        archetype = np.random.choice(ARCHETYPES, p=ARCHETYPE_WEIGHTS)

        # Each user has 1-3 preferred subjects drawn from SUBJECTS
        n_prefs = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        preferred_subjects = list(np.random.choice(SUBJECTS, size=n_prefs, replace=False))

        # Skill level: 1=beginner, 2=intermediate, 3=advanced
        skill_level = np.random.choice([1, 2, 3], p=[0.45, 0.40, 0.15])

        # Time available per week (hours)
        weekly_hours = round(np.random.lognormal(mean=2.0, sigma=0.6), 1)
        weekly_hours = np.clip(weekly_hours, 1.0, 40.0)

        users.append({
            "user_id":            f"U{uid:04d}",
            "archetype":          archetype.name,
            "skill_level":        skill_level,
            "preferred_subjects": "|".join(preferred_subjects),
            "weekly_hours":       weekly_hours,
            "goal":               np.random.choice(SUBJECTS),
            "completion_base":    round(archetype.completion_rate_base, 3),
            "quiz_mean":          round(archetype.quiz_score_mean, 1),
            "difficulty_tolerance": round(archetype.difficulty_tolerance, 3),
            "engagement_factor":  round(archetype.engagement_factor, 3),
        })

    return pd.DataFrame(users)


def compute_interaction_quality(
    user_row:   pd.Series,
    course_row: pd.Series,
    archetype:  LearnerArchetype,
) -> dict:
    """
    Compute realistic interaction metrics for a user-course pair.

    Models the key insight: engagement and performance are highest
    when course difficulty matches user skill level. Mismatch in
    either direction degrades the interaction.
    """
    user_skill   = user_row["skill_level"]
    course_diff  = DIFFICULTY_LEVELS.get(course_row["difficulty_label"], 0)
    diff_gap     = course_diff - user_skill   # positive = too hard, negative = too easy

    # ── Completion probability ────────────────────────────────────────────────
    # Peaks at diff_gap=0, falls off as gap increases
    difficulty_penalty = np.exp(-0.5 * diff_gap**2)  # Gaussian penalty
    is_preferred_subject = any(
        s in str(course_row["subject"])
        for s in user_row["preferred_subjects"].split("|")
    )
    subject_boost = 1.15 if is_preferred_subject else 0.85

    completion_prob = (
        archetype.completion_rate_base
        * difficulty_penalty
        * subject_boost
        * archetype.engagement_factor
    )
    completed = np.random.random() < np.clip(completion_prob, 0.05, 0.98)

    # ── Quiz score ────────────────────────────────────────────────────────────
    # Higher when difficulty matches skill, lower when too hard
    quiz_adjustment = -15 * diff_gap if diff_gap > 0 else 5 * abs(diff_gap)
    quiz_score = np.random.normal(
        archetype.quiz_score_mean + quiz_adjustment,
        archetype.quiz_score_std
    )
    quiz_score = float(np.clip(quiz_score, 0, 100))

    # ── Time spent (minutes) ─────────────────────────────────────────────────
    base_duration = float(course_row.get("duration_mins", 120))
    if completed:
        time_spent = base_duration * np.random.uniform(0.9, 1.2)
    else:
        # Dropped — only spent a fraction
        dropout_point = np.random.beta(2, 5)  # skewed toward early dropout
        time_spent    = base_duration * dropout_point

    time_spent = round(float(time_spent), 1)

    # ── Rating (1-5) ──────────────────────────────────────────────────────────
    # Users rate higher when course matches their level and they complete it
    if completed:
        base_rating = 3.5 + (quiz_score / 100) + (0.5 if is_preferred_subject else 0)
    else:
        base_rating = 2.0 + np.random.uniform(0, 1.5)
    rating = round(float(np.clip(np.random.normal(base_rating, 0.5), 1, 5)), 1)

    # ── Replay & pause counts ─────────────────────────────────────────────────
    # More replays when content is hard; more pauses when disengaged
    replay_count = int(np.random.poisson(max(0, diff_gap * 1.5 + 0.5)))
    pause_count  = int(np.random.poisson(
        3 if not completed else max(0, 1 - archetype.engagement_factor)
    ))

    # ── Implicit feedback score for collaborative filtering ───────────────────
    # This is the "rating" the CF model trains on
    # Explicit ratings have gaps so we build an implicit score from signals
    cf_score = (
        0.40 * (rating / 5)
        + 0.30 * float(completed)
        + 0.20 * (quiz_score / 100)
        + 0.10 * min(time_spent / max(base_duration, 1), 1.0)
    ) * 5  # scale to 0-5

    return {
        "completed":    int(completed),
        "quiz_score":   round(quiz_score, 1),
        "time_spent":   time_spent,
        "user_rating":  rating,
        "replay_count": replay_count,
        "pause_count":  pause_count,
        "cf_score":     round(float(cf_score), 3),
    }


def generate_interactions(
    users_df:   pd.DataFrame,
    courses_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate all user-course interaction records."""

    archetype_map = {a.name: a for a in ARCHETYPES}
    interactions  = []

    print(f"[Simulator] Generating interactions for {len(users_df)} users ...")

    for _, user in users_df.iterrows():
        archetype  = archetype_map[user["archetype"]]
        n_interact = max(5, int(np.random.normal(
            INTERACTIONS_PER_USER_MEAN, INTERACTIONS_PER_USER_STD
        )))

        # Subject sampling: 70% from preferred subjects, 30% exploration
        preferred_subjects = user["preferred_subjects"].split("|")
        subject_mask       = courses_df["subject"].isin(preferred_subjects)
        preferred_pool     = courses_df[subject_mask]
        explore_pool       = courses_df[~subject_mask]

        # Difficulty sampling: weighted toward user's comfort zone
        diff_weights = np.array([
            1.0,                               # Mixed
            2.0 if user["skill_level"] == 1 else 0.8,   # Beginner
            2.0 if user["skill_level"] == 2 else 1.0,   # Intermediate
            1.5 if user["skill_level"] == 3 else 0.4,   # Advanced
        ])
        diff_weights = diff_weights / diff_weights.sum()

        sampled_courses = []
        for _ in range(n_interact):
            from_preferred = np.random.random() < 0.70
            pool = preferred_pool if (from_preferred and len(preferred_pool) > 0) else explore_pool
            if len(pool) == 0:
                pool = courses_df
            course = pool.sample(1).iloc[0]
            sampled_courses.append(course)

        seen_courses = set()
        for course in sampled_courses:
            cid = course["course_id"]
            if cid in seen_courses:
                continue
            seen_courses.add(cid)

            metrics = compute_interaction_quality(user, course, archetype)
            interactions.append({
                "user_id":   user["user_id"],
                "course_id": cid,
                **metrics,
            })

    df = pd.DataFrame(interactions)
    print(f"[Simulator] Generated {len(df):,} interactions")
    print(f"[Simulator] Completion rate: {df['completed'].mean():.2%}")
    print(f"[Simulator] Mean quiz score: {df['quiz_score'].mean():.1f}")
    print(f"[Simulator] Mean CF score:   {df['cf_score'].mean():.3f}")
    return df


if __name__ == "__main__":
    courses_df = pd.read_csv("courses.csv")

    print("=== Generating synthetic users ===")
    users_df = generate_users(N_USERS)
    users_df.to_csv("data/users.csv", index=False)
    print(f"Saved {len(users_df)} users → data/users.csv")
    print(users_df["archetype"].value_counts().to_string())

    print("\n=== Generating interactions ===")
    interactions_df = generate_interactions(users_df, courses_df)
    interactions_df.to_csv("data/interactions.csv", index=False)
    print(f"Saved {len(interactions_df):,} interactions → data/interactions.csv")

    print("\n=== Sample interaction records ===")
    print(interactions_df.head(5).to_string())
