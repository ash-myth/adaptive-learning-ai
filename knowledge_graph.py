"""
knowledge_graph.py
==================
Adaptive AI Learning Intelligence Platform — Prerequisite Knowledge Graph

Fix #11 — consolidate duplicate prerequisite maps
---------------------------------------------------
The codebase had two separate prerequisite maps that had diverged:

  recommender.py        → PREREQUISITE_MAP  (12 subject-level entries)
  hybrid_recommender.py → PREREQ_MAP        (40+ query-level entries)

They were maintained independently, used different key granularities, and
were drifting apart silently.  This module is the single source of truth
for all prerequisite and synonym knowledge in the platform.

Both recommender.py and hybrid_recommender.py should import from here:

    from knowledge_graph import PREREQ_MAP, SUBJECT_PREREQ_MAP, get_prerequisites

Structure
---------
PREREQ_MAP          : dict[str, list[str]]
    Query-level map.  Keys are terse search strings (e.g. "machine learning").
    Values are ordered lists of prerequisite query strings — feed each into
    BM25 to find the best Beginner course for that prerequisite.
    Used by: HybridRecommender.skill_gap()

SUBJECT_PREREQ_MAP  : dict[str, list[str]]
    Subject-level map.  Keys are broad subject category names as they appear
    in courses.csv["subject"].  Values are lists of prerequisite subjects.
    Used by: AdaptiveLearningRecommender.detect_skill_gap()

get_prerequisites(goal, strategy) : list[str]
    Unified lookup function.  Tries PREREQ_MAP first (query-level, richer),
    falls back to SUBJECT_PREREQ_MAP (broader), then returns an empty list
    if the goal is genuinely unknown.
"""

from __future__ import annotations


# ════════════════════════════════════════════════════════════════════════════
# Query-level prerequisite map  (used by HybridRecommender.skill_gap)
# ════════════════════════════════════════════════════════════════════════════
# Keys  : lowercase query strings.  Substring matching — "dsa" matches
#         "intro to dsa" in the goal.  Longer/more-specific keys take
#         priority in get_prerequisites() (longest-match wins).
# Values: ordered list of prerequisite query strings.  BM25 searches each
#         to find the best Beginner course in the catalogue.

PREREQ_MAP: dict[str, list[str]] = {

    # ── CS Fundamentals ────────────────────────────────────────────────────
    "data structures and algorithms": [
        "introduction to programming", "python programming", "discrete mathematics",
    ],
    "data structures": [
        "introduction to programming", "python basics", "java programming basics",
    ],
    "algorithms": [
        "programming fundamentals", "mathematics for computing", "discrete mathematics",
    ],
    "dsa": [
        "introduction to programming", "python basics", "mathematics",
    ],
    "operating systems": [
        "c programming", "computer architecture", "programming basics",
    ],
    "computer networks": [
        "networking fundamentals", "linux basics", "programming",
    ],
    "computer architecture": [
        "digital logic", "electronics", "programming",
    ],
    "compilers": [
        "data structures", "programming language", "mathematics",
    ],
    "system design": [
        "databases", "networking", "backend programming",
    ],
    "software engineering": [
        "programming fundamentals", "version control", "databases",
    ],

    # ── ML / AI ───────────────────────────────────────────────────────────
    "machine learning": [
        "python programming", "statistics", "linear algebra",
    ],
    "deep learning": [
        "machine learning basics", "python", "linear algebra calculus",
    ],
    "neural networks": [
        "machine learning", "python", "mathematics statistics",
    ],
    "natural language processing": [
        "machine learning", "python", "text analytics",
    ],
    "nlp": [
        "machine learning", "python", "text analytics",
    ],
    "computer vision": [
        "deep learning", "python", "image processing",
    ],
    "reinforcement learning": [
        "machine learning", "python", "probability statistics",
    ],
    "data science": [
        "python programming", "statistics", "sql database",
    ],
    "data analysis": [
        "statistics", "excel", "sql",
    ],
    "data engineering": [
        "python", "sql database", "cloud computing",
    ],
    "mlops": [
        "machine learning", "docker", "cloud deployment",
    ],
    "llm": [
        "machine learning", "python", "natural language processing",
    ],
    "generative ai": [
        "machine learning", "python", "deep learning",
    ],
    "prompt engineering": [
        "python programming", "natural language processing", "api development",
    ],

    # ── Web / App Development ─────────────────────────────────────────────
    "web development": [
        "html css basics", "javascript fundamentals",
    ],
    "frontend development": [
        "html css", "javascript fundamentals",
    ],
    "backend development": [
        "programming fundamentals", "database sql", "api development",
    ],
    "full stack": [
        "html css javascript", "backend programming", "database",
    ],
    "react": [
        "javascript fundamentals", "html css", "web basics",
    ],
    "angular": [
        "javascript typescript", "html css", "web basics",
    ],
    "vue": [
        "javascript fundamentals", "html css",
    ],
    "node": [
        "javascript programming", "backend development",
    ],
    "django": [
        "python programming", "web basics", "database sql",
    ],
    "flask": [
        "python programming", "web basics",
    ],
    "rest api": [
        "programming", "http web", "database",
    ],
    "graphql": [
        "api development", "javascript", "backend",
    ],
    "android": [
        "java programming", "mobile development", "object oriented",
    ],
    "ios": [
        "swift programming", "mobile development", "xcode",
    ],
    "react native": [
        "javascript fundamentals", "react basics", "mobile development",
    ],
    "flutter": [
        "dart programming", "mobile development", "object oriented",
    ],

    # ── Cloud / DevOps ────────────────────────────────────────────────────
    "cloud computing": [
        "networking fundamentals", "linux", "programming basics",
    ],
    "aws": [
        "cloud fundamentals", "linux", "networking",
    ],
    "azure": [
        "cloud fundamentals", "windows server", "networking",
    ],
    "google cloud": [
        "cloud fundamentals", "linux", "networking",
    ],
    "devops": [
        "linux administration", "programming", "networking",
    ],
    "docker": [
        "linux basics", "command line", "programming",
    ],
    "kubernetes": [
        "docker containers", "linux", "networking",
    ],
    "terraform": [
        "cloud computing", "linux", "programming",
    ],
    "ci/cd": [
        "git version control", "programming", "linux",
    ],

    # ── Databases ─────────────────────────────────────────────────────────
    "database": [
        "sql fundamentals", "data modelling", "programming",
    ],
    "sql": [
        "database basics", "data concepts", "excel",
    ],
    "mongodb": [
        "database basics", "javascript", "programming",
    ],
    "postgresql": [
        "sql basics", "database", "programming",
    ],
    "redis": [
        "database basics", "programming", "networking",
    ],

    # ── Security ──────────────────────────────────────────────────────────
    "cybersecurity": [
        "networking fundamentals", "linux", "programming basics",
    ],
    "ethical hacking": [
        "networking", "linux", "cybersecurity basics",
    ],
    "penetration testing": [
        "networking", "linux", "python scripting",
    ],
    "cryptography": [
        "mathematics", "programming", "networking",
    ],

    # ── Mathematics ───────────────────────────────────────────────────────
    "linear algebra": [
        "mathematics basics", "calculus", "trigonometry",
    ],
    "calculus": [
        "algebra", "trigonometry", "mathematics",
    ],
    "statistics": [
        "mathematics basics", "probability", "algebra",
    ],
    "probability": [
        "mathematics basics", "algebra", "statistics",
    ],
    "discrete mathematics": [
        "algebra", "logic", "mathematics",
    ],

    # ── Design ────────────────────────────────────────────────────────────
    "graphic design": [
        "design fundamentals", "color theory", "typography",
    ],
    "ux design": [
        "design thinking", "user research", "wireframing",
    ],
    "ui design": [
        "ux design", "graphic design", "prototyping",
    ],
    "product design": [
        "design thinking", "ux research", "prototyping",
    ],

    # ── Business ──────────────────────────────────────────────────────────
    "finance": [
        "accounting basics", "economics", "mathematics",
    ],
    "investment": [
        "finance fundamentals", "economics", "mathematics",
    ],
    "marketing": [
        "business fundamentals", "communication", "analytics",
    ],
    "project management": [
        "communication", "business fundamentals", "planning",
    ],
    "product management": [
        "business fundamentals", "ux design", "analytics",
    ],
    "entrepreneurship": [
        "business fundamentals", "marketing", "finance",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# Subject-level prerequisite map  (used by AdaptiveLearningRecommender)
# ════════════════════════════════════════════════════════════════════════════
# Keys  : broad subject category names matching courses.csv["subject"].
# Values: list of prerequisite subject categories.

SUBJECT_PREREQ_MAP: dict[str, list[str]] = {
    "machine learning & ai":      [
        "Data Science & Analytics",
        "Programming & Software Dev",
        "Science & Engineering",
    ],
    "data science & analytics":   [
        "Programming & Software Dev",
        "Science & Engineering",
        "Business & Finance",
    ],
    "cloud & devops":             [
        "Programming & Software Dev",
        "Science & Engineering",
    ],
    "cybersecurity":              [
        "Cloud & DevOps",
        "Programming & Software Dev",
    ],
    "web development":            [
        "Programming & Software Dev",
        "Graphic Design & Media",
    ],
    "science & engineering":      [
        "Science & Engineering",
        "Health & Medicine",
    ],
    "business & finance":         [
        "Business & Finance",
        "Personal Development",
    ],
    "graphic design & media":     [
        "Graphic Design & Media",
        "Personal Development",
    ],
    "health & medicine":          [
        "Science & Engineering",
        "Personal Development",
    ],
    "law & social sciences":      [
        "Business & Finance",
        "Personal Development",
    ],
    "personal development":       ["Personal Development"],
    "music & arts":               ["Music & Arts"],
    "programming & software dev": [
        "Programming & Software Dev",
        "Science & Engineering",
    ],
    "mobile development":         [
        "Programming & Software Dev",
        "Web Development",
    ],
    "database":                   [
        "Programming & Software Dev",
        "Data Science & Analytics",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# Unified lookup function
# ════════════════════════════════════════════════════════════════════════════

def get_prerequisites(
    goal: str,
    strategy: str = "query",
) -> list[str]:
    """
    Return prerequisite strings for a learning goal.

    Parameters
    ----------
    goal     : raw goal string (e.g. "machine learning", "AWS DevOps")
    strategy : "query"   → search PREREQ_MAP (richer, BM25-ready terms)
               "subject" → search SUBJECT_PREREQ_MAP (broader categories)
               "auto"    → try query first, fall back to subject

    Returns
    -------
    List of prerequisite strings in learning order (most foundational first).
    Empty list if the goal is not recognised in the requested map.

    Algorithm
    ---------
    Longest-key-wins: when multiple PREREQ_MAP keys are substrings of the
    goal, the most specific (longest) matching key is used.  This prevents
    "machine learning" from matching just "learning" when "machine learning"
    is available.
    """
    goal_lower = goal.lower().strip()

    if strategy in ("query", "auto"):
        # Longest-key-wins matching
        matched_key   = None
        matched_score = 0
        for key in PREREQ_MAP:
            if key in goal_lower or goal_lower in key:
                score = len(key)
                if score > matched_score:
                    matched_key   = key
                    matched_score = score
        if matched_key:
            return PREREQ_MAP[matched_key]

    if strategy in ("subject", "auto"):
        for key in SUBJECT_PREREQ_MAP:
            if key in goal_lower or goal_lower in key:
                return SUBJECT_PREREQ_MAP[key]

    return []


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        ("machine learning", "query"),
        ("intro to dsa", "query"),
        ("aws devops", "query"),
        ("unknown topic xyz", "auto"),
        ("web development", "subject"),
    ]
    print("knowledge_graph.py smoke test")
    print("=" * 50)
    for goal, strategy in test_cases:
        prereqs = get_prerequisites(goal, strategy=strategy)
        print(f"  {goal!r:35s} ({strategy}) → {prereqs}")
    print(f"\nPREQ_MAP entries:         {len(PREREQ_MAP)}")
    print(f"SUBJECT_PREREQ_MAP entries: {len(SUBJECT_PREREQ_MAP)}")
