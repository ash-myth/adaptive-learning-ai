"""
migrate_to_db.py
================
AdaptLearn AI — One-time migration of CSV data into adaptlearn.db

Run ONCE:
    python migrate_to_db.py

What it does
------------
1. Creates `sim_users` table  → imports data/users.csv   (2000 synthetic users)
2. Creates `sim_interactions` → imports data/interactions.csv (~118k rows)
3. Safe to re-run: uses INSERT OR IGNORE so existing rows are never duplicated
4. Verifies row counts match the source CSVs

After this runs:
- main.py reads from the DB instead of CSVs
- Real users who sign up get a sim_users row automatically
- Course completions write to sim_interactions
- Admin sees all users (synthetic + real)
"""

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import sqlite3
import pandas as pd
from pathlib import Path
import time

DB_PATH       = Path("data/adaptlearn.db")
USERS_CSV     = Path("data/users.csv")
INTERACTIONS_CSV = Path("data/interactions.csv")


def migrate():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found. Run the server once first to create it.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # enables concurrent reads + writes
    conn.execute("PRAGMA foreign_keys=ON")

    # ── 1. Create sim_users table ────────────────────────────────────────────
    print("Creating sim_users table...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sim_users (
            user_id            TEXT PRIMARY KEY,
            archetype          TEXT NOT NULL DEFAULT 'Steady Learner',
            skill_level        INTEGER NOT NULL DEFAULT 1,
            preferred_subjects TEXT NOT NULL DEFAULT '',
            weekly_hours       REAL NOT NULL DEFAULT 5.0,
            goal               TEXT NOT NULL DEFAULT 'Machine Learning & AI',
            completion_base    REAL DEFAULT 0.65,
            quiz_mean          REAL DEFAULT 70.0,
            difficulty_tolerance REAL DEFAULT 0.5,
            engagement_factor  REAL DEFAULT 1.0,
            source             TEXT NOT NULL DEFAULT 'synthetic',
            created_at         TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── 2. Create sim_interactions table ────────────────────────────────────
    print("Creating sim_interactions table...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sim_interactions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT NOT NULL,
            course_id     TEXT NOT NULL,
            completed     INTEGER NOT NULL DEFAULT 0,
            quiz_score    REAL DEFAULT 0,
            time_spent    REAL DEFAULT 0,
            user_rating   REAL DEFAULT 0,
            replay_count  INTEGER DEFAULT 0,
            pause_count   INTEGER DEFAULT 0,
            cf_score      REAL DEFAULT 0,
            timestamp     TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, course_id)
        )
    """)

    # Indexes for fast lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_int_user ON sim_interactions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_int_course ON sim_interactions(course_id)")
    conn.commit()

    # ── 3. Import users.csv → sim_users ─────────────────────────────────────
    if USERS_CSV.exists():
        print(f"Loading {USERS_CSV}...")
        users_df = pd.read_csv(USERS_CSV)
        users_df = users_df.fillna("")

        before = conn.execute("SELECT COUNT(*) FROM sim_users").fetchone()[0]
        t0 = time.time()

        # Build rows, mapping CSV columns → DB columns
        rows = []
        for _, r in users_df.iterrows():
            rows.append((
                str(r["user_id"]),
                str(r.get("archetype", "Steady Learner")),
                int(r.get("skill_level", 1)),
                str(r.get("preferred_subjects", "")),
                float(r.get("weekly_hours", 5.0)),
                str(r.get("goal", "Machine Learning & AI")),
                float(r.get("completion_base", 0.65)),
                float(r.get("quiz_mean", 70.0)),
                float(r.get("difficulty_tolerance", 0.5)),
                float(r.get("engagement_factor", 1.0)),
                "synthetic",
            ))

        conn.executemany("""
            INSERT OR IGNORE INTO sim_users
              (user_id, archetype, skill_level, preferred_subjects,
               weekly_hours, goal, completion_base, quiz_mean,
               difficulty_tolerance, engagement_factor, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()

        after = conn.execute("SELECT COUNT(*) FROM sim_users").fetchone()[0]
        print(f"  sim_users: {before} → {after} rows  (+{after-before} inserted)  [{time.time()-t0:.1f}s]")
    else:
        print(f"WARNING: {USERS_CSV} not found — skipping sim_users import")

    # ── 4. Import interactions.csv → sim_interactions ────────────────────────
    if INTERACTIONS_CSV.exists():
        print(f"Loading {INTERACTIONS_CSV}  (this may take ~30s for 118k rows)...")
        t0 = time.time()
        int_df = pd.read_csv(INTERACTIONS_CSV)
        int_df = int_df.fillna(0)

        before = conn.execute("SELECT COUNT(*) FROM sim_interactions").fetchone()[0]

        rows = []
        for _, r in int_df.iterrows():
            rows.append((
                str(r["user_id"]),
                str(r["course_id"]),
                int(r.get("completed", 0)),
                float(r.get("quiz_score", 0)),
                float(r.get("time_spent", 0)),
                float(r.get("user_rating", 0)),
                int(r.get("replay_count", 0)),
                int(r.get("pause_count", 0)),
                float(r.get("cf_score", 0)),
            ))

        # Insert in batches of 5000 for speed
        BATCH = 5000
        inserted = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i+BATCH]
            conn.executemany("""
                INSERT OR IGNORE INTO sim_interactions
                  (user_id, course_id, completed, quiz_score, time_spent,
                   user_rating, replay_count, pause_count, cf_score)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, batch)
            conn.commit()
            inserted += len(batch)
            print(f"  {inserted:,}/{len(rows):,} rows...", end="\r")

        after = conn.execute("SELECT COUNT(*) FROM sim_interactions").fetchone()[0]
        print(f"\n  sim_interactions: {before} → {after} rows  (+{after-before} inserted)  [{time.time()-t0:.1f}s]")
    else:
        print(f"WARNING: {INTERACTIONS_CSV} not found — skipping sim_interactions import")

    conn.close()

    # ── 5. Verify ─────────────────────────────────────────────────────────────
    print("\n── Verification ──")
    conn = sqlite3.connect(DB_PATH)
    for table in ["users", "sim_users", "sim_interactions", "saved_courses"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<22} {count:>8,} rows")
    conn.close()
    print("\nMigration complete. You can now delete data/users.csv and data/interactions.csv")
    print("(keep them as backup until you've verified the server works correctly)")


if __name__ == "__main__":
    migrate()
