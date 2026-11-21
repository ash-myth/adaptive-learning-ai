"""
auth.py
=======
AdaptLearn AI — Authentication & User Management
-------------------------------------------------
Self-contained auth module using stdlib only (no extra pip installs).

  - SQLite database  → data/adaptlearn.db
  - PBKDF2-SHA256 password hashing (stdlib hashlib)
  - HMAC-SHA256 signed JWT tokens (stdlib hmac)
  - Onboarding quiz data stored per user
  - Profile read/write for real registered users

Tables
------
  users
    id            TEXT PK   (uuid4)
    email         TEXT UNIQUE
    display_name  TEXT
    password_hash TEXT
    salt          TEXT
    skill_level   INTEGER   (1=beginner, 2=intermediate, 3=advanced)
    weekly_hours  REAL
    goal          TEXT
    preferred_subjects TEXT  (pipe-separated)
    archetype     TEXT
    created_at    TEXT
    avatar_color  TEXT      (hex, random on signup)
    is_onboarded  INTEGER   (0/1)

Usage
-----
    from auth import router as auth_router
    app.include_router(auth_router, prefix="/api/auth")
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator

# ── Config ─────────────────────────────────────────────────────────────────
DB_PATH      = Path(os.environ.get("AUTH_DB_PATH", "data/adaptlearn.db"))
JWT_SECRET   = os.environ.get("JWT_SECRET", "adaptlearn-jwt-secret-change-in-prod-2025")
TOKEN_TTL    = int(os.environ.get("TOKEN_TTL_HOURS", "168")) * 3600  # 7 days default
PBKDF2_ITERS = 260_000   # NIST 2023 recommendation

AVATAR_COLORS = [
    "#6366f1","#8b5cf6","#ec4899","#f43f5e","#f97316",
    "#eab308","#22c55e","#06b6d4","#0ea5e9","#a855f7",
]

SUBJECTS = [
    "Machine Learning & AI", "Data Science & Analytics",
    "Programming & Software Dev", "Web Development", "Cloud & DevOps",
    "Cybersecurity", "Business & Finance", "Graphic Design & Media",
    "Health & Medicine", "Personal Development",
    "Science & Engineering", "Law & Social Sciences", "Music & Arts",
]

router   = APIRouter()
security = HTTPBearer(auto_error=False)


# ════════════════════════════════════════════════════════════════════════════
# Database
# ════════════════════════════════════════════════════════════════════════════

def _init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id               TEXT PRIMARY KEY,
                email            TEXT UNIQUE NOT NULL,
                display_name     TEXT NOT NULL,
                password_hash    TEXT NOT NULL,
                salt             TEXT NOT NULL,
                skill_level      INTEGER DEFAULT 1,
                weekly_hours     REAL    DEFAULT 5.0,
                goal             TEXT    DEFAULT 'Machine Learning & AI',
                preferred_subjects TEXT  DEFAULT '',
                archetype        TEXT    DEFAULT 'Steady Learner',
                avatar_color     TEXT    DEFAULT '#6366f1',
                is_onboarded     INTEGER DEFAULT 0,
                is_admin         INTEGER DEFAULT 0,
                created_at       TEXT    NOT NULL
            )
        """)
        # Add is_admin if upgrading from older schema (safe no-op if exists)
        try:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        except Exception:
            pass
        # Add created_at to saved_courses if upgrading from older schema
        try:
            conn.execute("ALTER TABLE saved_courses ADD COLUMN created_at TEXT DEFAULT (datetime('now'))")
        except Exception:
            pass
        # Create sim_users table (mirrors users.csv schema for ML model)
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
                source             TEXT NOT NULL DEFAULT 'real',
                created_at         TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                course_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, course_id)
            )
        """)
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_int_user ON sim_interactions(user_id)")
        conn.commit()

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

_init_db()


# ════════════════════════════════════════════════════════════════════════════
# Password hashing
# ════════════════════════════════════════════════════════════════════════════

def _hash_password(password: str, salt: str) -> str:
    key = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"),
        salt.encode("utf-8"), PBKDF2_ITERS
    )
    return base64.b64encode(key).decode()

def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    return hmac.compare_digest(
        _hash_password(password, salt), stored_hash
    )


# ════════════════════════════════════════════════════════════════════════════
# JWT (HMAC-SHA256, stdlib only)
# ════════════════════════════════════════════════════════════════════════════

def _create_token(user_id: str, email: str) -> str:
    payload = {
        "sub":   user_id,
        "email": email,
        "iat":   int(time.time()),
        "exp":   int(time.time()) + TOKEN_TTL,
    }
    header  = base64.urlsafe_b64encode(
        json.dumps({"alg":"HS256","typ":"JWT"}).encode()
    ).rstrip(b"=").decode()
    body    = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    signing = f"{header}.{body}"
    sig     = base64.urlsafe_b64encode(
        hmac.new(JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return f"{header}.{body}.{sig}"

def _decode_token(token: str) -> dict:
    try:
        header, body, sig = token.split(".")
        signing = f"{header}.{body}"
        expected = base64.urlsafe_b64encode(
            hmac.new(JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("Invalid signature")
        # Pad base64
        payload = json.loads(
            base64.urlsafe_b64decode(body + "==").decode()
        )
        if payload["exp"] < time.time():
            raise ValueError("Token expired")
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """FastAPI dependency — extracts user from Bearer token."""
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = _decode_token(creds.credentials)
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (payload["sub"],)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(row)


# ════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ════════════════════════════════════════════════════════════════════════════

class SignupRequest(BaseModel):
    email:        str
    display_name: str
    password:     str

    @field_validator("password")
    @classmethod
    def pwd_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("display_name")
    @classmethod
    def name_len(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters")
        return v

    @field_validator("email")
    @classmethod
    def email_fmt(cls, v):
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address")
        return v

class LoginRequest(BaseModel):
    email:    str
    password: str

class OnboardingRequest(BaseModel):
    skill_level:        int      # 1/2/3
    weekly_hours:       float
    goal:               str
    preferred_subjects: list[str]
    archetype:          str      # derived from quiz

class UpdateProfileRequest(BaseModel):
    display_name:       Optional[str] = None
    goal:               Optional[str] = None
    skill_level:        Optional[int] = None
    weekly_hours:       Optional[float] = None
    preferred_subjects: Optional[list[str]] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str

    @field_validator("new_password")
    @classmethod
    def pwd_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


# ════════════════════════════════════════════════════════════════════════════
# Helper: user → public dict
# ════════════════════════════════════════════════════════════════════════════

def _public(row: dict) -> dict:
    subs = row.get("preferred_subjects","") or ""
    return {
        "id":                row["id"],
        "email":             row["email"],
        "display_name":      row["display_name"],
        "skill_level":       row["skill_level"],
        "weekly_hours":      row["weekly_hours"],
        "goal":              row["goal"],
        "preferred_subjects":subs.split("|") if subs else [],
        "archetype":         row["archetype"],
        "avatar_color":      row["avatar_color"],
        "is_onboarded":      bool(row["is_onboarded"]),
        "is_admin":          bool(row.get("is_admin", 0)),
        "created_at":        row["created_at"],
        "initials":          "".join(
            w[0].upper() for w in row["display_name"].split()[:2]
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════════════

@router.post("/signup")
def signup(req: SignupRequest):
    """Create a new account."""
    with _db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (req.email,)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        uid   = str(uuid.uuid4())
        salt  = secrets.token_hex(32)
        phash = _hash_password(req.password, salt)
        color = secrets.choice(AVATAR_COLORS)
        ts    = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        conn.execute("""
            INSERT INTO users
              (id, email, display_name, password_hash, salt,
               avatar_color, created_at, is_onboarded)
            VALUES (?,?,?,?,?,?,?,0)
        """, (uid, req.email, req.display_name, phash, salt, color, ts))

        # Mirror into sim_users so this real user participates in ML recommendations
        conn.execute("""
            INSERT OR IGNORE INTO sim_users
              (user_id, archetype, skill_level, preferred_subjects,
               weekly_hours, goal, source)
            VALUES (?,?,1,'',5.0,'Machine Learning & AI','real')
        """, (uid,  "Steady Learner"))

    token = _create_token(uid, req.email)
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return {
        "token":       token,
        "user":        _public(dict(row)),
        "next":        "/onboarding.html",
    }


@router.post("/login")
def login(req: LoginRequest):
    """Authenticate and return a token."""
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (req.email.strip().lower(),)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    row = dict(row)
    if not _verify_password(req.password, row["salt"], row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(row["id"], row["email"])
    dest  = "/onboarding.html" if not row["is_onboarded"] else ("/index.html" if row.get("is_admin") else "/dashboard.html")
    return {
        "token": token,
        "user":  _public(row),
        "next":  dest,
    }


@router.post("/onboarding")
def save_onboarding(
    req:  OnboardingRequest,
    user: dict = Depends(get_current_user),
):
    """Save onboarding quiz results — called once after signup."""
    subs = "|".join(req.preferred_subjects[:3])
    with _db() as conn:
        conn.execute("""
            UPDATE users SET
              skill_level=?, weekly_hours=?, goal=?,
              preferred_subjects=?, archetype=?, is_onboarded=1
            WHERE id=?
        """, (
            req.skill_level, req.weekly_hours, req.goal,
            subs, req.archetype, user["id"]
        ))
        # Keep sim_users in sync so ML models see the real profile
        conn.execute("""
            UPDATE sim_users SET
              skill_level=?, weekly_hours=?, goal=?,
              preferred_subjects=?, archetype=?
            WHERE user_id=?
        """, (
            req.skill_level, req.weekly_hours, req.goal,
            subs, req.archetype, user["id"]
        ))
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    return {"ok": True, "user": _public(dict(row))}


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    """Return the authenticated user's public profile."""
    return _public(user)


@router.patch("/me")
def update_profile(
    req:  UpdateProfileRequest,
    user: dict = Depends(get_current_user),
):
    """Update mutable profile fields."""
    updates = {}
    if req.display_name is not None:
        updates["display_name"] = req.display_name.strip()
    if req.goal is not None:
        updates["goal"] = req.goal
    if req.skill_level is not None:
        updates["skill_level"] = req.skill_level
    if req.weekly_hours is not None:
        updates["weekly_hours"] = req.weekly_hours
    if req.preferred_subjects is not None:
        updates["preferred_subjects"] = "|".join(req.preferred_subjects[:3])

    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update")

    set_clause = ", ".join(f"{k}=?" for k in updates)
    values     = list(updates.values()) + [user["id"]]
    with _db() as conn:
        conn.execute(f"UPDATE users SET {set_clause} WHERE id=?", values)
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    return _public(dict(row))


@router.post("/change-password")
def change_password(
    req:  ChangePasswordRequest,
    user: dict = Depends(get_current_user),
):
    """Change password — requires current password verification."""
    if not _verify_password(req.current_password, user["salt"], user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    new_salt  = secrets.token_hex(32)
    new_hash  = _hash_password(req.new_password, new_salt)
    with _db() as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, salt=? WHERE id=?",
            (new_hash, new_salt, user["id"])
        )
    return {"ok": True, "message": "Password updated successfully"}


@router.post("/logout")
def logout(user: dict = Depends(get_current_user)):
    """
    Stateless JWT logout — token invalidation is handled client-side.
    This endpoint exists so the frontend has a clean call to make.
    """
    return {"ok": True}


@router.delete("/me")
def delete_account(user: dict = Depends(get_current_user)):
    """Permanently delete the account."""
    with _db() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (user["id"],))
    return {"ok": True, "message": "Account deleted"}


# ════════════════════════════════════════════════════════════════════════════
# Dev helper — list all users (remove in production)
# ════════════════════════════════════════════════════════════════════════════

@router.get("/dev/users")
def dev_list_users():
    """DEV ONLY — list registered users. Remove before deploying publicly."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, email, display_name, is_onboarded, created_at FROM users"
        ).fetchall()
    return {"users": [dict(r) for r in rows]}

class SaveCourseRequest(BaseModel):
    course_id: str
@router.post("/saved-courses")
def save_course(
    req: SaveCourseRequest,
    user: dict = Depends(get_current_user)
):
    with _db() as conn:
        try:
            conn.execute(
                "INSERT INTO saved_courses (user_id, course_id) VALUES (?, ?)",
                (user["id"], req.course_id)
            )
        except:
            pass

    return {"ok": True}
@router.get("/saved-courses")
def get_saved_courses(user: dict = Depends(get_current_user)):
    import pandas as pd

    with _db() as conn:
        rows = conn.execute(
            "SELECT course_id FROM saved_courses WHERE user_id=? ORDER BY rowid DESC",
            (user["id"],)
        ).fetchall()

    course_ids = [r["course_id"] for r in rows]

    if not course_ids:
        return {"courses": []}

    courses_df = pd.read_csv("data/courses.csv")

    saved = courses_df[courses_df["course_id"].isin(course_ids)]

    result = []
    for _, r in saved.iterrows():
        result.append({
            "course_id": r["course_id"],
            "title": r["title"],
            "difficulty": r.get("difficulty_label", "Mixed"),
            "source": r.get("source", ""),
            "source_label": r.get("source", "").title(),
            "rating": float(r.get("rating", 0)),
            "link": r.get("link", "")
        })

    return {"courses": result}
@router.delete("/saved-courses/{course_id}")
def remove_saved_course(
    course_id: str,
    user: dict = Depends(get_current_user)
):
    with _db() as conn:
        conn.execute(
            "DELETE FROM saved_courses WHERE user_id=? AND course_id=?",
            (user["id"], course_id)
        )

    return {"ok": True}