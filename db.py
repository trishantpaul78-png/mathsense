"""
SQLite persistence layer - plain sqlite3, no ORM.

Tables
------
students           : one row per learner
responses          : every answered question (the raw event log)
mastery_snapshots  : BKT mastery after every response (full history)

A fresh connection is opened per call. That is slightly slower than a shared
connection but is the only safe pattern with Streamlit's script re-runs.
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    question_id TEXT    NOT NULL,
    skill       TEXT    NOT NULL,
    correct     INTEGER NOT NULL,
    time_taken  REAL    NOT NULL DEFAULT 0,
    hints_used  INTEGER NOT NULL DEFAULT 0,
    timestamp   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_responses_student
    ON responses(student_id, id);

CREATE TABLE IF NOT EXISTS mastery_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    skill      TEXT    NOT NULL,
    mastery    REAL    NOT NULL,
    timestamp  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mastery_student
    ON mastery_snapshots(student_id, skill, id);
"""


def now_iso() -> str:
    """UTC timestamp, second precision - good enough for ordering."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they do not exist (idempotent)."""
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _execute(sql: str, params: tuple = ()) -> int:
    conn = _connect()
    try:
        cursor = conn.execute(sql, params)
        conn.commit()
        return int(cursor.lastrowid or 0)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------
def create_student(name: str) -> int:
    """Insert a student. Returns the existing id if the name is already taken."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Student name cannot be empty")
    _execute(
        "INSERT OR IGNORE INTO students (name, created_at) VALUES (?, ?)",
        (name, now_iso()),
    )
    rows = _query("SELECT id FROM students WHERE name = ?", (name,))
    return int(rows[0]["id"])


def get_or_create_student(name: str) -> int:
    """Convenience wrapper used by the UI."""
    return create_student(name)


def get_student(student_id: int) -> dict | None:
    rows = _query("SELECT * FROM students WHERE id = ?", (int(student_id),))
    return rows[0] if rows else None


def list_students() -> list[dict]:
    return _query("SELECT * FROM students ORDER BY name COLLATE NOCASE ASC")


def delete_students_like(pattern: str) -> int:
    """Delete matching students and their rows (used to reset the demo class)."""
    conn = _connect()
    try:
        ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM students WHERE name LIKE ?", (pattern,)
            ).fetchall()
        ]
        for sid in ids:
            conn.execute("DELETE FROM responses WHERE student_id = ?", (sid,))
            conn.execute("DELETE FROM mastery_snapshots WHERE student_id = ?", (sid,))
            conn.execute("DELETE FROM students WHERE id = ?", (sid,))
        conn.commit()
        return len(ids)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
def add_response(
    student_id: int,
    question_id: str,
    skill: str,
    correct: bool,
    time_taken: float = 0.0,
    hints_used: int = 0,
    timestamp: str | None = None,
) -> int:
    return _execute(
        """INSERT INTO responses
           (student_id, question_id, skill, correct, time_taken, hints_used, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            int(student_id),
            str(question_id),
            str(skill),
            1 if correct else 0,
            float(max(0.0, time_taken)),
            int(max(0, hints_used)),
            timestamp or now_iso(),
        ),
    )


def get_responses(student_id: int, limit: int | None = None) -> list[dict]:
    """Chronological response log. `correct` is normalised to bool."""
    if limit:
        sql = (
            "SELECT * FROM (SELECT * FROM responses WHERE student_id = ? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
        )
        rows = _query(sql, (int(student_id), int(limit)))
    else:
        rows = _query(
            "SELECT * FROM responses WHERE student_id = ? ORDER BY id ASC",
            (int(student_id),),
        )
    for row in rows:
        row["correct"] = bool(row["correct"])
    return rows


def count_responses(student_id: int) -> int:
    rows = _query(
        "SELECT COUNT(*) AS n FROM responses WHERE student_id = ?",
        (int(student_id),),
    )
    return int(rows[0]["n"]) if rows else 0


# ---------------------------------------------------------------------------
# Mastery snapshots
# ---------------------------------------------------------------------------
def save_mastery_snapshot(
    student_id: int, skill: str, mastery: float, timestamp: str | None = None
) -> None:
    _execute(
        """INSERT INTO mastery_snapshots (student_id, skill, mastery, timestamp)
           VALUES (?, ?, ?, ?)""",
        (int(student_id), str(skill), float(mastery), timestamp or now_iso()),
    )


def get_mastery_rows(student_id: int) -> list[dict]:
    """All snapshots for a student, oldest first."""
    return _query(
        "SELECT * FROM mastery_snapshots WHERE student_id = ? ORDER BY id ASC",
        (int(student_id),),
    )


def get_mastery_history(student_id: int, skill: str) -> list[dict]:
    return _query(
        """SELECT skill, mastery, timestamp FROM mastery_snapshots
           WHERE student_id = ? AND skill = ? ORDER BY id ASC""",
        (int(student_id), str(skill)),
    )


def get_all_mastery() -> dict[tuple[str, str], float]:
    """Latest mastery per (student name, skill) for the whole class."""
    rows = _query(
        """SELECT s.name AS name, m.skill AS skill, m.mastery AS mastery
           FROM mastery_snapshots m
           JOIN students s ON s.id = m.student_id
           ORDER BY m.id ASC"""
    )
    latest: dict[tuple[str, str], float] = {}
    for row in rows:
        latest[(row["name"], row["skill"])] = float(row["mastery"])
    return latest


# ---------------------------------------------------------------------------
# Convenience for the dashboard
# ---------------------------------------------------------------------------
def class_frame() -> pd.DataFrame:
    """Wide students x skills frame of latest mastery values."""
    rows = _query(
        """SELECT s.name AS name, m.skill AS skill, m.mastery AS mastery
           FROM mastery_snapshots m
           JOIN students s ON s.id = m.student_id
           ORDER BY m.id ASC"""
    )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates(subset=["name", "skill"], keep="last")
    return frame.pivot(index="name", columns="skill", values="mastery")


init_db()
