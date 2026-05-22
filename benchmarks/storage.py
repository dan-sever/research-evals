"""SQLite storage for eval runs.

Schema:
  runs(id, provider, benchmark, model, limit_n, workers, seed, judge_model,
       note, comparison_set, started_at, finished_at, config_json)
  results(id, run_id, q_index, question, expected_answer,
          research_status, research_content, research_sources_json,
          research_request_id, research_duration_seconds,
          extracted_answer, is_correct, confidence, reasoning,
          error, created_at)

`comparison_set` groups runs launched together by compare.py so the dashboard
can render a side-by-side matrix across providers.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


DB_PATH = Path(__file__).resolve().parent.parent / "results.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL DEFAULT 'tavily',
    benchmark TEXT NOT NULL,
    model TEXT NOT NULL,
    limit_n INTEGER,
    workers INTEGER NOT NULL,
    seed INTEGER,
    judge_model TEXT NOT NULL,
    note TEXT,
    comparison_set TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    q_index INTEGER NOT NULL,
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    research_status TEXT,
    research_content TEXT,
    research_sources_json TEXT,
    research_request_id TEXT,
    research_duration_seconds REAL,
    extracted_answer TEXT,
    is_correct INTEGER,
    confidence REAL,
    reasoning TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, q_index)
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_correct ON results(run_id, is_correct);
"""
# Indexes on `runs.provider` / `runs.comparison_set` are created in _migrate
# AFTER any ALTER TABLE statements, so this also works on older DBs that
# pre-date those columns.


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that newer code expects, for DBs created before they existed."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "provider" not in cols:
        conn.execute(
            "ALTER TABLE runs ADD COLUMN provider TEXT NOT NULL DEFAULT 'tavily'"
        )
    if "comparison_set" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN comparison_set TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_compare ON runs(comparison_set)"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_provider ON runs(provider)"
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run(config: dict, db_path: Path = DB_PATH) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO runs
               (provider, benchmark, model, limit_n, workers, seed,
                judge_model, note, comparison_set, started_at, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                config.get("provider", "tavily"),
                config["benchmark"],
                config["model"],
                config.get("limit"),
                config["workers"],
                config.get("sample_seed"),
                config["judge_model"],
                config.get("note", ""),
                config.get("comparison_set"),
                now_iso(),
                json.dumps(config),
            ),
        )
        return cur.lastrowid


def finish_run(run_id: int, db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET finished_at = ? WHERE id = ?",
            (now_iso(), run_id),
        )


def insert_result(
    run_id: int,
    q_index: int,
    question: str,
    expected_answer: str,
    research_status: Optional[str] = None,
    research_content: Optional[str] = None,
    research_sources: Optional[list] = None,
    research_request_id: Optional[str] = None,
    research_duration_seconds: Optional[float] = None,
    extracted_answer: Optional[str] = None,
    is_correct: Optional[bool] = None,
    confidence: Optional[float] = None,
    reasoning: Optional[str] = None,
    error: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO results
               (run_id, q_index, question, expected_answer,
                research_status, research_content, research_sources_json,
                research_request_id, research_duration_seconds,
                extracted_answer, is_correct, confidence, reasoning,
                error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                q_index,
                question,
                expected_answer,
                research_status,
                research_content,
                json.dumps(research_sources) if research_sources is not None else None,
                research_request_id,
                research_duration_seconds,
                extracted_answer,
                None if is_correct is None else int(bool(is_correct)),
                confidence,
                reasoning,
                error,
                now_iso(),
            ),
        )


def list_runs(db_path: Path = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT r.*,
                      COUNT(res.id) AS total,
                      SUM(CASE WHEN res.is_correct = 1 THEN 1 ELSE 0 END) AS correct,
                      SUM(CASE WHEN res.error IS NOT NULL THEN 1 ELSE 0 END) AS errors
               FROM runs r
               LEFT JOIN results res ON res.run_id = r.id
               GROUP BY r.id
               ORDER BY r.id DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int, db_path: Path = DB_PATH) -> Optional[dict]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def get_results(run_id: int, db_path: Path = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY q_index",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_comparison_sets(db_path: Path = DB_PATH) -> list[dict]:
    """One row per comparison_set with summary info."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT comparison_set,
                      MIN(started_at)               AS started_at,
                      MIN(benchmark)                AS benchmark,
                      MIN(limit_n)                  AS limit_n,
                      MIN(seed)                     AS seed,
                      COUNT(*)                      AS n_runs,
                      GROUP_CONCAT(DISTINCT provider) AS providers
               FROM runs
               WHERE comparison_set IS NOT NULL
               GROUP BY comparison_set
               ORDER BY started_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_coverage(benchmark: str, db_path: Path = DB_PATH) -> list[dict]:
    """For a benchmark, summarize which q_index has been run by which
    (provider, model, seed) combination.

    Returned rows look like:
      {provider, model, seed, run_ids: list[int], q_indices: list[int]}
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT r.provider, r.model, r.seed,
                      GROUP_CONCAT(DISTINCT r.id) AS run_ids,
                      GROUP_CONCAT(res.q_index)   AS q_indices
               FROM runs r
               JOIN results res ON res.run_id = r.id
               WHERE r.benchmark = ?
               GROUP BY r.provider, r.model, r.seed
               ORDER BY r.provider, r.model, r.seed""",
            (benchmark,),
        ).fetchall()

    out = []
    for row in rows:
        run_ids = sorted({int(x) for x in (row["run_ids"] or "").split(",") if x})
        q_indices = sorted({int(x) for x in (row["q_indices"] or "").split(",") if x})
        out.append({
            "provider": row["provider"],
            "model": row["model"],
            "seed": row["seed"],
            "run_ids": run_ids,
            "q_indices": q_indices,
        })
    return out


def get_question_status(benchmark: str, db_path: Path = DB_PATH) -> list[dict]:
    """Per-question outcome for every (provider, model, seed) combination
    that has touched this benchmark.

    Each row: {run_id, provider, model, seed, q_index, is_correct,
               extracted_answer, research_duration_seconds, error}.
    Sorted by run_id ascending so callers can dedup by overwriting (latest
    run wins) when the same question was answered more than once.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT r.id AS run_id, r.provider, r.model, r.seed,
                      res.q_index, res.is_correct,
                      res.extracted_answer,
                      res.research_duration_seconds,
                      res.error
               FROM runs r
               JOIN results res ON res.run_id = r.id
               WHERE r.benchmark = ?
               ORDER BY r.id ASC""",
            (benchmark,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_in_progress_runs(db_path: Path = DB_PATH) -> list[dict]:
    """Runs that started but never marked finished. Includes current row count."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT r.*,
                      COUNT(res.id) AS rows_so_far,
                      SUM(CASE WHEN res.is_correct = 1 THEN 1 ELSE 0 END) AS correct_so_far,
                      SUM(CASE WHEN res.error IS NOT NULL THEN 1 ELSE 0 END) AS errors_so_far
               FROM runs r
               LEFT JOIN results res ON res.run_id = r.id
               WHERE r.finished_at IS NULL
               GROUP BY r.id
               ORDER BY r.started_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_runs_in_set(comparison_set: str, db_path: Path = DB_PATH) -> list[dict]:
    """All runs in a comparison set, with totals + accuracy + avg duration."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT r.*,
                      COUNT(res.id) AS total,
                      SUM(CASE WHEN res.is_correct = 1 THEN 1 ELSE 0 END) AS correct,
                      SUM(CASE WHEN res.error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                      AVG(res.research_duration_seconds) AS avg_seconds
               FROM runs r
               LEFT JOIN results res ON res.run_id = r.id
               WHERE r.comparison_set = ?
               GROUP BY r.id
               ORDER BY r.provider""",
            (comparison_set,),
        ).fetchall()
        return [dict(r) for r in rows]
