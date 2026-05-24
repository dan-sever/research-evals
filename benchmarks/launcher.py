"""Spawn detached eval runs from the UI.

Each launch becomes a separate `python run.py ...` subprocess. The subprocess
is fully detached (new session) so it survives Streamlit reloads and browser
disconnects. Stdout + stderr are captured to `logs/{timestamp}_{provider}_{model}.log`
so failures can be inspected post-hoc.

The launcher returns the subprocess PID + log path. The caller does not block.
Run progress is observed by polling the DB (rows landing in `results`).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs"


def _log_path(provider: str, model: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = model.replace("/", "_").replace(":", "_")
    return LOG_DIR / f"{ts}_{provider}_{safe}.log"


def prune_logs(keep_n: int = 100, min_age_seconds: int = 3600) -> int:
    """Delete oldest `*.log` files under LOG_DIR, keeping at most `keep_n`.

    Files modified within the last `min_age_seconds` are always kept, so an
    in-flight run can never have its log deleted out from under it. Safe to
    call at app startup. Returns the count of files removed."""
    if not LOG_DIR.exists():
        return 0
    now = time.time()
    candidates = []
    for p in LOG_DIR.glob("*.log"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if now - mtime < min_age_seconds:
            continue
        candidates.append((mtime, p))
    # Newest first; everything past keep_n is fair game.
    candidates.sort(key=lambda x: x[0], reverse=True)
    removed = 0
    for _mtime, path in candidates[keep_n:]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def launch_run(
    *,
    benchmark: str,
    provider: str,
    model: str,
    offset: int = 0,
    limit: Optional[int] = None,
    q_indices: Optional[list[int]] = None,
    seed: Optional[int] = None,
    workers: int = 4,
    note: str = "",
    judge_model: str = "claude-haiku-4-5",
    comparison_set: Optional[str] = None,
) -> tuple[int, Path]:
    """Spawn a detached `python run.py ...` and return (pid, log_path).

    Pass either (`offset`, `limit`) for a range or `q_indices` to cherry-pick
    specific rows. `q_indices` wins if both are set.
    """
    cmd: list[str] = [
        sys.executable, "run.py",
        "--provider", provider,
        "--benchmark", benchmark,
        "--model", model,
        "--workers", str(workers),
        "--judge-model", judge_model,
    ]
    if q_indices:
        cmd += ["--q-indices", ",".join(str(i) for i in q_indices)]
    else:
        cmd += ["--offset", str(offset)]
        if limit is not None:
            cmd += ["--limit", str(limit)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if note:
        cmd += ["--note", note]
    if comparison_set:
        cmd += ["--comparison-set", comparison_set]

    log_path = _log_path(provider, model)
    log_file = open(log_path, "a", buffering=1)
    log_file.write(f"$ {shlex.join(cmd)}\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach so we survive Streamlit reload
        close_fds=True,
    )
    return proc.pid, log_path
