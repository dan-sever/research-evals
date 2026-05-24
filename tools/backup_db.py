"""Snapshot results.db to backups/results.db.bak.{timestamp}.

Run before risky migrations or schema changes. The `backups/` directory
is gitignored (see .gitignore matching `results.db.bak.*`) so snapshots
stay local.

Usage:
    python tools/backup_db.py
    python tools/backup_db.py --label preinsights   # writes results.db.bak.preinsights_{ts}
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "results.db"
BACKUP_DIR = REPO_ROOT / "backups"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot results.db to backups/.")
    parser.add_argument(
        "--label",
        default="",
        help="Optional short label embedded in the filename, e.g. 'preinsights'.",
    )
    args = parser.parse_args(argv)

    if not DB_PATH.exists():
        print(f"No database at {DB_PATH}. Nothing to back up.", file=sys.stderr)
        return 1

    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.label}_{ts}" if args.label else ts
    dest = BACKUP_DIR / f"results.db.bak.{suffix}"
    if dest.exists():
        print(f"Refusing to overwrite existing snapshot {dest}", file=sys.stderr)
        return 2
    shutil.copy2(DB_PATH, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Wrote {dest} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
