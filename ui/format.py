"""Small formatters shared across tabs."""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

# CJK ideographs, Hiragana/Katakana, Hangul, fullwidth forms. Any hit
# means the string is almost certainly not English. Benchmarks here are
# either entirely English or entirely a CJK language per row, so even
# a stray glyph is a strong signal.
NON_ENGLISH_RE = re.compile(
    "["
    "　-鿿"   # CJK Symbols, Hiragana, Katakana, CJK Unified Ideographs
    "가-힯"   # Hangul Syllables
    "＀-￯"   # Halfwidth/fullwidth forms
    "]"
)


def looks_non_english(text: str) -> bool:
    return bool(NON_ENGLISH_RE.search(text or ""))


def elapsed_human(started_at: str | None) -> str:
    """Human-readable elapsed since an ISO timestamp from SQLite."""
    if not started_at:
        return "—"
    try:
        t0 = datetime.fromisoformat(started_at)
        t1 = datetime.now(t0.tzinfo) if t0.tzinfo else datetime.now()
        secs = (t1 - t0).total_seconds()
        if secs < 0:
            return "—"
        if secs < 60:
            return f"{secs:.0f}s"
        if secs < 3600:
            return f"{secs / 60:.1f}m"
        return f"{secs / 3600:.1f}h"
    except Exception:
        return "—"


def fmt_duration(seconds) -> str:
    """`45.3s` under a minute, `128.0s (2.1 min)` above."""
    if seconds is None or pd.isna(seconds) or seconds == 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds:.1f}s ({seconds / 60:.1f} min)"


def fmt_duration_short(seconds) -> str:
    """Compact form for dense table cells: `12.3s` or `2.1m`."""
    if seconds is None or pd.isna(seconds) or seconds == 0:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def quote(text: str | None) -> str:
    """Render `text` as a markdown blockquote, preserving line breaks."""
    if not text:
        return "> —"
    return "\n".join(f"> {ln}" for ln in str(text).splitlines()) or "> —"


def ranges(sorted_ints: list[int]) -> str:
    """`[0,1,2,4,7,8,9]` -> `0-2, 4, 7-9`."""
    if not sorted_ints:
        return "—"
    parts: list[str] = []
    start = prev = sorted_ints[0]
    for n in sorted_ints[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)
