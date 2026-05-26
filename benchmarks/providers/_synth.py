"""Shared answer synthesizer for providers whose search endpoint returns
hits/excerpts but no synthesized answer (currently EXA and Parallel).

We need a gradeable string to pass to the judge, so we ask Claude Haiku
to write a short answer using only the provided excerpts. Reads
ANTHROPIC_API_KEY from the env — same key the runner already requires
for the judge, so callers don't need to plumb it through.
"""

from __future__ import annotations

import os

from anthropic import Anthropic

_SYSTEM = (
    "You are answering a question using only the provided search excerpts. "
    "Write a short, direct answer in 1-3 sentences. Cite supporting excerpts "
    "with bracket markers like [1], [2] referring to the numbered list. "
    "If the excerpts do not contain the answer, say so plainly."
)

_EXCERPT_CHAR_CAP = 1500


def synthesize_answer(
    question: str,
    sources: list[dict],
    *,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 600,
) -> str:
    """Synthesize a gradeable answer from `[{title, url, content}, ...]`.

    Raises if `ANTHROPIC_API_KEY` is missing; callers should pre-flight that
    same as they pre-flight the judge.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing — required for search-tier "
            "answer synthesis."
        )
    client = Anthropic(api_key=key)

    if sources:
        blocks: list[str] = []
        for i, s in enumerate(sources, 1):
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            content = (s.get("content") or "").strip()
            if len(content) > _EXCERPT_CHAR_CAP:
                content = content[:_EXCERPT_CHAR_CAP] + "…"
            blocks.append(f"[{i}] {title}\n{url}\n{content}")
        excerpt_block = "\n\n".join(blocks)
    else:
        excerpt_block = "(no search results returned)"

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"<question>\n{question}\n</question>\n\n"
                f"<search_results>\n{excerpt_block}\n</search_results>"
            ),
        }],
    )
    parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()
