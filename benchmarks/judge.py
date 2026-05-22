"""Anthropic-backed judge.

Two jobs:
  1. Extract the short final answer from a long research report.
  2. Grade that extracted answer against the dataset's expected answer,
     accounting for format differences ("12 players" vs "12", "$1.2B" vs "1.2 billion").

Returns structured output via tool use so we can store it without regex.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic


JUDGE_SYSTEM = """You are grading an AI research agent's answer against a known correct answer.

You will receive:
- the original question
- the expected answer (ground truth, from a benchmark dataset, treat as authoritative)
- the agent's full research report (often long; may include citations and reasoning)

Do two things:
1. EXTRACT the agent's final answer to the question as a short string (a name, number, date, phrase). Strip citations, prose, and hedging. If the report has no answer or refuses, return "" for extracted_answer.
2. GRADE whether the extracted answer matches the expected answer. Match liberally on equivalent values across formats:
   - "12 players" matches "12"
   - "$1.2B" matches "1.2 billion" matches "1,200,000,000"
   - "Serban Ghenea" matches "Šerban Ghenea"
   - small unit/precision differences are fine if the underlying value matches
   The match must be on the underlying fact, not on string similarity. Wrong entity = incorrect even if names look similar.

Return only the structured tool call."""


GRADE_TOOL = {
    "name": "record_grade",
    "description": "Record extraction + correctness for one research answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "extracted_answer": {
                "type": "string",
                "description": "Short final answer pulled from the report. Empty string if none.",
            },
            "is_correct": {
                "type": "boolean",
                "description": "True if extracted matches expected on the underlying fact.",
            },
            "confidence": {
                "type": "number",
                "description": "0..1 confidence in the grade.",
            },
            "reasoning": {
                "type": "string",
                "description": "One short sentence explaining the call.",
            },
        },
        "required": ["extracted_answer", "is_correct", "confidence", "reasoning"],
    },
}


@dataclass
class JudgeResult:
    extracted_answer: str
    is_correct: bool
    confidence: float
    reasoning: str


class Judge:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY missing")
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def grade(
        self,
        question: str,
        expected_answer: str,
        research_content: str,
    ) -> JudgeResult:
        user = (
            f"<question>\n{question}\n</question>\n\n"
            f"<expected_answer>\n{expected_answer}\n</expected_answer>\n\n"
            f"<research_report>\n{research_content}\n</research_report>"
        )
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            tools=[GRADE_TOOL],
            tool_choice={"type": "tool", "name": "record_grade"},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "record_grade":
                inp = block.input
                return JudgeResult(
                    extracted_answer=str(inp.get("extracted_answer", "")),
                    is_correct=bool(inp.get("is_correct", False)),
                    confidence=float(inp.get("confidence", 0.0)),
                    reasoning=str(inp.get("reasoning", "")),
                )
        raise RuntimeError(f"Judge did not return tool call. Raw: {msg}")
