"""Core data types.

A `Case` is one labeled example of what an agent should do with a caller
utterance. A `Prediction` is what an agent actually did. A `CaseResult` is the
scored comparison of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    """One labeled evaluation case.

    `expected_tool` of None means the correct behavior is to answer from the
    system prompt without calling a tool at all. That distinction matters: a
    phone agent that calls an inventory lookup to answer "what time do you
    close" burns a round-trip of latency mid-conversation.
    """

    id: str
    utterance: str
    category: str
    expected_tool: str | None = None
    expected_args: dict[str, Any] = field(default_factory=dict)
    arg_rules: dict[str, str] = field(default_factory=dict)
    forbidden_phrases: list[str] = field(default_factory=list)
    rubric: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        missing = {"id", "utterance", "category"} - d.keys()
        if missing:
            raise ValueError(f"case is missing required field(s): {sorted(missing)}")
        return cls(
            id=d["id"],
            utterance=d["utterance"],
            category=d["category"],
            expected_tool=d.get("expected_tool"),
            expected_args=d.get("expected_args") or {},
            arg_rules=d.get("arg_rules") or {},
            forbidden_phrases=d.get("forbidden_phrases") or [],
            rubric=d.get("rubric", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class Prediction:
    """What the agent under test actually did for one case."""

    case_id: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    response_text: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        return cls(
            case_id=d["case_id"],
            tool=d.get("tool"),
            args=d.get("args") or {},
            response_text=d.get("response_text", ""),
            latency_ms=float(d.get("latency_ms", 0.0)),
            cost_usd=float(d.get("cost_usd", 0.0)),
            error=d.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tool": self.tool,
            "args": self.args,
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


@dataclass
class CaseResult:
    """A scored case."""

    case: Case
    prediction: Prediction
    tool_correct: bool
    args_correct: bool
    arg_failures: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    judge_pass: bool | None = None
    judge_rationale: str = ""

    @property
    def errored(self) -> bool:
        return self.prediction.error is not None

    @property
    def task_success(self) -> bool:
        """Everything a human would need to see to call the turn correct.

        The judge can only veto. A case with no rubric is never failed by the
        judge, and a judge that passes a case cannot rescue a wrong tool call.
        """
        return (
            not self.errored
            and self.tool_correct
            and self.args_correct
            and not self.forbidden_hits
            and self.judge_pass is not False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case.id,
            "category": self.case.category,
            "utterance": self.case.utterance,
            "expected_tool": self.case.expected_tool,
            "predicted_tool": self.prediction.tool,
            "tool_correct": self.tool_correct,
            "args_correct": self.args_correct,
            "arg_failures": self.arg_failures,
            "forbidden_hits": self.forbidden_hits,
            "judge_pass": self.judge_pass,
            "judge_rationale": self.judge_rationale,
            "task_success": self.task_success,
            "error": self.prediction.error,
            "latency_ms": self.prediction.latency_ms,
            "cost_usd": self.prediction.cost_usd,
        }
