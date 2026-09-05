"""Run an adapter over a dataset and score the results."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, Sequence

from .matching import find_forbidden, match_args, match_tool
from .models import Case, CaseResult, Prediction


class Adapter(Protocol):
    """Anything that can answer a case. Real agent, replayed trace, or stub."""

    name: str

    def predict(self, case: Case) -> Prediction: ...


class Judge(Protocol):
    def verdict(self, case: Case, prediction: Prediction) -> tuple[bool | None, str]: ...


def score_one(case: Case, prediction: Prediction, judge: Judge | None = None) -> CaseResult:
    tool_correct = match_tool(case, prediction.tool)
    args_correct, arg_failures = match_args(case, prediction.args)
    forbidden = find_forbidden(case, prediction.response_text)

    judge_pass: bool | None = None
    rationale = ""
    # The judge is expensive and only adds signal where the correct answer is a
    # matter of wording rather than of which tool ran, so it is scoped to cases
    # that carry a rubric.
    if judge is not None and case.rubric and prediction.error is None:
        judge_pass, rationale = judge.verdict(case, prediction)

    return CaseResult(
        case=case,
        prediction=prediction,
        tool_correct=tool_correct,
        args_correct=args_correct,
        arg_failures=arg_failures,
        forbidden_hits=forbidden,
        judge_pass=judge_pass,
        judge_rationale=rationale,
    )


def run(
    cases: Sequence[Case],
    adapter: Adapter,
    judge: Judge | None = None,
    on_progress: Callable[[int, int, CaseResult], None] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        try:
            prediction = adapter.predict(case)
        except Exception as exc:  # an adapter blowing up is a data point, not a crash
            prediction = Prediction(case_id=case.id, error=f"{type(exc).__name__}: {exc}")
        result = score_one(case, prediction, judge)
        results.append(result)
        if on_progress:
            on_progress(i, len(cases), result)
    return results
