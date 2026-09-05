"""Aggregate metrics over scored cases.

Every rate here is reported with its denominator. A "94% tool accuracy" that
turns out to be 17 of 18 cases is a different claim than one over 300, and an
eval report that hides the denominator is not worth reading.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .models import CaseResult


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty input.

    Uses math.ceil rather than round(): Python's round() is banker's rounding,
    which sends an exact .5 rank to the nearest even integer and quietly shifts
    p50 by one position on even-sized samples.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p / 100.0 * len(ordered))))
    return float(ordered[rank - 1])


def _rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def _core(results: Sequence[CaseResult]) -> dict[str, Any]:
    n = len(results)
    judged = [r for r in results if r.judge_pass is not None]
    latencies = [r.prediction.latency_ms for r in results]
    costs = [r.prediction.cost_usd for r in results]

    return {
        "n": n,
        "tool_accuracy": _rate(sum(r.tool_correct for r in results), n),
        "arg_accuracy": _rate(sum(r.args_correct for r in results), n),
        "task_success_rate": _rate(sum(r.task_success for r in results), n),
        "forbidden_rate": _rate(sum(bool(r.forbidden_hits) for r in results), n),
        "error_rate": _rate(sum(r.errored for r in results), n),
        "judged_n": len(judged),
        "judge_pass_rate": _rate(sum(bool(r.judge_pass) for r in judged), len(judged)),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "cost_total_usd": round(sum(costs), 6),
        "cost_mean_usd": round(sum(costs) / n, 6) if n else 0.0,
    }


def aggregate(results: Iterable[CaseResult]) -> dict[str, Any]:
    results = list(results)
    summary = _core(results)

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({r.case.category for r in results}):
        subset = [r for r in results if r.case.category == category]
        by_category[category] = _core(subset)
    summary["by_category"] = by_category
    return summary
