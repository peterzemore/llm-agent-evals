"""Judge calibration.

An LLM judge is a measuring instrument, and an uncalibrated instrument produces
numbers that feel like evidence without being evidence. Before any judge score
is quoted, it gets compared against human labels on the same cases and the
agreement is published alongside it.

Cohen's kappa corrects for agreement that would happen by chance. On a suite
where 90% of cases pass, a judge that blindly answers "pass" scores 90% raw
agreement and kappa 0.0 - which is the whole point of reporting it.

Rough reading of kappa: <0.20 none, 0.21-0.40 fair, 0.41-0.60 moderate,
0.61-0.80 substantial, >0.80 near-perfect.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence


def agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b):
        raise ValueError("label sequences must be the same length")
    if not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters over the same items."""
    if len(a) != len(b):
        raise ValueError("label sequences must be the same length")
    n = len(a)
    if n == 0:
        return 0.0

    po = agreement(a, b)

    labels = set(a) | set(b)
    pe = sum((list(a).count(k) / n) * (list(b).count(k) / n) for k in labels)

    if pe == 1.0:
        # Both raters used a single label for everything; kappa is undefined.
        # Report perfect agreement as 1.0 and anything else as 0.0.
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def calibrate(
    judge_labels: dict[str, bool], human_labels: dict[str, bool]
) -> dict[str, Any]:
    """Compare judge verdicts against human labels on the overlapping cases."""
    shared = sorted(set(judge_labels) & set(human_labels))
    j = [judge_labels[k] for k in shared]
    h = [human_labels[k] for k in shared]

    false_pass = [k for k in shared if judge_labels[k] and not human_labels[k]]
    false_fail = [k for k in shared if not judge_labels[k] and human_labels[k]]

    return {
        "n_compared": len(shared),
        "n_judge_only": len(set(judge_labels) - set(human_labels)),
        "n_human_only": len(set(human_labels) - set(judge_labels)),
        "raw_agreement": agreement(j, h),
        "cohens_kappa": cohens_kappa(j, h),
        "false_pass": false_pass,
        "false_fail": false_fail,
        "human_pass_rate": (sum(h) / len(h)) if h else 0.0,
        "judge_pass_rate": (sum(j) / len(j)) if j else 0.0,
    }
