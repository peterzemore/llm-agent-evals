"""Render a run as Markdown - readable in a terminal, in a PR, and on GitHub."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from .models import CaseResult

_HEADLINE = [
    ("Cases", "n", "{:d}"),
    ("Task success", "task_success_rate", "{:.1%}"),
    ("Tool accuracy", "tool_accuracy", "{:.1%}"),
    ("Arg accuracy", "arg_accuracy", "{:.1%}"),
    ("Guarded-phrase hits", "forbidden_rate", "{:.1%}"),
    ("Errors", "error_rate", "{:.1%}"),
    ("Latency p50", "latency_p50_ms", "{:.0f} ms"),
    ("Latency p95", "latency_p95_ms", "{:.0f} ms"),
    ("Cost / case", "cost_mean_usd", "${:.4f}"),
    ("Cost / run", "cost_total_usd", "${:.4f}"),
]


def _fmt(fmt: str, value: Any) -> str:
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def render(
    results: Sequence[CaseResult],
    metrics: dict[str, Any],
    adapter_name: str,
    violations: Sequence[str] = (),
    calibration: dict[str, Any] | None = None,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Agent eval run",
        "",
        f"- **Adapter:** `{adapter_name}`",
        f"- **Run at:** {stamp}",
        f"- **Gate:** {'FAILED' if violations else 'passed'}",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for label, key, fmt in _HEADLINE:
        lines.append(f"| {label} | {_fmt(fmt, metrics.get(key, 0))} |")

    if metrics.get("judged_n"):
        lines.append(f"| Judge pass rate | {metrics['judge_pass_rate']:.1%} "
                     f"(n={metrics['judged_n']}) |")

    if violations:
        lines += ["", "## Gate violations", ""]
        lines += [f"- {v}" for v in violations]

    lines += [
        "",
        "## By category",
        "",
        "| Category | n | Task success | Tool acc | Arg acc | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, m in metrics.get("by_category", {}).items():
        lines.append(
            f"| {category} | {m['n']} | {m['task_success_rate']:.0%} | "
            f"{m['tool_accuracy']:.0%} | {m['arg_accuracy']:.0%} | {m['error_rate']:.0%} |"
        )

    if calibration:
        lines += [
            "",
            "## Judge calibration",
            "",
            f"- Compared on **{calibration['n_compared']}** human-labeled cases",
            f"- Raw agreement: **{calibration['raw_agreement']:.1%}**",
            f"- Cohen's kappa: **{calibration['cohens_kappa']:.3f}**",
            f"- Judge passed a case a human failed: {len(calibration['false_pass'])}",
            f"- Judge failed a case a human passed: {len(calibration['false_fail'])}",
        ]

    failures = [r for r in results if not r.task_success]
    lines += ["", f"## Failures ({len(failures)})", ""]
    if not failures:
        lines.append("None.")
    for r in failures:
        lines.append(f"### `{r.case.id}` - {r.case.category}")
        lines.append("")
        lines.append(f"> {r.case.utterance}")
        lines.append("")
        if r.errored:
            lines.append(f"- **error:** {r.prediction.error}")
        if not r.tool_correct:
            lines.append(
                f"- **wrong tool:** expected `{r.case.expected_tool}`, "
                f"got `{r.prediction.tool}`"
            )
        for f in r.arg_failures:
            lines.append(f"- **bad argument:** {f}")
        for p in r.forbidden_hits:
            lines.append(f"- **guarded phrase present:** {p!r}")
        if r.judge_pass is False:
            lines.append(f"- **judge:** {r.judge_rationale}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
