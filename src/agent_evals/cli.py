"""Command line entry point.

    agent-evals run       score a dataset against an adapter, optionally gate
    agent-evals calibrate compare judge verdicts against human labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .adapters.replay import ReplayAdapter
from .dataset import load_cases
from .gates import evaluate, load_gates
from .metrics import aggregate
from .report import render
from .runner import run


def _build_adapter(args: argparse.Namespace):
    if args.adapter == "replay":
        if not args.replay_file:
            raise SystemExit("--replay-file is required for the replay adapter")
        return ReplayAdapter(args.replay_file)

    if args.adapter == "anthropic":
        from .adapters.anthropic_agent import AnthropicAgentAdapter

        if not args.tools:
            raise SystemExit("--tools is required for the anthropic adapter")
        system_prompt = Path(args.system).read_text() if args.system else ""
        return AnthropicAgentAdapter(
            tools_path=args.tools,
            system_prompt=system_prompt,
            model=args.model,
            effort=args.effort,
        )

    raise SystemExit(f"unknown adapter: {args.adapter}")


def _cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    adapter = _build_adapter(args)

    judge = None
    if args.judge:
        from .judges.llm_judge import ClaudeJudge

        judge = ClaudeJudge(model=args.judge_model)

    def progress(i: int, total: int, result) -> None:
        if not args.quiet:
            mark = "." if result.task_success else "F"
            print(mark, end="", flush=True)
            if i == total:
                print()

    results = run(cases, adapter, judge, on_progress=progress)
    metrics = aggregate(results)
    if judge is not None:
        metrics["judge_cost_usd"] = round(judge.cost_usd, 6)

    violations: list[str] = []
    if args.gates:
        violations = evaluate(metrics, load_gates(args.gates))

    payload = {
        "version": __version__,
        "adapter": adapter.name,
        "dataset": str(args.dataset),
        "metrics": metrics,
        "violations": violations,
        "results": [r.to_dict() for r in results],
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
    markdown = render(results, metrics, adapter.name, violations)
    if args.report:
        Path(args.report).write_text(markdown)
    if not args.quiet:
        print(markdown)

    if violations:
        print("GATE FAILED:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import calibrate

    results = json.loads(Path(args.results).read_text())
    judge_labels = {
        r["case_id"]: bool(r["judge_pass"])
        for r in results["results"]
        if r.get("judge_pass") is not None
    }
    human_labels: dict[str, bool] = {}
    for line in Path(args.labels).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        record = json.loads(line)
        human_labels[record["case_id"]] = bool(record["passed"])

    if not judge_labels:
        raise SystemExit(
            "no judge verdicts in the results file - re-run with --judge before calibrating"
        )

    report = calibrate(judge_labels, human_labels)
    print(json.dumps(report, indent=2))

    if report["n_compared"] < args.min_compared:
        print(
            f"\nOnly {report['n_compared']} overlapping labels; "
            f"kappa is not meaningful below {args.min_compared}.",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-evals", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="score a dataset against an adapter")
    run_p.add_argument("--dataset", required=True)
    run_p.add_argument("--adapter", default="replay", choices=["replay", "anthropic"])
    run_p.add_argument("--replay-file", help="recorded predictions for the replay adapter")
    run_p.add_argument("--tools", help="JSON file of tool schemas for the live adapter")
    run_p.add_argument("--system", help="file holding the agent's system prompt")
    run_p.add_argument("--model", default="claude-opus-5")
    run_p.add_argument(
        "--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"]
    )
    run_p.add_argument("--judge", action="store_true", help="run the LLM judge on rubric cases")
    run_p.add_argument("--judge-model", default="claude-sonnet-5")
    run_p.add_argument("--gates", help="gates.toml; exit non-zero on violation")
    run_p.add_argument("--report", help="write the Markdown report here")
    run_p.add_argument("--json-out", help="write the full run payload here")
    run_p.add_argument("--quiet", action="store_true")
    run_p.set_defaults(func=_cmd_run)

    cal_p = sub.add_parser("calibrate", help="compare judge verdicts to human labels")
    cal_p.add_argument("--results", required=True, help="run payload written by --json-out")
    cal_p.add_argument("--labels", required=True, help="jsonl of {case_id, passed}")
    cal_p.add_argument("--min-compared", type=int, default=30)
    cal_p.set_defaults(func=_cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
