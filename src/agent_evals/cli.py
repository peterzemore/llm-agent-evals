"""Command line entry point.

    agent-evals run       score a dataset against an adapter, optionally gate
    agent-evals calibrate compare judge verdicts against human labels
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .adapters.replay import ReplayAdapter
from .dataset import load_cases
from .gates import evaluate, load_gates
from .metrics import aggregate
from .report import render
from .runner import run


def _load_dotenv(path: str | Path = ".env") -> None:
    """Fill unset environment variables from a local .env, if there is one.

    Deliberately dependency-free and deliberately non-overriding: an explicitly
    exported variable always wins over the file, so a checked-out .env cannot
    silently redirect a run at someone else's key.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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
        metrics["judge_error_n"] = len(judge.errors)

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


def _load_agent_turns(path: str | Path) -> dict[str, dict]:
    """Load what the agent did, from either a predictions jsonl or a results.json.

    Judge verdicts are deliberately dropped here. Grading is only worth doing
    blind - a human who has already read the judge's answer is checking the
    judge's work, not producing an independent label.
    """
    text = Path(path).read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and "results" in payload:
        return {
            r["case_id"]: {
                "tool": r.get("predicted_tool"),
                "args": r.get("predicted_args") or {},
                "response_text": r.get("response_text", ""),
            }
            for r in payload["results"]
        }

    from .dataset import load_predictions

    return {
        case_id: {"tool": p.tool, "args": p.args, "response_text": p.response_text}
        for case_id, p in load_predictions(path).items()
    }


WORKSHEET_HEADER = """# Grading worksheet

Replace each `verdict: ???` with `pass` or `fail`, and optionally fill in `note:`.
Leave `???` on any case you want to skip. Then run:

    agent-evals label --dataset {dataset} --from-worksheet {path} --out labels.jsonl

You are grading THE AGENT - the AI answering the phone. The customer is only the
input. Each block shows what the customer said, then what the agent did in
response; the rubric describes what the agent was required to do.

Grade against the rubric text only, not against how you would have phrased it.
If a case makes you hesitate, the rubric is ambiguous - say so in the note.
"""


def _write_worksheet(cases, turns, dataset: str, path: Path) -> int:
    blocks = [WORKSHEET_HEADER.format(dataset=dataset, path=path)]
    for case in cases:
        turn = turns.get(case.id, {})
        args = turn.get("args")
        blocks.append(
            f"""
## {case.id}  ({case.category})

CUSTOMER SAID: {case.utterance}

RUBRIC (what the agent was required to do): {case.rubric}

--- what the agent did in response ---
  tool called:   {turn.get('tool') or '(none - answered directly)'}
  arguments:     {args if args else '(none)'}
  agent replied: {turn.get('response_text') or '(nothing said)'}

verdict: ???
note:

---
"""
        )
    path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} case(s) to {path}.")
    print("Fill in the verdict lines, then re-run with --from-worksheet.")
    return 0


_PASS_WORDS = {"pass", "p", "yes", "y", "true", "ok"}
_FAIL_WORDS = {"fail", "f", "no", "n", "false"}


def _parse_worksheet(path: Path) -> tuple[list[dict], int]:
    """Return (records, ungraded_count). Raises on an unreadable verdict."""
    records: list[dict] = []
    ungraded = 0
    case_id: str | None = None
    verdict: bool | None = None
    note = ""

    def flush() -> None:
        nonlocal case_id, verdict, note, ungraded
        if case_id is None:
            return
        if verdict is None:
            ungraded += 1
        else:
            record: dict = {"case_id": case_id, "passed": verdict}
            if note:
                record["note"] = note
            records.append(record)
        case_id, verdict, note = None, None, ""

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line.startswith("## "):
            flush()
            case_id = line[3:].split()[0]
        elif line.lower().startswith("verdict:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in _PASS_WORDS:
                verdict = True
            elif value in _FAIL_WORDS:
                verdict = False
            elif value in ("", "???"):
                verdict = None
            else:
                raise SystemExit(
                    f"{path}:{lineno}: can't read verdict {value!r} - "
                    "use 'pass', 'fail', or leave '???' to skip"
                )
        elif line.lower().startswith("note:"):
            note = line.split(":", 1)[1].strip()
    flush()
    return records, ungraded


def _cmd_label(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    turns = _load_agent_turns(args.predictions)
    if not args.out and not args.worksheet:
        raise SystemExit("--out is required unless you are writing a --worksheet")
    out_path = Path(args.out) if args.out else Path("labels.jsonl")

    already: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                already.add(json.loads(line)["case_id"])

    if args.from_worksheet:
        records, ungraded = _parse_worksheet(Path(args.from_worksheet))
        new = [r for r in records if r["case_id"] not in already]
        with out_path.open("a", encoding="utf-8") as fh:
            for record in new:
                fh.write(json.dumps(record) + "\n")
        print(f"Read {len(records)} graded case(s); wrote {len(new)} new to {out_path}.")
        if ungraded:
            print(f"{ungraded} case(s) still marked ??? and were left ungraded.")
        if len(records) - len(new):
            print(f"{len(records) - len(new)} already had a label and were left alone.")
        return 0

    todo = [c for c in cases if c.rubric and c.id not in already]
    if not todo:
        print(f"Nothing left to grade - {len(already)} case(s) already labeled in {out_path}.")
        return 0

    if args.worksheet:
        return _write_worksheet(todo, turns, args.dataset, Path(args.worksheet))

    print(f"{len(todo)} case(s) to grade. Answers save as you go, so quitting keeps your work.")
    print("You are grading THE AGENT (the AI on the phone), not the customer.")
    print("Grade against the rubric only, not against what you would have said.\n")

    graded = 0
    for i, case in enumerate(todo, start=1):
        turn = turns.get(case.id, {})
        print("=" * 72)
        print(f"[{i}/{len(todo)}]  {case.id}  ({case.category})")
        print()
        print(f"  CUSTOMER SAID:  {case.utterance}")
        print()
        print(f"  RUBRIC (what the agent had to do):  {case.rubric}")
        print()
        print("  --- what the agent did in response ---")
        print(f"    tool called:   {turn.get('tool') or '(none - answered directly)'}")
        if turn.get("args"):
            print(f"    arguments:     {turn['args']}")
        print(f"    agent replied: {turn.get('response_text') or '(nothing said)'}")
        print()

        verdict: bool | None = None
        while verdict is None:
            try:
                answer = input(
                    "  Does this satisfy the rubric? [p]ass / [f]ail / [s]kip / [q]uit: "
                )
            except EOFError:
                # No usable stdin - running under a harness, a pipe, or an editor
                # shell. Point at the mode that works there instead of crashing.
                print("\n\n  No interactive input available here.")
                print("  Use worksheet mode instead:")
                print("    agent-evals label --dataset ... --predictions ... "
                      "--worksheet worksheet.md")
                print("  Fill in the verdict lines, then:")
                print("    agent-evals label --dataset ... --from-worksheet worksheet.md "
                      "--out labels.jsonl")
                return 2
            answer = answer.strip().lower()
            if answer in ("p", "pass"):
                verdict = True
            elif answer in ("f", "fail"):
                verdict = False
            elif answer in ("s", "skip"):
                break
            elif answer in ("q", "quit"):
                print(f"\nStopped. {graded} label(s) saved to {out_path}.")
                return 0
            else:
                print("  Please answer p, f, s, or q.")

        if verdict is None:
            print()
            continue

        record: dict[str, object] = {"case_id": case.id, "passed": verdict}
        try:
            note = input("  Why? (optional, enter to skip): ").strip()
        except EOFError:
            note = ""
        if note:
            record["note"] = note

        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        graded += 1
        print()

    print(f"Done. {graded} label(s) written to {out_path}.")
    print(f"Now run:  agent-evals calibrate --results <results.json> --labels {out_path}")
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

    lab_p = sub.add_parser("label", help="grade rubric cases by hand, blind to the judge")
    lab_p.add_argument("--dataset", required=True)
    lab_p.add_argument(
        "--predictions",
        required=True,
        help="predictions jsonl or a results.json - judge verdicts are not shown",
    )
    lab_p.add_argument("--out", help="jsonl to append labels to; resumable")
    lab_p.add_argument(
        "--worksheet",
        help="write a fill-in worksheet instead of prompting - for use without a terminal",
    )
    lab_p.add_argument("--from-worksheet", help="ingest a filled-in worksheet into --out")
    lab_p.set_defaults(func=_cmd_label)

    cal_p = sub.add_parser("calibrate", help="compare judge verdicts to human labels")
    cal_p.add_argument("--results", required=True, help="run payload written by --json-out")
    cal_p.add_argument("--labels", required=True, help="jsonl of {case_id, passed}")
    cal_p.add_argument("--min-compared", type=int, default=30)
    cal_p.set_defaults(func=_cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
