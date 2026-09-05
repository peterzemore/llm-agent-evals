#!/usr/bin/env python3
"""Scrub a production tool-call log so it can become public eval cases.

The live agent's log holds real customer emails, phone numbers, order numbers
and names. None of that can go in a public dataset, and "I'll just eyeball it"
does not survive the two hundredth line. This does the mechanical pass; a human
still reads every line before it is committed.

Usage:
    python scripts/anonymize_traces.py calls.jsonl --out scrubbed.jsonl
    python scripts/anonymize_traces.py calls.jsonl --to-cases stubs.jsonl

`--to-cases` emits case stubs with `expected_tool`/`expected_args` prefilled
from what the agent actually did. That is a starting point, not a label: the
agent's own past behavior is not ground truth, so every stub needs a human to
confirm or correct it before it counts as a case.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUBSTITUTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"\+?1?[\s.\-(]*\d{3}[\s.\-)]*\d{3}[\s.\-]*\d{4}\b"), "<PHONE>"),
    (re.compile(r"#\s?\d{3,}"), "<ORDER>"),
    (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "<POSTAL>"),
    (re.compile(r"\b\d{6,}\b"), "<NUMBER>"),
]

# Fields whose values are free text a caller supplied, so they get scrubbed
# whole rather than pattern-matched.
SENSITIVE_ARG_NAMES = {"name", "contact", "email", "phone"}

# Structural fields that never hold caller data. Left alone so that the digit
# rules below do not chew up timestamps.
SKIP_FIELDS = {"logged_at", "tool", "type", "status"}

# Note on the number rules: they over-match on purpose. A product number like
# "#893" gets replaced along with a real order number, which costs a little
# fidelity in the review pass. That is the correct direction to be wrong in -
# a mangled product number wastes a minute, a leaked order number is a customer.


def scrub_text(text: str) -> str:
    for pattern, replacement in SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def scrub_value(key: str, value):
    if isinstance(value, str):
        if key in SENSITIVE_ARG_NAMES and value.strip():
            return f"<{key.upper()}>"
        return scrub_text(value)
    if isinstance(value, dict):
        return {k: scrub_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_value(key, v) for v in value]
    return value


def scrub_record(record: dict) -> dict:
    return {
        k: (v if k in SKIP_FIELDS else scrub_value(k, v))
        for k, v in record.items()
    }


def to_case_stub(record: dict, index: int) -> dict:
    tool = record.get("tool")
    return {
        "id": f"replay-{index:04d}",
        "category": "REVIEW",
        "utterance": "TODO: what the caller said, from the transcript",
        "expected_tool": tool,
        "expected_args": record.get("arguments", {}),
        "arg_rules": {k: "icontains" for k in record.get("arguments", {})},
        "notes": "UNREVIEWED stub generated from a production trace.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="production tool-call log (jsonl)")
    parser.add_argument("--out", help="write scrubbed traces here")
    parser.add_argument("--to-cases", help="write unreviewed case stubs here")
    args = parser.parse_args()

    if not args.out and not args.to_cases:
        parser.error("give --out, --to-cases, or both")

    records = []
    for line in Path(args.source).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            records.append(json.loads(line))

    scrubbed = [scrub_record(r) for r in records]

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in scrubbed) + "\n")
    if args.to_cases:
        stubs = [to_case_stub(r, i) for i, r in enumerate(scrubbed, start=1)]
        Path(args.to_cases).write_text("\n".join(json.dumps(s) for s in stubs) + "\n")

    print(f"processed {len(records)} records", file=sys.stderr)
    print("REVIEW EVERY LINE BY HAND before committing anything.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
