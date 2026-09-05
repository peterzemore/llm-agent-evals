"""Deterministic argument matching.

Most tool-call failures are not subtle. The agent picks the wrong tool, drops a
required argument, or passes the caller's whole sentence as a search query
instead of the product name. All three are catchable without a model in the
loop, which keeps the bulk of the suite free, fast and non-flaky.

Match rules, declared per argument in a case's `arg_rules`:

    exact       str equality (default)
    iexact      case-insensitive equality, surrounding whitespace stripped
    icontains   expected value appears anywhere in the actual value
    numeric     both sides parse as numbers and compare equal
    email       case-insensitive equality after stripping (emails are not
                case-sensitive in the part that matters here)
    empty       actual must be empty/absent - used to assert that the agent did
                NOT guess a value it was never given
    regex:PAT   re.search(PAT, actual)
    any         argument may hold anything, including nothing
"""

from __future__ import annotations

import re
from typing import Any

from .models import Case


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def match_arg(rule: str, expected: Any, actual: Any) -> bool:
    """Apply a single match rule. Unknown rules raise rather than pass silently."""
    if rule == "any":
        return True

    if rule == "empty":
        return actual in (None, "", [], {})

    a = _as_text(actual)
    e = _as_text(expected)

    if rule == "exact":
        return a == e
    if rule == "iexact":
        return a.strip().lower() == e.strip().lower()
    if rule == "icontains":
        return e.strip().lower() in a.strip().lower()
    if rule == "email":
        return a.strip().lower() == e.strip().lower()
    if rule == "numeric":
        try:
            return float(a) == float(e)
        except (TypeError, ValueError):
            return False
    if rule.startswith("regex:"):
        return re.search(rule[len("regex:") :], a) is not None

    raise ValueError(f"unknown arg match rule: {rule!r}")


def match_tool(case: Case, predicted_tool: str | None) -> bool:
    return case.expected_tool == predicted_tool


def match_args(case: Case, actual: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check every declared expectation. Returns (ok, human-readable failures).

    Only arguments named in `expected_args` or `arg_rules` are checked. Extra
    arguments the agent supplies are ignored unless a rule names them, so that
    adding an optional parameter to a tool does not retroactively fail every
    stored case.
    """
    failures: list[str] = []
    checked = set(case.expected_args) | set(case.arg_rules)

    for name in sorted(checked):
        rule = case.arg_rules.get(name, "exact")
        expected = case.expected_args.get(name)
        got = actual.get(name)
        try:
            ok = match_arg(rule, expected, got)
        except ValueError as exc:
            failures.append(f"{name}: {exc}")
            continue
        if not ok:
            failures.append(f"{name}: expected {expected!r} ({rule}), got {got!r}")

    return (not failures), failures


def find_forbidden(case: Case, text: str) -> list[str]:
    """Substring guards against known hallucinations (invented policy, invented stock)."""
    lowered = (text or "").lower()
    return [p for p in case.forbidden_phrases if p.lower() in lowered]
