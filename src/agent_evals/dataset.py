"""Dataset loading. One JSON object per line; blank lines and # comments ignored."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Case, Prediction


def _iter_records(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON - {exc}") from exc


def load_cases(path: str | Path) -> list[Case]:
    cases = [Case.from_dict(r) for r in _iter_records(path)]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id: {c.id}")
        seen.add(c.id)
    if not cases:
        raise ValueError(f"no cases found in {path}")
    return cases


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    return {r["case_id"]: Prediction.from_dict(r) for r in _iter_records(path)}
