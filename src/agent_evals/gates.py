"""Regression gates: turn a metrics dict into a CI pass/fail."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_gates(path: str | Path) -> dict[str, dict[str, float]]:
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    return {
        "min": config.get("min", {}),
        "max": config.get("max", {}),
    }


def evaluate(metrics: dict[str, Any], gates: dict[str, dict[str, float]]) -> list[str]:
    """Return a list of violation strings. Empty means the run passes."""
    violations: list[str] = []

    for name, floor in gates.get("min", {}).items():
        if name not in metrics:
            violations.append(f"{name}: gate configured but metric not reported")
            continue
        if metrics[name] < floor:
            violations.append(f"{name} = {metrics[name]:.4g} is below the floor of {floor:.4g}")

    for name, ceiling in gates.get("max", {}).items():
        if name not in metrics:
            violations.append(f"{name}: gate configured but metric not reported")
            continue
        if metrics[name] > ceiling:
            violations.append(f"{name} = {metrics[name]:.4g} is above the ceiling of {ceiling:.4g}")

    return violations
