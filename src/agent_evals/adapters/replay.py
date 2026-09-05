"""Replay a recorded set of predictions instead of calling a model.

This is what makes the suite runnable in CI with no API key and no spend, and
it is also how you re-score an old run after changing the grading logic: the
model's behavior is frozen, so any metric movement is attributable to the
scoring change alone.
"""

from __future__ import annotations

from pathlib import Path

from ..dataset import load_predictions
from ..models import Case, Prediction


class ReplayAdapter:
    name = "replay"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.predictions = load_predictions(path)

    def predict(self, case: Case) -> Prediction:
        prediction = self.predictions.get(case.id)
        if prediction is None:
            return Prediction(
                case_id=case.id,
                error=f"no recorded prediction for case {case.id} in {self.path.name}",
            )
        return prediction
