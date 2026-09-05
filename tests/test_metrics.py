from agent_evals.metrics import aggregate, percentile
from agent_evals.models import Case, Prediction
from agent_evals.runner import score_one


def _scored(case_id, category, expected_tool, predicted_tool, **kw):
    case = Case(id=case_id, utterance="u", category=category, expected_tool=expected_tool, **kw)
    prediction = Prediction(case_id=case_id, tool=predicted_tool, latency_ms=100, cost_usd=0.001)
    return score_one(case, prediction)


def test_percentile_handles_empty_and_single():
    assert percentile([], 95) == 0.0
    assert percentile([42.0], 95) == 42.0


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10


def test_aggregate_splits_by_category():
    results = [
        _scored("a", "stock", "check_stock", "check_stock"),
        _scored("b", "stock", "check_stock", None),
        _scored("c", "order", "check_order_status", "check_order_status"),
    ]
    m = aggregate(results)
    assert m["n"] == 3
    assert m["tool_accuracy"] == 2 / 3
    assert m["by_category"]["stock"]["task_success_rate"] == 0.5
    assert m["by_category"]["order"]["task_success_rate"] == 1.0


def test_error_counts_as_failure_not_as_a_pass():
    case = Case(id="e", utterance="u", category="stock", expected_tool=None)
    prediction = Prediction(case_id="e", error="APIConnectionError")
    result = score_one(case, prediction)
    # The tool "matches" (both None) but an errored call is never a success.
    assert result.tool_correct is True
    assert result.task_success is False
    assert aggregate([result])["error_rate"] == 1.0
