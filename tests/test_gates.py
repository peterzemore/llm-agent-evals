from agent_evals.gates import evaluate


def test_floor_and_ceiling_violations():
    metrics = {"task_success_rate": 0.80, "forbidden_rate": 0.02}
    gates = {"min": {"task_success_rate": 0.88}, "max": {"forbidden_rate": 0.0}}
    violations = evaluate(metrics, gates)
    assert len(violations) == 2


def test_passing_run_has_no_violations():
    metrics = {"task_success_rate": 0.93, "forbidden_rate": 0.0}
    gates = {"min": {"task_success_rate": 0.88}, "max": {"forbidden_rate": 0.0}}
    assert evaluate(metrics, gates) == []


def test_missing_metric_is_a_violation_not_a_silent_pass():
    violations = evaluate({}, {"min": {"task_success_rate": 0.88}})
    assert violations and "not reported" in violations[0]
