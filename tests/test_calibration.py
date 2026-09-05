from agent_evals.calibration import calibrate, cohens_kappa


def test_perfect_agreement():
    assert cohens_kappa([True, False, True], [True, False, True]) == 1.0


def test_always_pass_judge_scores_zero_kappa():
    # The reason kappa is reported at all: a judge that rubber-stamps every case
    # gets 90% raw agreement on a 90%-pass suite and must score kappa 0.
    human = [True] * 9 + [False]
    judge = [True] * 10
    assert cohens_kappa(judge, human) == 0.0


def test_calibrate_separates_false_pass_from_false_fail():
    judge = {"a": True, "b": True, "c": False}
    human = {"a": True, "b": False, "c": True}
    report = calibrate(judge, human)
    assert report["n_compared"] == 3
    assert report["false_pass"] == ["b"]
    assert report["false_fail"] == ["c"]


def test_calibrate_reports_non_overlapping_labels():
    report = calibrate({"a": True}, {"a": True, "b": False})
    assert report["n_compared"] == 1
    assert report["n_human_only"] == 1
