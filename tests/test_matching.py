import pytest

from agent_evals.matching import find_forbidden, match_arg, match_args
from agent_evals.models import Case


def _case(**kw):
    base = dict(id="c1", utterance="u", category="test")
    base.update(kw)
    return Case(**base)


@pytest.mark.parametrize(
    "rule,expected,actual,want",
    [
        ("exact", "batman", "batman", True),
        ("exact", "batman", "Batman", False),
        ("iexact", "batman", "  Batman ", True),
        ("icontains", "batman", "the batman pop", True),
        ("icontains", "batman", "superman", False),
        ("email", "a@b.com", "A@B.com", True),
        ("numeric", "1043", 1043, True),
        ("numeric", "1043", "ten forty three", False),
        ("empty", None, "", True),
        ("empty", None, "1043", False),
        ("any", None, "anything at all", True),
        ("regex:^\\d{4}$", None, "1043", True),
        ("regex:^\\d{4}$", None, "104", False),
    ],
)
def test_match_arg(rule, expected, actual, want):
    assert match_arg(rule, expected, actual) is want


def test_unknown_rule_raises_rather_than_passing():
    # A typo in a rule name must never silently grade everything as correct.
    with pytest.raises(ValueError):
        match_arg("contians", "x", "x")


def test_unchecked_extra_args_are_ignored():
    case = _case(expected_args={"query": "batman"}, arg_rules={"query": "icontains"})
    ok, failures = match_args(case, {"query": "batman", "limit": 5})
    assert ok and not failures


def test_arg_failure_message_names_the_argument():
    case = _case(expected_args={"email": "jane@example.com"}, arg_rules={"email": "email"})
    ok, failures = match_args(case, {"email": "jane at example dot com"})
    assert not ok
    assert failures[0].startswith("email:")


def test_forbidden_phrases_are_case_insensitive():
    case = _case(forbidden_phrases=["use code"])
    assert find_forbidden(case, "Sure, USE CODE SAVE10") == ["use code"]
    assert find_forbidden(case, "I can't issue discounts.") == []
