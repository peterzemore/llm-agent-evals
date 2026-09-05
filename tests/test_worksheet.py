import pytest

from agent_evals.cli import _parse_worksheet

WORKSHEET = """# Grading worksheet

## order-003  (order)

Caller said: Hey, where's my order?

verdict: fail
note: called the lookup with two empty strings

---

## loyalty-002  (loyalty)

verdict: pass
note:

---

## static-001  (static_fact)

verdict: ???
note:
"""


def _write(tmp_path, text):
    path = tmp_path / "worksheet.md"
    path.write_text(text)
    return path


def test_parses_verdicts_and_notes(tmp_path):
    records, ungraded = _parse_worksheet(_write(tmp_path, WORKSHEET))
    assert records == [
        {
            "case_id": "order-003",
            "passed": False,
            "note": "called the lookup with two empty strings",
        },
        {"case_id": "loyalty-002", "passed": True},
    ]
    assert ungraded == 1


def test_unfilled_case_is_skipped_not_defaulted(tmp_path):
    # A case left as ??? must never silently become a pass - that would invent
    # ground truth the human never supplied.
    records, ungraded = _parse_worksheet(_write(tmp_path, WORKSHEET))
    assert "static-001" not in {r["case_id"] for r in records}
    assert ungraded == 1


@pytest.mark.parametrize("word,expected", [("p", True), ("yes", True), ("f", False), ("no", False)])
def test_accepts_common_shorthand(tmp_path, word, expected):
    text = f"## c1  (test)\n\nverdict: {word}\nnote:\n"
    records, _ = _parse_worksheet(_write(tmp_path, text))
    assert records == [{"case_id": "c1", "passed": expected}]


def test_unreadable_verdict_names_the_line(tmp_path):
    path = _write(tmp_path, "## c1  (test)\n\nverdict: maybe\nnote:\n")
    with pytest.raises(SystemExit) as exc:
        _parse_worksheet(path)
    assert "maybe" in str(exc.value)
    assert ":3:" in str(exc.value)
