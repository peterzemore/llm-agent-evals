# Design notes

## The problem this exists for

A tool-calling agent in production fails quietly. It does not throw. It picks
`check_order_status` when the caller asked about store hours, or it passes the
caller's entire sentence into a product search, or it agrees with a caller who
insists an order shipped. Every one of those is a normal-looking response, and
none of them show up in an error rate.

Prompts are also edited casually, because editing a prompt does not feel like
editing code. A one-line tone change ships with no test, and the behavior that
regresses is three intents away from the line that changed.

This harness makes that class of change measurable and blockable.

## Scope decisions

**Single turn, not whole conversations.** The failures above are decided on the
first turn: which tool, which arguments, or whether to answer directly. Full
conversation simulation costs more, is far noisier to grade, and would move the
suite out of the "runs on every PR" budget. Multi-turn cases are worth adding
later for the callback flow specifically, where the agent has to collect a name
and a number across turns.

**Deterministic grading first, model grading only where it earns its place.**
Tool selection and argument correctness are exact-match questions. Running a
model to answer them would be slower, more expensive, and less reliable than
comparing two strings. The judge is scoped to cases whose rubric is about
wording - declining without being rude, asking for an email instead of guessing
one - which is roughly a third of the suite.

**The judge can only veto.** A case passes when the deterministic checks pass
and the judge has not failed it. A judge that likes a response cannot rescue a
wrong tool call. This keeps the headline number anchored to things that are
objectively checkable.

**Guarded phrases are separate from the judge.** Some failures are catastrophic
enough to be worth a hard substring check that costs nothing and cannot be
talked out of its verdict: a quoted buy price, an invented discount code, a
system-prompt fragment. These are `forbidden_phrases`, and the gate on them is
zero tolerance.

The guards are written against leakage markers, not against topic words. The
case that tests prompt-injection resistance does not forbid the word
"instructions", because "I can't share my instructions" is the correct answer.
It forbids fragments that only appear if the prompt actually leaked.

## Why calibration is in the critical path

An LLM judge is an instrument, and an instrument nobody has checked produces
numbers that feel like evidence without being evidence. The specific failure is
easy to reach: on a suite where 90% of cases pass, a judge that answers "pass"
to everything scores 90% raw agreement and looks excellent.

Cohen's kappa is reported for exactly that reason - it corrects for agreement
that would happen by chance, and the rubber-stamp judge scores 0.0. There is a
unit test asserting that specific case, because it is the property that makes
the metric worth printing.

The workflow: grade a sample of cases by hand, blind to the judge's verdicts,
then `agent-evals calibrate`. `false_pass` is the list that matters most - the
cases the judge waved through that a human failed. If kappa is below ~0.6, the
rubric is ambiguous and needs rewriting before the judge's numbers mean
anything.

## Why the replay adapter exists

Three reasons, in order of how much they matter:

1. **CI runs with no API key and no spend.** A gate that costs money per PR
   gets disabled the first time someone opens five PRs in an afternoon.
2. **Grading changes are separable from model changes.** With behavior frozen,
   any metric movement is attributable to the scoring logic alone. Without
   this, changing a matcher and re-running the live agent conflates two
   variables and you cannot tell which one moved the number.
3. **Failures stay reproducible.** A case that fails intermittently against a
   live model can be captured once and debugged against forever.

## Gate thresholds

Set just under the current baseline, never at an aspirational number. A gate
pinned at the baseline goes red on noise; a gate set where you wish the agent
performed goes red permanently and gets ignored within a week. `forbidden_rate`
is the exception at zero tolerance, because those cases are not quality
regressions - they are the agent saying something untrue about money.

Raise thresholds after a real improvement lands, as a separate commit, so the
new floor is a deliberate decision with its own diff.

## Known gaps

- **Cases are synthetic.** The seed set is written from the shape of real
  calls, not from real calls. `scripts/anonymize_traces.py` handles the
  mechanical scrub, but every case still needs a human to write the utterance
  and confirm the label. The stub generator prefills the label from what the
  agent actually did, which is a starting point and explicitly not ground
  truth - grading an agent against its own past behavior measures consistency,
  not correctness.
- **Sample size.** 28 cases supports a headline number to roughly the nearest
  four points. Any per-category number here is directional only; those cells
  hold three to six cases each. 150-300 cases is where the category breakdown
  starts carrying real weight.
- **No multi-turn coverage** (see above).
- **Latency is measured against the API, not the phone.** It excludes
  transcription and text-to-speech, which dominate perceived latency on a real
  call. The number is useful for comparing prompt variants against each other,
  not for predicting what a caller experiences.
