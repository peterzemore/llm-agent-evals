# agent-evals

Offline evaluation and CI regression gating for tool-calling LLM agents.

A tool-calling agent in production fails quietly. It doesn't throw an exception
when it reaches for an order-lookup tool to answer a question about store
hours, or passes a caller's entire sentence into a product search, or agrees
with a caller who insists their order shipped. Those are normal-looking
responses. They do not appear in an error rate, and nobody notices until a
customer does.

This harness scores an agent against a labeled set of cases, reports metrics
with their denominators, and **blocks a merge when a prompt change moves the
numbers the wrong way**.

Built against a real deployment: a Vapi-backed phone agent answering calls for
a retail store, with four production tools (`check_stock`, `check_order_status`,
`check_loyalty_points`, `log_callback_request`).

## Results on the seed suite

```
agent-evals run --dataset datasets/sonny/cases.jsonl \
                --adapter replay \
                --replay-file datasets/sonny/fixtures/replay_baseline.jsonl \
                --gates gates.toml
```

| Metric | Value |
| --- | --- |
| Cases | 38 |
| Task success | 92.1% |
| Tool accuracy | 97.4% |
| Argument accuracy | 94.7% |
| Guarded-phrase hits | 0.0% |
| Latency p50 / p95 | 588 ms / 845 ms |
| Cost per case | $0.0035 |

| Category | n | Task success | Tool acc | Arg acc |
| --- | ---: | ---: | ---: | ---: |
| stock | 7 | 86% | 100% | 86% |
| order | 5 | 60% | 80% | 80% |
| loyalty | 3 | 100% | 100% | 100% |
| callback | 5 | 100% | 100% | 100% |
| grounding | 7 | 100% | 100% | 100% |
| static_fact | 4 | 100% | 100% | 100% |
| out_of_scope | 3 | 100% | 100% | 100% |
| adversarial | 4 | 100% | 100% | 100% |

38 cases supports the headline to roughly the nearest three points; the
per-category cells hold three to six cases each and are directional only. The
honest read of this table is the `order` row, not the headline.

## What the failures were

The three failing cases are the useful part of the output:

**`order-003` — "Hey, where's my order?"** The agent called the order-lookup
tool with two empty strings instead of asking which order. On a phone call that
becomes "I couldn't find an order matching that email" in response to a caller
who was never asked for an email.

**`order-005` — "I ordered last week under jane at example dot com."** Passed
through verbatim. Speech-to-text does not normalize a spoken email; the agent
has to, and it didn't.

**`stock-005` — "My son's birthday is Saturday and he really likes Spiderman…"**
The whole sentence went into the product search as the query string. The
case that catches this uses a negative lookahead on the argument
(`regex:(?i)^(?!.*birthday).*spider`) rather than a substring check, because
"contains spiderman" is true of the broken input too.

That last one is the general lesson: an assertion that the right thing is
present usually needs a matching assertion that the wrong thing is absent.

## Regression cases from real bugs

The `grounding` category encodes failures this agent actually shipped, so they
cannot come back quietly:

- **A flat denial off an inventory miss.** A caller asked about Pokemon Funkos
  and got "no." The inventory table is a periodically-synced snapshot, not a
  live read, so a miss was never proof of absence. Fixed by hedging and
  pointing at the website.
- **A same-franchise accessory read as a Pop.** Second round of the same bug:
  the search returned a Pokemon Loungefly backpack and crossbody bag, so the
  result was non-empty, and the agent answered confidently off a result that
  contained no Pop at all. This one is pinned by a *pair* of cases, because it
  has two halves. The tool's protection keys off the query string, so searching
  a bare franchise name silently disables it — that half is a deterministic
  check on the tool argument, no judge required. The other half grades what the
  agent says once the accessories are already in hand.
- **Claiming which grails are in stock**, and **promising a delivery day**,
  both of which store policy forbids for the same reason: the agent does not
  actually know.
- **Promising a callback the system never logged.** A spoken phone number
  arrived as its last seven digits, was stored as-is, and the caller was told
  someone would be in touch. Nobody could call that number back, and the record
  looked fine in the log.
- **Denying a policy it was never given.** Asked whether the store hosts
  birthday parties — a fact that appears nowhere — it answered "no, we don't."
  A denial is a policy claim exactly as much as an affirmation is.
- **Saying the store's web address two different ways.** The greeting said it
  correctly and a stock referral did not, because tool results are relayed
  close to verbatim and carried the bare domain. The hybrid it produced names a
  domain the store does not own.

Each was found by making a real call and listening, then written down as a case
before the fix was allowed to count. All of them are caught by guarded phrases
rather than by the judge, which is the cheaper half of the suite doing the
work.

All four pass the mechanical checks — right tool, right arguments — and fail on
what gets said. That is the class of bug a deterministic matcher cannot see,
and it is why the judge and the guarded phrases exist.

These are also **second-turn** cases: the failure happens after the tool
returns, so a case can prime the conversation with a tool result via a
`context` block and grade what the agent says once it has one.

## How it works

```
cases.jsonl ──▶ adapter ──▶ predictions ──▶ scoring ──▶ metrics ──▶ gate
                (replay                     (match +               (exit 1)
                 or live)                    judge)
```

**Deterministic checks carry the suite.** Tool selection and argument
correctness are exact-match questions; using a model to answer them would be
slower, dearer, and less reliable than comparing two strings.

**The LLM judge is scoped and can only veto.** It runs only on cases carrying a
rubric — where correctness is a matter of wording, like declining without being
rude — and a judge that likes a response cannot rescue a wrong tool call. It
never sees the answer key, returns a binary verdict rather than a 1–5 score
that would cluster on 4, and runs on a cheaper model than the agent under test.

**The judge is calibrated against human labels.** On a suite where 90% of cases
pass, a judge that answers "pass" to everything scores 90% raw agreement and
looks excellent. Cohen's kappa scores it 0.0, which is why kappa is what gets
reported — and there is a unit test asserting exactly that case.

```
agent-evals label     --dataset datasets/sonny/cases.jsonl \
                      --predictions results.json --out human_labels.jsonl
agent-evals calibrate --results results.json --labels human_labels.jsonl
```

`label` walks the rubric-carrying cases one at a time, showing the caller's
utterance, the rubric, and what the agent did — and deliberately **not** the
judge's verdict. A human who has already read the judge's answer is checking
the judge's work, not producing an independent label. Answers append as you go,
so quitting mid-way keeps them, and re-running resumes where you left off. `false_pass` — cases
the judge waved through that a human failed — is the list that matters. Below
about 0.6 kappa, the rubric is ambiguous and the judge's numbers don't mean
anything yet.

**What it currently measures — and why that isn't a claim yet.** Running the
judge over the frozen baseline and comparing it to the hand-labelled set:

| | |
|---|---|
| Cases judged | 19 (the rubric-carrying cases in the set at the time) |
| Cohen's kappa | **1.00** |
| Raw agreement | 1.00 |
| `false_pass` / `false_fail` | 0 / 0 |
| Human pass rate | 94.7% (18 of 19) |
| Judge errors | 0 |

**That kappa should not be quoted as evidence, and the tool refuses to quote it
either** — `calibrate` prints `Only 19 overlapping labels; kappa is not
meaningful below 30` and the threshold is a flag, not a footnote. Two things are
wrong with it. The sample is 19, well under the n≥30 floor. And the labels are
skewed 18:1 toward pass, so a single disagreement would swing kappa by roughly
0.4 — the estimate has no stability to speak of.

Perfect agreement on a small, skewed set is what an easy classification task
looks like, not what a good judge looks like. The honest reading is that the
judge and the human have not yet disagreed anywhere, which is a prerequisite for
trusting it, not proof of it.

Making it meaningful needs more labelled cases *and* more labelled failures —
kappa is driven by the off-diagonal, and there are currently zero entries there.
That is what the dataset-growth work is for.

**The replay adapter makes CI free.** Predictions are recorded once and
replayed, so the gate runs on every PR with no API key and no spend. It also
separates variables: with model behavior frozen, any metric movement is
attributable to a scoring change alone.

## The gate

`.github/workflows/evals.yml` scores the frozen baseline on every pull request
and fails the build on a violation:

```toml
[min]
task_success_rate = 0.88
tool_accuracy = 0.95
arg_accuracy = 0.90

[max]
forbidden_rate = 0.0    # invented policy or invented stock: zero tolerance
error_rate = 0.05
cost_mean_usd = 0.05
latency_p95_ms = 6000
```

Thresholds sit just under the current baseline, not at an aspirational number.
A gate pinned at the baseline goes red on noise; one set where you wish the
agent performed goes red permanently and gets ignored within a week.

## Install and run

```bash
pip install -e ".[dev]"      # core harness has no dependencies
pytest -q                    # 35 tests

# score the frozen baseline (no API key needed)
agent-evals run --dataset datasets/sonny/cases.jsonl \
                --adapter replay \
                --replay-file datasets/sonny/fixtures/replay_baseline.jsonl \
                --gates gates.toml --report report.md --json-out results.json

# score a live agent, with the judge on rubric cases
pip install -e ".[llm]"
agent-evals run --dataset datasets/sonny/cases.jsonl \
                --adapter anthropic \
                --tools datasets/sonny/tools.json \
                --system datasets/sonny/system_prompt.txt \
                --judge --gates gates.toml
```

The core harness is dependency-free on purpose: loading a dataset, scoring a
run, and gating CI must work on a bare Python install, so a regression run never
depends on a model provider being reachable.

## Growing the dataset

```bash
python scripts/anonymize_traces.py calls.jsonl --out scrubbed.jsonl --to-cases stubs.jsonl
```

Production logs hold real customer emails, phone numbers, order numbers and
names, and eyeballing it does not survive the two hundredth line. The scrubber
does the mechanical pass — verified against 133 real production traces with no
email or phone surviving — and deliberately over-matches on digit patterns: a
mangled product number wastes a minute of review, a leaked order number is a
customer.

`--to-cases` prefills a label from what the agent actually did. That is a
starting point and explicitly **not** ground truth — grading an agent against
its own past behavior measures consistency, not correctness. Every stub needs a
human before it counts.

## Layout

```
src/agent_evals/
  models.py        Case / Prediction / CaseResult
  matching.py      deterministic argument matching
  metrics.py       aggregation, denominators, percentiles
  calibration.py   Cohen's kappa, judge-vs-human comparison
  gates.py         thresholds → CI pass/fail
  runner.py        orchestration
  report.py        Markdown report
  adapters/        replay (frozen) and anthropic (live)
  judges/          rubric-scoped LLM judge
datasets/sonny/    cases, tool schemas, frozen baseline, example labels
docs/design.md     scope decisions and known gaps
```

## Status and gaps

Seed suite, honestly scoped. Documented in `docs/design.md`:

- Most cases are synthetic, written from the shape of real calls. The
  `grounding` four are modeled on real, documented production bugs.
- 38 cases is a seed; 150–300 is where per-category numbers carry weight.
- Single-turn, plus primed second-turn cases via `context`. A full multi-turn
  flow — collecting a name and email across turns — is not yet covered.
- Latency excludes transcription and text-to-speech, which dominate what a
  caller actually experiences.

MIT licensed.
