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
| Cases | 28 |
| Task success | 89.3% |
| Tool accuracy | 96.4% |
| Argument accuracy | 92.9% |
| Guarded-phrase hits | 0.0% |
| Latency p50 / p95 | 604 ms / 845 ms |
| Cost per case | $0.0036 |

| Category | n | Task success | Tool acc | Arg acc |
| --- | ---: | ---: | ---: | ---: |
| stock | 6 | 83% | 100% | 83% |
| order | 5 | 60% | 80% | 80% |
| loyalty | 3 | 100% | 100% | 100% |
| callback | 3 | 100% | 100% | 100% |
| static_fact | 4 | 100% | 100% | 100% |
| out_of_scope | 3 | 100% | 100% | 100% |
| adversarial | 4 | 100% | 100% | 100% |

28 cases supports the headline to roughly the nearest four points; the
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
agent-evals calibrate --results results.json --labels human_labels.jsonl
```

Grade a sample by hand, blind to the judge, then compare. `false_pass` — cases
the judge waved through that a human failed — is the list that matters. Below
about 0.6 kappa, the rubric is ambiguous and the judge's numbers don't mean
anything yet.

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
pytest -q                    # 28 tests

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

- Cases are synthetic, written from the shape of real calls.
- 28 cases is a seed; 150–300 is where per-category numbers carry weight.
- Single-turn only. The callback flow needs multi-turn cases to be covered.
- Latency excludes transcription and text-to-speech, which dominate what a
  caller actually experiences.

MIT licensed.
