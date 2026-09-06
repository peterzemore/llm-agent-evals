# llm-agent-evals — working notes

Read the README first. This is what a session needs that it doesn't say.

- **Invocation:** the package is not pip-installed in `.venv`. Run
  `PYTHONPATH=src .venv/bin/python -m agent_evals.cli ...` from the repo root.
- **Calibration must run against the replay baseline**, not a fresh live run. The
  human labels were made on those specific transcripts; judging different ones makes
  the pairing meaningless. `run --adapter replay --judge --json-out …` then
  `calibrate --results … --labels human_labels.jsonl`.
- **Do not quote a kappa the sample can't support.** The harness prints a warning
  under n=30 and the README reports the measurement together with that
  disqualification. Kappa is driven by the off-diagonal; the way to make it
  meaningful is more *labelled failures*, not more labelled passes.
- **Every pinned case comes from a real bug, and every fix in the live agent gets a
  case here before it counts.** Cases with a `context` block prime a tool result and
  grade the agent's *second* turn — that's where the real bugs happen.
- **Adding a case means adding its replay prediction** to
  `datasets/sonny/fixtures/replay_baseline.jsonl`, or CI fails on the next push. Then
  refresh the README's counts and per-category table from a run, not by hand.
- The dataset file allows `#` comment lines; parse with the project's reader, not raw
  `json.loads` per line.
- Selection criterion on validation, ties, guards, and gates are deliberate design
  calls documented in the README; don't re-litigate them in a fix.
