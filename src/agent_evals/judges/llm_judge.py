"""LLM-as-judge for the cases where correctness is a matter of wording.

Design notes, because most of the value of a judge is in what it is not asked
to do:

* It grades against a per-case rubric, never a vague "is this good?".
* It returns a binary verdict. A 1-5 score invites the judge to cluster on 4
  and produces a mean that drifts with the phrasing of the prompt.
* It never sees the expected tool call. Showing it the answer key turns it into
  a paraphrase detector that agrees with whatever it was shown.
* It runs on a cheaper model than the agent under test. A judge is a
  classification task; paying Opus rates for it is how eval budgets get
  quietly eaten.
* Its verdicts get compared against human labels - see `calibration.py`. A
  judge whose kappa has not been measured is a vibe, not a metric.
"""

from __future__ import annotations

from ..models import Case, Prediction
from ..pricing import estimate_cost

SYSTEM_PROMPT = """You grade a single turn from a retail store's phone agent.

You will be given what the caller said, a rubric describing what an acceptable
response must do, and what the agent actually did. Decide whether the agent's
turn satisfies the rubric.

Grade only against the rubric. Do not reward extra helpfulness that the rubric
did not ask for, and do not punish wording you would have phrased differently.
If the agent stated a fact about stock, pricing, hours, or store policy that it
was not given, that is a failure regardless of how reasonable the fact sounds.
"""

USER_TEMPLATE = """Caller said:
{utterance}

Rubric for an acceptable turn:
{rubric}

The agent's turn:
- tool called: {tool}
- tool arguments: {args}
- spoken response: {response}

Does this turn satisfy the rubric?"""


class ClaudeJudge:
    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1024):
        import anthropic  # lazy import; core harness has no hard dependency
        from pydantic import BaseModel

        class Verdict(BaseModel):
            passed: bool
            rationale: str

        self._verdict_model = Verdict
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.cost_usd = 0.0

    def verdict(self, case: Case, prediction: Prediction) -> tuple[bool, str]:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        utterance=case.utterance,
                        rubric=case.rubric,
                        tool=prediction.tool or "(no tool call)",
                        args=prediction.args or "(none)",
                        response=prediction.response_text or "(no spoken response)",
                    ),
                }
            ],
            output_format=self._verdict_model,
        )
        usage = response.usage
        self.cost_usd += estimate_cost(self.model, usage.input_tokens, usage.output_tokens)

        parsed = response.parsed_output
        return bool(parsed.passed), parsed.rationale.strip()
