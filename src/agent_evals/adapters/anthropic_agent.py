"""The live agent under test: one Claude turn with the production tool schemas.

This deliberately evaluates a single turn rather than a whole conversation.
The failure this suite exists to catch - the agent reaching for the wrong tool,
or inventing an order number the caller never said - is decided on the first
turn, and single-turn cases stay cheap enough to run on every prompt change.

Requires `pip install "agent-evals[llm]"` and credentials in the environment
(ANTHROPIC_API_KEY, or an `ant auth login` profile).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..models import Case, Prediction
from ..pricing import estimate_cost


class AnthropicAgentAdapter:
    name = "anthropic"

    def __init__(
        self,
        tools_path: str | Path,
        system_prompt: str,
        model: str = "claude-opus-5",
        effort: str = "low",
        max_tokens: int = 2048,
    ):
        import anthropic  # imported lazily so the core harness stays dependency-free

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.tools: list[dict[str, Any]] = json.loads(Path(tools_path).read_text())

    @staticmethod
    def _messages_for(case: Case) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": case.utterance}]
        if case.context:
            tool_use_id = "toolu_ctx_0"
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": case.context["tool"],
                            "input": case.context.get("args", {}),
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": case.context.get("result", ""),
                        }
                    ],
                }
            )
        return messages

    def predict(self, case: Case) -> Prediction:
        started = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            tools=self.tools,
            # `auto`, never forced: whether to call a tool at all is exactly what
            # is being measured. Forcing a call would paper over the out-of-scope
            # and static-fact cases.
            tool_choice={"type": "auto"},
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=self._messages_for(case),
        )
        latency_ms = (time.perf_counter() - started) * 1000

        tool_name: str | None = None
        tool_args: dict[str, Any] = {}
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "tool_use" and tool_name is None:
                tool_name = block.name
                # Tool inputs are already parsed objects; never string-match the
                # serialized form, escaping varies by model.
                tool_args = dict(block.input)
            elif block.type == "text":
                text_parts.append(block.text)

        usage = response.usage
        return Prediction(
            case_id=case.id,
            tool=tool_name,
            args=tool_args,
            response_text="\n".join(text_parts).strip(),
            latency_ms=latency_ms,
            cost_usd=estimate_cost(self.model, usage.input_tokens, usage.output_tokens),
        )
