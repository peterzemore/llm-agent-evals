"""Per-model token prices in USD per million tokens, for cost accounting.

Kept in one place so a model swap reprices every historical run consistently.
Verify against https://claude.com/pricing before quoting a number publicly.
"""

from __future__ import annotations

PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model id: (input, output)
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD for one request. Unknown models cost 0.0 rather than guessing."""
    if model not in PRICING_USD_PER_MTOK:
        return 0.0
    price_in, price_out = PRICING_USD_PER_MTOK[model]
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out
