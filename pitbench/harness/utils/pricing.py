"""Shared token-based model cost calculation."""

from __future__ import annotations

from pitbench.harness.utils.model_names import normalize_model_name_for_pricing


def cost_from_tokens(
    input_tokens: int, output_tokens: int, model_name: str | None
) -> float:
    if not model_name or (input_tokens == 0 and output_tokens == 0):
        return 0.0
    try:
        from pitbench.harness.llms.portkey_llm import MODEL_COST_OVERRIDES

        normalized_name = normalize_model_name_for_pricing(model_name)
        for candidate in (normalized_name, model_name):
            if candidate in MODEL_COST_OVERRIDES:
                input_cost, output_cost = MODEL_COST_OVERRIDES[candidate]
                return input_tokens * input_cost + output_tokens * output_cost
        import litellm

        input_cost, output_cost = litellm.cost_per_token(model=model_name)
        return input_tokens * input_cost + output_tokens * output_cost
    except Exception:
        return 0.0
