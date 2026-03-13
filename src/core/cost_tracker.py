"""
Cost tracking service for logging API usage and costs.
Records all API calls to data/costs.json with real token counts and pricing.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class CostTracker:
    """Service for tracking and logging API costs."""

    # Pricing table (input/output per 1M tokens)
    PRICING = {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    }

    def __init__(self, costs_file: str = "data/costs.json"):
        """
        Initialize the cost tracker.

        Args:
            costs_file: Path to the JSON file storing cost logs
        """
        self.costs_file = Path(costs_file)
        self._ensure_costs_file()

    def _ensure_costs_file(self) -> None:
        """Create costs file if it doesn't exist."""
        if not self.costs_file.exists():
            self.costs_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.costs_file, "w") as f:
                json.dump([], f)

    def _get_pricing(self, model: str, is_output: bool = False) -> float:
        """
        Get pricing per token for a model.

        Args:
            model: Model identifier
            is_output: If True, return output pricing; else input pricing

        Returns:
            Cost per token (in dollars)
        """
        # Fallback for unknown models
        model_key = model
        if model_key not in self.PRICING:
            # Try fuzzy match on Haiku
            if "haiku" in model.lower():
                model_key = "claude-haiku-4-5-20251001"
            elif "sonnet" in model.lower():
                model_key = "claude-sonnet-4-6"
            else:
                # Default to Haiku pricing
                model_key = "claude-haiku-4-5-20251001"

        pricing_entry = self.PRICING[model_key]
        key = "output" if is_output else "input"
        # Pricing is per 1M tokens, so convert to per-token
        return pricing_entry[key] / 1_000_000

    def log(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        source: str,
        context: str = "none",
        prompt_preview: str = "",
    ) -> None:
        """
        Log an API call to the costs file.

        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            source: Source of the call ("api_app", "api_script", or "claude_cli")
            context: Development context used (default: "none")
            prompt_preview: Short preview of the prompt (first N chars)
        """
        # Calculate costs
        input_cost = input_tokens * self._get_pricing(model, is_output=False)
        output_cost = output_tokens * self._get_pricing(model, is_output=True)
        total_cost = input_cost + output_cost

        # Create log entry
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "context": context,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "source": source,
            "prompt_preview": prompt_preview[:100] if prompt_preview else "",
        }

        # Append to costs file
        try:
            with open(self.costs_file, "r") as f:
                costs = json.load(f)

            costs.append(entry)

            with open(self.costs_file, "w") as f:
                json.dump(costs, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to log cost: {e}")
