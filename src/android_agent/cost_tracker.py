"""
Cost tracking for agent sessions.
Wraps src.core.cost_tracker.CostTracker with session-scoped accumulation.
"""

from src.core.cost_tracker import CostTracker
from src.core.config import PRICING


class AgentCostTracker:
    """Session-scoped cost accumulator."""

    def __init__(self, model: str):
        """
        Initialize cost tracker for a model.

        Args:
            model: Model identifier (e.g., "claude-3-haiku-20240307")
        """
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, prompt_tokens: int, response_tokens: int) -> None:
        """
        Record token usage from an LLM call.

        Args:
            prompt_tokens: Input tokens consumed
            response_tokens: Output tokens generated
        """
        self.input_tokens += prompt_tokens
        self.output_tokens += response_tokens

    def flush(self, task_preview: str = "") -> None:
        """
        Log accumulated usage to persistent storage.

        Args:
            task_preview: Optional preview of the task for context
        """
        if self.input_tokens + self.output_tokens == 0:
            return

        tracker = CostTracker()
        tracker.log(
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            source="android_agent",
            context="android",
            prompt_preview=task_preview
        )

    def summary(self) -> str:
        """
        Format usage summary for display.

        Returns:
            Formatted summary string
        """
        input_cost = self._cost_for(self.input_tokens, is_output=False)
        output_cost = self._cost_for(self.output_tokens, is_output=True)
        total_cost = input_cost + output_cost

        lines = [
            "=== Cost Summary ===",
            f"Model: {self.model}",
            f"Input tokens:  {self.input_tokens:,}   (${input_cost:.6f})",
            f"Output tokens:   {self.output_tokens:,}   (${output_cost:.6f})",
            f"Total cost:          ${total_cost:.6f}"
        ]

        return "\n".join(lines)

    def _cost_for(self, tokens: int, is_output: bool) -> float:
        """
        Calculate cost for a token count.

        Args:
            tokens: Number of tokens
            is_output: If True, use output pricing; else input pricing

        Returns:
            Cost in dollars
        """
        # Get pricing per token
        if self.model in PRICING:
            pricing_entry = PRICING[self.model]
        else:
            # Fallback to Haiku pricing for unknown models
            pricing_entry = PRICING.get("claude-3-haiku-20240307", {"input": 0.25, "output": 1.25})

        key = "output" if is_output else "input"
        price_per_million = pricing_entry[key]
        price_per_token = price_per_million / 1_000_000

        return tokens * price_per_token
