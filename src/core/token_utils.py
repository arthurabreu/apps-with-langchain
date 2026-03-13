"""
Utility module for token management and tracking.
Helps monitor token usage, costs, and limits for different models.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import os
from .services import get_brazil_time
from .config import PRICING, DEFAULT_MODEL, TOKEN_USAGE_LOG, MAX_TOKENS_PER_MODEL

@dataclass
class TokenUsage:
    """Stores token usage information for a specific operation"""
    timestamp: str
    model: str
    tokens_used: int
    estimated_cost: float
    operation_type: str  # e.g., "chat", "completion", "embedding"

class TokenManager:
    # Reference to global pricing and token limits from config
    COST_PER_1M_TOKENS = PRICING
    MAX_TOKENS = MAX_TOKENS_PER_MODEL

    # Common aliases mapped to canonical names with descriptions
    MODEL_ALIASES = {
        "claude-3-opus": {"model": "claude-3-opus-20240229", "description": "Most powerful Claude"},
        "claude-3-5-sonnet": {"model": "claude-3-5-sonnet-20240620", "description": "Best balance of speed and intelligence"},
        "claude-3-haiku": {"model": "claude-3-haiku-20240307", "description": "Fastest and most affordable Claude"},
    }
    
    # Best coding models (for different use cases)
    BEST_CODING_MODELS = {
        "general-purpose": "claude-3-5-sonnet-20240620",
        "fast-coding": "claude-3-haiku-20240307",
        "advanced-logic": "claude-3-opus-20240229",
    }

    def __init__(self, log_file: str = None):
        """Initialize TokenManager with optional log file path"""
        if log_file is None:
            log_file = TOKEN_USAGE_LOG
        self.log_file = log_file
        self.usage_history: List[TokenUsage] = self._load_usage_history()

    @classmethod
    def resolve_model(cls, model: str) -> str:
        """Map aliases to canonical model names if known."""
        if model in cls.MODEL_ALIASES:
            return cls.MODEL_ALIASES[model]["model"]
        return model

    def count_tokens(self, text: str, model: str = None) -> int:
        """Count tokens in a text string for a specific model"""
        if model is None:
            model = DEFAULT_MODEL
        if not isinstance(text, str):
            print(f"Warning: Expected string but got {type(text)}. Converting to string.")
            text = str(text) if text is not None else ""

        try:
            # For Claude models, a rough estimation is often used if the exact tokenizer isn't available
            # Claude's tokenizer is roughly 1 token per 4 characters for English text
            return int(len(text) / 4)
        except Exception as e:
            print(f"Error counting tokens for model '{model}': {e}")
            return int(len(text) // 4)  # Fallback: consistent character-based estimate

    def estimate_cost(self, num_tokens: int, model: str, is_output: bool = False) -> float:
        """Estimate cost for token usage"""
        model = self.resolve_model(model)
        if model not in self.COST_PER_1M_TOKENS:
            return 0.0
        
        rate_type = "output" if is_output else "input"
        rate = self.COST_PER_1M_TOKENS[model][rate_type]
        return (num_tokens / 1_000_000) * rate

    def get_token_breakdown(self, text: str, model: str = None) -> Dict:
        """Get detailed breakdown of token usage"""
        if model is None:
            model = DEFAULT_MODEL
        model = self.resolve_model(model)
        tokens_count = self.count_tokens(text, model)
        
        return {
            "total_tokens": tokens_count,
            "total_chars": len(text),
            "tokens_per_char": tokens_count / len(text) if text else 0,
            "estimated_cost": self.estimate_cost(tokens_count, model),
            "model": model,
        }

    def check_context_window(self, text: str, model: str) -> Tuple[int, int, float]:
        """Check token count against model's context window"""
        model_resolved = self.resolve_model(model)
        token_count = self.count_tokens(text, model_resolved)
        max_tokens = self.MAX_TOKENS.get(model_resolved, 0)
        remaining = max_tokens - token_count if max_tokens > 0 else 0
        usage_percentage = (token_count / max_tokens * 100) if max_tokens > 0 else 0
        
        return token_count, remaining, usage_percentage

    def log_usage(self, model: str, tokens_used: int, operation_type: str, is_output: bool = False):
        """Log token usage for tracking"""
        model_resolved = self.resolve_model(model)
        cost = self.estimate_cost(tokens_used, model_resolved, is_output=is_output)
        usage = TokenUsage(
            timestamp=get_brazil_time().isoformat(),
            model=model_resolved,
            tokens_used=tokens_used,
            estimated_cost=cost,
            operation_type=operation_type
        )
        self.usage_history.append(usage)
        self._save_usage_history()

    def get_coding_model(self, use_case: str = "general-purpose") -> str:
        """Get the best coding model for a specific use case"""
        return self.BEST_CODING_MODELS.get(use_case, "claude-3-5-sonnet-20240620")
    
    def list_all_models(self) -> List[str]:
        """List all available models"""
        return sorted(list(self.COST_PER_1M_TOKENS.keys()))
    
    def get_model_info(self, model: str) -> Dict:
        """Get detailed information about a model"""
        model = self.resolve_model(model)
        if model not in self.COST_PER_1M_TOKENS:
            return {"error": f"Model '{model}' not found"}
        
        pricing = self.COST_PER_1M_TOKENS[model]
        return {
            "model": model,
            "input_cost_per_1m": pricing["input"],
            "output_cost_per_1m": pricing["output"],
            "description": pricing.get("description", ""),
            "category": pricing.get("category", ""),
            "context_window": self.MAX_TOKENS.get(model, "Unknown")
        }
    
    def compare_models(self, models: List[str], tokens_count: int = 1000) -> Dict:
        """Compare costs for multiple models"""
        comparison = {}
        for model in models:
            resolved = self.resolve_model(model)
            input_cost = self.estimate_cost(tokens_count, resolved, is_output=False)
            output_cost = self.estimate_cost(tokens_count, resolved, is_output=True)
            comparison[model] = {
                "resolved_model": resolved,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_estimated": input_cost + output_cost,
                "description": self.COST_PER_1M_TOKENS.get(resolved, {}).get("description", "")
            }
        return comparison

    def get_usage_summary(self) -> Dict:
        """Get summary of token usage and costs"""
        summary = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "usage_by_model": {},
            "usage_by_operation": {}
        }
        
        for usage in self.usage_history:
            summary["total_tokens"] += usage.tokens_used
            summary["total_cost"] += usage.estimated_cost
            
            # Track by model
            if usage.model not in summary["usage_by_model"]:
                summary["usage_by_model"][usage.model] = {
                    "tokens": 0,
                    "cost": 0.0
                }
            summary["usage_by_model"][usage.model]["tokens"] += usage.tokens_used
            summary["usage_by_model"][usage.model]["cost"] += usage.estimated_cost
            
            # Track by operation type
            if usage.operation_type not in summary["usage_by_operation"]:
                summary["usage_by_operation"][usage.operation_type] = {
                    "tokens": 0,
                    "cost": 0.0
                }
            summary["usage_by_operation"][usage.operation_type]["tokens"] += usage.tokens_used
            summary["usage_by_operation"][usage.operation_type]["cost"] += usage.estimated_cost
        
        return summary

    def _load_usage_history(self) -> List[TokenUsage]:
        """Load usage history from file"""
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                return [TokenUsage(**usage) for usage in data]
        except Exception as e:
            print(f"Error loading usage history: {e}")
            return []

    def _save_usage_history(self):
        """Save usage history to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump([vars(usage) for usage in self.usage_history], f, indent=2)
        except Exception as e:
            print(f"Error saving usage history: {e}")

def main():
    """Example usage of TokenManager."""
    # Initialize token manager
    token_manager = TokenManager()

    # Example text
    example_text = """
    This is a sample text that we'll analyze for token usage.
    We'll see how different models handle it and track the usage.
    """

    # Analyze token usage for different models
    models = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]

    print("\n=== Token Analysis Examples ===")
    for model in models:
        print(f"\nAnalyzing for {model}:")

        # Get token breakdown
        breakdown = token_manager.get_token_breakdown(example_text, model)
        tokens_used = breakdown['total_tokens']
        print(f"Tokens used: {tokens_used}")
        print(f"Estimated input cost: ${breakdown['estimated_cost']:.6f}")

        # Check context window
        tokens, remaining, usage_percent = token_manager.check_context_window(example_text, model)
        print(f"Context window remaining: {remaining} tokens")
        print(f"Usage: {usage_percent:.3f}%")

        # Log this usage
        token_manager.log_usage(model, tokens_used, "analysis", is_output=False)

    # Get usage summary
    print("\n=== Usage Summary ===")
    summary = token_manager.get_usage_summary()
    print(f"Total tokens used: {summary['total_tokens']}")
    print(f"Total cost: ${summary['total_cost']:.6f}")

    print("\nUsage by model:")
    for model, usage in summary['usage_by_model'].items():
        tokens_used = usage['tokens']
        print(f"{model}: tokens used {tokens_used} (${usage['cost']:.6f})")

    # Show best coding models
    print("\n=== Best Coding Models ===")
    for use_case, model in token_manager.BEST_CODING_MODELS.items():
        info = token_manager.get_model_info(model)
        print(f"{use_case}: {model} ({info['description']})")

    # Compare models for coding
    print("\n=== Model Comparison (1000 tokens) ===")
    coding_models = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
    comparison = token_manager.compare_models(coding_models, tokens_count=1000)
    for model, costs in comparison.items():
        print(f"\n{model}:")
        print(f"  Input cost: ${costs['input_cost']:.6f}")
        print(f"  Output cost: ${costs['output_cost']:.6f}")
        print(f"  Total: ${costs['total_estimated']:.6f}")
        print(f"  Description: {costs['description']}")


if __name__ == "__main__":
    main()