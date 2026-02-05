"""
Utility module for token management and tracking using tiktoken.
Helps monitor token usage, costs, and limits for different models.
"""

import tiktoken
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import os

@dataclass
class TokenUsage:
    """Stores token usage information for a specific operation"""
    timestamp: str
    model: str
    tokens_used: int
    estimated_cost: float
    operation_type: str  # e.g., "chat", "completion", "embedding"

class TokenManager:
    # Updated prices per 1M tokens (Standard tier, as of 2026-01). Always verify with provider docs.
    # Source: https://platform.openai.com/docs/pricing
    COST_PER_1M_TOKENS = {
        # GPT-5 family (newest)
        "gpt-5.2": {"input": 1.75, "output": 14.00, "description": "Advanced reasoning", "category": "flagship"},
        "gpt-5.1": {"input": 1.25, "output": 10.00, "description": "Balanced capability", "category": "flagship"},
        "gpt-5": {"input": 1.25, "output": 10.00, "description": "Core reasoning", "category": "flagship"},
        "gpt-5-mini": {"input": 0.25, "output": 2.00, "description": "Fast inference", "category": "efficient"},
        "gpt-5-nano": {"input": 0.05, "output": 0.40, "description": "Ultra-light", "category": "efficient"},
        "gpt-5.2-pro": {"input": 21.00, "output": 168.00, "description": "Premium advanced", "category": "premium"},
        "gpt-5-pro": {"input": 15.00, "output": 120.00, "description": "Premium tier", "category": "premium"},
        
        # GPT-4.1 family (multimodal)
        "gpt-4.1": {"input": 2.00, "output": 8.00, "description": "Vision capable", "category": "multimodal"},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "description": "Fast vision", "category": "multimodal"},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "description": "Lite vision", "category": "multimodal"},
        
        # GPT-4o family (omni)
        "gpt-4o": {"input": 2.50, "output": 10.00, "description": "Audio-visual", "category": "multimodal"},
        "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00, "description": "Legacy omni", "category": "legacy"},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "description": "Lightweight omni", "category": "efficient"},
        
        # GPT-4 legacy
        "gpt-4": {"input": 30.00, "output": 60.00, "description": "Legacy reasoning", "category": "legacy"},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00, "description": "Legacy turbo", "category": "legacy"},
        
        # GPT-3.5 legacy
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "description": "Budget chat", "category": "legacy"},
        
        # O-series reasoning models
        "o1": {"input": 15.00, "output": 60.00, "description": "Extended reasoning", "category": "reasoning"},
        "o1-pro": {"input": 150.00, "output": 600.00, "description": "Advanced reasoning", "category": "reasoning"},
        "o3": {"input": 2.00, "output": 8.00, "description": "Fast reasoning", "category": "reasoning"},
        "o3-pro": {"input": 20.00, "output": 80.00, "description": "Expert reasoning", "category": "reasoning"},
        "o3-deep-research": {"input": 10.00, "output": 40.00, "description": "Research analysis", "category": "reasoning"},
        "o4-mini": {"input": 1.10, "output": 4.40, "description": "Quick decisions", "category": "reasoning"},
        "o4-mini-deep-research": {"input": 2.00, "output": 8.00, "description": "Deep analysis", "category": "reasoning"},
        "o3-mini": {"input": 1.10, "output": 4.40, "description": "Compact reasoning", "category": "reasoning"},
        "o1-mini": {"input": 1.10, "output": 4.40, "description": "Lean reasoning", "category": "reasoning"},
        
        # Audio/Visual/Realtime models
        "gpt-realtime": {"input": 4.00, "output": 16.00, "description": "Real-time audio", "category": "realtime"},
        "gpt-realtime-mini": {"input": 0.60, "output": 2.40, "description": "Lite realtime", "category": "realtime"},
        "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00, "description": "Preview realtime", "category": "realtime"},
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40, "description": "Mini realtime", "category": "realtime"},
        "gpt-audio": {"input": 2.50, "output": 10.00, "description": "Speech support", "category": "audio"},
        "gpt-audio-mini": {"input": 0.60, "output": 2.40, "description": "Speech lite", "category": "audio"},
        "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00, "description": "Audio preview", "category": "audio"},
        "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60, "description": "Mini audio", "category": "audio"},
        
        # Search & API models
        "gpt-4o-search-preview": {"input": 2.50, "output": 10.00, "description": "Web search", "category": "search"},
        "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.60, "description": "Search lite", "category": "search"},
        
        # Computer use
        "computer-use-preview": {"input": 3.00, "output": 12.00, "description": "Automation control", "category": "special"},
        
        # Embeddings (input only)
        "text-embedding-3-large": {"input": 0.13, "output": 0.0, "description": "Dense embeddings", "category": "embedding"},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0, "description": "Lite embeddings", "category": "embedding"},
    }
    
    # Maximum context windows (tokens). Values are typical defaults; verify per model variant.
    MAX_TOKENS = {
        "gpt-5.2": 200000,
        "gpt-5.1": 200000,
        "gpt-5": 200000,
        "gpt-5-mini": 128000,
        "gpt-5-nano": 128000,
        "gpt-5.2-pro": 200000,
        "gpt-5-pro": 200000,
        "gpt-4.1": 128000,
        "gpt-4.1-mini": 128000,
        "gpt-4.1-nano": 128000,
        "gpt-4o": 128000,
        "gpt-4o-2024-05-13": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 4096,
        "o1": 128000,
        "o1-pro": 128000,
        "o3": 128000,
        "o3-pro": 128000,
        "o3-deep-research": 128000,
        "o4-mini": 128000,
        "o4-mini-deep-research": 128000,
        "o3-mini": 128000,
        "o1-mini": 128000,
        "gpt-realtime": 128000,
        "gpt-realtime-mini": 128000,
        "gpt-4o-realtime-preview": 128000,
        "gpt-4o-mini-realtime-preview": 128000,
        "gpt-audio": 128000,
        "gpt-audio-mini": 128000,
        "gpt-4o-audio-preview": 128000,
        "gpt-4o-mini-audio-preview": 128000,
        "gpt-4o-search-preview": 128000,
        "gpt-4o-mini-search-preview": 128000,
        "computer-use-preview": 128000,
        "text-embedding-3-large": 8192,
        "text-embedding-3-small": 8192,
    }

    # Common aliases mapped to canonical names with descriptions
    MODEL_ALIASES = {
        "gpt5.2": {"model": "gpt-5.2", "description": "Advanced reasoning"},
        "gpt5.1": {"model": "gpt-5.1", "description": "Balanced capability"},
        "gpt5": {"model": "gpt-5", "description": "Core reasoning"},
        "gpt5-mini": {"model": "gpt-5-mini", "description": "Fast inference"},
        "gpt5-nano": {"model": "gpt-5-nano", "description": "Ultra-light"},
        "gpt4.1": {"model": "gpt-4.1", "description": "Vision capable"},
        "gpt4.1-mini": {"model": "gpt-4.1-mini", "description": "Fast vision"},
        "gpt4o": {"model": "gpt-4o", "description": "Audio-visual"},
        "gpt4o-mini": {"model": "gpt-4o-mini", "description": "Lightweight omni"},
        "gpt4": {"model": "gpt-4", "description": "Legacy reasoning"},
        "gpt35": {"model": "gpt-3.5-turbo", "description": "Budget chat"},
        "gpt-3.5": {"model": "gpt-3.5-turbo", "description": "Budget chat"},
        "o1": {"model": "o1", "description": "Extended reasoning"},
        "o3": {"model": "o3", "description": "Fast reasoning"},
        "o3-pro": {"model": "o3-pro", "description": "Expert reasoning"},
        "embedding-3-large": {"model": "text-embedding-3-large", "description": "Dense embeddings"},
        "embedding-3-small": {"model": "text-embedding-3-small", "description": "Lite embeddings"},
    }
    
    # Best coding models (for different use cases)
    BEST_CODING_MODELS = {
        "general-purpose": "gpt-4o",
        "fast-coding": "gpt-4o-mini",
        "advanced-logic": "o3",
        "deep-reasoning": "o1",
        "budget-coding": "gpt-3.5-turbo",
    }

    def __init__(self, log_file: str = "token_usage.json"):
        """Initialize TokenManager with optional log file path"""
        self.log_file = log_file
        self.usage_history: List[TokenUsage] = self._load_usage_history()

    @classmethod
    def resolve_model(cls, model: str) -> str:
        """Map aliases to canonical model names if known."""
        if model in cls.MODEL_ALIASES:
            return cls.MODEL_ALIASES[model]["model"]
        return model

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in a text string for a specific model"""
        if not isinstance(text, str):
            print(f"Warning: Expected string but got {type(text)}. Converting to string.")
            text = str(text) if text is not None else ""

        try:
            # Map newer models to their encoding base models
            model_to_encoding = {
                "gpt-4": "cl100k_base",
                "gpt-3.5-turbo": "cl100k_base",
                "gpt-4-turbo": "cl100k_base",
                "text-embedding-ada-002": "cl100k_base",
            }
            
            try:
                # Try getting encoding for the exact model
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # If that fails, use the base encoding for the model family
                base_encoding = model_to_encoding.get(model.split('-')[0], "cl100k_base")
                print(f"Using base encoding {base_encoding} for model {model}")
                encoding = tiktoken.get_encoding(base_encoding)
            
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting tokens for model '{model}': {e}")
            # Fallback to cl100k_base encoding
            try:
                print("Falling back to cl100k_base encoding")
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e2:
                print(f"Fallback encoding failed: {e2}")
                return len(text.split()) * 4  # Very rough estimation

    def estimate_cost(self, num_tokens: int, model: str, is_output: bool = False) -> float:
        """Estimate cost for token usage"""
        model = self.resolve_model(model)
        if model not in self.COST_PER_1M_TOKENS:
            return 0.0
        
        rate_type = "output" if is_output else "input"
        rate = self.COST_PER_1M_TOKENS[model][rate_type]
        return (num_tokens / 1_000_000) * rate

    def get_token_breakdown(self, text: str, model: str = "gpt-4o-mini") -> Dict:
        """Get detailed breakdown of token usage"""
        model = self.resolve_model(model)
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(text)
        
        return {
            "total_tokens": len(tokens),
            "total_chars": len(text),
            "tokens_per_char": len(tokens) / len(text) if text else 0,
            "first_tokens": [(token, encoding.decode([token])) for token in tokens[:10]],
            "estimated_cost": self.estimate_cost(len(tokens), model),
            "model": model,
            "encoding": encoding.name
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
            timestamp=datetime.now().isoformat(),
            model=model_resolved,
            tokens_used=tokens_used,
            estimated_cost=cost,
            operation_type=operation_type
        )
        self.usage_history.append(usage)
        self._save_usage_history()

    def get_coding_model(self, use_case: str = "general-purpose") -> str:
        """Get the best coding model for a specific use case"""
        return self.BEST_CODING_MODELS.get(use_case, "gpt-4o")
    
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

# Example usage
def main():
    # Initialize token manager
    token_manager = TokenManager()

    # Example text
    example_text = """
    This is a sample text that we'll analyze for token usage.
    We'll see how different models handle it and track the usage.
    """

    # Analyze token usage for different models
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
    
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

    # Embeddings example
    embed_text = "This is a sentence we want to embed."
    embed_tokens = token_manager.count_tokens(embed_text, "text-embedding-3-small")
    embed_cost = token_manager.estimate_cost(embed_tokens, "text-embedding-3-small", is_output=False)
    print("\n[Embeddings] tokens used:", embed_tokens, "estimated cost: $", f"{embed_cost:.6f}")
    token_manager.log_usage("text-embedding-3-small", embed_tokens, "embedding", is_output=False)

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
    coding_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "o1"]
    comparison = token_manager.compare_models(coding_models, tokens_count=1000)
    for model, costs in comparison.items():
        print(f"\n{model}:")
        print(f"  Input cost: ${costs['input_cost']:.6f}")
        print(f"  Output cost: ${costs['output_cost']:.6f}")
        print(f"  Total: ${costs['total_estimated']:.6f}")
        print(f"  Description: {costs['description']}")

if __name__ == "__main__":
    main()