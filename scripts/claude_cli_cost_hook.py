#!/usr/bin/env python3
"""
Claude CLI Stop hook for tracking API costs.
Reads session data from stdin when Claude Code stops and logs costs to data/costs.json.
"""

import sys
import json
import os
from datetime import datetime, timezone
from pathlib import Path


# Pricing table (input/output per 1M tokens)
PRICING = {
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 45.00},
}


def get_project_dir() -> Path:
    """Get the project directory."""
    # Try environment variable first
    if env_dir := os.getenv("CLAUDE_PROJECT_DIR"):
        return Path(env_dir)

    # Default to the app project directory
    return Path("/Users/mac/python/apps-with-langchain")


def get_costs_file() -> Path:
    """Get the path to the costs JSON file."""
    project_dir = get_project_dir()
    return project_dir / "data" / "costs.json"


def get_pricing(model: str, is_output: bool = False) -> float:
    """Get pricing per token for a model."""
    model_key = model
    if model_key not in PRICING:
        # Try fuzzy match
        if "haiku" in model.lower():
            model_key = "claude-haiku-4-5-20251001"
        elif "sonnet" in model.lower():
            model_key = "claude-sonnet-4-6"
        elif "opus" in model.lower():
            model_key = "claude-opus-4-6"
        else:
            model_key = "claude-haiku-4-5-20251001"

    pricing_entry = PRICING[model_key]
    key = "output" if is_output else "input"
    return pricing_entry[key] / 1_000_000


def extract_usage_from_stop_event(data: dict) -> tuple[dict, str, str]:
    """
    Extract usage data from Claude Code stop event.

    Returns:
        Tuple of (usage_dict, model_name, prompt_preview)
    """
    # Try different payload structures
    messages = data.get("messages", []) or data.get("transcript", [])
    usage = {}
    model = "claude-cli"
    prompt_preview = ""

    # Find last assistant message with usage data
    for msg in reversed(messages):
        if isinstance(msg, dict):
            # Check if this is an assistant message
            if msg.get("type") == "assistant" or msg.get("role") == "assistant":
                # Try to extract usage from different structures
                if "message" in msg:
                    message_obj = msg["message"]
                    if isinstance(message_obj, dict):
                        usage = message_obj.get("usage", {})
                        model = message_obj.get("model", "claude-cli")
                        content = message_obj.get("content", "")
                        if content:
                            prompt_preview = content[:100]
                elif "usage" in msg:
                    usage = msg["usage"]

                if usage:
                    break

            # Try to get prompt preview from user messages
            if msg.get("type") == "user" or msg.get("role") == "user":
                if "content" in msg:
                    prompt_preview = str(msg["content"])[:100]

    return usage, model, prompt_preview


def log_usage(input_tokens: int, output_tokens: int, model: str, prompt_preview: str = "") -> None:
    """Log usage to costs file."""
    costs_file = get_costs_file()

    # Ensure directory exists
    costs_file.parent.mkdir(parents=True, exist_ok=True)

    # Calculate costs
    input_cost = input_tokens * get_pricing(model, is_output=False)
    output_cost = output_tokens * get_pricing(model, is_output=True)
    total_cost = input_cost + output_cost

    # Create log entry
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "context": "none",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "source": "claude_cli",
        "prompt_preview": prompt_preview,
    }

    # Append to costs file
    try:
        if costs_file.exists():
            with open(costs_file, "r") as f:
                costs = json.load(f)
        else:
            costs = []

        costs.append(entry)

        with open(costs_file, "w") as f:
            json.dump(costs, f, indent=2)
    except Exception as e:
        # Silently fail - don't interrupt the CLI
        pass


def main():
    """Main entry point for the hook."""
    try:
        # Read stdin
        input_data = sys.stdin.read()
        if not input_data:
            return

        data = json.loads(input_data)

        # Extract usage
        usage, model, prompt_preview = extract_usage_from_stop_event(data)

        if not usage:
            return

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Account for cache creation tokens
        if "cache_creation_input_tokens" in usage:
            input_tokens += usage["cache_creation_input_tokens"]

        if input_tokens > 0 or output_tokens > 0:
            log_usage(input_tokens, output_tokens, model, prompt_preview)

    except Exception:
        # Silently fail - don't interrupt the CLI
        pass


if __name__ == "__main__":
    main()
