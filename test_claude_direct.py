"""
Direct Claude API test - for cost comparison without application context.
Run this alongside the app to compare token usage and costs.
"""

import os
import sys
import json
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '/Users/mac/python/apps-with-langchain')

from src.core.dependency_injection import get_container

def test_direct_api():
    """Call Claude API directly without system context."""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or "your-" in api_key.lower():
        print("[ERROR] ANTHROPIC_API_KEY not set in .env")
        return

    client = Anthropic(api_key=api_key)

    # Load prompt from test_prompts.json
    with open("src/prompts/test_prompts.json") as f:
        prompt = json.load(f)["comparison_prompt"]["text"]

    print("\n" + "=" * 70)
    print("DIRECT API TEST (No Context)")
    print("=" * 70)
    print(f"\nPrompt:\n{prompt}")
    print("\n" + "-" * 70)
    print("Calling Claude API...\n")

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract response
        response_text = response.content[0].text

        # Save response to file
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "test_output.txt", "w") as f:
            f.write(response_text)
        print(f"[INFO] Response saved to data/test_output.txt")

        # Token usage and cost
        print("\n" + "=" * 70)
        print("TOKEN USAGE & COST")
        print("=" * 70)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Use CostTracker for logging
        container = get_container()
        cost_tracker = container.get_cost_tracker()
        cost_tracker.log(
            model="claude-3-haiku-20240307",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source="api_script",
            context="none",
            prompt_preview=prompt
        )

        # Calculate and display costs
        input_cost_per_mtok = 0.25 / 1_000_000
        output_cost_per_mtok = 1.25 / 1_000_000

        input_cost = input_tokens * input_cost_per_mtok
        output_cost = output_tokens * output_cost_per_mtok
        total_cost = input_cost + output_cost

        print(f"Input tokens:     {input_tokens:,}")
        print(f"Output tokens:    {output_tokens:,}")
        print(f"Total tokens:     {total_tokens:,}")
        print(f"\nInput cost:       ${input_cost:.6f}")
        print(f"Output cost:      ${output_cost:.6f}")
        print(f"Total cost:       ${total_cost:.6f}")
        print(f"\n[INFO] Cost logged to data/costs.json")
        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_direct_api()
