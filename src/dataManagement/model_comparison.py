"""
Module for comparing different models.
"""

import os
from typing import List, Dict, Any
from datetime import datetime

class ModelComparison:
    """
    Compare different LLM models.
    """
    
    def __init__(self):
        self.results = []
        
    def run_comparison(self, prompts: List[Dict[str, str]] = None):
        """
        Run comparison with different models.
        
        Args:
            prompts: List of prompt dictionaries with 'name' and 'prompt' keys
        """
        if prompts is None:
            prompts = [
                {
                    "name": "Kotlin Palindrome",
                    "prompt": "Write a Kotlin function that checks if a string is a palindrome. Make it case-insensitive and ignore non-alphanumeric characters."
                },
                {
                    "name": "Kotlin Coroutine",
                    "prompt": "Write a Kotlin function that uses coroutines to fetch data from two APIs concurrently. Include error handling and timeouts."
                },
            ]
        
        print("\n" + "=" * 60)
        print("Model Comparison Test")
        print("=" * 60)
        
        models_to_test = []
        
        # Check which models are available
        try:
            from .langchain_huggingface_local import LocalHuggingFaceModel
            models_to_test.append(("Local Hugging Face", LocalHuggingFaceModel()))
        except Exception as e:
            print(f"[SKIP] Local model not available: {e}")
        
        try:
            from openai_model import OpenAIModel
            models_to_test.append(("OpenAI GPT", OpenAIModel()))
        except Exception as e:
            print(f"[SKIP] OpenAI model not available: {e}")
        
        try:
            from .claude_model import ClaudeModel
            models_to_test.append(("Anthropic Claude", ClaudeModel()))
        except Exception as e:
            print(f"[SKIP] Claude model not available: {e}")
        
        if not models_to_test:
            print("[ERROR] No models available for comparison!")
            return
        
        print(f"\n[INFO] Testing {len(models_to_test)} model(s)")
        
        # Test each model with each prompt
        for model_name, model in models_to_test:
            print(f"\n{'='*40}")
            print(f"Testing: {model_name}")
            print(f"{'='*40}")
            
            for prompt_info in prompts:
                print(f"\nPrompt: {prompt_info['name']}")
                print(f"Text: {prompt_info['prompt'][:80]}...")
                
                try:
                    start_time = datetime.now()
                    response = model.generate(prompt_info['prompt'])
                    end_time = datetime.now()
                    
                    duration = (end_time - start_time).total_seconds()
                    
                    print(f"Time: {duration:.2f}s")
                    print(f"Response (first 200 chars):\n{response[:200]}...")
                    
                    # Store results
                    self.results.append({
                        "model": model_name,
                        "prompt": prompt_info['name'],
                        "duration": duration,
                        "response_length": len(response),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    print(f"[ERROR] {e}")
        
        print("\n" + "=" * 60)
        print("Comparison Complete!")
        print("=" * 60)
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print comparison summary."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        
        # Group by model
        from collections import defaultdict
        model_stats = defaultdict(list)
        
        for result in self.results:
            model_stats[result['model']].append(result)
        
        for model, results in model_stats.items():
            avg_time = sum(r['duration'] for r in results) / len(results)
            avg_length = sum(r['response_length'] for r in results) / len(results)
            
            print(f"\n{model}:")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Average response length: {avg_length:.0f} chars")
            print(f"  Tests completed: {len(results)}")