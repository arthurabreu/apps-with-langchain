"""
Module for working with Anthropic Claude models in LangChain.
"""

import os
from typing import Optional, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from token_utils import TokenManager

class ClaudeModel:
    """
    A class to manage Claude models with LangChain.
    """
    
    def __init__(self, 
                 model_name: str = "claude-3-haiku-20240307",
                 temperature: float = 0.2,
                 max_tokens: int = 512):
        """
        Initialize the Claude model.
        
        Args:
            model_name: Claude model name
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize token manager
        self.token_manager = TokenManager()
        
        # Check API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or "your-" in api_key.lower():
            raise ValueError("Anthropic API key not configured. Add ANTHROPIC_API_KEY to .env file")
        
        print(f"\n[INFO] Initializing Claude model: {self.model_name}")
        print(f"[INFO] Temperature: {temperature}")
        print(f"[INFO] Max tokens: {max_tokens}")
        
        # Initialize the model
        self._model = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt with token tracking.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Token analysis before generation
            system_msg = "You are a helpful Kotlin programming assistant."
            full_prompt = f"{system_msg}\n{prompt}"
            
            # Try to count tokens - use a fallback model if Claude model isn't supported
            try:
                prompt_tokens = self.token_manager.count_tokens(full_prompt, self.model_name)
            except:
                # Fallback to GPT-4 tokenizer for estimation
                prompt_tokens = self.token_manager.count_tokens(full_prompt, "gpt-4")
            
            print("\n[TokenManager] Prompt Analysis:")
            print(f"- Input tokens: {prompt_tokens}")
            print(f"- Model: {self.model_name}")
            print("- Cost estimation: Not available for Claude models")
            
            print(f"\n[PROMPT] {prompt[:100]}...")
            print("[GENERATING] Using Claude...")
            
            # Create a simple prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful Kotlin programming assistant."),
                ("user", "{input}")
            ])
            
            # Format the prompt
            formatted_prompt = prompt_template.format_messages(input=prompt)
            
            # Generate response
            response = self._model.invoke(formatted_prompt)
            response_text = response.content
            
            # Token analysis after generation
            try:
                response_tokens = self.token_manager.count_tokens(response_text, self.model_name)
            except:
                # Fallback to GPT-4 tokenizer for estimation
                response_tokens = self.token_manager.count_tokens(response_text, "gpt-4")
            
            print("\n[TokenManager] Response Analysis:")
            print(f"- Output tokens: {response_tokens}")
            print("- Cost estimation: Not available for Claude models")
            
            # Log usage (using fallback model name for compatibility)
            fallback_model = "gpt-4"  # Use for logging since Claude models aren't in TokenManager
            self.token_manager.log_usage(fallback_model, prompt_tokens, "claude_prompt", is_output=False)
            self.token_manager.log_usage(fallback_model, response_tokens, "claude_response", is_output=True)
            
            # Show session summary
            summary = self.token_manager.get_usage_summary()
            print("\n[TokenManager] Session Summary:")
            print(f"- Total tokens used: {summary['total_tokens']}")
            print(f"- Total estimated cost: Not available for Claude models")
            
            print("[SUCCESS] Generation complete!")
            return response_text
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": "Anthropic",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "status": "ready"
        }