"""
Module for working with OpenAI models in LangChain.
Includes token tracking and usage analytics.
"""

import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from token_utils import TokenManager

class OpenAIModel:
    """
    A class to manage OpenAI models with LangChain.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.2,
                 max_tokens: int = 512):
        """
        Initialize the OpenAI model.
        
        Args:
            model_name: OpenAI model name (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize token manager
        self.token_manager = TokenManager()
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "your-" in api_key.lower():
            raise ValueError("OpenAI API key not configured. Add OPENAI_API_KEY to .env file")
        
        print(f"\n[INFO] Initializing OpenAI model: {self.model_name}")
        print(f"[INFO] Temperature: {temperature}")
        print(f"[INFO] Max tokens: {max_tokens}")
        
        # Initialize the model
        self._model = ChatOpenAI(
            model_name=self.model_name,
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
            
            prompt_tokens = self.token_manager.count_tokens(full_prompt, self.model_name)
            _, remaining, usage_percent = self.token_manager.check_context_window(full_prompt, self.model_name)
            est_cost = self.token_manager.estimate_cost(prompt_tokens, self.model_name, is_output=False)
            
            print("\n[TokenManager] Prompt Analysis:")
            print(f"- Input tokens: {prompt_tokens}")
            print(f"- Context window remaining: {remaining} tokens ({usage_percent:.2f}% used)")
            print(f"- Estimated input cost: ${est_cost:.6f}")
            
            print(f"\n[PROMPT] {prompt[:100]}...")
            print("[GENERATING] Using OpenAI...")
            
            # Create a simple prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("user", "{input}")
            ])
            
            # Format the prompt and generate
            formatted_prompt = prompt_template.format_messages(input=prompt)
            response = self._model.invoke(formatted_prompt)
            response_text = response.content
            
            # Token analysis after generation
            response_tokens = self.token_manager.count_tokens(response_text, self.model_name)
            response_cost = self.token_manager.estimate_cost(response_tokens, self.model_name, is_output=True)
            
            print("\n[TokenManager] Response Analysis:")
            print(f"- Output tokens: {response_tokens}")
            print(f"- Estimated output cost: ${response_cost:.6f}")
            
            # Log usage
            self.token_manager.log_usage(self.model_name, prompt_tokens, "openai_prompt", is_output=False)
            self.token_manager.log_usage(self.model_name, response_tokens, "openai_response", is_output=True)
            
            # Show summary
            summary = self.token_manager.get_usage_summary()
            print("\n[TokenManager] Session Summary:")
            print(f"- Total tokens used: {summary['total_tokens']}")
            print(f"- Total estimated cost: ${summary['total_cost']:.6f}")
            
            print("[SUCCESS] Generation complete!")
            return response_text
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": "OpenAI",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "status": "ready"
        }