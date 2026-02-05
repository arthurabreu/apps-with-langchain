"""
Module for working with Anthropic Claude models in LangChain.
"""

import os
from typing import Optional, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

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
    
    def generate(self, prompt: str, skip_prompt: bool = False, **kwargs) -> Optional[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Check if user wants to continue
            if not skip_prompt and not prompt_continue():
                print("[INFO] Generation skipped by user.")
                return None
                
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
            
            print("[SUCCESS] Generation complete!")
            return response.content
            
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