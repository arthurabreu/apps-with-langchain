"""
API key validation service for the LangChain application.
"""

import logging

from ..exceptions import ApiKeyError


class ApiKeyValidator:
    """
    Validates API keys for different providers.
    Analogous to a Validator class in Android (e.g., EmailValidator).
    """

    def validate_key(self, api_key: str, provider: str) -> bool:
        """
        Validate an API key for a specific provider.

        Args:
            api_key: The string key to check.
            provider: 'anthropic' or 'huggingface'.

        Returns:
            True if key looks okay.

        Raises:
            ApiKeyError: If validation fails (caught in the UI layer).
        """
        if not api_key:
            raise ApiKeyError(f"API key not provided for {provider}")

        # Check for placeholder values (like developer forgot to set .env)
        placeholder_indicators = ["your-", "placeholder", "example", "test"]
        if any(indicator in api_key.lower() for indicator in placeholder_indicators):
            if not api_key.startswith(("claude-", "hf_")):  # Allow valid prefixes
                raise ApiKeyError(f"API key appears to be a placeholder for {provider}")

        # Basic format validation
        if provider.lower() == "anthropic" and len(api_key) < 20:
            raise ApiKeyError("Anthropic API key appears too short")
        elif provider.lower() == "huggingface" and not api_key.startswith("hf_"):
            # HuggingFace keys are optional for some models
            logging.warning("HuggingFace API key doesn't start with 'hf_' - may be invalid")

        return True
