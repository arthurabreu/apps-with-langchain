"""
Prompt management service for loading and retrieving system prompts.
Provides context selection for different development domains.
"""

import yaml
import json
from typing import Dict, Any
from pathlib import Path


class PromptManager:
    """Service for managing development context prompts."""

    def __init__(self, prompts_file: str = "src/prompts/system_prompts.yaml"):
        """
        Initialize the prompt manager.

        Args:
            prompts_file: Path to the YAML file containing prompt definitions
        """
        self._prompts: Dict[str, Any] = self._load(prompts_file)
        self.export_contexts_to_json()

    def _load(self, path: str) -> Dict[str, Any]:
        """
        Load prompts from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary of prompts keyed by identifier
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            return data.get("prompts", {})

    def list_prompts(self) -> Dict[str, str]:
        """
        Get a dictionary of available prompts for display in menus.

        Returns:
            Dictionary mapping prompt keys to their descriptions
        """
        return {k: v["description"] for k, v in self._prompts.items()}

    def get_system_message(self, key: str) -> str:
        """
        Get the system message for a specific prompt context.

        Args:
            key: Prompt identifier

        Returns:
            System message text

        Raises:
            KeyError: If prompt key not found
        """
        if key not in self._prompts:
            raise KeyError(f"Prompt '{key}' not found. Available: {list(self._prompts.keys())}")
        return self._prompts[key]["system_message"]

    def get_config_overrides(self, key: str) -> Dict[str, Any]:
        """
        Get optional configuration overrides (max_tokens, temperature) for a prompt.

        Args:
            key: Prompt identifier

        Returns:
            Dictionary with optional max_tokens and temperature overrides
        """
        entry = self._prompts.get(key, {})
        overrides = {}

        if "max_tokens" in entry:
            overrides["max_tokens"] = entry["max_tokens"]
        if "temperature" in entry:
            overrides["temperature"] = entry["temperature"]

        return overrides

    def get_prompt_name(self, key: str) -> str:
        """
        Get the display name for a prompt.

        Args:
            key: Prompt identifier

        Returns:
            Prompt display name
        """
        if key not in self._prompts:
            raise KeyError(f"Prompt '{key}' not found")
        return self._prompts[key]["name"]

    def export_contexts_to_json(self, output_dir: str = "src/prompts/contexts") -> None:
        """
        Export all system prompts to individual JSON files.

        Args:
            output_dir: Directory to write context JSON files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for key, data in self._prompts.items():
            context_file = output_path / f"{key}.json"
            context_data = {"key": key, **data}

            with open(context_file, "w") as f:
                json.dump(context_data, f, indent=2)
