"""
Project registry for saving and loading Android project paths.
Persists to data/android_projects.json.
"""

import json
from pathlib import Path


REGISTRY_FILE = Path("data") / "android_projects.json"


class ProjectRegistry:
    """Manages saved Android projects."""

    def __init__(self, registry_path: Path = REGISTRY_FILE):
        """
        Initialize project registry.

        Args:
            registry_path: Path to JSON registry file
        """
        self.registry_path = Path(registry_path)

    def get_projects(self) -> list[dict]:
        """
        Load and return all saved projects.

        Returns:
            List of {name, path} dicts
        """
        return self._load()

    def add_project(self, project_root: Path) -> None:
        """
        Add a project to the registry (skips duplicates by path).

        Args:
            project_root: Path to Android project
        """
        project_root = Path(project_root).resolve()
        name = project_root.name
        path_str = str(project_root)

        projects = self._load()

        # Skip if path already exists
        if any(p["path"] == path_str for p in projects):
            return

        projects.append({"name": name, "path": path_str})
        self._save(projects)

    def _load(self) -> list[dict]:
        """
        Load projects from JSON file.

        Returns:
            List of projects, or [] if file doesn't exist
        """
        if not self.registry_path.exists():
            return []

        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, projects: list[dict]) -> None:
        """
        Save projects to JSON file.

        Args:
            projects: List of {name, path} dicts
        """
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.registry_path, "w") as f:
            json.dump(projects, f, indent=2)
