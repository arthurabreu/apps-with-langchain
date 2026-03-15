"""
File operation tools for the Android code-gen agent.
Closed over project_root for security and convenience.
"""

from pathlib import Path
from langchain_core.tools import tool


def make_file_tools(project_root: Path) -> list:
    """
    Create tool-decorated file operation functions scoped to project_root.

    Args:
        project_root: Base directory for all operations

    Returns:
        List of [write_file, make_directory, read_file, list_directory] tools
    """
    project_root = Path(project_root).resolve()

    def _check_path_safety(path: str) -> Path:
        """
        Ensure path is within project_root, resolve to absolute.

        Args:
            path: Relative path from project_root

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path escapes project_root
        """
        full_path = (project_root / path).resolve()

        # Security check: path must be under project_root
        try:
            full_path.relative_to(project_root)
        except ValueError:
            raise ValueError(f"Path escapes project root: {path}")

        return full_path

    @tool
    def write_file(path: str, content: str, mode: str = "overwrite") -> str:
        """
        Write content to a file relative to the Android project root.
        
        Args:
            path: Relative path from project root
            content: Content to write
            mode: "overwrite" (default) or "smart_merge"
        """
        try:
            full_path = _check_path_safety(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if mode == "smart_merge" and full_path.exists():
                existing_content = full_path.read_text(encoding="utf-8")
                
                # If content is already in existing_content, do nothing
                if content.strip() in existing_content:
                    return f"OK: {path} already contains the content"
                
                # Simple append for all files when in smart_merge mode
                # Ensure there's a newline between existing and new
                separator = "\n\n"
                if existing_content.endswith("\n\n"):
                    separator = ""
                elif existing_content.endswith("\n"):
                    separator = "\n"
                elif not existing_content:
                    separator = ""
                
                full_path.write_text(existing_content + separator + content.strip() + "\n", encoding="utf-8")
                return f"OK: appended to {path}"

            full_path.write_text(content, encoding="utf-8")
            return f"OK: wrote {path}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    @tool
    def make_directory(path: str) -> str:
        """Create a directory relative to the Android project root."""
        try:
            full_path = _check_path_safety(path)
            full_path.mkdir(parents=True, exist_ok=True)
            return f"OK: created directory {path}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    @tool
    def read_file(path: str) -> str:
        """Read a file relative to the Android project root."""
        try:
            full_path = _check_path_safety(path)
            if not full_path.exists():
                return f"ERROR: File not found: {path}"
            content = full_path.read_text(encoding="utf-8")
            return content
        except Exception as e:
            return f"ERROR: {str(e)}"

    @tool
    def list_directory(path: str) -> str:
        """List contents of a directory relative to the Android project root."""
        try:
            full_path = _check_path_safety(path)
            if not full_path.is_dir():
                return f"ERROR: Not a directory: {path}"

            items = sorted([item.name for item in full_path.iterdir()])
            return "\n".join(items) if items else "(empty directory)"
        except Exception as e:
            return f"ERROR: {str(e)}"

    return [write_file, make_directory, read_file, list_directory]
