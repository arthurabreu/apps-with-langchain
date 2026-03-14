"""
Context engine for Android projects.
Detects project structure, architecture, and generates markdown context.
"""

import re
from pathlib import Path
from xml.etree import ElementTree as ET


class ContextEngine:
    """Analyzes Android projects and generates context markdown."""

    def __init__(self, project_root: Path):
        """
        Initialize context engine for a project.

        Args:
            project_root: Path to Android project root
        """
        self.project_root = Path(project_root)
        self.context_file = self.project_root / "claude_init.md"

    def load_context(self) -> str:
        """
        Load project context from file or generate if missing.
        Invalidates stale cache containing com.example.app.

        Returns:
            Markdown context string
        """
        if self.context_file.exists():
            content = self.context_file.read_text(encoding="utf-8")
            if "com.example.app" not in content:
                return content
            self.context_file.unlink()

        return self.generate_context()

    def load_claude_md(self) -> str:
        """
        Load CLAUDE.md guidelines from project root.

        Returns:
            Content of CLAUDE.md, or empty string if not found
        """
        claude_md = self.project_root / "CLAUDE.md"
        if claude_md.exists():
            return claude_md.read_text(encoding="utf-8")
        return ""

    def generate_context(self) -> str:
        """
        Analyze project and generate context markdown.

        Returns:
            Markdown context string (also written to claude_init.md)
        """
        package = self._detect_package_name()
        modules = self._detect_modules()
        architecture = self._detect_architecture()
        source_tree = self._detect_source_tree()

        markdown = self._build_markdown(package, modules, architecture, source_tree)

        # Write for future reference
        self.context_file.write_text(markdown, encoding="utf-8")

        return markdown

    def _detect_package_name(self) -> str:
        """
        Detect package name from AndroidManifest.xml (legacy) or build.gradle.kts (modern AGP 7.0+).

        Returns:
            Package name or "com.example.app" if not found
        """
        manifest_path = self.project_root / "app" / "src" / "main" / "AndroidManifest.xml"
        if not manifest_path.exists():
            manifest_path = self.project_root / "src" / "main" / "AndroidManifest.xml"

        if manifest_path.exists():
            try:
                tree = ET.parse(manifest_path)
                root = tree.getroot()
                package = root.get("package")
                if package:
                    return package
            except Exception:
                pass

        package = self._detect_namespace_from_gradle()
        return package if package else "com.example.app"

    def _detect_namespace_from_gradle(self) -> str:
        """
        Detect namespace from app/build.gradle.kts or app/build.gradle.

        Returns:
            Namespace string or empty string if not found
        """
        gradle_kts = self.project_root / "app" / "build.gradle.kts"
        if gradle_kts.exists():
            try:
                content = gradle_kts.read_text(encoding="utf-8")
                match = re.search(r'namespace\s*=\s*"([^"]+)"', content)
                if match:
                    return match.group(1)
            except Exception:
                pass

        gradle = self.project_root / "app" / "build.gradle"
        if gradle.exists():
            try:
                content = gradle.read_text(encoding="utf-8")
                match = re.search(r"namespace\s*=\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    return match.group(1)
            except Exception:
                pass

        return ""

    def _detect_modules(self) -> list[str]:
        """
        Detect gradle modules (subdirs with build.gradle or build.gradle.kts).

        Returns:
            List of module names
        """
        modules = []
        for item in self.project_root.iterdir():
            if not item.is_dir() or item.name.startswith("."):
                continue

            if (item / "build.gradle").exists() or (item / "build.gradle.kts").exists():
                modules.append(item.name)

        return sorted(modules)

    def _detect_architecture(self) -> str:
        """
        Detect architecture patterns from Kotlin files.

        Returns:
            Architecture pattern (e.g., "MVVM + Compose", "MVP")
        """
        patterns = {
            "MVVM": r"class\s+\w+ViewModel|@HiltViewModel|LiveData|StateFlow",
            "MVP": r"class\s+\w+Presenter|interface\s+\w+View",
            "Compose": r"@Composable|@Preview|rememberState|Column|Row",
            "Fragment": r"class\s+\w+Fragment|Fragment\(\)",
        }

        found_patterns = set()

        # Scan Kotlin files
        for kt_file in self.project_root.rglob("*.kt"):
            if "build" in kt_file.parts:
                continue

            try:
                content = kt_file.read_text(encoding="utf-8", errors="ignore")
                for pattern_name, regex in patterns.items():
                    if re.search(regex, content):
                        found_patterns.add(pattern_name)
            except Exception:
                continue

        if found_patterns:
            return " + ".join(sorted(found_patterns))
        return "Unknown"

    def _detect_source_tree(self) -> str:
        """
        Detect existing Kotlin files in the project to show package structure.

        Returns:
            Formatted list of relative file paths
        """
        java_root = self.project_root / "app" / "src" / "main" / "java"
        if not java_root.exists():
            return "(no source files detected)"

        kt_files = []
        for kt_file in java_root.rglob("*.kt"):
            try:
                rel_path = kt_file.relative_to(self.project_root)
                kt_files.append(str(rel_path))
            except ValueError:
                pass

        if not kt_files:
            return "(no source files detected)"

        kt_files.sort()
        limited = kt_files[:8]
        file_list = "\n".join(f"- {f}" for f in limited)
        if len(kt_files) > 8:
            file_list += f"\n- ... and {len(kt_files) - 8} more files"

        return file_list

    def _build_markdown(self, package: str, modules: list[str], architecture: str, source_tree: str) -> str:
        """
        Build markdown context document.

        Args:
            package: Package name
            modules: List of module names
            architecture: Architecture pattern
            source_tree: Existing source file listing

        Returns:
            Formatted markdown string
        """
        module_list = "\n".join(f"- {m}" for m in modules) if modules else "- (none detected)"

        markdown = f"""# Android Project Context

## Package
{package}

## Modules
{module_list}

## Architecture
{architecture}

## Source Files (Existing Structure)
{source_tree}

## Key Files
- app/src/main/AndroidManifest.xml
- build.gradle or build.gradle.kts
"""
        return markdown
