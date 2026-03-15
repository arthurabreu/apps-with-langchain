"""
CLI entry point for the Android code-gen agent.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Make src.core importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load .env file from project root
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Set HF_TOKEN from HUGGINGFACE_API_KEY for HuggingFace library compatibility
hf_key = os.getenv("HUGGINGFACE_API_KEY")
if hf_key:
    os.environ["HF_TOKEN"] = hf_key

from .agent import AndroidAgent
from .provider_factory import _list_model_files
from .context_engine import ContextEngine
from .project_registry import ProjectRegistry
from src.core.prompt_manager import PromptManager


def _select_development_context() -> str:
    """
    Prompt user to select a development context from available prompts.

    Returns:
        System message string for the selected context, or empty string if skipped
    """
    try:
        workspace_root = Path(__file__).resolve().parents[2]
        prompts_file = workspace_root / "src" / "prompts" / "system_prompts.yaml"
        prompt_manager = PromptManager(str(prompts_file))
    except Exception:
        return ""

    prompts = prompt_manager.list_prompts()
    if not prompts:
        return ""

    print("\n" + "=" * 40)
    print("🎨 Development Context (optional):")
    print("=" * 40)
    options = list(prompts.items())
    for i, (key, description) in enumerate(options, 1):
        print(f"  {i}. {description}")
    print(f"  {len(options) + 1}. Skip\n")

    while True:
        choice = input("Select context [number] (or skip): ").strip()
        if choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                key = options[choice_idx][0]
                print(f"✓ Context selected: {options[choice_idx][1]}\n")
                return prompt_manager.get_system_message(key)
            elif choice_idx == len(options):
                print("✓ Skipping context\n")
                return ""
            else:
                print("Invalid selection. Try again.\n")
        else:
            print("Invalid input. Try again.\n")


def _prompt_new_directory() -> Path:
    """
    Prompt user for Android project directory and validate.

    Returns:
        Path to validated Android project directory
    """
    while True:
        project_path = input("Enter Android project root directory: ").strip()
        project_root = Path(project_path).expanduser()

        if not project_root.is_dir():
            print(f"Error: {project_root} is not a directory. Try again.\n")
            continue

        has_manifest = (project_root / "app" / "src" / "main" / "AndroidManifest.xml").exists()
        has_build = (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists()

        if not (has_manifest or has_build):
            print("Warning: Project doesn't look like an Android project (no AndroidManifest.xml or build.gradle)")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != "y":
                continue

        return project_root


def main():
    """Main CLI entry point."""
    print("\n" + "=" * 40)
    print("🤖 ANDROID CODE-GEN AGENT")
    print("=" * 40 + "\n")

    # 1. Project selection from registry or new directory
    registry = ProjectRegistry()
    projects = registry.get_projects()

    if projects:
        print("📁 Saved Projects:")
        for i, p in enumerate(projects, 1):
            print(f"  {i}. {p['name']}  ({p['path']})")
        print(f"  {len(projects) + 1}. Enter new directory\n")

        while True:
            choice = input("Select project [number]: ").strip()
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(projects):
                    project_root = Path(projects[choice_idx]["path"])
                    print(f"✓ Using project: {project_root}\n")
                    break
                elif choice_idx == len(projects):
                    project_root = _prompt_new_directory()
                    registry.add_project(project_root)
                    print()
                    break
                else:
                    print("Invalid selection. Try again.\n")
            else:
                print("Invalid input. Try again.\n")
    else:
        project_root = _prompt_new_directory()
        registry.add_project(project_root)
        print()

    # Load CLAUDE.md context for saved projects
    engine = ContextEngine(project_root)
    claude_md_context = engine.load_claude_md()

    # 1.5. Select development context
    system_context = _select_development_context()
    print()

    # 2. Choose model provider
    print("=" * 40)
    print("🚀 Model Provider:")
    print("=" * 40)
    print("1. Claude API")
    print("2. Local HuggingFace Model\n")

    while True:
        choice = input("Select [1 or 2]: ").strip()
        if choice == "1":
            model_provider = "claude"
            print()
            model_name = input("Enter model name (default: claude-3-haiku-20240307): ").strip()
            if not model_name:
                model_name = "claude-3-haiku-20240307"
            model_path_or_name = model_name
            print(f"✓ Selected Claude: {model_name}\n")
            break
        elif choice == "2":
            # Import HuggingFace model manager
            from src.core.hf_model_manager import select_hf_model

            model_provider = "huggingface"
            print()

            # Use new interactive model selection with download capability
            result = select_hf_model()

            if result is None:
                print("[INFO] Returning to model selection...\n")
                continue

            model_id, model_folder = result
            model_path_or_name = str(model_folder)
            print(f"✓ Selected: {model_id}\n")
            break
        else:
            print("Invalid choice. Enter 1 or 2.\n")

    # 3. Get task
    print("=" * 40)
    print("✍️  Task Description:")
    print("=" * 40)
    task = input("What would you like to generate? ").strip()
    if not task:
        print("Error: Task cannot be empty.")
        return
    print()

    # 4. Run agent
    print("=" * 40)
    print("⚡ Starting Generation")
    print("=" * 40)
    print(f"Provider: {model_provider}")
    print(f"Model: {model_path_or_name}")
    print(f"Task: {task}")
    print("=" * 40 + "\n")

    agent = AndroidAgent(
        project_root=project_root,
        model_provider=model_provider,
        model_path_or_name=model_path_or_name,
        system_context=system_context
    )

    agent.run(task, claude_md_context=claude_md_context)


if __name__ == "__main__":
    main()
