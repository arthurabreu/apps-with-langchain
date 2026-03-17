"""
Android code-gen agent orchestrator.
Single-shot agent using LangChain's new agents API.
Falls back to text-based generation + file parsing for local HuggingFace models
that don't support structured tool-calling.
"""

import re
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

from .provider_factory import get_chat_model
from .file_tools import make_file_tools
from .context_engine import ContextEngine
from .cost_tracker import AgentCostTracker
from src.core.config import CODE_TEMPERATURE, INTERACTIVE_MAX_TOKENS


class AndroidAgent:
    """Single-shot agent for Android code generation."""

    SYSTEM_PROMPT = """{dev_context_section}{guidelines_section}You are a file-system agent for an Android project.

    ## Project Context
    {project_context}
    ## Rules
    - Do NOT explain, chat, or produce any prose.
    - ONLY call the provided tools to write, read, or create files and directories.
    - BEFORE writing any file, call read_file on that path first.
    - If a file exists, prefer using write_file(path=..., content=..., mode="smart_merge")
      to append new functions or add composable routes at the bottom (end) of the file instead of overwriting.
    - If the file does not exist, use the default "overwrite" mode.
    - For Kotlin files, provide only the new functions or code blocks that need to be added.
    - Do NOT remove, rename, or restructure any existing code.
    - When ALL required files are written, respond with exactly: Success"""

    HF_PROMPT = """{dev_context_section}{guidelines_section}You are a code generator for an Android project using Kotlin and Jetpack Compose.

## Project Context
{project_context}

## Rules
- Output ONLY a single valid JSON object. No explanations, no prose.
- The JSON keys must be the relative file paths from the project root.
- The JSON values must be the complete, compilable Kotlin code for that file.
- Use Jetpack Compose for UI.
- Use the correct package name from the project context above.

## Task
{task}

## Response Format Example:
{{
  "app/src/main/java/com/example/MyFile.kt": "package com.example\n\nimport ...\n\nclass MyFile {{ ... }}",
  "app/src/main/res/layout/activity_main.xml": "..."
}}
"""

    def __init__(
        self,
        project_root: Path,
        model_provider: str,
        model_path_or_name: str,
        temperature: float = CODE_TEMPERATURE,
        max_tokens: int = INTERACTIVE_MAX_TOKENS,
        system_context: str = ""
    ):
        self.project_root = Path(project_root)
        self.model_provider = model_provider
        self.model_path_or_name = model_path_or_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_context = system_context

    def _build_context_sections(self, claude_md_context: str) -> tuple[str, str]:
        """Build dev_context_section and guidelines_section strings."""
        dev_context_section = ""
        if self.system_context.strip():
            dev_context_section = (
                "## Development Expertise\n"
                f"{self.system_context}\n\n"
                "---\n\n"
            )

        guidelines_section = ""
        if claude_md_context.strip():
            guidelines_section = (
                "## MANDATORY PROJECT GUIDELINES\n"
                "The following rules from CLAUDE.md MUST be followed exactly. "
                "Do NOT invent package names, DI patterns, or naming conventions — "
                "use only what is specified below.\n\n"
                f"{claude_md_context}\n\n"
                "---\n\n"
            )

        return dev_context_section, guidelines_section

    def run(self, task: str, claude_md_context: str = "") -> None:
        """Execute single-shot agent task."""
        engine = ContextEngine(self.project_root)
        context = engine.load_context()

        chat_model = get_chat_model(
            self.model_provider,
            self.model_path_or_name,
            self.temperature,
            self.max_tokens
        )

        dev_context_section, guidelines_section = self._build_context_sections(claude_md_context)

        if self.model_provider == "huggingface":
            self._run_hf_text_mode(chat_model, task, context, dev_context_section, guidelines_section)
        else:
            self._run_tool_agent(chat_model, task, context, dev_context_section, guidelines_section, claude_md_context)

    def _run_tool_agent(
        self, chat_model, task, context, dev_context_section, guidelines_section, claude_md_context
    ) -> None:
        """Run using LangChain's tool-calling agent (Claude and compatible models)."""
        tools = make_file_tools(self.project_root)

        system_prompt = self.SYSTEM_PROMPT.format(
            dev_context_section=dev_context_section,
            project_context=context,
            guidelines_section=guidelines_section
        )

        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            debug=False
        )

        cost_tracker = AgentCostTracker(model=self.model_path_or_name)

        try:
            input_state = {"messages": [{"role": "user", "content": task}]}

            for event in agent_graph.stream(input_state):
                for key, value in event.items():
                    if isinstance(value, dict) and "messages" in value:
                        for msg in value["messages"]:
                            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                                usage = msg.usage_metadata
                                cost_tracker.record(
                                    usage.get("input_tokens", 0),
                                    usage.get("output_tokens", 0)
                                )

        except Exception as e:
            print(f"Agent error: {e}")

        cost_tracker.flush(task_preview=task[:100])
        print("\nJob Done")
        print(cost_tracker.summary())

    def _run_hf_text_mode(
        self, chat_model, task, context, dev_context_section, guidelines_section
    ) -> None:
        """Run using text generation + JSON parsing for HuggingFace models."""
        prompt = self.HF_PROMPT.format(
            dev_context_section=dev_context_section,
            guidelines_section=guidelines_section,
            project_context=context,
            task=task
        )

        print("[INFO] Generating code with local model (JSON mode)...")
        print("[INFO] This may take a while depending on your hardware...\n")

        try:
            response = chat_model.invoke(prompt)

            # Extract text content from response
            if hasattr(response, "content"):
                text = response.content
            else:
                text = str(response)

            if not text or not text.strip():
                print("[ERROR] Model returned empty response.")
                return

            print("[INFO] Generation complete. Parsing output...\n")

            # Parse and write files
            files_written = self._parse_json_and_write_files(text)

            if files_written:
                print(f"\n[SUCCESS] Written {files_written} file(s) to {self.project_root}")
            else:
                print("\n[WARNING] No files could be parsed from model output.")
                print("[DEBUG] Raw model output (first 500 chars):")
                print(text[:500])

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")

        print("\nJob Done")

    def _parse_json_and_write_files(self, text: str) -> int:
        """
        Parse model output for JSON mapping and write to disk.

        Expected format:
            {
                "path/to/File.kt": "package ...\n\ncode"
            }

        Returns:
            Number of files written
        """
        import json
        
        # Try to find JSON block in the response
        json_content = text.strip()
        
        # Clean up Markdown markers if present (some models still wrap in ```json)
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(json_content)
            if not isinstance(data, dict):
                print(f"  [ERROR] Expected JSON object, got {type(data).__name__}")
                return 0
                
            files_written = 0
            for rel_path, content in data.items():
                rel_path = rel_path.strip().strip('"').strip("'")

                # Security: ensure path stays within project root
                full_path = (self.project_root / rel_path).resolve()
                try:
                    full_path.relative_to(self.project_root.resolve())
                except ValueError:
                    print(f"  [SKIP] Path escapes project root: {rel_path}")
                    continue

                full_path.parent.mkdir(parents=True, exist_ok=True)

                # Smart merge if file exists (appending extracted code)
                if full_path.exists():
                    existing = full_path.read_text(encoding="utf-8")
                    if content.strip() in existing:
                        print(f"  [SKIP] {rel_path} — content already present")
                        continue
                    # Append new content
                    separator = "\n\n" if not existing.endswith("\n\n") else ""
                    full_path.write_text(existing + separator + content.strip() + "\n", encoding="utf-8")
                    print(f"  [MERGED] {rel_path}")
                else:
                    full_path.write_text(content.strip() + "\n", encoding="utf-8")
                    print(f"  [CREATED] {rel_path}")

                files_written += 1
            
            return files_written

        except json.JSONDecodeError:
            print("  [ERROR] Failed to decode JSON from model output.")
            return 0

    def _infer_file_path(self, code: str, index: int) -> str:
        """
        Try to infer a file path from Kotlin code content.

        Args:
            code: Kotlin source code
            index: Fallback index for naming

        Returns:
            Inferred relative path or empty string
        """
        # Try to find package declaration
        pkg_match = re.search(r'^package\s+([\w.]+)', code, re.MULTILINE)
        # Try to find class/object name
        class_match = re.search(r'(?:class|object|fun)\s+(\w+)', code)

        if class_match:
            class_name = class_match.group(1)
            if pkg_match:
                pkg_path = pkg_match.group(1).replace(".", "/")
                return f"app/src/main/java/{pkg_path}/{class_name}.kt"
            return f"app/src/main/java/{class_name}.kt"

        return ""
