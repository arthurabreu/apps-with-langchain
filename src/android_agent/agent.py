"""
Android code-gen agent orchestrator.
Single-shot agent using LangChain's new agents API.
"""

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

    def __init__(
        self,
        project_root: Path,
        model_provider: str,
        model_path_or_name: str,
        temperature: float = CODE_TEMPERATURE,
        max_tokens: int = INTERACTIVE_MAX_TOKENS,
        system_context: str = ""
    ):
        """
        Initialize Android agent.

        Args:
            project_root: Path to Android project
            model_provider: "claude" or "huggingface"
            model_path_or_name: Model name or path
            temperature: Generation temperature
            max_tokens: Max tokens for response
            system_context: Optional development context (expertise prompt)
        """
        self.project_root = Path(project_root)
        self.model_provider = model_provider
        self.model_path_or_name = model_path_or_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_context = system_context

    def run(self, task: str, claude_md_context: str = "") -> None:
        """
        Execute single-shot agent task.

        Args:
            task: Description of what to generate
            claude_md_context: Optional CLAUDE.md guidelines content
        """
        # 1. Load project context
        engine = ContextEngine(self.project_root)
        context = engine.load_context()

        # 2. Create file tools
        tools = make_file_tools(self.project_root)

        # 3. Build chat model
        chat_model = get_chat_model(
            self.model_provider,
            self.model_path_or_name,
            self.temperature,
            self.max_tokens
        )

        # 4. Build system prompt (f-string substitution, not template)
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

        system_prompt = self.SYSTEM_PROMPT.format(
            dev_context_section=dev_context_section,
            project_context=context,
            guidelines_section=guidelines_section
        )

        # 5. Create agent using new LangChain API
        agent_graph = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            debug=False
        )

        # 6. Initialize cost tracker
        cost_tracker = AgentCostTracker(model=self.model_path_or_name)

        # 7. Execute task
        try:
            input_state = {"messages": [{"role": "user", "content": task}]}

            # Stream and track tokens
            for event in agent_graph.stream(input_state):
                # Extract token usage from LLM responses if available
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

        # 8. Log costs
        cost_tracker.flush(task_preview=task[:100])

        # 9. Print completion
        print("\nJob Done")
        print(cost_tracker.summary())
