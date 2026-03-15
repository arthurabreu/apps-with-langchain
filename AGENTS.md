# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
source .venv/bin/activate
pip install -r requirements.txt

# Run the application (interactive CLI)
python src/main.py

# Run tests
python test_claude.py
```

Single test: there is no pytest config — run `python test_claude.py` directly.

Requires `.env` with `ANTHROPIC_API_KEY` set. Copy `.env.example` if present, or create `.env` manually.

## Architecture

This is a LangChain + Anthropic Claude application with an interactive CLI (`src/main.py`) for testing and comparing LLM models. The architecture follows SOLID principles with explicit dependency injection.

### Core Module (`src/core/`)

| File | Purpose |
|------|---------|
| `interfaces.py` | All abstract contracts (`ILanguageModel`, `ITokenManager`, `IModelFactory`, `IGenerationStrategy`, etc.) |
| `dependency_injection.py` | `DIContainer` — singleton/factory registration, accessed via `get_container()` |
| `services.py` | `ConfigurationManager`, `ApiKeyValidator`, `ConsoleUserInteraction`, `LoggingService` |
| `token_utils.py` | Token counting, cost estimation, usage logging to `token_usage.json` |
| `exceptions.py` | Custom exception hierarchy |
| `models/model_factory.py` | `ModelFactory` — registry-based creation; currently maps `"anthropic"` → `ClaudeModel` |
| `models/claude_model.py` | `ClaudeModel` implementing `ILanguageModel` via `langchain_anthropic.ChatAnthropic` |
| `strategies/` | `StandardGenerationStrategy` and `StreamingGenerationStrategy` — pluggable via constructor injection |

### Key Design Decisions

- **`main.py` is orchestration only** — no business logic there; delegate to `core/`.
- **Adding a new LLM provider**: implement `ILanguageModel`, register in `ModelFactory`, no other changes needed (Open/Closed).
- **Testing with mocks**: call `reset_container()`, then `container.register_instance(IUserInteraction, MockClass())` before instantiating models. The DI container is the seam for test isolation.
- **LangChain chains** should live in `src/services/` (feature-specific), not in `core/` (which holds provider-agnostic business logic).
- Token pricing constants are in `token_utils.py` and may need updating as Anthropic changes rates.
