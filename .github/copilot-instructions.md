# AI Coding Agent Instructions

## Project Overview
A Python application built with LangChain for AI-powered functionality. The codebase is organized into core business logic and service modules.

## Architecture

### Directory Structure
- **`src/main.py`**: Application entry point
- **`src/core/`**: Core business logic and domain models
- **`src/services/`**: Service layer for external integrations (e.g., LangChain chains, API clients)

### Key Design Patterns
1. **Separation of Concerns**: Core logic isolated from service dependencies
2. **Service Layer Pattern**: Services handle LangChain integration and external communication
3. **Modular Organization**: Each feature should have corresponding core logic and service layer

## LangChain Integration
- Services layer (`src/services/`) manages LangChain chains and models
- Keep chains and prompts organized by feature/domain
- Example structure when adding features:
  ```
  src/
  ├── services/
  │   └── document_processor.py    # LangChain chains for docs
  └── core/
      └── processors.py            # Business logic (without LangChain)
  ```

## Development Conventions
- Use type hints throughout the codebase
- Python 3.8+ features expected (f-strings, type hints, dataclasses)
- Virtual environment: `venv/` folder should contain dependencies

## Common Tasks

### Adding a New Feature
1. Create core logic in `src/core/` (business rules, data models)
2. Create service in `src/services/` for LangChain/external calls
3. Wire components in `src/main.py` or relevant handler

### Running the Application
- Activate virtual environment: `source venv/Scripts/activate` (Windows) or `venv/bin/activate` (Unix)
- Execute: `python src/main.py`

## Dependencies & Setup
- LangChain is the primary framework (check `venv/` for version)
- Create `requirements.txt` if not present for reproducible environments
- Consider adding config files: `pyproject.toml` or `setup.py` for packaging

## Important Notes
- Keep `src/main.py` as orchestration layer
- Avoid business logic in `main.py` - delegate to core/services
- Document LangChain chain purposes and configurations
