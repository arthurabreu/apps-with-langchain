"""
Global configuration and constants for the application.
All model names, tokens, temperatures, and other settings are defined here.
"""

# ============================================================================
# MODEL IDENTIFIERS
# ============================================================================

# Primary Claude model for interactive use
DEFAULT_MODEL = "claude-3-haiku-20240307"

# Available Claude models
CLAUDE_MODELS = {
    "opus": "claude-3-opus-20240229",
    "sonnet": "claude-3-5-sonnet-20240620",
    "haiku": "claude-3-haiku-20240307",
}

# ============================================================================
# TOKEN & CONTEXT WINDOWS
# ============================================================================

# Max tokens per model (input + output window)
MAX_TOKENS_PER_MODEL = {
    "claude-3-opus-20240229": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-haiku-20240307": 200000,
}

# Default max tokens for generation
DEFAULT_MAX_TOKENS = 512
INTERACTIVE_MAX_TOKENS = 4096

# ============================================================================
# MODEL PRICING (per 1M tokens)
# ============================================================================

PRICING = {
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

# Default temperature for deterministic responses
DEFAULT_TEMPERATURE = 0.2

# Temperature for creative responses
CREATIVE_TEMPERATURE = 0.7

# Temperature for precise code generation
CODE_TEMPERATURE = 0.1

# ============================================================================
# CONTEXT-SPECIFIC DEFAULTS
# ============================================================================

# Default config overrides per context
CONTEXT_DEFAULTS = {
    "default": {
        "model": DEFAULT_MODEL,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
    },
    "android": {
        "model": DEFAULT_MODEL,
        "max_tokens": INTERACTIVE_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
    },
    "python": {
        "model": DEFAULT_MODEL,
        "max_tokens": INTERACTIVE_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
    },
}

# ============================================================================
# FILE PATHS
# ============================================================================

TOKEN_USAGE_LOG = "token_usage.json"
COSTS_LOG = "data/costs.json"
RESPONSES_DIR = "responses"

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Anthropic API settings
ANTHROPIC_TIMEOUT = 60  # seconds
ANTHROPIC_MAX_RETRIES = 3

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = "INFO"
