"""Runtime configuration for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Claude Opus 5 is the default. Override with --model or AGENT_MODEL.
DEFAULT_MODEL = "claude-opus-5"

# Streaming is always on, so we can afford a large output ceiling.
DEFAULT_MAX_TOKENS = 64000

# low | medium | high | xhigh | max
DEFAULT_EFFORT = "high"

# Safety valve: stop after this many model round-trips within a single turn.
DEFAULT_MAX_STEPS = 60

# Price per million tokens, used only for the local /cost estimate.
PRICING = {
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.40},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
}


@dataclass
class Config:
    """Everything the agent needs to know about how it should behave."""

    workspace: Path = field(default_factory=Path.cwd)
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str = DEFAULT_EFFORT
    max_steps: int = DEFAULT_MAX_STEPS

    # UI language: "en", "es", or None to auto-detect from the environment.
    lang: str | None = None

    # Capabilities
    enable_shell: bool = True
    enable_web: bool = True

    # Permissions
    auto_approve: bool = False
    allow_outside_workspace: bool = False

    # Command execution
    command_timeout: int = 120

    # Streaming tool inputs eagerly; disabled automatically if the endpoint
    # rejects the field (older proxies / Bedrock deployments).
    eager_tool_input: bool = True

    # Show a summary of the model's reasoning while it works.
    show_thinking: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        if model := os.environ.get("AGENT_MODEL"):
            cfg.model = model
        if lang := os.environ.get("AGENT_LANG"):
            cfg.lang = lang
        if effort := os.environ.get("AGENT_EFFORT"):
            cfg.effort = effort
        return cfg

    def pricing(self) -> dict[str, float] | None:
        return PRICING.get(self.model)
