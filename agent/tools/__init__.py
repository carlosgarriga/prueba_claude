"""Tool registry."""

from __future__ import annotations

from typing import Any

from . import files, shell, web
from .base import (
    Approver,
    Tool,
    ToolContext,
    ToolError,
    ToolInputError,
    Workspace,
    execute,
    truncate,
    validate_input,
)

__all__ = [
    "Approver",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolInputError",
    "Workspace",
    "build_registry",
    "execute",
    "tool_definitions",
    "truncate",
    "validate_input",
    "web",
]


def build_registry(enable_shell: bool = True) -> dict[str, Tool]:
    """Client-side tools, keyed by name. Order is stable for cache reuse."""
    tools: list[Tool] = list(files.TOOLS)
    if enable_shell:
        tools.extend(shell.TOOLS)
    return {tool.name: tool for tool in tools}


def tool_definitions(
    registry: dict[str, Tool],
    model: str,
    enable_web: bool = True,
    eager_input_streaming: bool = False,
) -> list[dict[str, Any]]:
    """The `tools` payload: client tools first, then server tools."""
    definitions: list[dict[str, Any]] = [
        tool.definition(eager_input_streaming) for tool in registry.values()
    ]
    if enable_web:
        definitions.extend(web.definitions(model))
    return definitions
