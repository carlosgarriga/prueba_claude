"""Shell tool: runs a command after the user approves it."""

from __future__ import annotations

import subprocess
from typing import Any

from .base import Tool, ToolContext, ToolError, truncate

MAX_OUTPUT_CHARS = 30_000


def run_command(ctx: ToolContext, params: dict[str, Any]) -> str:
    command = params["command"].strip()
    if not command:
        raise ToolError("command must not be empty")

    description = params.get("description")
    timeout = int(params.get("timeout") or ctx.command_timeout)
    timeout = max(1, min(timeout, 600))

    if not ctx.always_approve and not ctx.approve(command, description):
        # A refusal is a normal outcome, not a failure: let the model adapt.
        return "The user declined to run this command. Do not retry it as-is; propose an alternative or ask why."

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.workspace.root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout}s") from None
    except OSError as exc:
        raise ToolError(f"could not start the command: {exc}") from None

    parts = [f"exit code: {completed.returncode}"]
    if completed.stdout.strip():
        parts.append(f"stdout:\n{truncate(completed.stdout, MAX_OUTPUT_CHARS, 'stdout')}")
    if completed.stderr.strip():
        parts.append(f"stderr:\n{truncate(completed.stderr, MAX_OUTPUT_CHARS, 'stderr')}")
    if len(parts) == 1:
        parts.append("(no output)")
    return "\n\n".join(parts)


TOOLS = [
    Tool(
        name="run_command",
        description=(
            "Run a shell command in the working directory and return its exit code, "
            "stdout and stderr. The user is asked to approve each command, so explain "
            "anything destructive before calling this. Commands do not share state "
            "between calls: use `cmd1 && cmd2` rather than a separate `cd`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "description": {
                    "type": "string",
                    "description": "One short sentence, in the user's language, saying what the command does. Shown in the approval prompt.",
                },
                "timeout": {"type": "integer", "description": "Timeout in seconds (max 600)."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=run_command,
        summary=lambda p: str(p.get("command", ""))[:60],
    ),
]
