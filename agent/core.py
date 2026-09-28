"""The agentic loop."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from .config import Config
from .prompts import build_system
from .tools import Tool, ToolContext, Workspace, build_registry, execute, tool_definitions


class UI(Protocol):
    """Everything the loop needs from the presentation layer."""

    def on_text(self, chunk: str) -> None: ...
    def on_thinking(self, chunk: str) -> None: ...
    def on_tool_start(self, name: str, summary: str) -> None: ...
    def on_tool_end(self, name: str, ok: bool) -> None: ...
    def on_server_tool(self, name: str) -> None: ...
    def on_notice(self, key: str, **kwargs: Any) -> None: ...
    def approve(self, command: str, description: str | None) -> bool: ...
    def turn_end(self) -> None: ...


@dataclass
class Usage:
    """Running token totals for the session."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def cost(self, pricing: dict[str, float] | None) -> float | None:
        if pricing is None:
            return None
        return (
            self.input_tokens * pricing["input"]
            + self.output_tokens * pricing["output"]
            + self.cache_read_tokens * pricing["cache_read"]
            + self.cache_write_tokens * pricing["cache_write"]
        ) / 1_000_000


class Agent:
    """Holds the conversation and drives request -> tools -> request."""

    def __init__(self, client: anthropic.Anthropic, config: Config, ui: UI) -> None:
        self.client = client
        self.config = config
        self.ui = ui
        self.messages: list[dict[str, Any]] = []
        self.usage = Usage()

        self.workspace = Workspace(config.workspace, config.allow_outside_workspace)
        self.registry: dict[str, Tool] = build_registry(enable_shell=config.enable_shell)
        self.ctx = ToolContext(
            workspace=self.workspace,
            approve=ui.approve,
            command_timeout=config.command_timeout,
            always_approve=config.auto_approve,
        )
        self._eager = config.eager_tool_input

    # ---------------------------------------------------------------- public

    def reset(self) -> None:
        self.messages.clear()

    def tool_names(self) -> list[str]:
        names = list(self.registry)
        if self.config.enable_web:
            names += ["web_search", "web_fetch"]
        return names

    def run(self, user_input: str) -> None:
        """Handle one user turn, looping over tool calls until the model stops."""
        self._repair_history()
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.config.max_steps):
            message = self._request()
            if message is None:
                return

            if message.content:
                self.messages.append({"role": "assistant", "content": message.content})
            self.usage.add(message.usage)

            stop = message.stop_reason
            if stop == "pause_turn":
                # A server tool hit its per-turn limit; resend to continue.
                continue
            if stop == "refusal":
                detail = ""
                details = getattr(message, "stop_details", None)
                if details is not None and getattr(details, "category", None):
                    detail = f" ({details.category})"
                self.ui.on_notice("refusal", detail=detail)
                return
            if stop == "max_tokens":
                self.ui.on_notice("truncated")
                return

            tool_uses = [b for b in message.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                return

            self.messages.append({"role": "user", "content": self._run_tools(tool_uses)})
        else:
            self.ui.on_notice("max_steps", n=self.config.max_steps)

    # --------------------------------------------------------------- request

    def _request(self) -> Any:
        params = self._request_params()
        try:
            return self._stream(params)
        except anthropic.BadRequestError as exc:
            # Some endpoints (older proxies, some Bedrock deployments) reject
            # eager tool-input streaming. Drop it once and retry.
            if self._eager and "eager_input_streaming" in str(exc):
                self._eager = False
                return self._stream(self._request_params())
            raise

    def _request_params(self) -> dict[str, Any]:
        today = _dt.date.today().isoformat()
        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": build_system(
                workspace=str(self.workspace.root),
                today=today,
                shell_enabled=self.config.enable_shell,
                web_enabled=self.config.enable_web,
            ),
            "messages": self.messages,
            "tools": tool_definitions(
                self.registry,
                model=self.config.model,
                enable_web=self.config.enable_web,
                eager_input_streaming=self._eager,
            ),
            "output_config": {"effort": self.config.effort},
            # Two cache breakpoints. The one inside `system` covers the frozen
            # tools + prompt prefix (~1.8K tokens); this one auto-caches the
            # last block of `messages`, so the conversation itself is read from
            # cache on the next step instead of being re-billed in full. That
            # matters far more: a turn with a dozen tool calls resends the whole
            # (growing) history on every one of them.
            "cache_control": {"type": "ephemeral"},
        }
        if self.config.show_thinking:
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
        return params

    def _stream(self, params: dict[str, Any]) -> Any:
        with self.client.messages.stream(**params) as stream:
            for event in stream:
                kind = getattr(event, "type", None)
                if kind == "text":
                    self.ui.on_text(event.text)
                elif kind == "thinking" and self.config.show_thinking:
                    self.ui.on_thinking(event.thinking)
                elif kind == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "server_tool_use":
                        self.ui.on_server_tool(getattr(block, "name", "web"))
            return stream.get_final_message()

    # ----------------------------------------------------------------- tools

    def _run_tools(self, tool_uses: list[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for block in tool_uses:
            tool = self.registry.get(block.name)
            if tool is None:
                results.append(self._result(block.id, f"Error: unknown tool '{block.name}'", True))
                continue

            self.ui.on_tool_start(tool.name, self._summarize(tool, block.input))
            text, failed = execute(tool, self.ctx, block.input)
            self.ui.on_tool_end(tool.name, not failed)
            results.append(self._result(block.id, text, failed))
        return results

    @staticmethod
    def _summarize(tool: Tool, raw_input: Any) -> str:
        if not isinstance(raw_input, dict):
            return ""
        try:
            return tool.summary(raw_input)
        except Exception:  # noqa: BLE001 - a bad summary must not break the call
            return ""

    @staticmethod
    def _result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        return result

    # --------------------------------------------------------------- history

    def _repair_history(self) -> None:
        """Close off tool calls left dangling by an interrupted turn.

        The API rejects a conversation where an assistant turn requested tools
        that never got results, so Ctrl-C mid-turn would otherwise poison the
        rest of the session.
        """
        if not self.messages:
            return
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return
        pending = [
            block
            for block in last.get("content", [])
            if getattr(block, "type", None) == "tool_use"
        ]
        if pending:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        self._result(block.id, "Error: interrupted by the user.", True)
                        for block in pending
                    ],
                }
            )
