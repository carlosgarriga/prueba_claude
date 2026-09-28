"""Server-side web tools.

These run on Anthropic's infrastructure -- there is nothing to execute
locally, we only declare them. The `_20260209` variants include dynamic
filtering and require Opus 4.6 / Sonnet 4.6 or newer.
"""

from __future__ import annotations

from typing import Any

WEB_SEARCH: dict[str, Any] = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 10,
}

WEB_FETCH: dict[str, Any] = {
    "type": "web_fetch_20260209",
    "name": "web_fetch",
    "max_uses": 10,
    "citations": {"enabled": True},
}

# Models older than Opus 4.6 / Sonnet 4.6 only support the basic variants.
LEGACY_WEB_SEARCH: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 10}
LEGACY_WEB_FETCH: dict[str, Any] = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 10}

_MODERN_MODEL_PREFIXES = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def definitions(model: str) -> list[dict[str, Any]]:
    if model.startswith(_MODERN_MODEL_PREFIXES):
        return [WEB_SEARCH, WEB_FETCH]
    return [LEGACY_WEB_SEARCH, LEGACY_WEB_FETCH]


# Result block types the CLI should render as "the agent used the web".
RESULT_BLOCK_TYPES = frozenset({"web_search_tool_result", "web_fetch_tool_result"})
