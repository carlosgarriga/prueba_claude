"""Tool plumbing: definitions, input validation, execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class ToolError(Exception):
    """A tool failed in a way the model should see and can recover from."""


class ToolInputError(ToolError):
    """The model sent arguments that don't match the tool's schema."""


class Approver(Protocol):
    """Asks the human whether a command may run."""

    def __call__(self, command: str, description: str | None) -> bool: ...


class Workspace:
    """Resolves paths, keeping them inside the working directory by default."""

    def __init__(self, root: Path, allow_outside: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.allow_outside = allow_outside

    def resolve(self, raw: str) -> Path:
        if not raw or not raw.strip():
            raise ToolInputError("path must not be empty")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # resolve() without strict=True so we can also target files yet to be
        # created; it still collapses any '..' used to climb out.
        resolved = candidate.resolve()
        if not self.allow_outside and not self._inside(resolved):
            raise ToolError(
                f"path '{raw}' is outside the working directory ({self.root}). "
                "Only paths inside it are allowed."
            )
        return resolved

    def _inside(self, path: Path) -> bool:
        return path == self.root or self.root in path.parents

    def display(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


@dataclass
class ToolContext:
    """Shared state handed to every tool handler."""

    workspace: Workspace
    approve: Approver
    command_timeout: int = 120
    always_approve: bool = False


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], str]
    summary: Callable[[dict[str, Any]], str] = field(
        default=lambda params: ", ".join(f"{k}={v!r}" for k, v in list(params.items())[:2])
    )

    def definition(self, eager_input_streaming: bool = False) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        if eager_input_streaming:
            # Streaming tool inputs means the SDK no longer guarantees a
            # complete, coerced input -- validate_input() below is what makes
            # that safe.
            spec["eager_input_streaming"] = True
        return spec


_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_input(schema: dict[str, Any], value: Any) -> dict[str, Any]:
    """Check a tool input against its schema.

    Necessary because eager input streaming can deliver a truncated or
    partially-parsed object instead of raising.
    """
    if not isinstance(value, dict):
        raise ToolInputError("tool input must be a JSON object")

    properties: dict[str, Any] = schema.get("properties", {})

    for key in schema.get("required", []):
        if key not in value or value[key] is None:
            raise ToolInputError(f"missing required parameter '{key}'")

    for key, item in value.items():
        spec = properties.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                raise ToolInputError(f"unknown parameter '{key}'")
            continue
        if item is None:
            continue
        expected = spec.get("type")
        if expected is None:
            continue
        python_type = _JSON_TYPES.get(expected)
        if python_type is None:
            continue
        # bool is a subclass of int in Python; don't let True pass as an integer.
        if expected in ("integer", "number") and isinstance(item, bool):
            raise ToolInputError(f"parameter '{key}' must be a {expected}, got boolean")
        if not isinstance(item, python_type):
            raise ToolInputError(
                f"parameter '{key}' must be a {expected}, got {type(item).__name__}"
            )
        if expected == "string" and spec.get("enum") and item not in spec["enum"]:
            allowed = ", ".join(repr(option) for option in spec["enum"])
            raise ToolInputError(f"parameter '{key}' must be one of: {allowed}")

    return value


def execute(tool: Tool, ctx: ToolContext, raw_input: Any) -> tuple[str, bool]:
    """Run a tool. Returns (result_text, is_error) -- never raises."""
    try:
        params = validate_input(tool.input_schema, raw_input)
        result = tool.handler(ctx, params)
        return (result if result.strip() else "(no output)"), False
    except ToolError as exc:
        return f"Error: {exc}", True
    except Exception as exc:  # noqa: BLE001 - the model handles the failure
        return f"Error: {tool.name} failed unexpectedly: {type(exc).__name__}: {exc}", True


def truncate(text: str, limit: int, note: str = "output") -> str:
    """Clip long tool output, telling the model what it lost."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n\n[{note} truncated: {dropped} more characters]"
