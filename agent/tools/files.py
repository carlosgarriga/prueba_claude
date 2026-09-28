"""File tools: read, write, edit, list."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolError, truncate

MAX_READ_CHARS = 120_000
MAX_LIST_ENTRIES = 400
DEFAULT_READ_LIMIT = 2000

# Directories that are almost never what the user means when they say "list".
NOISY_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache"}


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        return False
    return b"\x00" in chunk


def read_file(ctx: ToolContext, params: dict[str, Any]) -> str:
    path = ctx.workspace.resolve(params["path"])
    if not path.exists():
        raise ToolError(f"no such file: {ctx.workspace.display(path)}")
    if path.is_dir():
        raise ToolError(f"{ctx.workspace.display(path)} is a directory; use list_directory")
    if _looks_binary(path):
        raise ToolError(f"{ctx.workspace.display(path)} looks like a binary file")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    offset = max(int(params.get("offset", 1)), 1)
    limit = max(int(params.get("limit", DEFAULT_READ_LIMIT)), 1)
    window = lines[offset - 1 : offset - 1 + limit]

    if not window:
        return f"{ctx.workspace.display(path)} has {len(lines)} lines; nothing at line {offset}."

    body = "\n".join(f"{offset + i}\t{line}" for i, line in enumerate(window))
    header = f"{ctx.workspace.display(path)} (lines {offset}-{offset + len(window) - 1} of {len(lines)})"
    remaining = len(lines) - (offset - 1 + len(window))
    if remaining > 0:
        header += f", {remaining} more below"
    return truncate(f"{header}\n{body}", MAX_READ_CHARS, "file")


def write_file(ctx: ToolContext, params: dict[str, Any]) -> str:
    path = ctx.workspace.resolve(params["path"])
    content = params["content"]
    if path.is_dir():
        raise ToolError(f"{ctx.workspace.display(path)} is a directory")

    existed = path.exists()
    previous = path.stat().st_size if existed else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    action = "Overwrote" if existed else "Created"
    detail = f" (was {previous} bytes)" if existed else ""
    return f"{action} {ctx.workspace.display(path)}{detail}: {len(content)} characters written."


def edit_file(ctx: ToolContext, params: dict[str, Any]) -> str:
    path = ctx.workspace.resolve(params["path"])
    old = params["old_text"]
    new = params["new_text"]

    if not path.exists():
        raise ToolError(f"no such file: {ctx.workspace.display(path)}")
    if old == new:
        raise ToolError("old_text and new_text are identical")
    if not old:
        raise ToolError("old_text must not be empty; use write_file to create a file")

    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    replace_all = bool(params.get("replace_all", False))

    if count == 0:
        raise ToolError(
            f"old_text not found in {ctx.workspace.display(path)}. "
            "Read the file and copy the exact text, including indentation."
        )
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_text appears {count} times in {ctx.workspace.display(path)}. "
            "Include more surrounding context to make it unique, or pass replace_all."
        )

    path.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1), encoding="utf-8")
    occurrences = count if replace_all else 1
    return f"Edited {ctx.workspace.display(path)}: replaced {occurrences} occurrence(s)."


def list_directory(ctx: ToolContext, params: dict[str, Any]) -> str:
    path = ctx.workspace.resolve(params.get("path") or ".")
    if not path.exists():
        raise ToolError(f"no such directory: {ctx.workspace.display(path)}")
    if not path.is_dir():
        raise ToolError(f"{ctx.workspace.display(path)} is a file, not a directory")

    show_hidden = bool(params.get("show_hidden", False))
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

    rows: list[str] = []
    hidden_count = 0
    for entry in entries:
        if entry.name.startswith(".") and not show_hidden:
            hidden_count += 1
            continue
        if entry.is_dir():
            marker = "/" if entry.name not in NOISY_DIRS else "/  (skipped by default)"
            rows.append(f"{entry.name}{marker}")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            rows.append(f"{entry.name}  ({size} bytes)")

    if not rows:
        return f"{ctx.workspace.display(path)} is empty."

    header = f"{ctx.workspace.display(path)} ({len(rows)} entries"
    if hidden_count:
        header += f", {hidden_count} hidden omitted"
    header += ")"
    listing = "\n".join(rows[:MAX_LIST_ENTRIES])
    if len(rows) > MAX_LIST_ENTRIES:
        listing += f"\n[{len(rows) - MAX_LIST_ENTRIES} more entries not shown]"
    return f"{header}\n{listing}"


TOOLS = [
    Tool(
        name="read_file",
        description=(
            "Read a text file from the working directory. Returns the contents with "
            "line numbers. Use offset/limit to page through a large file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "offset": {"type": "integer", "description": "First line to read (1-based). Default 1."},
                "limit": {"type": "integer", "description": "How many lines to read. Default 2000."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=read_file,
        summary=lambda p: str(p.get("path", "")),
    ),
    Tool(
        name="write_file",
        description=(
            "Write a text file, creating parent directories as needed. Overwrites the "
            "whole file -- read it first if it already exists, and prefer edit_file for "
            "changing part of a file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "content": {"type": "string", "description": "Full contents to write."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=write_file,
        summary=lambda p: str(p.get("path", "")),
    ),
    Tool(
        name="edit_file",
        description=(
            "Replace an exact string in an existing file. old_text must match the file "
            "byte for byte, including indentation, and must be unique unless replace_all "
            "is set. Read the file before editing it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "old_text": {"type": "string", "description": "Exact text to replace."},
                "new_text": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring a unique match.",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        handler=edit_file,
        summary=lambda p: str(p.get("path", "")),
    ),
    Tool(
        name="list_directory",
        description="List the files and subdirectories of a directory.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Defaults to the working directory."},
                "show_hidden": {"type": "boolean", "description": "Include dotfiles. Default false."},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=list_directory,
        summary=lambda p: str(p.get("path", ".")),
    ),
]
