"""Tests for the tool layer: sandboxing, validation, file edits, shell."""

from __future__ import annotations

import pytest

from agent.tools import ToolContext, ToolError, ToolInputError, Workspace, execute, validate_input
from agent.tools.files import TOOLS as FILE_TOOLS
from agent.tools.shell import TOOLS as SHELL_TOOLS

FILE_TOOLS_BY_NAME = {tool.name: tool for tool in FILE_TOOLS}
RUN_COMMAND = SHELL_TOOLS[0]


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=Workspace(tmp_path), approve=lambda command, description: True)


# ------------------------------------------------------------------ workspace

def test_relative_paths_resolve_inside_the_workspace(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace.resolve("notes/todo.md") == tmp_path / "notes" / "todo.md"


def test_escaping_the_workspace_is_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    with pytest.raises(ToolError):
        workspace.resolve("../secrets.txt")
    with pytest.raises(ToolError):
        workspace.resolve("a/b/../../../etc/passwd")
    with pytest.raises(ToolError):
        workspace.resolve("/etc/passwd")


def test_allow_outside_lifts_the_restriction(tmp_path):
    workspace = Workspace(tmp_path, allow_outside=True)
    assert workspace.resolve("/etc/hostname").as_posix() == "/etc/hostname"


def test_empty_path_is_rejected(tmp_path):
    with pytest.raises(ToolInputError):
        Workspace(tmp_path).resolve("   ")


# ----------------------------------------------------------------- validation

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "limit": {"type": "integer"},
        "flag": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_validate_accepts_a_well_formed_input():
    assert validate_input(SCHEMA, {"path": "a.txt", "limit": 5})["limit"] == 5


def test_validate_rejects_missing_required_field():
    with pytest.raises(ToolInputError, match="missing required parameter 'path'"):
        validate_input(SCHEMA, {"limit": 5})


def test_validate_rejects_a_truncated_non_object_input():
    # What a cut-off eagerly-streamed tool input can look like.
    with pytest.raises(ToolInputError):
        validate_input(SCHEMA, "{\"path\": \"a.tx")


def test_validate_rejects_wrong_types_and_unknown_keys():
    with pytest.raises(ToolInputError, match="must be a integer"):
        validate_input(SCHEMA, {"path": "a", "limit": "5"})
    with pytest.raises(ToolInputError, match="unknown parameter"):
        validate_input(SCHEMA, {"path": "a", "nope": 1})


def test_validate_does_not_accept_a_boolean_as_an_integer():
    with pytest.raises(ToolInputError, match="got boolean"):
        validate_input(SCHEMA, {"path": "a", "limit": True})


def test_validate_enforces_enums():
    with pytest.raises(ToolInputError, match="must be one of"):
        validate_input(SCHEMA, {"path": "a", "mode": "medium"})


# ---------------------------------------------------------------- file tools

def test_read_file_numbers_lines_and_pages(ctx, tmp_path):
    (tmp_path / "poem.txt").write_text("uno\ndos\ntres\ncuatro\n", encoding="utf-8")
    tool = FILE_TOOLS_BY_NAME["read_file"]

    text, failed = execute(tool, ctx, {"path": "poem.txt"})
    assert not failed
    assert "1\tuno" in text and "4\tcuatro" in text

    text, failed = execute(tool, ctx, {"path": "poem.txt", "offset": 2, "limit": 2})
    assert not failed
    assert "2\tdos" in text and "3\ttres" in text
    assert "cuatro" not in text
    assert "1 more below" in text


def test_read_file_reports_a_missing_file_as_an_error(ctx):
    text, failed = execute(FILE_TOOLS_BY_NAME["read_file"], ctx, {"path": "ghost.txt"})
    assert failed
    assert "no such file" in text


def test_read_file_refuses_binary(ctx, tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    text, failed = execute(FILE_TOOLS_BY_NAME["read_file"], ctx, {"path": "blob.bin"})
    assert failed
    assert "binary" in text


def test_write_file_creates_parent_directories(ctx, tmp_path):
    text, failed = execute(
        FILE_TOOLS_BY_NAME["write_file"], ctx, {"path": "docs/notas/a.md", "content": "hola"}
    )
    assert not failed
    assert (tmp_path / "docs" / "notas" / "a.md").read_text(encoding="utf-8") == "hola"
    assert "Created" in text


def test_edit_file_requires_a_unique_match(ctx, tmp_path):
    (tmp_path / "code.py").write_text("x = 1\ny = 1\n", encoding="utf-8")
    tool = FILE_TOOLS_BY_NAME["edit_file"]

    text, failed = execute(tool, ctx, {"path": "code.py", "old_text": "= 1", "new_text": "= 2"})
    assert failed
    assert "appears 2 times" in text

    text, failed = execute(tool, ctx, {"path": "code.py", "old_text": "x = 1", "new_text": "x = 2"})
    assert not failed
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 2\ny = 1\n"


def test_edit_file_replace_all(ctx, tmp_path):
    (tmp_path / "code.py").write_text("a\na\na\n", encoding="utf-8")
    text, failed = execute(
        FILE_TOOLS_BY_NAME["edit_file"],
        ctx,
        {"path": "code.py", "old_text": "a", "new_text": "b", "replace_all": True},
    )
    assert not failed
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "b\nb\nb\n"


def test_edit_file_missing_text_explains_itself(ctx, tmp_path):
    (tmp_path / "code.py").write_text("hola\n", encoding="utf-8")
    text, failed = execute(
        FILE_TOOLS_BY_NAME["edit_file"], ctx, {"path": "code.py", "old_text": "adiós", "new_text": "x"}
    )
    assert failed
    assert "not found" in text


def test_list_directory_hides_dotfiles_by_default(ctx, tmp_path):
    (tmp_path / "visible.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    tool = FILE_TOOLS_BY_NAME["list_directory"]

    text, failed = execute(tool, ctx, {})
    assert not failed
    assert "visible.txt" in text and "sub/" in text and ".hidden" not in text

    text, _ = execute(tool, ctx, {"show_hidden": True})
    assert ".hidden" in text


# --------------------------------------------------------------- shell tool

def test_run_command_returns_exit_code_and_output(ctx):
    text, failed = execute(RUN_COMMAND, ctx, {"command": "echo hola"})
    assert not failed
    assert "exit code: 0" in text
    assert "hola" in text


def test_run_command_reports_failures_without_raising(ctx):
    text, failed = execute(RUN_COMMAND, ctx, {"command": "exit 3"})
    assert not failed  # the tool worked; the command is what failed
    assert "exit code: 3" in text


def test_run_command_runs_in_the_workspace(ctx, tmp_path):
    text, _ = execute(RUN_COMMAND, ctx, {"command": "pwd"})
    assert str(tmp_path.resolve()) in text


def test_declined_command_is_not_executed(tmp_path):
    ctx = ToolContext(workspace=Workspace(tmp_path), approve=lambda command, description: False)
    text, failed = execute(RUN_COMMAND, ctx, {"command": "touch created.txt"})
    assert not failed
    assert "declined" in text
    assert not (tmp_path / "created.txt").exists()


def test_always_approve_skips_the_prompt(tmp_path):
    def refuse(command, description):  # pragma: no cover - must never be called
        raise AssertionError("approval should have been skipped")

    ctx = ToolContext(workspace=Workspace(tmp_path), approve=refuse, always_approve=True)
    _, failed = execute(RUN_COMMAND, ctx, {"command": "echo ok"})
    assert not failed


def test_run_command_times_out(ctx):
    text, failed = execute(RUN_COMMAND, ctx, {"command": "sleep 5", "timeout": 1})
    assert failed
    assert "timed out" in text
