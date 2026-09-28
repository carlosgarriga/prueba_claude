"""End-to-end tests of the CLI, with the Anthropic client stubbed out."""

from __future__ import annotations

import pytest
from fakes import FakeClient, message, text_block, tool_block

from agent import cli
from agent.config import Config
from agent.core import Agent
from agent.i18n import Translator


@pytest.fixture(autouse=True)
def plain_output(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def stub_client(monkeypatch, responses):
    """Make `anthropic.Anthropic()` return a scripted fake."""
    holder = {}

    def factory(*args, **kwargs):
        holder["client"] = FakeClient(responses)
        return holder["client"]

    monkeypatch.setattr(cli.anthropic, "Anthropic", factory)
    return holder


def run_cli(monkeypatch, argv, responses, stdin_lines=None, tty=True):
    holder = stub_client(monkeypatch, responses)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: tty)
    if stdin_lines is not None:
        pending = list(stdin_lines)
        monkeypatch.setattr("builtins.input", lambda *a: pending.pop(0) if pending else exec("raise EOFError"))
    code = cli.main(argv)
    return code, holder.get("client")


# ------------------------------------------------------------------ one-shot

def test_a_task_on_the_command_line_runs_once_and_exits(monkeypatch, capsys, tmp_path):
    code, client = run_cli(
        monkeypatch,
        ["-C", str(tmp_path), "resume este repo"],
        [message([text_block("Aquí tienes el resumen.")])],
    )
    assert code == 0
    assert "Aquí tienes el resumen." in capsys.readouterr().out
    assert client.messages.calls[0]["messages"][0]["content"] == "resume este repo"


def test_a_piped_task_is_read_from_stdin(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.sys.stdin, "read", lambda: "summarise this repo\n")
    code, client = run_cli(
        monkeypatch,
        ["-C", str(tmp_path)],
        [message([text_block("Here is the summary.")])],
        tty=False,
    )
    assert code == 0
    assert "Here is the summary." in capsys.readouterr().out
    assert client.messages.calls[0]["messages"][0]["content"] == "summarise this repo"


def test_an_api_failure_exits_non_zero(monkeypatch, capsys, tmp_path):
    def exploding(*args, **kwargs):
        raise cli.anthropic.APIConnectionError(request=None)

    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda *a, **k: _ClientRaising(exploding))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)

    code = cli.main(["-C", str(tmp_path), "--lang", "es", "haz algo"])
    assert code == 1
    assert "conexión" in capsys.readouterr().err


class _ClientRaising:
    def __init__(self, fn):
        self.messages = type("M", (), {"stream": staticmethod(fn)})()


def test_missing_credentials_are_reported_not_traced(monkeypatch, capsys, tmp_path):
    def no_auth(*args, **kwargs):
        raise TypeError("Could not resolve authentication method. Expected one of api_key, ...")

    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda *a, **k: _ClientRaising(no_auth))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)

    code = cli.main(["-C", str(tmp_path), "--lang", "en", "do something"])
    assert code == 1
    assert "No Anthropic credentials found" in capsys.readouterr().err


def test_an_unrelated_type_error_still_propagates(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise TypeError("unsupported operand type(s)")

    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda *a, **k: _ClientRaising(boom))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)

    with pytest.raises(TypeError, match="unsupported operand"):
        cli.main(["-C", str(tmp_path), "do something"])


def test_a_bad_working_directory_is_rejected(monkeypatch, capsys, tmp_path):
    code = cli.main(["-C", str(tmp_path / "nope"), "hola"])
    assert code == 2
    assert "Not a directory" in capsys.readouterr().err


# ---------------------------------------------------------------------- repl

def test_the_repl_runs_turns_until_exit(monkeypatch, capsys, tmp_path):
    code, client = run_cli(
        monkeypatch,
        ["-C", str(tmp_path)],
        [message([text_block("uno")]), message([text_block("dos")])],
        stdin_lines=["primera", "segunda", "/exit"],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "uno" in out and "dos" in out
    assert len(client.messages.calls) == 2


def test_the_repl_keeps_conversation_history(monkeypatch, tmp_path):
    _, client = run_cli(
        monkeypatch,
        ["-C", str(tmp_path)],
        [message([text_block("hola")]), message([text_block("Carlos")])],
        stdin_lines=["me llamo Carlos", "¿cómo me llamo?", "/exit"],
    )
    second_request = client.messages.calls[1]["messages"]
    assert second_request[0]["content"] == "me llamo Carlos"
    assert second_request[-1]["content"] == "¿cómo me llamo?"


def test_blank_input_does_not_call_the_api(monkeypatch, tmp_path):
    _, client = run_cli(
        monkeypatch, ["-C", str(tmp_path)], [], stdin_lines=["", "   ", "/exit"]
    )
    assert client.messages.calls == []


# ----------------------------------------------------------- slash commands

def make_agent(tmp_path, responses=()):
    return Agent(FakeClient(list(responses)), Config(workspace=tmp_path), _NullUI())


class _NullUI:
    def on_text(self, chunk): pass
    def on_thinking(self, chunk): pass
    def on_tool_start(self, name, summary): pass
    def on_tool_end(self, name, ok): pass
    def on_server_tool(self, name): pass
    def on_notice(self, key, **kwargs): pass
    def approve(self, command, description): return False
    def turn_end(self): pass


def make_ui(lang="en"):
    return cli.TerminalUI(Translator(lang), interactive=True, auto_approve=False)


def test_exit_command_ends_the_session(tmp_path):
    assert cli.handle_command("/exit", make_ui(), make_agent(tmp_path)) is False
    assert cli.handle_command("/salir", make_ui("es"), make_agent(tmp_path)) is False


def test_clear_command_wipes_history(tmp_path):
    agent = make_agent(tmp_path)
    agent.messages.append({"role": "user", "content": "hola"})
    assert cli.handle_command("/clear", make_ui(), agent) is True
    assert agent.messages == []


def test_lang_command_switches_the_interface(tmp_path, capsys):
    ui = make_ui("en")
    cli.handle_command("/lang es", ui, make_agent(tmp_path))
    assert ui.t.lang == "es"
    assert "español" in capsys.readouterr().out


def test_lang_command_rejects_an_unsupported_language(tmp_path, capsys):
    ui = make_ui("en")
    cli.handle_command("/lang de", ui, make_agent(tmp_path))
    assert ui.t.lang == "en"
    assert "Unknown language 'de'" in capsys.readouterr().out


def test_tools_command_lists_the_tools(tmp_path, capsys):
    cli.handle_command("/tools", make_ui(), make_agent(tmp_path))
    out = capsys.readouterr().out
    assert "read_file" in out and "run_command" in out and "web_search" in out


def test_cost_command_reports_usage(tmp_path, capsys):
    agent = make_agent(tmp_path)
    agent.usage.input_tokens = 1_000_000
    agent.usage.output_tokens = 1_000_000
    cli.handle_command("/cost", make_ui(), agent)
    assert "$30.0000" in capsys.readouterr().out


def test_cost_command_without_a_price_list(tmp_path, capsys):
    agent = make_agent(tmp_path)
    agent.config.model = "some-custom-deployment"
    cli.handle_command("/cost", make_ui(), agent)
    assert "No price list" in capsys.readouterr().out


def test_an_unknown_command_shows_the_help(tmp_path, capsys):
    cli.handle_command("/wat", make_ui(), make_agent(tmp_path))
    assert "/help" in capsys.readouterr().out


# ------------------------------------------------------------------ approval

def test_approval_accepts_yes_in_both_languages(monkeypatch, tmp_path):
    ui = make_ui("es")
    for answer in ("y", "yes", "s", "sí", "SI"):
        monkeypatch.setattr("builtins.input", lambda *a, _a=answer: _a)
        assert ui.approve("ls", "lista archivos") is True


def test_approval_defaults_to_no(monkeypatch):
    ui = make_ui()
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert ui.approve("rm -rf /", None) is False


def test_always_answer_stops_asking(monkeypatch):
    ui = make_ui()
    monkeypatch.setattr("builtins.input", lambda *a: "a")
    assert ui.approve("ls", None) is True

    def refuse(*a):  # pragma: no cover - must not be called again
        raise AssertionError("the user should not be asked twice")

    monkeypatch.setattr("builtins.input", refuse)
    assert ui.approve("pwd", None) is True


def test_approval_is_denied_without_a_terminal(capsys):
    ui = cli.TerminalUI(Translator("en"), interactive=False, auto_approve=False)
    assert ui.approve("ls", None) is False
    assert "--yes" in capsys.readouterr().out


def test_yes_flag_approves_everything(capsys):
    ui = cli.TerminalUI(Translator("en"), interactive=False, auto_approve=True)
    assert ui.approve("rm -rf build", None) is True


def test_ctrl_c_at_the_prompt_declines(monkeypatch):
    ui = make_ui()

    def interrupt(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert ui.approve("ls", None) is False


# ----------------------------------------------------------------- arguments

def test_flags_map_onto_the_config(tmp_path):
    args = cli.build_parser().parse_args(
        ["-C", str(tmp_path), "--model", "claude-sonnet-5", "--effort", "low", "--lang", "es",
         "--yes", "--no-shell", "--no-web", "--no-thinking", "--allow-outside", "--max-steps", "7"]
    )
    config = cli.config_from_args(args)

    assert config.workspace == tmp_path.resolve()
    assert config.model == "claude-sonnet-5"
    assert config.effort == "low"
    assert config.lang == "es"
    assert config.auto_approve is True
    assert config.enable_shell is False
    assert config.enable_web is False
    assert config.show_thinking is False
    assert config.allow_outside_workspace is True
    assert config.max_steps == 7


def test_defaults_are_sane(tmp_path):
    config = cli.config_from_args(cli.build_parser().parse_args(["-C", str(tmp_path)]))
    assert config.model == "claude-opus-5"
    assert config.effort == "high"
    assert config.enable_shell and config.enable_web and config.show_thinking
    assert config.auto_approve is False
    assert config.allow_outside_workspace is False


def test_a_declined_command_flows_back_to_the_model(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    code, client = run_cli(
        monkeypatch,
        ["-C", str(tmp_path), "--lang", "es"],
        [
            message([tool_block("t1", "run_command", {"command": "rm -rf ."})], stop_reason="tool_use"),
            message([text_block("Entendido, no lo he ejecutado.")]),
        ],
        stdin_lines=["borra todo", "/exit"],
    )
    result = client.messages.calls[1]["messages"][2]["content"][0]
    assert "declined" in result["content"]
    assert "Entendido" in capsys.readouterr().out
