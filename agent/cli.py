"""Command-line interface: argument parsing, the REPL, and rendering."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import anthropic

from .config import Config
from .core import Agent
from .i18n import SUPPORTED, Translator

try:  # optional: arrow-key history at the prompt
    import readline  # noqa: F401
except ImportError:  # pragma: no cover - Windows without pyreadline
    pass


# --------------------------------------------------------------------- colour

class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)


def _colour_enabled(stream: Any) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


# ------------------------------------------------------------------- terminal

class TerminalUI:
    """Renders the agent's work to a terminal and asks for approvals."""

    def __init__(self, translator: Translator, interactive: bool, auto_approve: bool) -> None:
        self.t = translator
        self.interactive = interactive
        self.auto_approve = auto_approve
        self.colour = Palette(_colour_enabled(sys.stdout))
        self.always_approve = auto_approve
        self._in_thinking = False
        self._line_open = False

    # -- streaming output

    def on_text(self, chunk: str) -> None:
        self._close_thinking()
        sys.stdout.write(chunk)
        sys.stdout.flush()
        self._line_open = not chunk.endswith("\n")

    def on_thinking(self, chunk: str) -> None:
        if not self._in_thinking:
            self._break_line()
            sys.stdout.write(self.colour.dim(f"[{self.t('thinking_label')}] "))
            self._in_thinking = True
        sys.stdout.write(self.colour.dim(chunk))
        sys.stdout.flush()

    def _close_thinking(self) -> None:
        if self._in_thinking:
            sys.stdout.write("\n\n")
            self._in_thinking = False
            self._line_open = False

    def _break_line(self) -> None:
        if self._line_open:
            sys.stdout.write("\n")
            self._line_open = False

    # -- tool activity

    def on_tool_start(self, name: str, summary: str) -> None:
        self._close_thinking()
        self._break_line()
        label = self.t("tool_running", name=name, summary=summary)
        sys.stdout.write(self.colour.cyan(f"  · {label} "))
        sys.stdout.flush()
        self._line_open = True

    def on_tool_end(self, name: str, ok: bool) -> None:
        mark = self.colour.green(self.t("tool_ok")) if ok else self.colour.red(self.t("tool_failed"))
        sys.stdout.write(f"{mark}\n")
        sys.stdout.flush()
        self._line_open = False

    def on_server_tool(self, name: str) -> None:
        self._close_thinking()
        self._break_line()
        sys.stdout.write(self.colour.cyan(f"  · {name}\n"))
        sys.stdout.flush()

    def on_notice(self, key: str, **kwargs: Any) -> None:
        self._close_thinking()
        self._break_line()
        sys.stdout.write(self.colour.yellow(f"  {self.t(key, **kwargs)}\n"))
        sys.stdout.flush()

    def turn_end(self) -> None:
        self._close_thinking()
        self._break_line()
        sys.stdout.write("\n")
        sys.stdout.flush()

    # -- approvals

    def approve(self, command: str, description: str | None) -> bool:
        if self.always_approve:
            return True
        self._close_thinking()
        self._break_line()

        if not self.interactive:
            self.on_notice("denied_noninteractive")
            return False

        print()
        print(self.colour.bold(self.t("confirm_command")))
        if description:
            print(f"  {description}")
        for line in command.splitlines():
            print(self.colour.yellow(f"  $ {line}"))
        print(self.colour.dim(f"  {self.t('confirm_choices')}"))

        try:
            answer = input(f"  {self.t('confirm_prompt')}> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = "n"

        # Accept the natural first letter in both languages: y/yes/sí/s, a/always.
        if answer in ("a", "always", "siempre"):
            self.always_approve = True
            return True
        if answer in ("y", "yes", "s", "si", "sí", "ok"):
            return True
        self.on_notice("denied")
        return False

    # -- plain messages

    def say(self, text: str) -> None:
        print(text)

    def error(self, text: str) -> None:
        print(self.colour.red(text), file=sys.stderr)


# -------------------------------------------------------------- slash commands

def _print_help(ui: TerminalUI) -> None:
    t = ui.t
    rows = [
        ("/help", t("help_help")),
        ("/clear", t("help_clear")),
        ("/lang <en|es>", t("help_lang")),
        ("/tools", t("help_tools")),
        ("/cost", t("help_cost")),
        ("/exit", t("help_exit")),
    ]
    width = max(len(name) for name, _ in rows)
    ui.say(ui.colour.bold(t("help_title")))
    for name, text in rows:
        ui.say(f"  {name.ljust(width)}  {ui.colour.dim(text)}")
    ui.say("")
    ui.say(ui.colour.dim(t("help_language_note")))


def _print_cost(ui: TerminalUI, agent: Agent) -> None:
    t = ui.t
    usage = agent.usage
    pricing = agent.config.pricing()
    ui.say(ui.colour.bold(t("cost_title")))
    ui.say(f"  {t('cost_input')}: {usage.input_tokens:,}")
    ui.say(f"  {t('cost_output')}: {usage.output_tokens:,}")
    ui.say(f"  {t('cost_cache_read')}: {usage.cache_read_tokens:,}")
    ui.say(f"  {t('cost_cache_write')}: {usage.cache_write_tokens:,}")
    cost = usage.cost(pricing)
    if cost is None:
        ui.say(ui.colour.dim(f"  {t('cost_unknown_model', model=agent.config.model)}"))
    else:
        ui.say(f"  {t('cost_total')}: ${cost:.4f}")


def handle_command(line: str, ui: TerminalUI, agent: Agent) -> bool:
    """Run a slash command. Returns False when the session should end."""
    parts = line.strip().split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "/salir"):
        ui.say(ui.t("goodbye"))
        return False
    if command in ("/help", "/ayuda", "/?"):
        _print_help(ui)
    elif command in ("/clear", "/limpiar"):
        agent.reset()
        ui.say(ui.colour.dim(ui.t("cleared")))
    elif command in ("/lang", "/idioma"):
        if ui.t.set_language(argument):
            ui.say(ui.colour.dim(ui.t("lang_set")))
        else:
            ui.say(ui.t("lang_unknown", lang=argument, options="/".join(SUPPORTED)))
    elif command in ("/tools", "/herramientas"):
        ui.say(ui.colour.bold(ui.t("tools_title")))
        for name in agent.tool_names():
            ui.say(f"  {name}")
    elif command in ("/cost", "/coste"):
        _print_cost(ui, agent)
    else:
        _print_help(ui)
    return True


# ------------------------------------------------------------------ execution

def run_turn(agent: Agent, ui: TerminalUI, text: str) -> bool:
    """One user turn, with every expected failure translated for the user.

    Returns False if the turn ended in an error.
    """
    try:
        agent.run(text)
        return True
    except KeyboardInterrupt:
        ui.on_notice("interrupted")
    except anthropic.AuthenticationError:
        ui.error(ui.t("no_api_key"))
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "60")
        ui.error(ui.t("rate_limited", seconds=retry_after))
    except anthropic.APIConnectionError:
        ui.error(ui.t("connection_error"))
    except anthropic.APIStatusError as exc:
        ui.error(ui.t("api_error", detail=f"{exc.status_code} {exc.message}"))
    except TypeError as exc:
        # With no credentials at all the SDK raises a TypeError when it builds
        # the request, not an AuthenticationError when the client is created.
        if "authentication" not in str(exc).lower():
            raise
        ui.error(ui.t("no_api_key"))
    finally:
        ui.turn_end()
    return False


def repl(agent: Agent, ui: TerminalUI) -> int:
    ui.say(ui.colour.bold(ui.t("banner")))
    ui.say(ui.colour.dim(ui.t("workspace", path=agent.workspace.root)))
    ui.say(ui.colour.dim(ui.t("banner_hint")))
    ui.say("")

    while True:
        try:
            line = input(ui.colour.bold(f"{ui.t('prompt')}> "))
        except (EOFError, KeyboardInterrupt):
            ui.say("")
            ui.say(ui.t("goodbye"))
            return 0

        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            if not handle_command(text, ui, agent):
                return 0
            ui.say("")
            continue

        run_turn(agent, ui, text)


# ------------------------------------------------------------------ arguments

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description=(
            "A bilingual (English/Spanish) AI agent that completes tasks using files, "
            "shell commands and web search.\n"
            "Un agente de IA bilingüe (español/inglés) que completa tareas usando "
            "archivos, comandos de shell y búsqueda web."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", nargs="*", help="Task to run non-interactively / tarea a ejecutar")
    parser.add_argument("-C", "--directory", default=".", help="Working directory / directorio de trabajo")
    parser.add_argument("--model", help="Model id / identificador del modelo")
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        help="Reasoning effort / esfuerzo de razonamiento",
    )
    parser.add_argument("--lang", choices=SUPPORTED, help="Interface language / idioma de la interfaz")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Run commands without asking / ejecutar comandos sin preguntar"
    )
    parser.add_argument(
        "--no-shell", action="store_true", help="Disable command execution / desactivar comandos"
    )
    parser.add_argument("--no-web", action="store_true", help="Disable web access / desactivar acceso web")
    parser.add_argument(
        "--no-thinking", action="store_true", help="Hide reasoning summaries / ocultar el razonamiento"
    )
    parser.add_argument(
        "--allow-outside",
        action="store_true",
        help="Allow file access outside the working directory / permitir acceso fuera del directorio",
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max tool round-trips per turn / máximo de pasos por turno"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config.from_env()
    config.workspace = Path(args.directory).expanduser().resolve()
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    if args.lang:
        config.lang = args.lang
    if args.max_steps:
        config.max_steps = max(1, args.max_steps)
    config.auto_approve = args.yes
    config.enable_shell = not args.no_shell
    config.enable_web = not args.no_web
    config.show_thinking = not args.no_thinking
    config.allow_outside_workspace = args.allow_outside
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    translator = Translator(config.lang)

    if not config.workspace.is_dir():
        print(f"Not a directory: {config.workspace}", file=sys.stderr)
        return 2

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    ui = TerminalUI(translator, interactive=interactive, auto_approve=config.auto_approve)

    try:
        client = anthropic.Anthropic()
    except Exception:  # noqa: BLE001 - missing or malformed credentials
        ui.error(translator("no_api_key"))
        return 1

    agent = Agent(client, config, ui)

    # Non-interactive: a task on the command line, or piped stdin.
    task = " ".join(args.task).strip()
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()

    if task:
        return 0 if run_turn(agent, ui, task) else 1

    return repl(agent, ui)
