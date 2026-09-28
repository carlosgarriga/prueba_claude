"""Tests for the agentic loop, using a fake Anthropic client."""

from __future__ import annotations

import pytest
from fakes import FakeClient, RecordingUI, message, text_block, thinking_block, tool_block

from agent.config import Config
from agent.core import Agent, Usage
from agent.prompts import build_system


@pytest.fixture
def config(tmp_path):
    return Config(workspace=tmp_path, auto_approve=True)


def build(responses, config, ui=None):
    ui = ui or RecordingUI()
    return Agent(FakeClient(responses), config, ui), ui


# ------------------------------------------------------------------ the loop

def test_a_plain_answer_ends_the_turn(config):
    agent, ui = build([message([text_block("Hola.")])], config)
    agent.run("hola")

    assert ui.output == "Hola."
    assert agent.messages[0] == {"role": "user", "content": "hola"}
    assert agent.messages[-1]["role"] == "assistant"
    assert len(agent.client.messages.calls) == 1


def test_a_tool_call_is_executed_and_fed_back(config, tmp_path):
    (tmp_path / "data.txt").write_text("contenido\n", encoding="utf-8")
    agent, ui = build(
        [
            message([tool_block("t1", "read_file", {"path": "data.txt"})], stop_reason="tool_use"),
            message([text_block("Dice: contenido")]),
        ],
        config,
    )
    agent.run("¿qué dice data.txt?")

    assert ui.tools == [("read_file", True)]
    results = agent.messages[2]["content"]
    assert results[0]["tool_use_id"] == "t1"
    assert "contenido" in results[0]["content"]
    assert "is_error" not in results[0]
    assert ui.output == "Dice: contenido"


def test_parallel_tool_calls_return_in_one_user_message(config, tmp_path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    agent, ui = build(
        [
            message(
                [
                    tool_block("t1", "read_file", {"path": "a.txt"}),
                    tool_block("t2", "read_file", {"path": "b.txt"}),
                ],
                stop_reason="tool_use",
            ),
            message([text_block("listo")]),
        ],
        config,
    )
    agent.run("lee ambos")

    results = agent.messages[2]["content"]
    assert len(results) == 2
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]


def test_a_failing_tool_is_reported_to_the_model_not_raised(config):
    agent, ui = build(
        [
            message([tool_block("t1", "read_file", {"path": "missing.txt"})], stop_reason="tool_use"),
            message([text_block("No existe.")]),
        ],
        config,
    )
    agent.run("lee missing.txt")

    assert ui.tools == [("read_file", False)]
    result = agent.messages[2]["content"][0]
    assert result["is_error"] is True
    assert "no such file" in result["content"]


def test_an_unknown_tool_name_is_an_error_result(config):
    agent, ui = build(
        [
            message([tool_block("t1", "teleport", {})], stop_reason="tool_use"),
            message([text_block("ok")]),
        ],
        config,
    )
    agent.run("hazlo")

    result = agent.messages[2]["content"][0]
    assert result["is_error"] is True
    assert "unknown tool" in result["content"]


def test_malformed_tool_input_is_rejected_before_the_handler_runs(config):
    agent, ui = build(
        [
            message([tool_block("t1", "write_file", {"path": "a.txt"})], stop_reason="tool_use"),
            message([text_block("ok")]),
        ],
        config,
    )
    agent.run("escribe")

    result = agent.messages[2]["content"][0]
    assert result["is_error"] is True
    assert "missing required parameter 'content'" in result["content"]


def test_pause_turn_resends_without_losing_history(config):
    agent, ui = build(
        [
            message([text_block("buscando... ")], stop_reason="pause_turn"),
            message([text_block("listo")]),
        ],
        config,
    )
    agent.run("busca algo")

    assert len(agent.client.messages.calls) == 2
    assert ui.output == "buscando... listo"
    # Append-only: the paused turn stays in the history.
    assert len(agent.messages) == 3


def test_refusal_stops_the_turn_and_tells_the_user(config):
    agent, ui = build([message([text_block("")], stop_reason="refusal")], config)
    agent.run("algo")

    assert [key for key, _ in ui.notices] == ["refusal"]


def test_hitting_the_output_limit_is_surfaced(config):
    agent, ui = build([message([text_block("a medias")], stop_reason="max_tokens")], config)
    agent.run("escribe mucho")

    assert [key for key, _ in ui.notices] == ["truncated"]


def test_the_step_limit_stops_a_runaway_loop(config, tmp_path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    config.max_steps = 3
    responses = [
        message([tool_block(f"t{i}", "read_file", {"path": "a.txt"})], stop_reason="tool_use")
        for i in range(3)
    ]
    agent, ui = build(responses, config)
    agent.run("bucle")

    assert len(agent.client.messages.calls) == 3
    assert [key for key, _ in ui.notices] == ["max_steps"]


def test_an_interrupted_tool_call_does_not_poison_the_next_turn(config):
    agent, ui = build([message([text_block("ok")])], config)
    # Simulate Ctrl-C right after the model asked for a tool.
    agent.messages = [
        {"role": "user", "content": "primera"},
        {"role": "assistant", "content": [tool_block("t1", "read_file", {"path": "a.txt"})]},
    ]
    agent.run("segunda")

    repair = agent.messages[2]
    assert repair["role"] == "user"
    assert repair["content"][0]["tool_use_id"] == "t1"
    assert repair["content"][0]["is_error"] is True
    assert agent.messages[3] == {"role": "user", "content": "segunda"}


def test_reset_clears_the_conversation(config):
    agent, _ = build([message([text_block("ok")])], config)
    agent.run("hola")
    assert agent.messages
    agent.reset()
    assert agent.messages == []


# ------------------------------------------------------------ request shaping

def test_the_request_advertises_client_and_server_tools(config):
    agent, _ = build([message([text_block("ok")])], config)
    agent.run("hola")

    tools = agent.client.messages.calls[0]["tools"]
    names = [t.get("name") for t in tools]
    assert {"read_file", "write_file", "edit_file", "list_directory", "run_command"} <= set(names)
    assert {"web_search", "web_fetch"} <= set(names)
    # Eager input streaming applies to client tools only.
    assert all(t.get("eager_input_streaming") for t in tools if t.get("name") == "read_file")
    assert not any("eager_input_streaming" in t for t in tools if t.get("name") == "web_search")


def test_disabling_capabilities_removes_their_tools(tmp_path):
    config = Config(workspace=tmp_path, enable_shell=False, enable_web=False)
    agent, _ = build([message([text_block("ok")])], config)
    agent.run("hola")

    names = [t.get("name") for t in agent.client.messages.calls[0]["tools"]]
    assert "run_command" not in names
    assert "web_search" not in names
    assert "read_file" in names


def test_the_request_uses_adaptive_thinking_and_the_configured_effort(config):
    config.effort = "xhigh"
    agent, _ = build([message([text_block("ok")])], config)
    agent.run("hola")

    call = agent.client.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call["output_config"] == {"effort": "xhigh"}
    assert call["model"] == "claude-opus-5"


def test_reasoning_summaries_reach_the_ui_and_can_be_turned_off(config):
    agent, ui = build([message([thinking_block("Voy a mirar el archivo."), text_block("Listo.")])], config)
    agent.run("hola")
    assert "".join(ui.thinking) == "Voy a mirar el archivo."

    config.show_thinking = False
    agent, ui = build([message([thinking_block("oculto"), text_block("Listo.")])], config)
    agent.run("hola")
    assert ui.thinking == []
    assert "thinking" not in agent.client.messages.calls[0]


def test_the_system_prefix_is_cached_and_stable(config):
    agent, _ = build([message([text_block("ok")]), message([text_block("ok")])], config)
    agent.run("una")
    agent.run("dos")

    first, second = (call["system"] for call in agent.client.messages.calls)
    assert first[0]["cache_control"] == {"type": "ephemeral"}
    assert first[0]["text"] == second[0]["text"]


def test_usage_accumulates_across_steps(config, tmp_path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    agent, _ = build(
        [
            message([tool_block("t1", "read_file", {"path": "a.txt"})], stop_reason="tool_use"),
            message([text_block("A")], input_tokens=100, output_tokens=20),
        ],
        config,
    )
    agent.run("lee")

    assert agent.usage.input_tokens == 110
    assert agent.usage.output_tokens == 25


# -------------------------------------------------------------------- prompts

def test_the_system_prompt_states_the_bilingual_requirement():
    blocks = build_system("/tmp/work", "2026-01-01", shell_enabled=True, web_enabled=True)
    prompt = blocks[0]["text"]
    assert "natively fluent in English and in Spanish" in prompt
    assert "language of the user's latest message" in prompt
    assert "/tmp/work" in blocks[1]["text"]


def test_disabled_capabilities_are_stated_in_the_session_block():
    blocks = build_system("/tmp/work", "2026-01-01", shell_enabled=False, web_enabled=False)
    context = blocks[1]["text"]
    assert "run_command` is disabled" in context
    assert "Web search and web fetch are disabled" in context


# ---------------------------------------------------------------------- usage

def test_cost_uses_the_price_list():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    pricing = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25}
    assert usage.cost(pricing) == pytest.approx(30.0)


def test_cost_is_unknown_for_an_unlisted_model():
    assert Usage(input_tokens=10).cost(None) is None
