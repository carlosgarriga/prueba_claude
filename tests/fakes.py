"""Test doubles for the Anthropic client and the UI."""

from __future__ import annotations

from types import SimpleNamespace


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def thinking_block(text):
    return SimpleNamespace(type="thinking", thinking=text)


def tool_block(tool_id, name, params):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=params)


def message(content, stop_reason="end_turn", **usage):
    counts = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    counts.update(usage)
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(**counts),
    )


class FakeStream:
    """Replays a scripted message as the SDK's streaming events would."""

    def __init__(self, final):
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for block in self._final.content:
            if block.type == "text":
                yield SimpleNamespace(type="text", text=block.text)
            elif block.type == "thinking":
                yield SimpleNamespace(type="thinking", thinking=block.thinking)

    def get_final_message(self):
        return self._final


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def stream(self, **params):
        # Snapshot the message list: the agent keeps mutating its own copy.
        self.calls.append({**params, "messages": list(params.get("messages", []))})
        if not self._responses:
            raise AssertionError("the agent made more requests than the test scripted")
        return FakeStream(self._responses.pop(0))


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


class RecordingUI:
    """Captures everything the loop sends to the presentation layer."""

    def __init__(self, approve=True):
        self.text: list[str] = []
        self.thinking: list[str] = []
        self.tools: list[tuple[str, bool]] = []
        self.notices: list[tuple[str, dict]] = []
        self.server_tools: list[str] = []
        self._approve = approve

    def on_text(self, chunk):
        self.text.append(chunk)

    def on_thinking(self, chunk):
        self.thinking.append(chunk)

    def on_tool_start(self, name, summary):
        pass

    def on_tool_end(self, name, ok):
        self.tools.append((name, ok))

    def on_server_tool(self, name):
        self.server_tools.append(name)

    def on_notice(self, key, **kwargs):
        self.notices.append((key, kwargs))

    def approve(self, command, description):
        return self._approve

    def turn_end(self):
        pass

    @property
    def output(self):
        return "".join(self.text)
