from __future__ import annotations

import hashlib
import json

import pytest

from censure.actors import ActorTurn, ScriptedActor
from censure.actors.tool_calls import (
    ToolCallParseError,
    normalize_structured_tool_calls,
    parse_text_tool_calls,
)
from censure.actors.transformers_backend import (
    _huggingface_tool_schemas,
    _tokenize_text_chat,
)


def test_no_tool_call_and_final_answer() -> None:
    assert parse_text_tool_calls("The task is complete.") == []


def test_one_tagged_tool_call() -> None:
    calls = parse_text_tool_calls('<tool_call>{"name":"send","arguments":{"to":"a"}}</tool_call>')
    assert [(call.index, call.name, call.arguments) for call in calls] == [(0, "send", {"to": "a"})]


def test_multiple_calls_preserve_ids_and_order() -> None:
    calls = normalize_structured_tool_calls(
        [
            {"id": "first", "function": {"name": "read", "arguments": '{"id": 1}'}},
            {"id": "second", "function": {"name": "write", "arguments": {"id": 2}}},
        ]
    )
    assert [call.call_id for call in calls] == ["first", "second"]
    assert [call.index for call in calls] == [0, 1]


def test_llama_python_tag_tool_call_is_normalized() -> None:
    calls = parse_text_tool_calls('<|python_tag|>{"name":"read","parameters":{"id":1}}<|eom_id|>')
    assert [(call.name, call.arguments) for call in calls] == [("read", {"id": 1})]


def test_llama_bare_json_tool_call_with_eot_is_normalized() -> None:
    calls = parse_text_tool_calls('{"name":"read","parameters":{"id":1}}<|eot_id|>')
    assert [(call.name, call.arguments) for call in calls] == [("read", {"id": 1})]


def test_environment_tools_are_projected_to_huggingface_schemas() -> None:
    schemas = _huggingface_tool_schemas(
        [{"name": "read", "description": "Read one item.", "parameters": {"type": "object"}}]
    )
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read one item.",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_text_chat_tokenization_requests_and_requires_attention_mask() -> None:
    class RecordingTokenizer:
        kwargs: dict[str, object]

        def apply_chat_template(self, messages: object, **kwargs: object) -> dict[str, object]:
            self.kwargs = kwargs
            return {"input_ids": object(), "attention_mask": object()}

    tokenizer = RecordingTokenizer()
    encoded = _tokenize_text_chat(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tools=None,
        template_args={},
    )
    assert set(encoded) == {"input_ids", "attention_mask"}
    assert tokenizer.kwargs["return_dict"] is True
    assert tokenizer.kwargs["return_tensors"] == "pt"

    tokenizer.apply_chat_template = lambda *args, **kwargs: {"input_ids": object()}  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="attention_mask"):
        _tokenize_text_chat(
            tokenizer,
            [{"role": "user", "content": "hello"}],
            tools=None,
            template_args={},
        )


def test_malformed_json_arguments_are_invalid() -> None:
    with pytest.raises(ToolCallParseError):
        normalize_structured_tool_calls([{"name": "read", "arguments": "{"}])
    with pytest.raises(ToolCallParseError):
        parse_text_tool_calls('<tool_call>{"name":"read","arguments":</tool_call>')


def test_malformed_bare_tool_call_preserves_bounded_raw_provenance() -> None:
    raw = '{"name":"read","parameters":' + "x" * 3000
    with pytest.raises(ToolCallParseError) as caught:
        parse_text_tool_calls(raw)

    message = str(caught.value)
    assert f"raw_length={len(raw)}" in message
    assert f"raw_sha256={hashlib.sha256(raw.encode()).hexdigest()}" in message
    preview = json.loads(message.partition("raw_preview=")[2])
    assert preview.startswith(raw[:1024])
    assert preview.endswith(raw[-1024:])
    assert "...<truncated>..." in preview
    assert len(message) < 2400


def test_scripted_actor_supports_replanning_turns() -> None:
    actor = ScriptedActor(
        [
            ActorTurn(tool_calls=normalize_structured_tool_calls([{"id": "x", "name": "unsafe"}])),
            ActorTurn(content="I will not do that.", finish_reason="stop"),
        ]
    )
    first = actor.respond(messages=[], tools=[], decoding_seed=3)
    second = actor.respond(
        messages=[{"role": "tool", "content": "blocked"}], tools=[], decoding_seed=3
    )
    assert first.tool_calls[0].call_id == "x"
    assert second.content == "I will not do that."
