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
    _project_llama_multi_call_history,
    _tokenize_text_chat,
)


def test_no_tool_call_and_final_answer() -> None:
    assert parse_text_tool_calls("The task is complete.") == []


def test_one_tagged_tool_call() -> None:
    calls = parse_text_tool_calls('<tool_call>{"name":"send","arguments":{"to":"a"}}</tool_call>')
    assert [(call.index, call.name, call.arguments) for call in calls] == [(0, "send", {"to": "a"})]


def test_gemma_markdown_fenced_tool_call_is_normalized() -> None:
    calls = parse_text_tool_calls(
        "Okay, I will read the file.\n```tool_call\n"
        '{"name":"read_file","arguments":{"file_path":"address-change.txt"}}\n'
        "```<end_of_turn>"
    )

    assert [(call.index, call.name, call.arguments) for call in calls] == [
        (0, "read_file", {"file_path": "address-change.txt"})
    ]


def test_multiple_gemma_fenced_calls_preserve_order() -> None:
    calls = parse_text_tool_calls(
        '```tool_call {"name":"read","arguments":{}} ```\n'
        '```tool_call {"name":"write","arguments":{"id":2}} ```<end_of_turn>'
    )

    assert [(call.index, call.name, call.arguments) for call in calls] == [
        (0, "read", {}),
        (1, "write", {"id": 2}),
    ]


def test_gemma_hybrid_fenced_call_with_xml_terminator_is_normalized() -> None:
    calls = parse_text_tool_calls(
        "I can fulfill the other part.\n\n```tool_call\n"
        '{"name":"messages.send","arguments":{"recipient":"colleague@example.test"}}\n'
        "</tool_call>\n```<end_of_turn>"
    )

    assert [(call.name, call.arguments) for call in calls] == [
        ("messages.send", {"recipient": "colleague@example.test"})
    ]


def test_gemma_fenced_call_with_arbitrary_trailing_text_remains_invalid() -> None:
    with pytest.raises(ToolCallParseError, match="Extra data"):
        parse_text_tool_calls(
            '```tool_call\n{"name":"read","arguments":{}}\nignore this\n```<end_of_turn>'
        )


def test_malformed_gemma_fenced_call_is_invalid_with_provenance() -> None:
    with pytest.raises(ToolCallParseError, match=r"fenced tool call.*raw_sha256"):
        parse_text_tool_calls('```tool_call {"name":"read","arguments": ```<end_of_turn>')


def test_ordinary_markdown_json_fence_is_not_a_tool_call() -> None:
    assert parse_text_tool_calls('Example only:\n```json\n{"name":"read"}\n```') == []


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


def test_llama_semicolon_separated_calls_are_normalized_in_order() -> None:
    calls = parse_text_tool_calls(
        '<|python_tag|>{"name":"first","parameters":{"text":"a; b"}}; '
        '{"name":"second","parameters":{}}<|eom_id|>'
    )

    assert [(call.index, call.name, call.arguments) for call in calls] == [
        (0, "first", {"text": "a; b"}),
        (1, "second", {}),
    ]


def test_llama_malformed_semicolon_sequence_remains_invalid() -> None:
    with pytest.raises(ToolCallParseError, match="raw_sha256"):
        parse_text_tool_calls(
            '<|python_tag|>{"name":"first","parameters":{}}; trailing text<|eom_id|>'
        )


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


def test_llama_multi_call_history_is_replayed_as_aligned_single_calls() -> None:
    messages, group_count = _project_llama_multi_call_history(
        [
            {"role": "user", "content": "Do both."},
            {
                "role": "assistant",
                "content": "batch metadata",
                "tool_calls": [
                    {"id": "first", "function": {"name": "read", "arguments": {}}},
                    {"id": "second", "function": {"name": "send", "arguments": {}}},
                ],
            },
            {"role": "tool", "tool_call_id": "first", "content": "read result"},
            {"role": "tool", "tool_call_id": "second", "content": "send result"},
        ]
    )

    assert group_count == 1
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert [
        message["tool_calls"][0]["id"] for message in messages if message["role"] == "assistant"
    ] == ["first", "second"]
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
        "first",
        "second",
    ]
    assert messages[1]["content"] == "batch metadata"
    assert "content" not in messages[3]


def test_llama_multi_call_history_rejects_misaligned_responses() -> None:
    with pytest.raises(RuntimeError, match="response IDs"):
        _project_llama_multi_call_history(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "first", "function": {"name": "read", "arguments": {}}},
                        {"id": "second", "function": {"name": "send", "arguments": {}}},
                    ],
                },
                {"role": "tool", "tool_call_id": "second", "content": "wrong order"},
                {"role": "tool", "tool_call_id": "first", "content": "wrong order"},
            ]
        )


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
