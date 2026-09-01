from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from contextlib import nullcontext
from typing import Any

import pytest

from censure.actors.gpt_oss_harmony import (
    HARMONY_CALL_TOKEN_ID,
    HARMONY_RETURN_TOKEN_ID,
    HarmonyProjectionError,
    HarmonyToolNameProjection,
    build_gpt_oss_system_content,
    build_gpt_oss_tool_descriptions,
    build_gpt_oss_tool_result_message,
    parse_gpt_oss_harmony_completion,
)
from censure.actors.tool_calls import ToolCallParseError
from censure.actors.transformers_backend import TransformersActor

_HARMONY_AVAILABLE = importlib.util.find_spec("openai_harmony") is not None
_requires_harmony = pytest.mark.skipif(
    not _HARMONY_AVAILABLE, reason="openai-harmony is an optional GPT-OSS dependency"
)


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "payments.create",
            "description": "Create a payment.",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
                "required": ["amount"],
            },
        },
        {
            "type": "function",
            "function": {
                "name": "safe_lookup",
                "description": "Read a public value.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _encoding() -> Any:
    import openai_harmony as harmony

    return harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)


def _completion_tokens(text: str) -> list[int]:
    return list(_encoding().encode(text, allowed_special="all"))


def test_tool_name_projection_is_deterministic_identifier_safe_and_reversible() -> None:
    tools = _tools()
    forward = HarmonyToolNameProjection.from_tools(tools)
    reverse_order = HarmonyToolNameProjection.from_tools(list(reversed(tools)))

    assert forward.entries == reverse_order.entries
    assert forward.sha256 == reverse_order.sha256
    assert forward.to_alias("safe_lookup") == "safe_lookup"
    dotted_alias = forward.to_alias("payments.create")
    assert dotted_alias != "payments.create"
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", dotted_alias)
    assert forward.to_canonical(dotted_alias) == "payments.create"
    assert len(forward.sha256) == 64


def test_tool_schema_and_history_projection_round_trip_without_mutation() -> None:
    tools = _tools()
    original_tools = json.loads(json.dumps(tools))
    projection = HarmonyToolNameProjection.from_tools(tools)
    alias = projection.to_alias("payments.create")
    projected_tools = projection.project_tool_schemas(tools)

    assert tools == original_tools
    assert projected_tools[0]["name"] == alias
    assert projected_tools[1]["function"]["name"] == "safe_lookup"
    assert projection.restore_tool_schemas(projected_tools) == tools

    history = [
        {"role": "user", "content": "Pay the invoice."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "payments.create", "arguments": '{"amount":7}'},
                }
            ],
        },
        {
            "role": "tool",
            "name": "functions.payments.create",
            "recipient": "assistant",
            "tool_call_id": "call-1",
            "content": "ok",
        },
        {
            "role": "assistant",
            "channel": "commentary",
            "recipient": "functions.payments.create",
            "content": '{"amount":7}',
        },
    ]
    original_history = json.loads(json.dumps(history))
    projected_history = projection.project_history(history)

    assert history == original_history
    assert projected_history[1]["tool_calls"][0]["function"]["name"] == alias
    assert projected_history[2]["name"] == f"functions.{alias}"
    assert projected_history[2]["recipient"] == "assistant"
    assert projected_history[3]["recipient"] == f"functions.{alias}"
    assert projection.restore_history(projected_history) == history


def test_projection_fails_closed_for_duplicates_unknown_names_and_partial_schemas() -> None:
    with pytest.raises(HarmonyProjectionError, match="unique"):
        HarmonyToolNameProjection.from_tools([_tools()[0], _tools()[0]])

    projection = HarmonyToolNameProjection.from_tools(_tools())
    with pytest.raises(HarmonyProjectionError, match="unknown"):
        projection.to_alias("payments.delete")
    with pytest.raises(HarmonyProjectionError, match="exactly match"):
        projection.project_tool_schemas(_tools()[:1])
    with pytest.raises(HarmonyProjectionError, match="unknown"):
        projection.project_history(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "payments.delete", "arguments": {}}],
                }
            ]
        )


def test_frozen_system_arguments_are_validated_before_optional_import() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_gpt_oss_system_content(conversation_start_date="2026-9-01", reasoning_effort="medium")
    with pytest.raises(ValueError, match="reasoning_effort"):
        build_gpt_oss_system_content(
            conversation_start_date="2026-09-01",
            reasoning_effort="maximum",  # type: ignore[arg-type]
        )


@_requires_harmony
def test_system_tool_description_and_tool_result_helpers_use_projected_names() -> None:
    projection = HarmonyToolNameProjection.from_tools(_tools())
    alias = projection.to_alias("payments.create")

    system = build_gpt_oss_system_content(
        conversation_start_date="2026-09-01", reasoning_effort="low"
    )
    assert system.conversation_start_date == "2026-09-01"
    assert system.reasoning_effort.value == "Low"

    descriptions = build_gpt_oss_tool_descriptions(_tools(), projection)
    assert [description.name for description in descriptions] == [alias, "safe_lookup"]
    assert descriptions[0].parameters["required"] == ["amount"]

    result_message = build_gpt_oss_tool_result_message(
        projection=projection,
        canonical_tool_name="payments.create",
        content='{"ok":true}',
    ).to_dict()
    assert result_message["role"] == "tool"
    assert result_message["name"] == f"functions.{alias}"
    assert result_message["channel"] == "commentary"
    assert result_message["recipient"] == "assistant"


@_requires_harmony
def test_final_completion_keeps_analysis_private() -> None:
    secret_analysis = "PRIVATE_CHAIN_OF_THOUGHT_7f0a"
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>"
        + secret_analysis
        + "<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    )
    projection = HarmonyToolNameProjection.from_tools(_tools())

    result = parse_gpt_oss_harmony_completion(tokens, projection=projection, turn_index=4)
    turn = result.to_actor_turn()

    assert result.content == "Done."
    assert result.tool_calls == ()
    assert result.finish_reason == "stop"
    assert secret_analysis in json.dumps(result.private_harmony_messages)
    assert result.private_analysis_texts == (secret_analysis,)
    public = json.dumps(
        {
            "content": turn.content,
            "tool_calls": [call.model_dump() for call in turn.tool_calls],
            "raw_text": turn.raw_text,
            "metadata": turn.model_metadata,
        }
    )
    assert secret_analysis not in public
    assert turn.raw_text is None
    assert result.model_metadata["harmony_analysis_message_count"] == 1
    assert result.model_metadata["harmony_analysis_character_count"] == len(secret_analysis)
    assert result.model_metadata["harmony_stop_token_id"] == HARMONY_RETURN_TOKEN_ID


@_requires_harmony
def test_tool_completion_restores_canonical_name_and_normalizes_call() -> None:
    projection = HarmonyToolNameProjection.from_tools(_tools())
    alias = projection.to_alias("payments.create")
    raw_arguments = '{"amount":7,"memo":"invoice"}'
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>private plan<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>I will check this."
        "<|end|><|start|>assistant<|channel|>commentary to=functions."
        + alias
        + "<|constrain|>json<|message|>"
        + raw_arguments
        + "<|call|>"
    )

    result = parse_gpt_oss_harmony_completion(tokens, projection=projection, turn_index=8)
    repeated = parse_gpt_oss_harmony_completion(tokens, projection=projection, turn_index=8)

    assert result.content == "I will check this."
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "payments.create"
    assert call.arguments == {"amount": 7, "memo": "invoice"}
    assert call.raw_arguments == raw_arguments
    assert call.call_id == repeated.tool_calls[0].call_id
    assert result.model_metadata["harmony_terminal_action"] == "tool_call"
    assert result.model_metadata["harmony_stop_token_id"] == HARMONY_CALL_TOKEN_ID
    assert result.model_metadata["harmony_commentary_preamble_message_count"] == 1
    metadata_json = json.dumps(result.model_metadata)
    assert "private plan" not in metadata_json
    assert "invoice" not in metadata_json


@_requires_harmony
@pytest.mark.parametrize(
    ("completion", "reason"),
    [
        (
            "<|channel|>final<|message|>Done.<|call|>",
            "final message ended with the tool-call token",
        ),
        (
            "<|channel|>commentary to=functions.UNKNOWN_TOOL<|constrain|>json"
            '<|message|>{"amount":7}<|call|>',
            "not a projected function",
        ),
        (
            "<|channel|>commentary to=functions.safe_lookup<|message|>{}<|call|>",
            "required JSON constraint",
        ),
        (
            "<|channel|>commentary to=functions.safe_lookup<|constrain|>json"
            '<|message|>{"x":1,"x":2}<|call|>',
            "arguments could not be normalized",
        ),
        (
            "<|channel|>commentary to=functions.safe_lookup<|constrain|>json"
            '<|message|>{"x":1e999}<|call|>',
            "arguments could not be normalized",
        ),
        (
            "<|channel|>commentary to=functions.safe_lookup<|constrain|>json"
            '<|message|>{"nested":[-1e999]}<|call|>',
            "arguments could not be normalized",
        ),
    ],
)
def test_invalid_harmony_actions_fail_closed_with_token_provenance(
    completion: str, reason: str
) -> None:
    tokens = _completion_tokens(completion)
    digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()

    with pytest.raises(ToolCallParseError, match=reason) as caught:
        parse_gpt_oss_harmony_completion(
            tokens, projection=HarmonyToolNameProjection.from_tools(_tools())
        )

    message = str(caught.value)
    assert f"completion_token_count={len(tokens)}" in message
    assert f"completion_token_sha256={digest}" in message


@_requires_harmony
def test_malformed_tool_json_diagnostics_do_not_leak_reasoning_or_arguments() -> None:
    projection = HarmonyToolNameProjection.from_tools(_tools())
    alias = projection.to_alias("payments.create")
    secret_reasoning = "DO_NOT_LOG_REASONING_1049"
    secret_argument = "DO_NOT_LOG_ARGUMENT_8052"
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>"
        + secret_reasoning
        + "<|end|><|start|>assistant<|channel|>commentary to=functions."
        + alias
        + '<|constrain|>json<|message|>{"secret":"'
        + secret_argument
        + '",}<|call|>'
    )

    with pytest.raises(ToolCallParseError) as caught:
        parse_gpt_oss_harmony_completion(tokens, projection=projection)

    message = str(caught.value)
    assert secret_reasoning not in message
    assert secret_argument not in message
    assert "completion_token_sha256=" in message


def test_missing_or_embedded_terminal_tokens_fail_before_harmony_import() -> None:
    projection = HarmonyToolNameProjection.from_tools(_tools())
    with pytest.raises(ToolCallParseError, match="missing terminal"):
        parse_gpt_oss_harmony_completion([42, 43], projection=projection)
    with pytest.raises(ToolCallParseError, match="embedded terminal"):
        parse_gpt_oss_harmony_completion(
            [42, HARMONY_RETURN_TOKEN_ID, 43, HARMONY_CALL_TOKEN_ID],
            projection=projection,
        )


@_requires_harmony
def test_transformers_actor_parses_harmony_and_replays_private_analysis_in_memory() -> None:
    class FakeIds:
        shape = (1, 2)

    class FakeBatch(dict[str, Any]):
        def to(self, _device: str) -> FakeBatch:
            return self

    class FakeTokenSlice:
        def __init__(self, tokens: list[int]) -> None:
            self.tokens = tokens

        def detach(self) -> FakeTokenSlice:
            return self

        def cpu(self) -> FakeTokenSlice:
            return self

        def tolist(self) -> list[int]:
            return list(self.tokens)

    class FakeOutput:
        def __init__(self, tokens: list[int]) -> None:
            self.tokens = tokens

        def __getitem__(self, key: object) -> FakeTokenSlice:
            assert key == (0, slice(2, None, None))
            return FakeTokenSlice(self.tokens)

    class FakeTokenizer:
        def __init__(self) -> None:
            self.rendered: list[dict[str, Any]] = []
            self.tools: list[dict[str, Any]] = []

        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> FakeBatch:
            self.rendered = json.loads(json.dumps(messages))
            self.tools = json.loads(json.dumps(kwargs["tools"]))
            return FakeBatch(input_ids=FakeIds(), attention_mask=object())

        def decode(self, _tokens: FakeTokenSlice, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is False
            return "decoded private completion must not be persisted"

    class FakeModel:
        device = "cuda:0"

        def __init__(self, tokens: list[int]) -> None:
            self.tokens = tokens

        def generate(self, **_kwargs: Any) -> FakeOutput:
            return FakeOutput(self.tokens)

    actor = object.__new__(TransformersActor)
    actor._torch = type(  # type: ignore[attr-defined]
        "FakeTorch",
        (),
        {
            "manual_seed": staticmethod(lambda _seed: None),
            "cuda": type("FakeCuda", (), {"manual_seed_all": staticmethod(lambda _seed: None)}),
            "inference_mode": staticmethod(nullcontext),
        },
    )
    actor._config = {  # type: ignore[attr-defined]
        "dtype": "bfloat16",
        "quantization": None,
        "native_weight_format": "mxfp4_safetensors",
        "response_parser_version": "test-v1",
        "template_current_date": "2026-09-01",
        "chat_template_args": {"reasoning_effort": "low"},
        "generation": {"max_new_tokens": 32, "max_input_tokens": 128},
    }
    actor.actor_id = "openai/gpt-oss-20b"
    actor.actor_revision = "a" * 40
    actor.chat_template_hash = "b" * 64
    actor._checkpoint_load_mode = "dequantize_mxfp4_to_bfloat16"  # type: ignore[attr-defined]
    actor._is_llama = False  # type: ignore[attr-defined]
    actor._is_multimodal = False  # type: ignore[attr-defined]
    actor._processor = None  # type: ignore[attr-defined]
    actor._template_supports_tools = True  # type: ignore[attr-defined]
    actor._tool_protocol = "openai_harmony_v1"  # type: ignore[attr-defined]
    actor._history_projection = "harmony_tool_name_alias_v1"  # type: ignore[attr-defined]
    actor._tool_name_projection = "none"  # type: ignore[attr-defined]
    actor._turn_index = 0  # type: ignore[attr-defined]
    actor._harmony_private_analysis_by_call_id = {}  # type: ignore[attr-defined]
    actor._tokenizer = FakeTokenizer()  # type: ignore[attr-defined]

    projection = HarmonyToolNameProjection.from_tools(_tools())
    alias = projection.to_alias("payments.create")
    private_analysis = "PRIVATE_REPLAY_ONLY_319"
    actor._model = FakeModel(  # type: ignore[attr-defined]
        _completion_tokens(
            f"<|channel|>analysis<|message|>{private_analysis}<|end|>"
            f"<|start|>assistant<|channel|>commentary to=functions.{alias}"
            '<|constrain|>json<|message|>{"amount":7}<|call|>'
        )
    )
    first = actor.respond(
        messages=[
            {"role": "system", "content": "Follow the user."},
            {"role": "user", "content": "Pay seven."},
        ],
        tools=_tools(),
        decoding_seed=7,
    )
    assert first.tool_calls[0].name == "payments.create"
    assert first.raw_text is None
    assert private_analysis not in json.dumps(first.model_dump(mode="json"))

    actor._model.tokens = _completion_tokens(  # type: ignore[attr-defined]
        "<|channel|>analysis<|message|>finish privately<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    )
    second = actor.respond(
        messages=[
            {"role": "system", "content": "Follow the user."},
            {"role": "user", "content": "Pay seven."},
            {
                "role": "assistant",
                "content": "public preamble",
                "tool_calls": [
                    {
                        "id": first.tool_calls[0].call_id,
                        "type": "function",
                        "function": {
                            "name": "payments.create",
                            "arguments": {"amount": 7},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": first.tool_calls[0].call_id,
                "content": '{"ok":true}',
            },
        ],
        tools=_tools(),
        decoding_seed=7,
    )

    replayed_assistant = actor._tokenizer.rendered[2]  # type: ignore[attr-defined]
    assert replayed_assistant["content"] == private_analysis
    assert replayed_assistant["tool_calls"][0]["function"]["name"] == alias
    assert actor._tokenizer.tools[0]["function"]["name"] == alias  # type: ignore[attr-defined]
    assert second.content == "Done."
    assert second.raw_text is None
