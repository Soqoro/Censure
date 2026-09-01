from __future__ import annotations

import json
import re
from contextlib import nullcontext
from typing import Any

import pytest

from censure.actors.base import NormalizedToolCall
from censure.actors.mistral_tool_names import (
    MISTRAL_TOOL_NAME_PROJECTION_VERSION,
    MistralToolNameProjection,
    MistralToolNameProjectionError,
)
from censure.actors.transformers_backend import TransformersActor


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "payments.verify_invoice",
                "description": "Verify one invoice.",
                "parameters": {
                    "type": "object",
                    "properties": {"invoice_id": {"type": "string"}},
                    "required": ["invoice_id"],
                },
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


def test_projection_is_deterministic_wire_safe_and_reversible() -> None:
    tools = _tools()
    forward = MistralToolNameProjection.from_tools(tools)
    reverse_order = MistralToolNameProjection.from_tools(list(reversed(tools)))

    assert forward.entries == reverse_order.entries
    assert forward.sha256 == reverse_order.sha256
    assert forward.to_alias("safe_lookup") == "safe_lookup"
    dotted_alias = forward.to_alias("payments.verify_invoice")
    assert dotted_alias != "payments.verify_invoice"
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", dotted_alias)
    assert forward.to_canonical(dotted_alias) == "payments.verify_invoice"
    assert len(forward.sha256) == 64


def test_projection_avoids_a_reserved_alias_collision() -> None:
    dotted = "payments.verify_invoice"
    initial_alias = MistralToolNameProjection.from_tools(
        [{"name": dotted, "parameters": {}}]
    ).to_alias(dotted)

    projection = MistralToolNameProjection.from_tools(
        [
            {"name": dotted, "parameters": {}},
            {"name": initial_alias, "parameters": {}},
        ]
    )

    assert projection.to_alias(initial_alias) == initial_alias
    assert projection.to_alias(dotted) != initial_alias
    assert projection.to_canonical(projection.to_alias(dotted)) == dotted


def test_schema_history_and_calls_round_trip_without_mutation() -> None:
    tools = _tools()
    original_tools = json.loads(json.dumps(tools))
    projection = MistralToolNameProjection.from_tools(tools)
    alias = projection.to_alias("payments.verify_invoice")
    projected_tools = projection.project_tool_schemas(tools)

    assert tools == original_tools
    assert projected_tools[0]["function"]["name"] == alias
    assert projected_tools[1]["function"]["name"] == "safe_lookup"
    assert projection.restore_tool_schemas(projected_tools) == tools

    history = [
        {"role": "user", "content": "Verify it."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "payments.verify_invoice",
                        "arguments": {"invoice_id": "invoice-1"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]
    original_history = json.loads(json.dumps(history))
    projected_history = projection.project_history(history)

    assert history == original_history
    assert projected_history[1]["tool_calls"][0]["function"]["name"] == alias

    calls = [
        NormalizedToolCall(
            call_id="call-1",
            name=alias,
            arguments={"invoice_id": "invoice-1"},
            index=0,
        )
    ]
    restored = projection.restore_calls(calls)
    assert restored[0].name == "payments.verify_invoice"
    assert calls[0].name == alias


def test_projection_rejects_duplicate_and_unknown_names() -> None:
    with pytest.raises(MistralToolNameProjectionError, match="unique"):
        MistralToolNameProjection.from_tools(
            [
                {"name": "duplicate", "parameters": {}},
                {"name": "duplicate", "parameters": {}},
            ]
        )

    projection = MistralToolNameProjection.from_tools(_tools())
    with pytest.raises(MistralToolNameProjectionError, match="unknown Mistral"):
        projection.to_canonical("invented_tool")


def test_transformers_actor_projects_mistral_tools_and_restores_generated_names() -> None:
    projection = MistralToolNameProjection.from_tools(_tools())
    alias = projection.to_alias("payments.verify_invoice")

    class FakeIds:
        shape = (1, 2)

    class FakeBatch(dict[str, Any]):
        def to(self, _device: str) -> FakeBatch:
            return self

    class FakeTokenSlice:
        pass

    class FakeOutput:
        def __getitem__(self, key: object) -> FakeTokenSlice:
            assert key == (0, slice(2, None, None))
            return FakeTokenSlice()

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
            return f'[TOOL_CALLS]{alias}[ARGS]{{"invoice_id":"invoice-1"}}</s>'

    class FakeModel:
        device = "cuda:0"

        def generate(self, **_kwargs: Any) -> FakeOutput:
            return FakeOutput()

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
        "native_weight_format": "bfloat16_safetensors",
        "response_parser_version": "test-v1",
        "generation": {"max_new_tokens": 32, "max_input_tokens": 128},
    }
    actor.actor_id = "mistralai/example"
    actor.actor_revision = "a" * 40
    actor.chat_template_hash = "b" * 64
    actor._checkpoint_load_mode = "native"  # type: ignore[attr-defined]
    actor._is_llama = False  # type: ignore[attr-defined]
    actor._is_multimodal = False  # type: ignore[attr-defined]
    actor._processor = None  # type: ignore[attr-defined]
    actor._template_supports_tools = True  # type: ignore[attr-defined]
    actor._tool_protocol = "mistral_tool_calls_v1"  # type: ignore[attr-defined]
    actor._history_projection = "mistral_call_id_alias_v1"  # type: ignore[attr-defined]
    actor._tool_name_projection = "mistral_tool_name_alias_v1"  # type: ignore[attr-defined]
    actor._turn_index = 0  # type: ignore[attr-defined]
    actor._harmony_private_analysis_by_call_id = {}  # type: ignore[attr-defined]
    actor._tokenizer = FakeTokenizer()  # type: ignore[attr-defined]
    actor._model = FakeModel()  # type: ignore[attr-defined]

    turn = actor.respond(
        messages=[
            {"role": "system", "content": "Follow the user."},
            {"role": "user", "content": "Verify invoice-1."},
        ],
        tools=_tools(),
        decoding_seed=7,
    )

    assert actor._tokenizer.tools[0]["function"]["name"] == alias  # type: ignore[attr-defined]
    assert turn.tool_calls[0].name == "payments.verify_invoice"
    assert turn.model_metadata["tool_name_projection"] == "mistral_tool_name_alias_v1"
    assert (
        turn.model_metadata["tool_name_projection_version"] == MISTRAL_TOOL_NAME_PROJECTION_VERSION
    )
    assert turn.model_metadata["tool_name_projection_sha256"] == projection.sha256
