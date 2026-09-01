from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

import pytest

from censure.actors.glm_tool_calls import (
    GLM4_HISTORY_PROJECTION_VERSION,
    parse_glm4_response,
    project_glm4_history,
)
from censure.actors.tool_calls import ToolCallParseError
from censure.actors.transformers_backend import TransformersActor


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "payments.create",
            "description": "Create one payment.",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
                "required": ["amount"],
            },
        }
    ]


def test_glm4_native_call_is_strictly_parsed() -> None:
    content, calls = parse_glm4_response(
        'payments.create\n{"amount":7,"nested":{"currency":"USD"}}<|observation|>',
        turn_index=3,
    )

    assert content == ""
    assert [(call.name, call.arguments, call.index) for call in calls] == [
        ("payments.create", {"amount": 7, "nested": {"currency": "USD"}}, 0)
    ]
    assert calls[0].call_id.startswith("call_")


@pytest.mark.parametrize(
    "raw",
    [
        "<|observation|>",
        "payments.create<|observation|>",
        "payments.create\n[]<|observation|>",
        'payments.create\n{"amount":1,"amount":2}<|observation|>',
        'payments.create\n{"amount":NaN}<|observation|>',
        'payments.create\n{"amount":1e999}<|observation|>',
        'payments.create\n{"amount":7} trailing<|observation|>',
        'payments.create\n{"amount":7}<|observation|> trailing',
        'payments.create\n{"amount":7}<|assistant|><|observation|>',
    ],
)
def test_glm4_malformed_native_calls_fail_closed_with_provenance(raw: str) -> None:
    with pytest.raises(ToolCallParseError, match=r"GLM-4.*raw_sha256"):
        parse_glm4_response(raw)


def test_glm4_final_content_strips_one_native_terminal_token() -> None:
    assert parse_glm4_response("Done.<|user|>") == ("Done.", [])
    assert parse_glm4_response("Done.<|endoftext|>") == ("Done.", [])
    assert parse_glm4_response("Done without a terminal token.") == (
        "Done without a terminal token.",
        [],
    )


def test_glm4_history_projects_calls_and_validates_ids() -> None:
    original = [
        {"role": "system", "content": "Follow the user."},
        {"role": "user", "content": "Pay and then verify."},
        {
            "role": "assistant",
            "content": "I will do both.",
            "tool_calls": [
                {
                    "id": "first",
                    "type": "function",
                    "function": {"name": "payments.create", "arguments": {"amount": 7}},
                },
                {
                    "id": "second",
                    "type": "function",
                    "function": {"name": "payments.verify", "arguments": {"id": "p-1"}},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "first", "content": '{"id":"p-1"}'},
        {"role": "tool", "tool_call_id": "second", "content": '{"ok":true}'},
    ]

    projected, count = project_glm4_history(original)

    assert count == 2
    assert original[2]["tool_calls"][0]["id"] == "first"
    assert [message["role"] for message in projected] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "observation",
        "assistant",
        "observation",
    ]
    assert projected[3] == {
        "role": "assistant",
        "metadata": "payments.create",
        "content": '{"amount":7}',
    }
    assert projected[4] == {"role": "observation", "content": '{"id":"p-1"}'}
    assert projected[5]["metadata"] == "payments.verify"


def test_glm4_history_rejects_unaligned_or_orphaned_tool_results() -> None:
    with pytest.raises(RuntimeError, match="response IDs"):
        project_glm4_history(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "first", "function": {"name": "read", "arguments": {}}}
                    ],
                },
                {"role": "tool", "tool_call_id": "wrong", "content": "result"},
            ]
        )
    with pytest.raises(RuntimeError, match="no aligned prior"):
        project_glm4_history(
            [{"role": "tool", "tool_call_id": "orphan", "content": "result"}]
        )


def test_transformers_actor_projects_glm4_history_and_parses_native_call() -> None:
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
            return 'payments.create\n{"amount":7}<|observation|>'

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
        "response_parser_version": "censure-glm4-function-call-parser-v1",
        "generation": {"max_new_tokens": 32, "max_input_tokens": 128},
    }
    actor.actor_id = "zai-org/GLM-4-32B-0414"
    actor.actor_revision = "a" * 40
    actor.chat_template_hash = "b" * 64
    actor._checkpoint_load_mode = "native"  # type: ignore[attr-defined]
    actor._is_llama = False  # type: ignore[attr-defined]
    actor._is_multimodal = False  # type: ignore[attr-defined]
    actor._processor = None  # type: ignore[attr-defined]
    actor._template_supports_tools = True  # type: ignore[attr-defined]
    actor._tool_protocol = "glm4_function_calls_v1"  # type: ignore[attr-defined]
    actor._history_projection = "glm4_observation_v1"  # type: ignore[attr-defined]
    actor._tool_name_projection = "none"  # type: ignore[attr-defined]
    actor._turn_index = 1  # type: ignore[attr-defined]
    actor._harmony_private_analysis_by_call_id = {}  # type: ignore[attr-defined]
    actor._tokenizer = FakeTokenizer()  # type: ignore[attr-defined]
    actor._model = FakeModel()  # type: ignore[attr-defined]

    turn = actor.respond(
        messages=[
            {"role": "system", "content": "Follow the user."},
            {"role": "user", "content": "Pay seven."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "prior-call",
                        "type": "function",
                        "function": {
                            "name": "payments.create",
                            "arguments": {"amount": 3},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "prior-call", "content": '{"ok":true}'},
        ],
        tools=_tools(),
        decoding_seed=7,
    )

    assert actor._tokenizer.rendered[-2] == {  # type: ignore[attr-defined]
        "role": "assistant",
        "metadata": "payments.create",
        "content": '{"amount":3}',
    }
    assert actor._tokenizer.rendered[-1] == {  # type: ignore[attr-defined]
        "role": "observation",
        "content": '{"ok":true}',
    }
    assert actor._tokenizer.tools[0]["function"]["name"] == "payments.create"  # type: ignore[attr-defined]
    assert turn.tool_calls[0].name == "payments.create"
    assert turn.raw_text is not None
    assert turn.model_metadata["history_projection"] == "glm4_observation_v1"
    assert turn.model_metadata["history_projection_version"] == GLM4_HISTORY_PROJECTION_VERSION
    assert turn.model_metadata["projected_call_count"] == 1
