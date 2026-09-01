from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from censure.actors import ActorTurn, ScriptedActor
from censure.actors.tool_calls import (
    ToolCallParseError,
    normalize_structured_tool_calls,
    parse_mistral_response,
    parse_mistral_tool_calls,
    parse_text_tool_calls,
)
from censure.actors.transformers_backend import (
    TransformersActor,
    _huggingface_tool_schemas,
    _project_llama_multi_call_history,
    _project_mistral_history,
    _required_module_symbol,
    _tokenize_text_chat,
    validate_transformers_runtime_api,
)


class _FakeLoadedModel:
    device = "cuda:0"

    def eval(self) -> _FakeLoadedModel:
        return self


class _FakeLoader:
    calls: list[tuple[str, dict[str, object]]]

    def __init__(self, result: object) -> None:
        self.calls = []
        self.result = result

    def from_pretrained(self, model_id: str, **kwargs: object) -> object:
        self.calls.append((model_id, kwargs))
        return self.result


def _fake_model_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.ModuleType, types.ModuleType]:
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    torch.float16 = object()  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = _FakeLoader(_FakeLoadedModel())  # type: ignore[attr-defined]
    transformers.AutoModelForImageTextToText = _FakeLoader(_FakeLoadedModel())  # type: ignore[attr-defined]
    transformers.AutoProcessor = _FakeLoader(object())  # type: ignore[attr-defined]
    transformers.AutoTokenizer = _FakeLoader(types.SimpleNamespace(chat_template="tools"))  # type: ignore[attr-defined]
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs  # type: ignore[attr-defined]

    class FakeMxfp4Config:
        def __init__(self, *, dequantize: bool) -> None:
            self.dequantize = dequantize

    transformers.Mxfp4Config = FakeMxfp4Config  # type: ignore[attr-defined]
    transformers.Glm4ForCausalLM = _FakeLoader(_FakeLoadedModel())  # type: ignore[attr-defined]
    transformers.Mistral3ForConditionalGeneration = _FakeLoader(_FakeLoadedModel())  # type: ignore[attr-defined]
    transformers.MistralCommonBackend = _FakeLoader(  # type: ignore[attr-defined]
        types.SimpleNamespace(chat_template=None)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return torch, transformers


def test_required_module_symbol_resolves_lazy_exports() -> None:
    sentinel = object()

    class LazyModule(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            if name == "Mxfp4Config":
                return sentinel
            raise AttributeError(name)

    module = LazyModule("lazy_transformers")
    assert "Mxfp4Config" not in vars(module)
    assert _required_module_symbol(module, "Mxfp4Config") is sentinel


def test_transformers_runtime_api_checks_lazy_mxfp4_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base = _fake_model_modules(monkeypatch)
    sentinel = base.Mxfp4Config  # type: ignore[attr-defined]
    delattr(base, "Mxfp4Config")

    class LazyTransformers(types.ModuleType):
        def __getattr__(self, name: str) -> object:
            if name == "Mxfp4Config":
                return sentinel
            raise AttributeError(name)

    transformers = LazyTransformers("transformers")
    transformers.__dict__.update(vars(base))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    resolved = validate_transformers_runtime_api(
        {"checkpoint_load_mode": "dequantize_mxfp4_to_bfloat16"}
    )

    assert "Mxfp4Config" in resolved


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


def test_mistral_native_calls_preserve_order_and_nested_arguments() -> None:
    calls = parse_mistral_tool_calls(
        '[TOOL_CALLS]calendar.read[ARGS]{"query":{"text":"[TOOL_CALLS]"}}\n'
        '[TOOL_CALLS]messages.send[ARGS]{"to":"a@example.test"}</s>'
    )

    assert [(call.index, call.name, call.arguments) for call in calls] == [
        (0, "calendar.read", {"query": {"text": "[TOOL_CALLS]"}}),
        (1, "messages.send", {"to": "a@example.test"}),
    ]


def test_mistral_native_calls_allow_public_preamble() -> None:
    content, calls = parse_mistral_response(
        "I will check both sources.\n"
        '[TOOL_CALLS]calendar.read[ARGS]{"query":{"text":"[TOOL_CALLS]"}}\n'
        '[TOOL_CALLS]messages.send[ARGS]{"to":"a@example.test"}</s>'
    )

    assert content == "I will check both sources."
    assert [(call.index, call.name, call.arguments) for call in calls] == [
        (0, "calendar.read", {"query": {"text": "[TOOL_CALLS]"}}),
        (1, "messages.send", {"to": "a@example.test"}),
    ]


@pytest.mark.parametrize(
    "raw",
    [
        "[TOOL_CALLS]read{}",
        "[TOOL_CALLS][ARGS]{}",
        "[TOOL_CALLS]read[ARGS][]",
        '[TOOL_CALLS]read[ARGS]{"id":1,"id":2}',
        '[TOOL_CALLS]read[ARGS]{"value":NaN}',
        '[TOOL_CALLS]read[ARGS]{"value":Infinity}',
        '[TOOL_CALLS]read[ARGS]{"value":1e999}',
        '[TOOL_CALLS]read[ARGS]{"nested":[-1e999]}',
        "[TOOL_CALLS]read[ARGS]{} trailing",
        "[TOOL_CALLS]read[ARGS]{} </s> trailing",
    ],
)
def test_mistral_native_malformed_calls_fail_closed_with_provenance(raw: str) -> None:
    with pytest.raises(ToolCallParseError, match=r"malformed Mistral tool call.*raw_sha256"):
        parse_mistral_tool_calls(raw)


def test_mistral_parser_ignores_ordinary_prose() -> None:
    assert parse_mistral_tool_calls("I did not call a tool.") == []


def test_mistral_content_response_strips_one_terminal_eos() -> None:
    content, calls = parse_mistral_response("Done.</s>")

    assert content == "Done."
    assert calls == []
    assert parse_mistral_response("Done.</s></s>")[0] == "Done.</s>"


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


def test_mistral_history_uses_aligned_nine_character_wire_ids() -> None:
    original = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_0123456789abcdef0123", "function": {"name": "read"}},
                {"id": "call_fedcba9876543210fedc", "function": {"name": "send"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_0123456789abcdef0123", "content": "one"},
        {"role": "tool", "tool_call_id": "call_fedcba9876543210fedc", "content": "two"},
    ]

    projected, aliases = _project_mistral_history(original)

    assert original[0]["tool_calls"][0]["id"] == "call_0123456789abcdef0123"
    assert set(aliases) == {"call_0123456789abcdef0123", "call_fedcba9876543210fedc"}
    assert len(set(aliases.values())) == 2
    assert all(len(alias) == 9 and alias.isalnum() for alias in aliases.values())
    assert [call["id"] for call in projected[0]["tool_calls"]] == [
        aliases["call_0123456789abcdef0123"],
        aliases["call_fedcba9876543210fedc"],
    ]
    assert [message["tool_call_id"] for message in projected[1:]] == [
        aliases["call_0123456789abcdef0123"],
        aliases["call_fedcba9876543210fedc"],
    ]


def test_mistral_history_rejects_unaligned_tool_result() -> None:
    with pytest.raises(RuntimeError, match="no aligned prior"):
        _project_mistral_history([{"role": "tool", "tool_call_id": "unknown", "content": "result"}])


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


def test_ministral_loader_uses_mistral_common_and_conditional_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch, transformers = _fake_model_modules(monkeypatch)
    revision = "a" * 40
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("mistral template", encoding="utf-8")
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: str(template_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    actor = TransformersActor(
        {
            "actor_id": "ministral",
            "model_id": "mistralai/example",
            "model_revision": revision,
            "tokenizer_revision": revision,
            "chat_template_sha256": hashlib.sha256(b"mistral template").hexdigest(),
            "device": "cuda",
            "dtype": "bfloat16",
            "quantization": None,
            "model_loader": "mistral3_conditional_generation",
            "tokenizer_backend": "mistral_common",
            "native_tools": True,
            "tool_protocol": "mistral_tool_calls_v1",
        }
    )

    tokenizer_loader = transformers.MistralCommonBackend
    model_loader = transformers.Mistral3ForConditionalGeneration
    assert len(tokenizer_loader.calls) == 1
    assert len(model_loader.calls) == 1
    assert model_loader.calls[0][1]["dtype"] is torch.bfloat16
    assert transformers.AutoTokenizer.calls == []
    assert transformers.AutoProcessor.calls == []
    assert actor.chat_template_hash == hashlib.sha256(b"mistral template").hexdigest()


def test_gpt_oss_load_path_explicitly_dequantizes_mxfp4_to_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, transformers = _fake_model_modules(monkeypatch)
    revision = "c" * 40
    TransformersActor(
        {
            "actor_id": "gpt-oss",
            "model_id": "openai/gpt-oss-20b",
            "model_revision": revision,
            "tokenizer_revision": revision,
            "chat_template_sha256": hashlib.sha256(b"tools").hexdigest(),
            "device": "cuda",
            "dtype": "bfloat16",
            "quantization": None,
            "checkpoint_load_mode": "dequantize_mxfp4_to_bfloat16",
            "model_loader": "auto_causal_lm",
            "tokenizer_backend": "auto_tokenizer",
            "native_tools": True,
            "tool_protocol": "openai_harmony_v1",
        }
    )

    load_kwargs = transformers.AutoModelForCausalLM.calls[0][1]
    assert load_kwargs["quantization_config"].dequantize is True


def test_glm4_loader_uses_frozen_native_transformers_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, transformers = _fake_model_modules(monkeypatch)
    revision = "d" * 40
    actor = TransformersActor(
        {
            "actor_id": "glm4",
            "model_id": "zai-org/GLM-4-32B-0414",
            "model_revision": revision,
            "tokenizer_revision": revision,
            "chat_template_sha256": hashlib.sha256(b"tools").hexdigest(),
            "device": "cuda",
            "dtype": "bfloat16",
            "quantization": None,
            "checkpoint_load_mode": "native",
            "model_loader": "glm4_causal_lm",
            "tokenizer_backend": "auto_tokenizer",
            "native_tools": True,
            "tool_protocol": "glm4_function_calls_v1",
            "history_projection": "glm4_observation_v1",
        }
    )

    loader = transformers.Glm4ForCausalLM
    assert len(loader.calls) == 1
    assert loader.calls[0][1]["dtype"] is torch.bfloat16
    assert len(transformers.AutoTokenizer.calls) == 1
    assert transformers.AutoModelForCausalLM.calls == []
    assert actor.chat_template_hash == hashlib.sha256(b"tools").hexdigest()


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
