"""Reference direct Hugging Face Transformers actor backend.

Heavy imports and model downloads occur only when the class is instantiated.
There is deliberately no CPU fallback when a CUDA model was requested.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from censure.actors.base import Actor, ActorTurn
from censure.actors.tool_calls import parse_mistral_response, parse_text_tool_calls

_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _required_module_symbol(module: Any, name: str) -> Any:
    """Resolve a version-gated runtime symbol, including lazy module exports."""

    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(name) from exc


def validate_transformers_runtime_api(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve every Transformers symbol required by an actor without loading weights."""

    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - extension environment only
        raise RuntimeError("Transformers is not importable") from exc

    required = [
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoProcessor",
        "AutoTokenizer",
        "BitsAndBytesConfig",
    ]
    if config.get("checkpoint_load_mode") == "dequantize_mxfp4_to_bfloat16":
        required.append("Mxfp4Config")
    if config.get("model_loader") == "mistral3_conditional_generation":
        required.append("Mistral3ForConditionalGeneration")
    if config.get("tokenizer_backend") == "mistral_common":
        required.append("MistralCommonBackend")

    for name in required:
        try:
            _required_module_symbol(transformers, name)
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(f"Transformers cannot resolve required symbol {name}") from exc
    return tuple(required)


def _huggingface_tool_schemas(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize environment schemas to the Transformers chat-template contract."""

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
            normalized.append(copy.deepcopy(dict(tool)))
        else:
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool["name"]),
                        "description": str(tool.get("description", "")),
                        "parameters": copy.deepcopy(dict(tool.get("parameters", {}))),
                    },
                }
            )
    return normalized


def _tokenize_text_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None,
    template_args: Mapping[str, Any],
) -> Any:
    """Return generation inputs with an explicit, tokenizer-derived mask."""

    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        **template_args,
    )
    if (
        not isinstance(encoded, Mapping)
        or "input_ids" not in encoded
        or "attention_mask" not in encoded
    ):
        raise RuntimeError("tokenizer chat template did not return input_ids and attention_mask")
    return encoded


def _project_llama_multi_call_history(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Replay Llama batches through its single-call-only released template.

    The execution trace retains the original assistant message and ordered call
    IDs. For model replay only, each call is paired with its already-produced
    response so the tokenizer template sees the one-call turns it supports.
    """

    projected: list[dict[str, Any]] = []
    group_count = 0
    cursor = 0
    while cursor < len(messages):
        message = copy.deepcopy(dict(messages[cursor]))
        raw_calls = message.get("tool_calls")
        calls = (
            list(raw_calls)
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes))
            else []
        )
        if message.get("role") != "assistant" or len(calls) <= 1:
            projected.append(message)
            cursor += 1
            continue

        response_start = cursor + 1
        response_end = response_start + len(calls)
        if response_end > len(messages):
            raise RuntimeError("Llama multi-call history is missing tool responses")
        responses = [copy.deepcopy(dict(item)) for item in messages[response_start:response_end]]
        for call_index, (call, response) in enumerate(zip(calls, responses, strict=True)):
            if not isinstance(call, Mapping):
                raise RuntimeError("Llama multi-call history contains a non-object call")
            if response.get("role") != "tool" or response.get("tool_call_id") != call.get("id"):
                raise RuntimeError("Llama multi-call history response IDs are not aligned")
            single_call = copy.deepcopy(message)
            single_call["tool_calls"] = [copy.deepcopy(dict(call))]
            if call_index > 0:
                single_call.pop("content", None)
            projected.extend((single_call, response))
        group_count += 1
        cursor = response_end
    return projected, group_count


def _base62_digest(value: str, *, length: int = 9, salt: int = 0) -> str:
    """Return a deterministic fixed-width alphanumeric digest."""

    digest = hashlib.sha256(f"{salt}:{value}".encode()).digest()
    integer = int.from_bytes(digest, "big")
    characters: list[str] = []
    for _ in range(length):
        integer, remainder = divmod(integer, len(_BASE62_ALPHABET))
        characters.append(_BASE62_ALPHABET[remainder])
    return "".join(characters)


def _project_mistral_history(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Alias internal call IDs to Mistral's nine-character wire IDs.

    The returned projection is prompt-only. Persisted trace IDs remain the
    backend-neutral CENSURE IDs, and every tool result must align with a prior
    assistant call before serialization.
    """

    projected = [copy.deepcopy(dict(message)) for message in messages]
    aliases: dict[str, str] = {}
    reverse: dict[str, str] = {}

    def alias_for(raw_id: Any) -> str:
        if not isinstance(raw_id, str) or not raw_id:
            raise RuntimeError("Mistral history contains a tool call without an ID")
        existing = aliases.get(raw_id)
        if existing is not None:
            return existing
        salt = 0
        while True:
            alias = _base62_digest(raw_id, salt=salt)
            owner = reverse.get(alias)
            if owner is None or owner == raw_id:
                aliases[raw_id] = alias
                reverse[alias] = raw_id
                return alias
            salt += 1

    seen_calls: set[str] = set()
    for message in projected:
        raw_calls = message.get("tool_calls")
        if message.get("role") == "assistant" and raw_calls:
            if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
                raise RuntimeError("Mistral assistant tool_calls must be a sequence")
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    raise RuntimeError("Mistral assistant tool call must be an object")
                raw_id = raw_call.get("id")
                alias = alias_for(raw_id)
                raw_call["id"] = alias
                seen_calls.add(str(raw_id))
        elif message.get("role") == "tool":
            raw_id = message.get("tool_call_id")
            if not isinstance(raw_id, str) or raw_id not in seen_calls:
                raise RuntimeError("Mistral tool response has no aligned prior assistant call")
            message["tool_call_id"] = alias_for(raw_id)
    return projected, aliases


class TransformersActor(Actor):
    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover - exercised only in Colab
            raise RuntimeError(
                "Install the 'models' extra and a CUDA-compatible PyTorch build"
            ) from exc

        self._torch = torch
        self._config = dict(config)
        self.actor_id = str(config["actor_id"])
        self.actor_revision = str(config["model_revision"])
        if self.actor_revision == "resolve_at_doctor":
            raise RuntimeError("model revision is unresolved; run the doctor stage first")
        if config.get("device", "cuda") != "cuda":
            raise RuntimeError("Experiment 1 reference models require device=cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
        dtype_name = str(config.get("dtype"))
        if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support BF16")
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if dtype_name not in dtype_by_name:
            raise RuntimeError(f"unsupported model dtype: {dtype_name}")
        model_dtype = dtype_by_name[dtype_name]
        load_dtype_kwargs = (
            {"dtype": model_dtype} if "model_loader" in config else {"torch_dtype": model_dtype}
        )
        quantization = config.get("quantization")
        quantization_config = None
        if quantization is not None:
            if quantization != "bitsandbytes_nf4_4bit":
                raise RuntimeError(f"unsupported quantization mode: {quantization}")
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "bitsandbytes_nf4_4bit requires the pinned bitsandbytes package"
                ) from exc
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=True,
            )

        checkpoint_load_mode = str(config.get("checkpoint_load_mode", "native"))
        self._checkpoint_load_mode = checkpoint_load_mode
        if checkpoint_load_mode == "dequantize_mxfp4_to_bfloat16":
            if quantization_config is not None:
                raise RuntimeError("MXFP4 dequantization cannot be combined with bitsandbytes")
            if model_dtype is not torch.bfloat16:
                raise RuntimeError("MXFP4 dequantization requires dtype=bfloat16")
            try:
                import transformers

                mxfp4_config_type = _required_module_symbol(transformers, "Mxfp4Config")
            except ImportError as exc:  # pragma: no cover - extension environment only
                raise RuntimeError(
                    "the GPT-OSS BF16 load path requires Transformers with Mxfp4Config"
                ) from exc
            except AttributeError as exc:  # pragma: no cover - extension environment only
                raise RuntimeError(
                    "the GPT-OSS BF16 load path requires Transformers with Mxfp4Config"
                ) from exc
            quantization_config = mxfp4_config_type(dequantize=True)
        elif checkpoint_load_mode != "native":
            raise RuntimeError(f"unsupported checkpoint load mode: {checkpoint_load_mode}")

        model_id = str(config["model_id"])
        self._is_llama = model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self._tool_protocol = str(config.get("tool_protocol", "generic_text_v1"))
        self._history_projection = str(config.get("history_projection", "none"))
        model_loader = str(config.get("model_loader", "auto"))
        tokenizer_backend = str(config.get("tokenizer_backend", "auto"))
        if model_loader == "auto_causal_lm":
            model_loader = "auto"
        if tokenizer_backend == "auto_tokenizer":
            tokenizer_backend = "auto"
        tokenizer_revision = str(config.get("tokenizer_revision", self.actor_revision))
        token = config.get("token")
        common_load = {
            "revision": tokenizer_revision,
            "token": token,
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        }
        self._is_multimodal = model_id == "google/gemma-3-12b-it"
        self._processor = None
        if model_loader == "mistral3_conditional_generation":
            if tokenizer_backend != "mistral_common":
                raise RuntimeError(
                    "mistral3_conditional_generation requires tokenizer_backend=mistral_common"
                )
            try:
                import transformers

                mistral_model_type = _required_module_symbol(
                    transformers, "Mistral3ForConditionalGeneration"
                )
                mistral_tokenizer_type = _required_module_symbol(
                    transformers, "MistralCommonBackend"
                )
            except (ImportError, AttributeError) as exc:  # pragma: no cover
                raise RuntimeError(
                    "Ministral 3 requires the extension Transformers/mistral-common lock"
                ) from exc
            self._tokenizer = mistral_tokenizer_type.from_pretrained(model_id, **common_load)
            self._model = mistral_model_type.from_pretrained(
                model_id,
                revision=self.actor_revision,
                token=token,
                trust_remote_code=bool(config.get("trust_remote_code", False)),
                device_map="auto",
                quantization_config=quantization_config,
                **load_dtype_kwargs,
            ).eval()
        elif tokenizer_backend != "auto":
            raise RuntimeError(f"unsupported tokenizer backend: {tokenizer_backend}")
        elif model_loader not in {"auto", "auto_image_text"}:
            raise RuntimeError(f"unsupported model loader: {model_loader}")
        elif self._is_multimodal or model_loader == "auto_image_text":
            self._processor = AutoProcessor.from_pretrained(model_id, **common_load)
            self._tokenizer = self._processor.tokenizer
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                revision=self.actor_revision,
                token=token,
                trust_remote_code=bool(config.get("trust_remote_code", False)),
                device_map="auto",
                quantization_config=quantization_config,
                **load_dtype_kwargs,
            ).eval()
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, **common_load)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=self.actor_revision,
                token=token,
                trust_remote_code=bool(config.get("trust_remote_code", False)),
                device_map="auto",
                quantization_config=quantization_config,
                **load_dtype_kwargs,
            ).eval()
        template = getattr(self._tokenizer, "chat_template", None) or ""
        expected_template_hash = config.get("chat_template_sha256")
        actual_template_hash = hashlib.sha256(template.encode()).hexdigest() if template else None
        if expected_template_hash and actual_template_hash is None:
            try:
                from huggingface_hub import hf_hub_download

                template_path = hf_hub_download(
                    repo_id=model_id,
                    filename="chat_template.jinja",
                    revision=tokenizer_revision,
                    token=token,
                )
                actual_template_hash = hashlib.sha256(Path(template_path).read_bytes()).hexdigest()
            except Exception as exc:
                raise RuntimeError(
                    "could not verify the configured chat template at runtime"
                ) from exc
        if expected_template_hash and actual_template_hash != expected_template_hash:
            raise RuntimeError(
                "downloaded chat template does not match frozen chat_template_sha256: "
                f"{actual_template_hash} != {expected_template_hash}"
            )
        self.chat_template_hash = str(expected_template_hash or actual_template_hash or "")
        explicit_native_tools = config.get("native_tools")
        self._template_supports_tools = (
            bool(explicit_native_tools)
            if explicit_native_tools is not None
            else "tools" in template
        )
        self._turn_index = 0
        self._harmony_private_analysis_by_call_id: dict[str, str] = {}

    def reset(self) -> None:
        """Reset the only conversational state retained by the shared model backend."""

        self._turn_index = 0
        self._harmony_private_analysis_by_call_id.clear()

    def respond(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        decoding_seed: int,
    ) -> ActorTurn:
        torch = self._torch
        torch.manual_seed(decoding_seed)
        torch.cuda.manual_seed_all(decoding_seed)
        template_args = dict(self._config.get("chat_template_args", {}))
        frozen_template_date = self._config.get("template_current_date")
        if frozen_template_date is not None:
            frozen_date = str(frozen_template_date)
            template_args["strftime_now"] = lambda _format: frozen_date
        rendered_messages: list[dict[str, Any]] = [copy.deepcopy(dict(item)) for item in messages]
        normalized_tools = _huggingface_tool_schemas(tools)
        harmony_projection: Any | None = None
        if self._history_projection == "harmony_tool_name_alias_v1":
            from censure.actors.gpt_oss_harmony import HarmonyToolNameProjection

            harmony_projection = HarmonyToolNameProjection.from_tools(normalized_tools)
            rendered_messages = harmony_projection.project_history(rendered_messages)
            for message in rendered_messages:
                if message.get("role") != "assistant" or not message.get("tool_calls"):
                    continue
                first_call = message["tool_calls"][0]
                if not isinstance(first_call, Mapping):
                    raise RuntimeError("Harmony assistant history contains a non-object call")
                call_id = first_call.get("id")
                if not isinstance(call_id, str):
                    raise RuntimeError("Harmony assistant history contains a call without an ID")
                private_analysis = self._harmony_private_analysis_by_call_id.get(call_id)
                if private_analysis is not None:
                    # The released template interprets content on a tool-call
                    # assistant message as analysis. Replace any public
                    # commentary preamble with the private in-memory CoT for
                    # correct tool-loop replay without persisting that CoT.
                    message["content"] = private_analysis
            normalized_tools = harmony_projection.project_tool_schemas(normalized_tools)
        projected_multi_call_groups = 0
        projected_call_id_count = 0
        if self._is_llama:
            rendered_messages, projected_multi_call_groups = _project_llama_multi_call_history(
                rendered_messages
            )
        if self._history_projection == "mistral_call_id_alias_v1":
            rendered_messages, call_id_aliases = _project_mistral_history(rendered_messages)
            projected_call_id_count = len(call_id_aliases)
        if self._is_multimodal:
            # Gemma 3's released template has no tool role. Preserve every call
            # ID and observation while projecting tool results to its alternating
            # user/model dialogue contract.
            for message in rendered_messages:
                if message.get("role") == "assistant" and message.get("tool_calls"):
                    serialized_calls = json.dumps(
                        message["tool_calls"], sort_keys=True, ensure_ascii=False
                    )
                    message["content"] = (
                        str(message.get("content") or "")
                        + f"\n<tool_calls_emitted>{serialized_calls}</tool_calls_emitted>"
                    ).strip()
                    message.pop("tool_calls", None)
                elif message.get("role") == "tool":
                    call_id = message.get("tool_call_id", "unknown")
                    message["role"] = "user"
                    message["content"] = (
                        f"<tool_response id={json.dumps(call_id)}>{message.get('content', '')}"
                        "</tool_response>"
                    )
                    message.pop("tool_call_id", None)
            merged_messages: list[dict[str, Any]] = []
            for message in rendered_messages:
                is_tool_response = str(message.get("content", "")).startswith("<tool_response")
                previous_is_tool_response = bool(
                    merged_messages
                    and str(merged_messages[-1].get("content", "")).startswith("<tool_response")
                )
                if is_tool_response and previous_is_tool_response:
                    merged_messages[-1]["content"] = (
                        str(merged_messages[-1]["content"]) + "\n" + str(message["content"])
                    )
                else:
                    merged_messages.append(message)
            rendered_messages = merged_messages
        if normalized_tools and not self._template_supports_tools:
            tool_instruction = (
                "\n\n# Tools\nYou may call one or more tools. Available JSON schemas:\n"
                + json.dumps(normalized_tools, sort_keys=True, ensure_ascii=False)
                + '\nEmit each call as <tool_call>{"name":"tool.name",'
                '"arguments":{...}}</tool_call>. Do not invent tools.'
            )
            if rendered_messages and rendered_messages[0].get("role") == "system":
                rendered_messages[0]["content"] = (
                    str(rendered_messages[0].get("content", "")) + tool_instruction
                )
            else:
                rendered_messages.insert(
                    0, {"role": "system", "content": tool_instruction.lstrip()}
                )
        template_tools = normalized_tools if self._template_supports_tools else None
        if self._processor is not None:
            prompt = self._processor.apply_chat_template(
                rendered_messages,
                tools=template_tools,
                add_generation_prompt=True,
                tokenize=False,
                **template_args,
            )
            encoded = self._processor(text=prompt, return_tensors="pt").to(self._model.device)
            input_length = encoded["input_ids"].shape[-1]
        else:
            encoded = _tokenize_text_chat(
                self._tokenizer,
                rendered_messages,
                tools=template_tools,
                template_args=template_args,
            ).to(self._model.device)
            input_length = encoded["input_ids"].shape[-1]
        generation = dict(self._config.get("generation", {}))
        max_input = int(generation.pop("max_input_tokens", 24576))
        if input_length > max_input:
            raise OverflowError(f"context_overflow: {input_length} > {max_input} input tokens")
        generation = {key: value for key, value in generation.items() if value is not None}
        with torch.inference_mode():
            output = self._model.generate(**encoded, **generation)
        new_tokens = output[0, input_length:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=False)
        harmony_metadata: dict[str, Any] = {}
        harmony_content: str | None = None
        mistral_content: str | None = None
        if self._tool_protocol == "mistral_tool_calls_v1":
            mistral_content, calls = parse_mistral_response(text, turn_index=self._turn_index)
        elif self._tool_protocol in {"generic_text_v1", "generic_text"}:
            calls = parse_text_tool_calls(text, turn_index=self._turn_index)
        elif self._tool_protocol != "openai_harmony_v1":
            raise RuntimeError(f"unsupported tool protocol: {self._tool_protocol}")
        else:
            if harmony_projection is None:
                raise RuntimeError(
                    "openai_harmony_v1 requires history_projection=harmony_tool_name_alias_v1"
                )
            from censure.actors.gpt_oss_harmony import parse_gpt_oss_harmony_completion

            parsed_harmony = parse_gpt_oss_harmony_completion(
                new_tokens.detach().cpu().tolist(),
                projection=harmony_projection,
                turn_index=self._turn_index,
            )
            calls = list(parsed_harmony.tool_calls)
            harmony_content = parsed_harmony.content
            harmony_metadata = parsed_harmony.model_metadata
            if calls:
                self._harmony_private_analysis_by_call_id[calls[0].call_id] = "\n\n".join(
                    parsed_harmony.private_analysis_texts
                )
        self._turn_index += 1
        model_metadata = {
            "actor_id": self.actor_id,
            "model_revision": self.actor_revision,
            "chat_template_hash": self.chat_template_hash,
            "decoding_seed": decoding_seed,
            "dtype": str(self._config.get("dtype")),
            "quantization": self._config.get("quantization"),
            "native_weight_format": self._config.get("native_weight_format"),
            "checkpoint_load_mode": self._checkpoint_load_mode,
            "tool_protocol": self._tool_protocol,
            "response_parser_version": self._config.get("response_parser_version"),
            "generation": generation,
        }
        if self._is_llama:
            model_metadata.update(
                {
                    "history_projection": "llama31_single_call_replay_v1",
                    "projected_multi_call_groups": projected_multi_call_groups,
                }
            )
        if self._history_projection == "mistral_call_id_alias_v1":
            model_metadata.update(
                {
                    "history_projection": self._history_projection,
                    "projected_call_id_count": projected_call_id_count,
                }
            )
        model_metadata.update(harmony_metadata)
        protocol_content = harmony_content if harmony_content is not None else mistral_content
        turn_content = (
            protocol_content if protocol_content is not None else ("" if calls else text.strip())
        )
        return ActorTurn(
            content=turn_content,
            tool_calls=calls,
            raw_text=None if self._tool_protocol == "openai_harmony_v1" else text,
            finish_reason="tool_calls" if calls else "stop",
            model_metadata=model_metadata,
        )
