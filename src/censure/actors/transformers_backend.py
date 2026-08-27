"""Reference direct Hugging Face Transformers actor backend.

Heavy imports and model downloads occur only when the class is instantiated.
There is deliberately no CPU fallback when a CUDA model was requested.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from censure.actors.base import Actor, ActorTurn
from censure.actors.tool_calls import parse_text_tool_calls


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

        model_id = str(config["model_id"])
        tokenizer_revision = str(config.get("tokenizer_revision", self.actor_revision))
        token = config.get("token")
        common_load = {
            "revision": tokenizer_revision,
            "token": token,
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        }
        self._is_multimodal = model_id == "google/gemma-3-12b-it"
        self._processor = None
        if self._is_multimodal:
            self._processor = AutoProcessor.from_pretrained(model_id, **common_load)
            self._tokenizer = self._processor.tokenizer
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                revision=self.actor_revision,
                token=token,
                trust_remote_code=bool(config.get("trust_remote_code", False)),
                torch_dtype=model_dtype,
                device_map="auto",
                quantization_config=quantization_config,
            ).eval()
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, **common_load)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=self.actor_revision,
                token=token,
                trust_remote_code=bool(config.get("trust_remote_code", False)),
                torch_dtype=model_dtype,
                device_map="auto",
                quantization_config=quantization_config,
            ).eval()
        template = self._tokenizer.chat_template or ""
        self.chat_template_hash = hashlib.sha256(template.encode()).hexdigest()
        expected_template_hash = config.get("chat_template_sha256")
        if expected_template_hash and self.chat_template_hash != expected_template_hash:
            raise RuntimeError(
                "downloaded chat template does not match frozen chat_template_sha256: "
                f"{self.chat_template_hash} != {expected_template_hash}"
            )
        self._template_supports_tools = "tools" in template
        self._turn_index = 0

    def reset(self) -> None:
        """Reset the only conversational state retained by the shared model backend."""

        self._turn_index = 0

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
        rendered_messages: list[dict[str, Any]] = [copy.deepcopy(dict(item)) for item in messages]
        normalized_tools = _huggingface_tool_schemas(tools)
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
        calls = parse_text_tool_calls(text, turn_index=self._turn_index)
        self._turn_index += 1
        return ActorTurn(
            content="" if calls else text.strip(),
            tool_calls=calls,
            raw_text=text,
            finish_reason="tool_calls" if calls else "stop",
            model_metadata={
                "actor_id": self.actor_id,
                "model_revision": self.actor_revision,
                "chat_template_hash": self.chat_template_hash,
                "decoding_seed": decoding_seed,
                "dtype": str(self._config.get("dtype")),
                "quantization": self._config.get("quantization"),
                "generation": generation,
            },
        )
