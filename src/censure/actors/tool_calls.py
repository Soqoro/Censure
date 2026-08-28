"""Strict, backend-independent parsing of model-emitted tool calls."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from censure.actors.base import NormalizedToolCall


class ToolCallParseError(ValueError):
    """The actor attempted a tool call whose payload was malformed."""


_TAGGED_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_LLAMA_PYTHON_TAG = "<|python_tag|>"
_TRAILING_SPECIAL_TOKENS = re.compile(r"(?:<\|(?:eom|eot|end_of_text)_id\|>)+\s*$")


def _stable_call_id(payload: Mapping[str, Any], index: int, turn_index: int) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(f"{turn_index}:{index}:{encoded}".encode()).hexdigest()[:20]
    return f"call_{digest}"


def _normalize_one(payload: Mapping[str, Any], index: int, turn_index: int) -> NormalizedToolCall:
    function = payload.get("function")
    if isinstance(function, Mapping):
        outer_id = payload.get("id")
        payload = function
    else:
        outer_id = payload.get("id")

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolCallParseError(f"tool call {index} has no non-empty name")
    raw_args = payload.get("arguments", payload.get("parameters", {}))
    raw_arguments: str | None = None
    if isinstance(raw_args, str):
        raw_arguments = raw_args
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"tool call {index} arguments are malformed JSON: {exc.msg}"
            ) from exc
    else:
        arguments = raw_args
    if not isinstance(arguments, dict):
        raise ToolCallParseError(f"tool call {index} arguments must be a JSON object")
    call_id = (
        outer_id
        if isinstance(outer_id, str) and outer_id
        else _stable_call_id(payload, index, turn_index)
    )
    return NormalizedToolCall(
        call_id=call_id,
        name=name,
        arguments=dict(arguments),
        index=index,
        raw_arguments=raw_arguments,
    )


def normalize_structured_tool_calls(
    calls: Sequence[Mapping[str, Any]], *, turn_index: int = 0
) -> list[NormalizedToolCall]:
    """Normalize OpenAI/Hugging Face-style structured tool-call dictionaries."""

    return [_normalize_one(call, index, turn_index) for index, call in enumerate(calls)]


def parse_text_tool_calls(text: str, *, turn_index: int = 0) -> list[NormalizedToolCall]:
    """Parse Qwen tags or a bare JSON call emitted by a model chat template.

    Ordinary prose is not an error. A present tool-call marker with invalid JSON is.
    """

    tagged = _TAGGED_CALL.findall(text)
    if tagged:
        parsed: list[NormalizedToolCall] = []
        for index, raw in enumerate(tagged):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolCallParseError(f"tool call {index} is malformed JSON: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                raise ToolCallParseError(f"tool call {index} must be a JSON object")
            parsed.append(_normalize_one(value, index, turn_index))
        return parsed

    stripped = text.strip()
    if stripped.startswith(_LLAMA_PYTHON_TAG):
        stripped = stripped[len(_LLAMA_PYTHON_TAG) :].lstrip()
    # Llama 3.1 uses two released custom-tool forms: a ``<|python_tag|>``
    # payload ending in ``<|eom_id|>`` and bare JSON ending in
    # ``<|eot_id|>``. Decoding with ``skip_special_tokens=False`` preserves
    # either terminator, so remove it independently of the optional prefix.
    stripped = _TRAILING_SPECIAL_TOKENS.sub("", stripped).rstrip()
    if not stripped.startswith("{") and not stripped.startswith("["):
        return []
    try:
        value: Any = json.loads(stripped)
    except json.JSONDecodeError as exc:
        # A natural-language answer may begin with punctuation. Only classify it
        # as an attempted call when recognizable call keys are present.
        if '"name"' in stripped and ('"arguments"' in stripped or '"parameters"' in stripped):
            raise ToolCallParseError("tool-call-looking response contains malformed JSON") from exc
        return []
    if isinstance(value, Mapping) and "name" in value:
        return [_normalize_one(value, 0, turn_index)]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, Mapping) for item in value)
        and all("name" in item or isinstance(item.get("function"), Mapping) for item in value)
    ):
        return [_normalize_one(item, index, turn_index) for index, item in enumerate(value)]
    return []
