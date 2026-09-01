"""Strict, backend-independent parsing of model-emitted tool calls."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from censure.actors.base import NormalizedToolCall


class ToolCallParseError(ValueError):
    """The actor attempted a tool call whose payload was malformed."""


_TAGGED_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_MARKDOWN_FENCED_CALL = re.compile(r"```tool_call\s+(.*?)\s*```", re.DOTALL)
_FENCED_TRAILING_TOOL_TAG = re.compile(r"\s*</tool_call>\s*$")
_LLAMA_PYTHON_TAG = "<|python_tag|>"
_MISTRAL_TOOL_CALL = "[TOOL_CALLS]"
_MISTRAL_ARGS = "[ARGS]"
_MISTRAL_TRAILING_TOKENS = ("</s>",)
_GRANITE_TERMINAL_TOKEN = "<|end_of_text|>"
_TRAILING_SPECIAL_TOKENS = re.compile(r"(?:<\|(?:eom|eot|end_of_text)_id\|>)+\s*$")
_RAW_DIAGNOSTIC_EDGE_CHARS = 1024


def _raw_diagnostic(text: str) -> str:
    """Return bounded, control-character-safe provenance for a malformed emission."""

    if len(text) <= 2 * _RAW_DIAGNOSTIC_EDGE_CHARS:
        preview = text
    else:
        preview = (
            text[:_RAW_DIAGNOSTIC_EDGE_CHARS]
            + "...<truncated>..."
            + text[-_RAW_DIAGNOSTIC_EDGE_CHARS:]
        )
    digest = hashlib.sha256(text.encode()).hexdigest()
    return (
        f"raw_length={len(text)}; raw_sha256={digest}; "
        f"raw_preview={json.dumps(preview, ensure_ascii=True)}"
    )


def _parse_semicolon_json_sequence(text: str) -> list[Mapping[str, Any]]:
    """Parse an explicit sequence of JSON objects separated at top level by semicolons."""

    decoder = json.JSONDecoder()
    values: list[Mapping[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        value, cursor = decoder.raw_decode(text, cursor)
        if not isinstance(value, Mapping):
            raise ToolCallParseError("semicolon-separated tool calls must all be JSON objects")
        values.append(value)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        if text[cursor] != ";":
            raise ToolCallParseError(
                "semicolon-separated tool calls contain non-separator trailing content"
            )
        cursor += 1
        if not text[cursor:].strip():
            raise ToolCallParseError("semicolon-separated tool calls have a trailing separator")
    if len(values) < 2:
        raise ToolCallParseError("semicolon-separated tool calls require at least two objects")
    return values


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


def parse_mistral_tool_calls(text: str, *, turn_index: int = 0) -> list[NormalizedToolCall]:
    """Strictly parse Mistral's native ``[TOOL_CALLS]name[ARGS]{...}`` format.

    The native format permits multiple adjacent calls. ``JSONDecoder.raw_decode``
    is used deliberately so nested objects and marker-looking strings inside a
    JSON value cannot be confused with structural delimiters.
    """

    if _MISTRAL_TOOL_CALL not in text:
        return []
    cursor = 0
    length = len(text)
    calls: list[NormalizedToolCall] = []

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_non_finite_numbers(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        if isinstance(value, Mapping):
            for nested in value.values():
                reject_non_finite_numbers(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_non_finite_numbers(nested)

    decoder = json.JSONDecoder(
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )

    def fail(reason: str) -> ToolCallParseError:
        return ToolCallParseError(f"malformed Mistral tool call: {reason}; {_raw_diagnostic(text)}")

    # The released Mistral template permits public assistant content before
    # one or more native tool-call blocks. Parsing starts at the first marker;
    # the backend retains the prefix as ActorTurn content.
    cursor = text.find(_MISTRAL_TOOL_CALL)
    if cursor < 0:  # guarded above, retained defensively
        return []

    while cursor < length:
        if not text.startswith(_MISTRAL_TOOL_CALL, cursor):
            raise fail("expected [TOOL_CALLS] marker")
        cursor += len(_MISTRAL_TOOL_CALL)
        args_marker = text.find(_MISTRAL_ARGS, cursor)
        if args_marker < 0:
            raise fail("missing [ARGS] marker")
        name = text[cursor:args_marker].strip()
        if not name:
            raise fail("tool name is empty")
        if _MISTRAL_TOOL_CALL in name:
            raise fail("encountered a second call before [ARGS]")
        cursor = args_marker + len(_MISTRAL_ARGS)
        while cursor < length and text[cursor].isspace():
            cursor += 1
        try:
            arguments, cursor = decoder.raw_decode(text, cursor)
        except (json.JSONDecodeError, ValueError) as exc:
            reason = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise fail(f"arguments are malformed JSON: {reason}") from exc
        try:
            reject_non_finite_numbers(arguments)
        except ValueError as exc:
            raise fail(f"arguments are malformed JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise fail("arguments must be a JSON object")
        calls.append(
            _normalize_one(
                {"name": name, "arguments": arguments},
                len(calls),
                turn_index,
            )
        )
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor == length:
            break
        if text.startswith(_MISTRAL_TOOL_CALL, cursor):
            continue
        matched_eos = next(
            (token for token in _MISTRAL_TRAILING_TOKENS if text.startswith(token, cursor)),
            None,
        )
        if matched_eos is None:
            raise fail("unexpected trailing content")
        cursor += len(matched_eos)
        if text[cursor:].strip():
            raise fail("content follows the terminal token")
        break
    return calls


def parse_mistral_response(
    text: str, *, turn_index: int = 0
) -> tuple[str, list[NormalizedToolCall]]:
    """Return public assistant content and strict native Mistral calls together."""

    calls = parse_mistral_tool_calls(text, turn_index=turn_index)
    content = text.partition(_MISTRAL_TOOL_CALL)[0].strip() if calls else text.strip()
    if not calls:
        terminal_token = next(
            (token for token in _MISTRAL_TRAILING_TOKENS if content.endswith(token)),
            None,
        )
        if terminal_token is not None:
            content = content[: -len(terminal_token)].rstrip()
    return content, calls


def parse_granite_response(
    text: str, *, turn_index: int = 0
) -> tuple[str, list[NormalizedToolCall]]:
    """Parse Granite 4.1's released XML-wrapped JSON tool protocol.

    The published template permits public assistant text before one or more
    adjacent ``<tool_call>`` blocks. After the first block, only additional
    blocks, whitespace, and one model terminal token are valid. This keeps
    malformed or mixed tool-looking output from silently becoming a final
    natural-language answer.
    """

    matches = list(_TAGGED_CALL.finditer(text))
    has_tool_marker = "<tool_call" in text or "</tool_call>" in text
    if not matches:
        if has_tool_marker:
            raise ToolCallParseError(
                "malformed Granite tool-call structure; " + _raw_diagnostic(text)
            )
        content = text.strip()
        if content.endswith(_GRANITE_TERMINAL_TOKEN):
            content = content[: -len(_GRANITE_TERMINAL_TOKEN)].rstrip()
        return content, []

    prefix = text[: matches[0].start()]
    if "<tool_call" in prefix or "</tool_call>" in prefix:
        raise ToolCallParseError("malformed Granite tool-call prefix; " + _raw_diagnostic(text))
    cursor = matches[0].start()
    for match in matches:
        if text[cursor : match.start()].strip():
            raise ToolCallParseError(
                "content appears between Granite tool calls; " + _raw_diagnostic(text)
            )
        cursor = match.end()
    trailing = text[cursor:].strip()
    if trailing not in {"", _GRANITE_TERMINAL_TOKEN}:
        raise ToolCallParseError(
            "unexpected content follows Granite tool calls; " + _raw_diagnostic(text)
        )
    try:
        calls = parse_text_tool_calls(text, turn_index=turn_index)
    except ToolCallParseError as exc:
        raise ToolCallParseError(
            f"malformed Granite tool call: {exc}; {_raw_diagnostic(text)}"
        ) from exc
    return prefix.strip(), calls


def parse_text_tool_calls(text: str, *, turn_index: int = 0) -> list[NormalizedToolCall]:
    """Parse tagged, fenced, or bare JSON calls emitted by model chat templates.

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

    # Gemma 3 may follow a text-described tool contract by placing each JSON
    # call in an exact ``tool_call`` Markdown fence. Ordinary Markdown/JSON
    # fences remain natural-language output and are deliberately not parsed.
    fenced = _MARKDOWN_FENCED_CALL.findall(text)
    if fenced:
        parsed = []
        for index, raw in enumerate(fenced):
            # Gemma also emits a deterministic hybrid form whose Markdown
            # fence contains one redundant XML closing tag after the JSON.
            # Remove only that exact terminal tag; arbitrary trailing content
            # must continue to fail closed.
            raw = _FENCED_TRAILING_TOOL_TAG.sub("", raw)
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolCallParseError(
                    f"fenced tool call {index} is malformed JSON: {exc.msg}; "
                    + _raw_diagnostic(text)
                ) from exc
            if not isinstance(value, Mapping):
                raise ToolCallParseError(f"fenced tool call {index} must be a JSON object")
            parsed.append(_normalize_one(value, index, turn_index))
        return parsed

    stripped = text.strip()
    has_llama_python_tag = stripped.startswith(_LLAMA_PYTHON_TAG)
    if has_llama_python_tag:
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
        if has_llama_python_tag and ";" in stripped:
            try:
                sequence = _parse_semicolon_json_sequence(stripped)
            except (json.JSONDecodeError, ToolCallParseError):
                pass
            else:
                return [
                    _normalize_one(item, index, turn_index) for index, item in enumerate(sequence)
                ]
        # A natural-language answer may begin with punctuation. Only classify it
        # as an attempted call when recognizable call keys are present.
        if '"name"' in stripped and ('"arguments"' in stripped or '"parameters"' in stripped):
            raise ToolCallParseError(
                "tool-call-looking response contains malformed JSON; " + _raw_diagnostic(text)
            ) from exc
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
