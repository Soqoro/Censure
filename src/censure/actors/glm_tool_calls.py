"""Pure GLM-4 native tool-protocol helpers.

GLM-4 represents a function call as an assistant ``metadata`` value containing
the function name plus JSON assistant content containing the arguments.  Tool
results are replayed with the model-specific ``observation`` role.  CENSURE's
persisted trace remains backend-neutral; this module performs only the
prompt/output wire projection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from censure.actors.base import NormalizedToolCall
from censure.actors.tool_calls import ToolCallParseError, normalize_structured_tool_calls

GLM4_TOOL_PROTOCOL_VERSION = "censure-glm4-function-call-parser-v1"
GLM4_HISTORY_PROJECTION_VERSION = "glm4_observation_v1"

_OBSERVATION_TOKEN = "<|observation|>"
_FINAL_TOKENS = ("<|user|>", "<|endoftext|>")
_RESERVED_TOKENS = (_OBSERVATION_TOKEN, *_FINAL_TOKENS, "<|assistant|>")
_RAW_DIAGNOSTIC_EDGE_CHARS = 1024


def _raw_diagnostic(text: str) -> str:
    if len(text) <= 2 * _RAW_DIAGNOSTIC_EDGE_CHARS:
        preview = text
    else:
        preview = (
            text[:_RAW_DIAGNOSTIC_EDGE_CHARS]
            + "...<truncated>..."
            + text[-_RAW_DIAGNOSTIC_EDGE_CHARS:]
        )
    return (
        f"raw_length={len(text)}; raw_sha256={hashlib.sha256(text.encode()).hexdigest()}; "
        f"raw_preview={json.dumps(preview, ensure_ascii=True)}"
    )


def _strict_json_object(raw: str, *, source_text: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_non_finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        if isinstance(value, Mapping):
            for nested in value.values():
                reject_non_finite(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_non_finite(nested)

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        reject_non_finite(value)
    except (json.JSONDecodeError, ValueError) as exc:
        reason = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise ToolCallParseError(
            f"malformed GLM-4 tool call arguments: {reason}; {_raw_diagnostic(source_text)}"
        ) from exc
    if not isinstance(value, dict):
        raise ToolCallParseError(
            "malformed GLM-4 tool call arguments: expected a JSON object; "
            + _raw_diagnostic(source_text)
        )
    return value


def parse_glm4_response(
    text: str, *, turn_index: int = 0
) -> tuple[str, list[NormalizedToolCall]]:
    """Strictly parse one native GLM-4 completion.

    The frozen model ends function-call generations with ``<|observation|>``
    and ordinary assistant generations with ``<|user|>`` or
    ``<|endoftext|>``.  The model's released generation configuration stops at
    the first such token, so multiple calls must be emitted across replanning
    turns rather than guessed from ambiguous text.
    """

    stripped = text.strip()
    if stripped.endswith(_OBSERVATION_TOKEN):
        payload = stripped[: -len(_OBSERVATION_TOKEN)].rstrip()
        if any(token in payload for token in _RESERVED_TOKENS):
            raise ToolCallParseError(
                "malformed GLM-4 tool call: unexpected reserved token; "
                + _raw_diagnostic(text)
            )
        name, separator, raw_arguments = payload.partition("\n")
        name = name.strip()
        if not separator or not name:
            raise ToolCallParseError(
                "malformed GLM-4 tool call: expected function name followed by JSON; "
                + _raw_diagnostic(text)
            )
        arguments = _strict_json_object(raw_arguments.strip(), source_text=text)
        calls = normalize_structured_tool_calls(
            [{"name": name, "arguments": arguments}], turn_index=turn_index
        )
        return "", calls

    if _OBSERVATION_TOKEN in stripped:
        raise ToolCallParseError(
            "malformed GLM-4 tool call: observation token is not terminal; "
            + _raw_diagnostic(text)
        )
    for token in _FINAL_TOKENS:
        if stripped.endswith(token):
            content = stripped[: -len(token)].rstrip()
            if any(reserved in content for reserved in _RESERVED_TOKENS):
                raise ToolCallParseError(
                    "malformed GLM-4 response: unexpected reserved token; "
                    + _raw_diagnostic(text)
                )
            return content, []
    if any(token in stripped for token in _RESERVED_TOKENS):
        raise ToolCallParseError(
            "malformed GLM-4 response: unexpected reserved token; " + _raw_diagnostic(text)
        )
    return stripped, []


def project_glm4_history(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Project normalized CENSURE call/result history to GLM-4 messages.

    Call IDs are validated before being omitted from the prompt because the
    released GLM-4 template has no call-ID field.  Batched normalized calls are
    replayed as ordered assistant-metadata/observation pairs.
    """

    projected: list[dict[str, Any]] = []
    projected_call_count = 0
    cursor = 0
    while cursor < len(messages):
        message = copy.deepcopy(dict(messages[cursor]))
        raw_calls = message.get("tool_calls")
        calls = (
            list(raw_calls)
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes))
            else []
        )
        if message.get("role") == "tool":
            raise RuntimeError("GLM-4 tool response has no aligned prior assistant call")
        if message.get("role") != "assistant" or not calls:
            projected.append(message)
            cursor += 1
            continue

        response_start = cursor + 1
        response_end = response_start + len(calls)
        if response_end > len(messages):
            raise RuntimeError("GLM-4 call history is missing tool responses")
        responses = [copy.deepcopy(dict(item)) for item in messages[response_start:response_end]]

        public_content = message.get("content")
        if public_content:
            projected.append({"role": "assistant", "content": str(public_content)})

        for call, response in zip(calls, responses, strict=True):
            if not isinstance(call, Mapping):
                raise RuntimeError("GLM-4 assistant tool call must be an object")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError("GLM-4 assistant tool call has no ID")
            if response.get("role") != "tool" or response.get("tool_call_id") != call_id:
                raise RuntimeError("GLM-4 call history response IDs are not aligned")
            function = call.get("function")
            payload = function if isinstance(function, Mapping) else call
            name = payload.get("name")
            arguments = payload.get("arguments", payload.get("parameters", {}))
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError("GLM-4 assistant tool call has no function name")
            if not isinstance(arguments, Mapping):
                raise RuntimeError("GLM-4 assistant tool-call arguments must be an object")
            projected.append(
                {
                    "role": "assistant",
                    "metadata": name,
                    "content": json.dumps(
                        dict(arguments),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                }
            )
            projected.append(
                {
                    "role": "observation",
                    "content": str(response.get("content", "")),
                }
            )
            projected_call_count += 1
        cursor = response_end
    return projected, projected_call_count


__all__ = [
    "GLM4_HISTORY_PROJECTION_VERSION",
    "GLM4_TOOL_PROTOCOL_VERSION",
    "parse_glm4_response",
    "project_glm4_history",
]
