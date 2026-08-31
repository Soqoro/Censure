"""Pure GPT-OSS/Harmony protocol helpers.

The generic Transformers backend deliberately does not import ``openai_harmony``.
This module keeps that optional dependency lazy, projects CENSURE's dotted tool
names onto identifiers accepted by Harmony's function namespace, and converts
one generated Harmony action into backend-neutral CENSURE values.

Harmony ``analysis`` messages are required for correct private conversational
replay.  They are therefore returned in ``private_harmony_messages`` but are
never copied into public content, tool calls, model metadata, or diagnostics.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from censure.actors.base import ActorTurn, NormalizedToolCall
from censure.actors.tool_calls import ToolCallParseError, normalize_structured_tool_calls

OPENAI_HARMONY_VERSION = "0.0.8"
HARMONY_TOOL_NAME_PROJECTION_VERSION = "censure.gpt-oss-tool-name-projection.v1"
GPT_OSS_HARMONY_PARSE_VERSION = "censure.gpt-oss-harmony-parse.v1"
HARMONY_RETURN_TOKEN_ID = 200002
HARMONY_CALL_TOKEN_ID = 200012

_HARMONY_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FUNCTION_RECIPIENT_PREFIX = "functions."
_ALIAS_HASH_CHARACTERS = 16
_ALIAS_BASE_CHARACTERS = 48
_MAX_ALIAS_COLLISION_SALTS = 4096


class HarmonyProjectionError(ValueError):
    """A schema or history cannot be projected without ambiguity."""


def _schema_function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool.get("function")
    if function is not None:
        if tool.get("type") != "function" or not isinstance(function, Mapping):
            raise HarmonyProjectionError(
                "wrapped Harmony tool schemas require type='function' and an object function"
            )
        if "name" in tool:
            raise HarmonyProjectionError("tool schema has ambiguous outer and function names")
        return function
    if "name" not in tool:
        raise HarmonyProjectionError("tool schema has no name")
    return tool


def _schema_name(tool: Mapping[str, Any]) -> str:
    raw_name = _schema_function(tool).get("name")
    if not isinstance(raw_name, str) or not raw_name or raw_name != raw_name.strip():
        raise HarmonyProjectionError("tool schema name must be a non-empty, unpadded string")
    return raw_name


def _candidate_alias(canonical_name: str, salt: int) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", canonical_name).strip("_") or "tool"
    if not re.match(r"[A-Za-z_]", base):
        base = f"tool_{base}"
    base = base[:_ALIAS_BASE_CHARACTERS].rstrip("_") or "tool"
    digest_input = (
        f"{HARMONY_TOOL_NAME_PROJECTION_VERSION}:{canonical_name}"
        if salt == 0
        else f"{HARMONY_TOOL_NAME_PROJECTION_VERSION}:{canonical_name}:{salt}"
    )
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:_ALIAS_HASH_CHARACTERS]
    return f"{base}__censure_{digest}"


def _deterministic_projection_entries(
    canonical_names: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    names = tuple(sorted(canonical_names))
    if len(set(names)) != len(names):
        raise HarmonyProjectionError("tool schema names must be unique")
    for name in names:
        if not name or name != name.strip():
            raise HarmonyProjectionError("tool names must be non-empty, unpadded strings")

    reserved = {name for name in names if _HARMONY_IDENTIFIER.fullmatch(name)}
    aliases: dict[str, str] = {name: name for name in reserved}
    used_aliases = set(reserved)
    for canonical_name in names:
        if canonical_name in aliases:
            continue
        for salt in range(_MAX_ALIAS_COLLISION_SALTS):
            alias = _candidate_alias(canonical_name, salt)
            if alias not in used_aliases:
                break
        else:  # pragma: no cover - requires thousands of deliberate hash collisions
            raise HarmonyProjectionError("could not allocate a collision-free Harmony alias")
        if not _HARMONY_IDENTIFIER.fullmatch(alias):  # defense in depth
            raise HarmonyProjectionError("generated Harmony alias is not an identifier")
        aliases[canonical_name] = alias
        used_aliases.add(alias)
    return tuple((name, aliases[name]) for name in names)


@dataclass(frozen=True, slots=True)
class HarmonyToolNameProjection:
    """An immutable, deterministic, reversible tool-name mapping.

    Construct projections with :meth:`from_tools`.  Direct construction is
    validated against the deterministic v1 algorithm so a forged or
    order-dependent mapping cannot silently enter provenance.
    """

    entries: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        canonical_names: list[str] = []
        for entry in self.entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not all(isinstance(item, str) for item in entry)
            ):
                raise HarmonyProjectionError("projection entries must be string pairs")
            canonical_names.append(entry[0])
        expected = _deterministic_projection_entries(canonical_names)
        if self.entries != expected:
            raise HarmonyProjectionError(
                "projection entries do not match the deterministic projection algorithm"
            )

    @classmethod
    def from_tools(cls, tools: Sequence[Mapping[str, Any]]) -> HarmonyToolNameProjection:
        """Build a projection from flat CENSURE or wrapped HF function schemas."""

        names: list[str] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                raise HarmonyProjectionError("tool schemas must be objects")
            names.append(_schema_name(tool))
        return cls(_deterministic_projection_entries(names))

    @property
    def canonical_to_alias(self) -> dict[str, str]:
        """Return a mutable copy of the canonical-to-wire mapping."""

        return dict(self.entries)

    @property
    def alias_to_canonical(self) -> dict[str, str]:
        """Return a mutable copy of the wire-to-canonical mapping."""

        return {alias: canonical for canonical, alias in self.entries}

    @property
    def sha256(self) -> str:
        """Hash the complete versioned projection for run provenance."""

        payload = {
            "entries": [list(entry) for entry in self.entries],
            "version": HARMONY_TOOL_NAME_PROJECTION_VERSION,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_alias(self, canonical_name: str) -> str:
        try:
            return self.canonical_to_alias[canonical_name]
        except (KeyError, TypeError) as exc:
            raise HarmonyProjectionError("unknown canonical tool name") from exc

    def to_canonical(self, alias: str) -> str:
        try:
            return self.alias_to_canonical[alias]
        except (KeyError, TypeError) as exc:
            raise HarmonyProjectionError("unknown Harmony tool alias") from exc

    def project_tool_schemas(self, tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Deep-copy schemas and replace canonical names with Harmony aliases."""

        return self._rewrite_tool_schemas(tools, self.to_alias, set(self.canonical_to_alias))

    def restore_tool_schemas(self, tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Reverse :meth:`project_tool_schemas` without mutating its input."""

        return self._rewrite_tool_schemas(tools, self.to_canonical, set(self.alias_to_canonical))

    def _rewrite_tool_schemas(
        self,
        tools: Sequence[Mapping[str, Any]],
        rewrite: Callable[[str], str],
        expected_names: set[str],
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        observed_names: list[str] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                raise HarmonyProjectionError("tool schemas must be objects")
            name = _schema_name(tool)
            observed_names.append(name)
            copied = copy.deepcopy(dict(tool))
            function = copied.get("function")
            if function is not None:
                if not isinstance(function, dict):  # checked before copy, retained defensively
                    raise HarmonyProjectionError("wrapped tool function must be an object")
                function["name"] = rewrite(name)
            else:
                copied["name"] = rewrite(name)
            projected.append(copied)
        if len(set(observed_names)) != len(observed_names):
            raise HarmonyProjectionError("tool schema names must be unique")
        if set(observed_names) != expected_names:
            raise HarmonyProjectionError("tool schemas do not exactly match the projection")
        return projected

    def project_history(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Deep-copy history and map every represented function name to its alias."""

        return self._rewrite_history(messages, self.to_alias)

    def restore_history(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Reverse :meth:`project_history` without mutating its input."""

        return self._rewrite_history(messages, self.to_canonical)

    @staticmethod
    def _rewrite_history(
        messages: Sequence[Mapping[str, Any]], rewrite: Callable[[str], str]
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []

        def rewrite_function_recipient(raw: Any) -> str:
            if not isinstance(raw, str) or not raw.startswith(_FUNCTION_RECIPIENT_PREFIX):
                raise HarmonyProjectionError("Harmony function recipient is malformed")
            name = raw[len(_FUNCTION_RECIPIENT_PREFIX) :]
            if not name:
                raise HarmonyProjectionError("Harmony function recipient has no name")
            return _FUNCTION_RECIPIENT_PREFIX + rewrite(name)

        for raw_message in messages:
            if not isinstance(raw_message, Mapping):
                raise HarmonyProjectionError("history messages must be objects")
            message = copy.deepcopy(dict(raw_message))
            role = message.get("role")
            if "tool_calls" in message:
                if role != "assistant":
                    raise HarmonyProjectionError("only assistant history may contain tool_calls")
                raw_calls = message["tool_calls"]
                if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
                    raise HarmonyProjectionError("assistant tool_calls must be a sequence")
                calls = list(raw_calls)
                for call in calls:
                    if not isinstance(call, dict):
                        raise HarmonyProjectionError("assistant tool calls must be objects")
                    function = call.get("function")
                    if function is not None:
                        if not isinstance(function, dict) or "name" in call:
                            raise HarmonyProjectionError("assistant tool call shape is ambiguous")
                        name = function.get("name")
                        if not isinstance(name, str) or not name:
                            raise HarmonyProjectionError("assistant tool call has no name")
                        function["name"] = rewrite(name)
                    else:
                        name = call.get("name")
                        if not isinstance(name, str) or not name:
                            raise HarmonyProjectionError("assistant tool call has no name")
                        call["name"] = rewrite(name)
                message["tool_calls"] = calls
            if "recipient" in message:
                raw_recipient = message["recipient"]
                if isinstance(raw_recipient, str) and raw_recipient.startswith(
                    _FUNCTION_RECIPIENT_PREFIX
                ):
                    message["recipient"] = rewrite_function_recipient(raw_recipient)
            if role == "tool" and "name" in message:
                raw_name = message["name"]
                if isinstance(raw_name, str) and raw_name.startswith(_FUNCTION_RECIPIENT_PREFIX):
                    message["name"] = rewrite_function_recipient(raw_name)
                elif isinstance(raw_name, str) and raw_name:
                    message["name"] = rewrite(raw_name)
                else:
                    raise HarmonyProjectionError("tool history author name is malformed")
            projected.append(message)
        return projected


def _load_openai_harmony() -> Any:
    """Import the audited optional Harmony release or fail as configuration error."""

    try:
        version = importlib_metadata.version("openai-harmony")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"GPT-OSS requires openai-harmony=={OPENAI_HARMONY_VERSION}") from exc
    if version != OPENAI_HARMONY_VERSION:
        raise RuntimeError(
            "unsupported openai-harmony version: "
            f"{version}; expected exactly {OPENAI_HARMONY_VERSION}"
        )
    try:
        harmony = importlib.import_module("openai_harmony")
    except ImportError as exc:
        raise RuntimeError(f"GPT-OSS requires openai-harmony=={OPENAI_HARMONY_VERSION}") from exc
    required = (
        "Author",
        "DeveloperContent",
        "HarmonyEncodingName",
        "Message",
        "ReasoningEffort",
        "Role",
        "SystemContent",
        "ToolDescription",
        "load_harmony_encoding",
    )
    if any(not hasattr(harmony, name) for name in required):
        raise RuntimeError("installed openai-harmony package lacks the audited API")
    return harmony


def build_gpt_oss_system_content(
    *, conversation_start_date: str, reasoning_effort: Literal["low", "medium", "high"]
) -> Any:
    """Build the exact dated GPT-OSS system content used in a frozen run."""

    if not isinstance(conversation_start_date, str):
        raise ValueError("conversation_start_date must be an ISO date string")
    try:
        parsed_date = date.fromisoformat(conversation_start_date)
    except ValueError as exc:
        raise ValueError("conversation_start_date must use exact YYYY-MM-DD format") from exc
    if parsed_date.isoformat() != conversation_start_date:
        raise ValueError("conversation_start_date must use exact YYYY-MM-DD format")
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be low, medium, or high")
    harmony = _load_openai_harmony()
    effort = getattr(harmony.ReasoningEffort, reasoning_effort.upper())
    return (
        harmony.SystemContent.new()
        .with_reasoning_effort(effort)
        .with_conversation_start_date(conversation_start_date)
    )


def build_gpt_oss_tool_descriptions(
    tools: Sequence[Mapping[str, Any]], projection: HarmonyToolNameProjection
) -> list[Any]:
    """Build Harmony function descriptions from projected CENSURE schemas."""

    harmony = _load_openai_harmony()
    projected = projection.project_tool_schemas(tools)
    descriptions: list[Any] = []
    for tool in projected:
        function = _schema_function(tool)
        name = _schema_name(tool)
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        if not isinstance(description, str):
            raise HarmonyProjectionError("tool description must be a string")
        if not isinstance(parameters, Mapping):
            raise HarmonyProjectionError("tool parameters must be an object")
        descriptions.append(
            harmony.ToolDescription.new(name, description, copy.deepcopy(dict(parameters)))
        )
    return descriptions


def build_gpt_oss_tool_result_message(
    *,
    projection: HarmonyToolNameProjection,
    canonical_tool_name: str,
    content: str,
) -> Any:
    """Build the private Harmony tool-result message for the next model turn."""

    if not isinstance(content, str):
        raise TypeError("Harmony tool result content must be a string")
    harmony = _load_openai_harmony()
    alias = projection.to_alias(canonical_tool_name)
    author = harmony.Author(role=harmony.Role.TOOL, name=f"functions.{alias}")
    return (
        harmony.Message.from_author_and_content(author, content)
        .with_channel("commentary")
        .with_recipient("assistant")
    )


@dataclass(frozen=True, slots=True)
class GptOssHarmonyParseResult:
    """One parsed assistant action plus private messages needed for replay."""

    content: str
    tool_calls: tuple[NormalizedToolCall, ...]
    finish_reason: Literal["stop", "tool_calls"]
    model_metadata: dict[str, Any]
    private_harmony_messages: tuple[dict[str, Any], ...]

    @property
    def private_analysis_texts(self) -> tuple[str, ...]:
        """Return analysis texts for ephemeral replay, never for trace persistence."""

        return tuple(
            _message_text(message)
            for message in self.private_harmony_messages
            if message.get("channel") == "analysis"
        )

    def to_actor_turn(self) -> ActorTurn:
        """Return the public actor turn; private analysis is intentionally omitted."""

        return ActorTurn(
            content=self.content,
            tool_calls=[call.model_copy(deep=True) for call in self.tool_calls],
            raw_text=None,
            finish_reason=self.finish_reason,
            model_metadata=copy.deepcopy(self.model_metadata),
        )


def _completion_fingerprint(tokens: Sequence[int]) -> tuple[str, str]:
    encoded = json.dumps(list(tokens), separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return digest, f"completion_token_count={len(tokens)}; completion_token_sha256={digest}"


def _json_message(message: Any) -> dict[str, Any]:
    raw = message.to_dict()
    if not isinstance(raw, Mapping):
        raise TypeError("Harmony message serialization was not an object")
    serialized = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    restored = json.loads(serialized)
    if not isinstance(restored, dict):  # defense in depth
        raise TypeError("Harmony message serialization was not an object")
    return restored


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], Mapping)
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
    ):
        raise ValueError("Harmony action must contain exactly one text content")
    return str(content[0]["text"])


class _StrictJsonError(ValueError):
    pass


def _strict_json_object(raw: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise _StrictJsonError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StrictJsonError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_non_finite_numbers(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise _StrictJsonError("non-finite JSON number")
        if isinstance(value, Mapping):
            for nested in value.values():
                reject_non_finite_numbers(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_non_finite_numbers(nested)

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, _StrictJsonError, RecursionError) as exc:
        raise _StrictJsonError("tool arguments are not strict JSON") from exc
    reject_non_finite_numbers(value)
    if not isinstance(value, dict):
        raise _StrictJsonError("tool arguments must be a JSON object")
    return value


def parse_gpt_oss_harmony_completion(
    completion_tokens: Sequence[int],
    *,
    projection: HarmonyToolNameProjection,
    turn_index: int = 0,
) -> GptOssHarmonyParseResult:
    """Strictly parse one GPT-OSS Harmony action into CENSURE values.

    ``completion_tokens`` must include exactly one terminal ``<|return|>`` or
    ``<|call|>`` token at the end.  The terminal token is removed before the
    audited Harmony strict parser is invoked.  Any model-output failure is a
    :class:`ToolCallParseError` containing only token-level provenance.
    Dependency/version errors remain :class:`RuntimeError` because they are
    infrastructure failures, not invalid model actions.
    """

    if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
        raise ValueError("turn_index must be a non-negative integer")
    if not isinstance(completion_tokens, Sequence) or isinstance(completion_tokens, (str, bytes)):
        raise TypeError("completion_tokens must be a sequence of token IDs")
    tokens = list(completion_tokens)
    if not tokens or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens
    ):
        raise ValueError("completion_tokens must contain non-negative integer token IDs")
    completion_sha256, diagnostic = _completion_fingerprint(tokens)

    def fail(reason: str) -> ToolCallParseError:
        return ToolCallParseError(f"invalid GPT-OSS Harmony completion: {reason}; {diagnostic}")

    stop_token = tokens[-1]
    if stop_token not in {HARMONY_RETURN_TOKEN_ID, HARMONY_CALL_TOKEN_ID}:
        raise fail("missing terminal assistant-action token")
    body_tokens = tokens[:-1]
    if not body_tokens:
        raise fail("assistant action is empty")
    if any(token in {HARMONY_RETURN_TOKEN_ID, HARMONY_CALL_TOKEN_ID} for token in body_tokens):
        raise fail("assistant action contains an embedded terminal token")

    harmony = _load_openai_harmony()
    encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
    if set(encoding.stop_tokens_for_assistant_actions()) != {
        HARMONY_RETURN_TOKEN_ID,
        HARMONY_CALL_TOKEN_ID,
    }:
        raise RuntimeError("openai-harmony terminal token contract has changed")
    try:
        parsed_messages = encoding.parse_messages_from_completion_tokens(
            body_tokens, harmony.Role.ASSISTANT, strict=True
        )
        private_messages = tuple(_json_message(message) for message in parsed_messages)
    except Exception as exc:
        # Harmony parser errors may quote decoded model output.  Suppress the
        # cause so even traceback-style diagnostics retain only the fingerprint.
        raise fail(f"strict Harmony parser rejected the action ({type(exc).__name__})") from None
    if not private_messages:
        raise fail("strict Harmony parser returned no messages")

    analysis_count = 0
    analysis_characters = 0
    preamble_texts: list[str] = []
    channels: list[str] = []
    terminal_kind: Literal["final", "tool_call"] | None = None
    terminal_text = ""
    terminal_alias: str | None = None
    seen_non_analysis = False

    for index, message in enumerate(private_messages):
        if message.get("role") != "assistant" or message.get("name") is not None:
            raise fail("parsed action contains a non-assistant author")
        channel = message.get("channel")
        if not isinstance(channel, str):
            raise fail("parsed assistant message has no channel")
        channels.append(channel)
        try:
            text = _message_text(message)
        except (TypeError, ValueError):
            raise fail("parsed assistant message has invalid content") from None
        recipient = message.get("recipient")
        content_type = message.get("content_type")
        is_last = index == len(private_messages) - 1

        if channel == "analysis":
            if seen_non_analysis or recipient is not None or content_type is not None:
                raise fail("analysis messages are out of order or addressed")
            analysis_count += 1
            analysis_characters += len(text)
            continue

        seen_non_analysis = True
        if channel == "final":
            if not is_last or terminal_kind is not None:
                raise fail("final message is not the unique terminal message")
            if recipient is not None or content_type is not None:
                raise fail("final message is addressed or constrained")
            terminal_kind = "final"
            terminal_text = text
            continue

        if channel != "commentary":
            raise fail("parsed action uses an unsupported channel")
        if recipient is None:
            if content_type is not None or terminal_kind is not None or is_last:
                raise fail("commentary preamble is malformed or terminal")
            preamble_texts.append(text)
            continue
        if not is_last or terminal_kind is not None:
            raise fail("tool call is not the unique terminal message")
        if content_type != "<|constrain|>json":
            raise fail("tool call does not use the required JSON constraint")
        if not isinstance(recipient, str) or not recipient.startswith(_FUNCTION_RECIPIENT_PREFIX):
            raise fail("tool call recipient is not in the functions namespace")
        alias = recipient[len(_FUNCTION_RECIPIENT_PREFIX) :]
        if not alias or not _HARMONY_IDENTIFIER.fullmatch(alias):
            raise fail("tool call recipient has an invalid function alias")
        try:
            projection.to_canonical(alias)
        except HarmonyProjectionError:
            raise fail("tool call recipient is not a projected function") from None
        terminal_kind = "tool_call"
        terminal_text = text
        terminal_alias = alias

    if terminal_kind is None:
        raise fail("parsed action has no terminal final message or tool call")
    if terminal_kind == "final" and stop_token != HARMONY_RETURN_TOKEN_ID:
        raise fail("final message ended with the tool-call token")
    if terminal_kind == "tool_call" and stop_token != HARMONY_CALL_TOKEN_ID:
        raise fail("tool call ended with the return token")

    tool_calls: tuple[NormalizedToolCall, ...] = ()
    public_content = terminal_text
    if terminal_kind == "tool_call":
        if terminal_alias is None:  # defense in depth
            raise fail("tool call has no projected function alias")
        try:
            _strict_json_object(terminal_text)
            canonical_name = projection.to_canonical(terminal_alias)
            normalized = normalize_structured_tool_calls(
                [{"name": canonical_name, "arguments": terminal_text}], turn_index=turn_index
            )
        except (HarmonyProjectionError, ToolCallParseError, _StrictJsonError):
            raise fail("tool call arguments could not be normalized") from None
        tool_calls = tuple(normalized)
        public_content = "\n\n".join(preamble_texts)

    preamble_characters = sum(len(text) for text in preamble_texts)
    metadata: dict[str, Any] = {
        "gpt_oss_harmony_parse_version": GPT_OSS_HARMONY_PARSE_VERSION,
        "openai_harmony_version": OPENAI_HARMONY_VERSION,
        "harmony_strict": True,
        "completion_token_count": len(tokens),
        "completion_token_sha256": completion_sha256,
        "harmony_stop_token_id": stop_token,
        "harmony_channels": channels,
        "harmony_analysis_message_count": analysis_count,
        "harmony_analysis_character_count": analysis_characters,
        "harmony_commentary_preamble_message_count": len(preamble_texts),
        "harmony_commentary_preamble_character_count": preamble_characters,
        "harmony_terminal_action": terminal_kind,
        "harmony_tool_name_projection_version": HARMONY_TOOL_NAME_PROJECTION_VERSION,
        "harmony_tool_name_projection_sha256": projection.sha256,
    }
    return GptOssHarmonyParseResult(
        content=public_content,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        model_metadata=metadata,
        private_harmony_messages=private_messages,
    )


__all__ = [
    "GPT_OSS_HARMONY_PARSE_VERSION",
    "HARMONY_CALL_TOKEN_ID",
    "HARMONY_RETURN_TOKEN_ID",
    "HARMONY_TOOL_NAME_PROJECTION_VERSION",
    "OPENAI_HARMONY_VERSION",
    "GptOssHarmonyParseResult",
    "HarmonyProjectionError",
    "HarmonyToolNameProjection",
    "build_gpt_oss_system_content",
    "build_gpt_oss_tool_descriptions",
    "build_gpt_oss_tool_result_message",
    "parse_gpt_oss_harmony_completion",
]
