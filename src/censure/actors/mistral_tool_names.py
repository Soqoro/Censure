"""Deterministic tool-name projection for Mistral's native tool protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from censure.actors.base import NormalizedToolCall

MISTRAL_TOOL_NAME_PROJECTION_VERSION = "censure.mistral-tool-name-projection.v1"

_MISTRAL_WIRE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_ALIAS_HASH_CHARACTERS = 16
_ALIAS_SUFFIX = "__censure_"
_ALIAS_BASE_CHARACTERS = 64 - len(_ALIAS_SUFFIX) - _ALIAS_HASH_CHARACTERS
_MAX_ALIAS_COLLISION_SALTS = 4096


class MistralToolNameProjectionError(ValueError):
    """A schema, history, or generated name cannot be projected safely."""


def _schema_function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool.get("function")
    if function is not None:
        if tool.get("type") != "function" or not isinstance(function, Mapping):
            raise MistralToolNameProjectionError(
                "wrapped Mistral tool schemas require type='function' and an object function"
            )
        if "name" in tool:
            raise MistralToolNameProjectionError(
                "tool schema has ambiguous outer and function names"
            )
        return function
    if "name" not in tool:
        raise MistralToolNameProjectionError("tool schema has no name")
    return tool


def _schema_name(tool: Mapping[str, Any]) -> str:
    raw_name = _schema_function(tool).get("name")
    if not isinstance(raw_name, str) or not raw_name or raw_name != raw_name.strip():
        raise MistralToolNameProjectionError(
            "tool schema name must be a non-empty, unpadded string"
        )
    return raw_name


def _candidate_alias(canonical_name: str, salt: int) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]", "_", canonical_name).strip("_-") or "tool"
    base = base[:_ALIAS_BASE_CHARACTERS].rstrip("_-") or "tool"
    digest_input = (
        f"{MISTRAL_TOOL_NAME_PROJECTION_VERSION}:{canonical_name}"
        if salt == 0
        else f"{MISTRAL_TOOL_NAME_PROJECTION_VERSION}:{canonical_name}:{salt}"
    )
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:_ALIAS_HASH_CHARACTERS]
    return f"{base}{_ALIAS_SUFFIX}{digest}"


def _deterministic_projection_entries(
    canonical_names: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    names = tuple(sorted(canonical_names))
    if len(set(names)) != len(names):
        raise MistralToolNameProjectionError("tool schema names must be unique")
    for name in names:
        if not name or name != name.strip():
            raise MistralToolNameProjectionError("tool names must be non-empty, unpadded strings")

    reserved = {name for name in names if _MISTRAL_WIRE_NAME.fullmatch(name)}
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
            raise MistralToolNameProjectionError(
                "could not allocate a collision-free Mistral alias"
            )
        if not _MISTRAL_WIRE_NAME.fullmatch(alias):  # defense in depth
            raise MistralToolNameProjectionError(
                "generated Mistral alias violates the native tool-name constraint"
            )
        aliases[canonical_name] = alias
        used_aliases.add(alias)
    return tuple((name, aliases[name]) for name in names)


@dataclass(frozen=True, slots=True)
class MistralToolNameProjection:
    """Immutable mapping between canonical CENSURE names and Mistral wire names."""

    entries: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        canonical_names: list[str] = []
        for entry in self.entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not all(isinstance(item, str) for item in entry)
            ):
                raise MistralToolNameProjectionError("projection entries must be string pairs")
            canonical_names.append(entry[0])
        expected = _deterministic_projection_entries(canonical_names)
        if self.entries != expected:
            raise MistralToolNameProjectionError(
                "projection entries do not match the deterministic projection algorithm"
            )

    @classmethod
    def from_tools(cls, tools: Sequence[Mapping[str, Any]]) -> MistralToolNameProjection:
        names: list[str] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                raise MistralToolNameProjectionError("tool schemas must be objects")
            names.append(_schema_name(tool))
        return cls(_deterministic_projection_entries(names))

    @property
    def canonical_to_alias(self) -> dict[str, str]:
        return dict(self.entries)

    @property
    def alias_to_canonical(self) -> dict[str, str]:
        return {alias: canonical for canonical, alias in self.entries}

    @property
    def sha256(self) -> str:
        payload = {
            "entries": [list(entry) for entry in self.entries],
            "version": MISTRAL_TOOL_NAME_PROJECTION_VERSION,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_alias(self, canonical_name: str) -> str:
        try:
            return self.canonical_to_alias[canonical_name]
        except (KeyError, TypeError) as exc:
            raise MistralToolNameProjectionError("unknown canonical tool name") from exc

    def to_canonical(self, alias: str) -> str:
        try:
            return self.alias_to_canonical[alias]
        except (KeyError, TypeError) as exc:
            raise MistralToolNameProjectionError("unknown Mistral tool alias") from exc

    def project_tool_schemas(self, tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return self._rewrite_tool_schemas(tools, self.to_alias, set(self.canonical_to_alias))

    def restore_tool_schemas(self, tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                raise MistralToolNameProjectionError("tool schemas must be objects")
            name = _schema_name(tool)
            observed_names.append(name)
            copied = copy.deepcopy(dict(tool))
            function = copied.get("function")
            if function is not None:
                if not isinstance(function, dict):  # checked before copy
                    raise MistralToolNameProjectionError("wrapped tool function must be an object")
                function["name"] = rewrite(name)
            else:
                copied["name"] = rewrite(name)
            projected.append(copied)
        if len(set(observed_names)) != len(observed_names):
            raise MistralToolNameProjectionError("tool schema names must be unique")
        if set(observed_names) != expected_names:
            raise MistralToolNameProjectionError("tool schemas do not exactly match the projection")
        return projected

    def project_history(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        for raw_message in messages:
            if not isinstance(raw_message, Mapping):
                raise MistralToolNameProjectionError("history messages must be objects")
            message = copy.deepcopy(dict(raw_message))
            raw_calls = message.get("tool_calls")
            if raw_calls is not None:
                if message.get("role") != "assistant":
                    raise MistralToolNameProjectionError(
                        "only assistant history may contain tool_calls"
                    )
                if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
                    raise MistralToolNameProjectionError("assistant tool_calls must be a sequence")
                calls = list(raw_calls)
                for call in calls:
                    if not isinstance(call, dict):
                        raise MistralToolNameProjectionError("assistant tool calls must be objects")
                    function = call.get("function")
                    if function is not None:
                        if not isinstance(function, dict) or "name" in call:
                            raise MistralToolNameProjectionError(
                                "assistant tool call shape is ambiguous"
                            )
                        name = function.get("name")
                        if not isinstance(name, str) or not name:
                            raise MistralToolNameProjectionError("assistant tool call has no name")
                        function["name"] = self.to_alias(name)
                    else:
                        name = call.get("name")
                        if not isinstance(name, str) or not name:
                            raise MistralToolNameProjectionError("assistant tool call has no name")
                        call["name"] = self.to_alias(name)
                message["tool_calls"] = calls
            projected.append(message)
        return projected

    def restore_calls(self, calls: Sequence[NormalizedToolCall]) -> list[NormalizedToolCall]:
        return [call.model_copy(update={"name": self.to_canonical(call.name)}) for call in calls]
