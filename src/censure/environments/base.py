"""Environment-side contracts used by CENSURE trajectory runners.

The action guard deliberately does not appear in this module.  A runner captures
an actor proposal, evaluates the proposal for audit purposes, asks a guard for
an intervention, and only then supplies an allowed operation to ``execute``.
Keeping the mutation boundary this small makes that ordering testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from censure.schemas import StateSnapshot, ToolCall, ToolObservation

JsonObject = dict[str, Any]


class EnvironmentError(RuntimeError):
    """Base error for environment reconstruction or execution failures."""


class SnapshotRestoreError(EnvironmentError):
    """A snapshot was corrupt, incompatible, or belonged to another scenario."""


class InvalidToolCallError(EnvironmentError):
    """A proposed operation was not syntactically valid for an available tool."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Durable tool metadata exposed to an actor backend."""

    name: str
    description: str
    parameters: Mapping[str, Any]

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _deep_json_copy(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """One deterministic environment response and its mutation status."""

    tool_call_id: str
    tool_name: str
    ok: bool
    output: Mapping[str, Any]
    state_changed: bool
    observation: ToolObservation


@runtime_checkable
class StatefulEnvironment(Protocol):
    """Minimal stateful environment interface required by paired execution."""

    @property
    def environment_id(self) -> str: ...

    @property
    def available_tools(self) -> Sequence[ToolDefinition]: ...

    def canonical_state(self) -> JsonObject: ...

    def snapshot(self) -> StateSnapshot: ...

    def restore(self, snapshot: StateSnapshot) -> None: ...

    def execute(self, proposed_call: ToolCall) -> ToolExecutionResult: ...


def _deep_json_copy(value: Any) -> Any:
    """Copy a JSON tree without retaining mutable references.

    ``copy.deepcopy`` accepts many opaque runtime objects.  This intentionally
    narrow copier fails instead, preventing such objects from leaking into a
    durable environment snapshot.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("environment JSON mappings require string keys")
        return {key: _deep_json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_json_copy(item) for item in value]
    raise TypeError(f"environment state is not JSON-compatible: {type(value).__qualname__}")
