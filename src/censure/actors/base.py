"""Backend-neutral actor protocol used by every environment runner."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NormalizedToolCall(BaseModel):
    """One syntactically parsed tool call, ordered as emitted by the actor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    name: str
    arguments: dict[str, Any]
    index: int = Field(ge=0)
    raw_arguments: str | None = None


class ActorTurn(BaseModel):
    """A normalized assistant turn."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[NormalizedToolCall] = Field(default_factory=list)
    raw_text: str | None = None
    finish_reason: str | None = None
    model_metadata: dict[str, Any] = Field(default_factory=dict)


class Actor(ABC):
    """Small interface shared by scripted tests and local model backends."""

    actor_id: str
    actor_revision: str

    @abstractmethod
    def respond(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        decoding_seed: int,
    ) -> ActorTurn:
        """Produce the next normalized assistant turn."""


class ScriptedActor(Actor):
    """Deterministic actor for CPU tests and the synthetic doctor round trip."""

    def __init__(
        self,
        turns: Sequence[ActorTurn | Mapping[str, Any]],
        *,
        actor_id: str = "scripted",
        actor_revision: str = "scripted-v1",
        repeat_last: bool = False,
    ) -> None:
        if not turns:
            raise ValueError("ScriptedActor requires at least one turn")
        self.actor_id = actor_id
        self.actor_revision = actor_revision
        self._turns = [
            turn if isinstance(turn, ActorTurn) else ActorTurn.model_validate(turn)
            for turn in turns
        ]
        self._cursor = 0
        self._repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    def respond(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        decoding_seed: int,
    ) -> ActorTurn:
        self.calls.append(
            {
                "messages": copy.deepcopy(list(messages)),
                "tools": copy.deepcopy(list(tools)),
                "decoding_seed": decoding_seed,
            }
        )
        if self._cursor >= len(self._turns):
            if not self._repeat_last:
                raise RuntimeError("ScriptedActor exhausted")
            turn = self._turns[-1]
        else:
            turn = self._turns[self._cursor]
            self._cursor += 1
        return turn.model_copy(deep=True)

    def reset(self) -> None:
        self._cursor = 0
        self.calls.clear()
