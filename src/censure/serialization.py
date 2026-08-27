"""Deterministic JSON serialization and SHA-256 helpers.

Opaque pickle payloads are intentionally unsupported.  Non-string mapping keys,
non-finite floats, and other non-JSON values fail loudly so a checkpoint cannot
silently acquire platform-dependent representations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel

if TYPE_CHECKING:
    from censure.schemas import StateSnapshot

CANONICAL_SERIALIZATION_VERSION = "censure-canonical-json-v1"


def _canonical_value(value: Any) -> Any:
    """Convert supported values to a deterministic JSON-compatible tree."""

    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json", round_trip=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit NaN or infinity")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical JSON does not permit non-finite decimals")
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, Mapping):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError("canonical JSON mappings require string keys")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=lambda item: canonical_json(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__qualname__}")


def canonical_json(value: Any) -> str:
    """Return compact UTF-8-safe JSON with recursively sorted object keys."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the exact bytes used for durable JSON hashing."""

    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON bytes of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_state_snapshot(state: Any) -> StateSnapshot:
    """Create a version-tagged state checkpoint and digest."""

    # Local import keeps low-level serialization independent of schema imports.
    from censure.schemas import StateSnapshot

    normalized = _canonical_value(state)
    return StateSnapshot(
        serialization_version=CANONICAL_SERIALIZATION_VERSION,
        state=normalized,
        sha256=canonical_sha256(normalized),
    )


def verify_state_snapshot(snapshot: StateSnapshot) -> bool:
    """Return whether a stored snapshot still matches its canonical state."""

    return (
        snapshot.serialization_version == CANONICAL_SERIALIZATION_VERSION
        and snapshot.sha256 == canonical_sha256(snapshot.state)
    )
