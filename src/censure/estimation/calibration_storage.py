"""Atomic per-repetition storage for sharded Phase 2 calibration runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from censure.estimation.calibration import (
    CalibrationBudgetSummary,
    CalibrationCellSpec,
    CalibrationReplicateResult,
)
from censure.serialization import canonical_sha256
from censure.storage import CorruptArtifactError, atomic_write_bytes, atomic_write_json


def _write_artifact(path: Path, value: Any) -> str:
    digest = atomic_write_json(path, value)
    atomic_write_bytes(path.with_suffix(".sha256"), f"{digest}\n".encode())
    return digest


def _read_artifact(path: Path) -> Any:
    digest_path = path.with_suffix(".sha256")
    if not path.is_file() or not digest_path.is_file():
        raise CorruptArtifactError(f"calibration artifact is absent: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest_path.read_text(encoding="utf-8").strip():
        raise CorruptArtifactError(f"calibration artifact checksum mismatch: {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptArtifactError(f"calibration artifact is invalid JSON: {path}") from exc


class CalibrationRunStore:
    def __init__(self, out_root: str | Path, experiment_id: str) -> None:
        self.root = (
            Path(out_root).expanduser().resolve()
            / experiment_id
            / "phase2"
            / "calibration"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _cell_root(self, cell_id: str) -> Path:
        return self.root / "cells" / cell_id

    def write_catalog(self, specs: tuple[CalibrationCellSpec, ...]) -> str:
        if not specs:
            raise ValueError("calibration catalog cannot be empty")
        cell_ids = [spec.cell_id for spec in specs]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("calibration catalog contains duplicate cells")
        payload = {
            "schema_version": "censure.calibration-catalog.v1",
            "cell_count": len(specs),
            "cell_ids": cell_ids,
            "specs": [spec.model_dump(mode="json") for spec in specs],
        }
        path = self.root / "catalog" / "frozen_cells.json"
        if path.exists():
            existing = _read_artifact(path)
            if existing != payload:
                raise FileExistsError("the existing frozen calibration catalog differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write_artifact(path, payload)

    def write_resolved_catalog(self, catalog: Any) -> str:
        payload = (
            catalog.model_dump(mode="json")
            if hasattr(catalog, "model_dump")
            else catalog
        )
        path = self.root / "catalog" / "resolved_catalog.json"
        if path.exists():
            existing = _read_artifact(path)
            if existing != payload:
                raise FileExistsError("the existing resolved calibration catalog differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write_artifact(path, payload)

    def write_cell_spec(self, spec: CalibrationCellSpec) -> str:
        path = self._cell_root(spec.cell_id) / "cell_spec.json"
        if path.exists():
            existing = CalibrationCellSpec.model_validate(_read_artifact(path))
            if existing != spec:
                raise FileExistsError("the existing calibration cell spec differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write_artifact(path, spec)

    def _repetition_path(self, spec: CalibrationCellSpec, repetition_index: int) -> Path:
        return self._cell_root(spec.cell_id) / "repetitions" / f"{repetition_index:06d}.json"

    def _chunk_path(self, spec: CalibrationCellSpec, chunk_index: int) -> Path:
        return self._cell_root(spec.cell_id) / "chunks" / f"{chunk_index:04d}.json"

    def write_chunk(
        self,
        spec: CalibrationCellSpec,
        *,
        chunk_index: int,
        repetitions_per_chunk: int,
        rows: tuple[CalibrationReplicateResult, ...],
    ) -> str:
        expected_indices = calibration_chunk_repetition_indices(
            repetitions=spec.repetitions,
            repetitions_per_chunk=repetitions_per_chunk,
            chunk_index=chunk_index,
        )
        if len(rows) != len(expected_indices) * len(spec.budget_fractions):
            raise ValueError("calibration chunk does not contain every repetition/budget row")
        observed_indices = {row.repetition_index for row in rows}
        if observed_indices != set(expected_indices):
            raise ValueError("calibration chunk repetition indices differ from the frozen chunk")
        if any(row.cell_id != spec.cell_id or row.policy != spec.policy for row in rows):
            raise ValueError("calibration chunk identity differs from the cell")
        payload = {
            "schema_version": "censure.calibration-chunk-artifact.v1",
            "cell_id": spec.cell_id,
            "chunk_index": chunk_index,
            "repetitions_per_chunk": repetitions_per_chunk,
            "repetition_indices": list(expected_indices),
            "rows": [row.model_dump(mode="json") for row in rows],
        }
        path = self._chunk_path(spec, chunk_index)
        if path.exists():
            existing = _read_artifact(path)
            if existing != payload:
                raise FileExistsError("the completed calibration chunk differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write_artifact(path, payload)

    def is_chunk_complete(
        self,
        spec: CalibrationCellSpec,
        *,
        chunk_index: int,
        repetitions_per_chunk: int,
    ) -> bool:
        expected_indices = calibration_chunk_repetition_indices(
            repetitions=spec.repetitions,
            repetitions_per_chunk=repetitions_per_chunk,
            chunk_index=chunk_index,
        )
        try:
            raw = _read_artifact(self._chunk_path(spec, chunk_index))
            return bool(
                isinstance(raw, dict)
                and raw.get("schema_version") == "censure.calibration-chunk-artifact.v1"
                and raw.get("cell_id") == spec.cell_id
                and raw.get("chunk_index") == chunk_index
                and raw.get("repetitions_per_chunk") == repetitions_per_chunk
                and raw.get("repetition_indices") == list(expected_indices)
                and isinstance(raw.get("rows"), list)
                and len(raw["rows"]) == len(expected_indices) * len(spec.budget_fractions)
            )
        except CorruptArtifactError:
            return False

    def read_chunk(
        self,
        spec: CalibrationCellSpec,
        *,
        chunk_index: int,
        repetitions_per_chunk: int,
    ) -> tuple[CalibrationReplicateResult, ...]:
        if not self.is_chunk_complete(
            spec,
            chunk_index=chunk_index,
            repetitions_per_chunk=repetitions_per_chunk,
        ):
            raise CorruptArtifactError("calibration chunk is absent, corrupt, or incomplete")
        raw = _read_artifact(self._chunk_path(spec, chunk_index))
        return tuple(CalibrationReplicateResult.model_validate(row) for row in raw["rows"])

    def read_completed_cell_chunks(
        self,
        spec: CalibrationCellSpec,
        *,
        repetitions_per_chunk: int,
        require_all: bool = True,
    ) -> tuple[CalibrationReplicateResult, ...]:
        rows: list[CalibrationReplicateResult] = []
        missing: list[int] = []
        for chunk_index in range(
            calibration_chunk_count(spec.repetitions, repetitions_per_chunk)
        ):
            if self.is_chunk_complete(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=repetitions_per_chunk,
            ):
                rows.extend(
                    self.read_chunk(
                        spec,
                        chunk_index=chunk_index,
                        repetitions_per_chunk=repetitions_per_chunk,
                    )
                )
            else:
                missing.append(chunk_index)
        if require_all and missing:
            raise FileNotFoundError(
                f"calibration cell {spec.cell_id} is missing {len(missing)} chunks"
            )
        return tuple(rows)

    def write_repetition(
        self,
        spec: CalibrationCellSpec,
        repetition_index: int,
        rows: tuple[CalibrationReplicateResult, ...],
    ) -> str:
        if len(rows) != len(spec.budget_fractions):
            raise ValueError("calibration repetition does not contain every frozen budget")
        if {row.budget_fraction for row in rows} != set(spec.budget_fractions):
            raise ValueError("calibration repetition budget set differs from the cell")
        if any(
            row.cell_id != spec.cell_id
            or row.repetition_index != repetition_index
            or row.policy != spec.policy
            for row in rows
        ):
            raise ValueError("calibration repetition identity differs from the cell")
        path = self._repetition_path(spec, repetition_index)
        payload = {
            "schema_version": "censure.calibration-repetition-artifact.v1",
            "cell_id": spec.cell_id,
            "repetition_index": repetition_index,
            "rows": [row.model_dump(mode="json") for row in rows],
        }
        if path.exists():
            existing = _read_artifact(path)
            if existing != payload:
                raise FileExistsError("the completed calibration repetition differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write_artifact(path, payload)

    def is_repetition_complete(
        self, spec: CalibrationCellSpec, repetition_index: int
    ) -> bool:
        path = self._repetition_path(spec, repetition_index)
        try:
            raw = _read_artifact(path)
            return bool(
                isinstance(raw, dict)
                and raw.get("schema_version")
                == "censure.calibration-repetition-artifact.v1"
                and raw.get("cell_id") == spec.cell_id
                and raw.get("repetition_index") == repetition_index
                and isinstance(raw.get("rows"), list)
                and len(raw["rows"]) == len(spec.budget_fractions)
            )
        except CorruptArtifactError:
            return False

    def read_repetition(
        self, spec: CalibrationCellSpec, repetition_index: int
    ) -> tuple[CalibrationReplicateResult, ...]:
        raw = _read_artifact(self._repetition_path(spec, repetition_index))
        if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list):
            raise CorruptArtifactError("calibration repetition has an invalid envelope")
        rows = tuple(CalibrationReplicateResult.model_validate(row) for row in raw["rows"])
        if any(row.cell_id != spec.cell_id for row in rows):
            raise CorruptArtifactError("calibration repetition contains the wrong cell")
        return rows

    def read_completed_cell(
        self, spec: CalibrationCellSpec, *, require_all: bool = True
    ) -> tuple[CalibrationReplicateResult, ...]:
        rows: list[CalibrationReplicateResult] = []
        missing: list[int] = []
        for repetition_index in range(spec.repetitions):
            if self.is_repetition_complete(spec, repetition_index):
                rows.extend(self.read_repetition(spec, repetition_index))
            else:
                missing.append(repetition_index)
        if require_all and missing:
            raise FileNotFoundError(
                f"calibration cell {spec.cell_id} is missing {len(missing)} repetitions"
            )
        return tuple(rows)

    def write_summaries(
        self, spec: CalibrationCellSpec, summaries: tuple[CalibrationBudgetSummary, ...]
    ) -> str:
        if any(summary.cell_id != spec.cell_id for summary in summaries):
            raise ValueError("calibration summary contains the wrong cell")
        return _write_artifact(
            self._cell_root(spec.cell_id) / "summary.json",
            {
                "schema_version": "censure.calibration-cell-summary.v1",
                "cell_id": spec.cell_id,
                "summary_sha256": canonical_sha256(summaries),
                "summaries": [summary.model_dump(mode="json") for summary in summaries],
            },
        )


def calibration_shard(
    *, cell_id: str, repetition_index: int, num_shards: int
) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return int(
        canonical_sha256(
            {
                "schema_version": "censure.calibration-shard.v1",
                "cell_id": cell_id,
                "repetition_index": repetition_index,
            }
        ),
        16,
    ) % num_shards


def calibration_chunk_count(repetitions: int, repetitions_per_chunk: int) -> int:
    if repetitions < 1 or repetitions_per_chunk < 1:
        raise ValueError("repetitions and repetitions_per_chunk must be positive")
    return math.ceil(repetitions / repetitions_per_chunk)


def calibration_chunk_repetition_indices(
    *, repetitions: int, repetitions_per_chunk: int, chunk_index: int
) -> tuple[int, ...]:
    chunk_count = calibration_chunk_count(repetitions, repetitions_per_chunk)
    if not 0 <= chunk_index < chunk_count:
        raise ValueError("chunk_index is outside the frozen repetition range")
    start = chunk_index * repetitions_per_chunk
    stop = min(repetitions, start + repetitions_per_chunk)
    return tuple(range(start, stop))


def calibration_chunk_shard(*, cell_id: str, chunk_index: int, num_shards: int) -> int:
    return calibration_shard(
        cell_id=cell_id, repetition_index=chunk_index, num_shards=num_shards
    )


__all__ = [
    "CalibrationRunStore",
    "calibration_chunk_count",
    "calibration_chunk_repetition_indices",
    "calibration_chunk_shard",
    "calibration_shard",
]
