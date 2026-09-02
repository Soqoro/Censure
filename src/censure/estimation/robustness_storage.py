"""Atomic chunk persistence for Phase 2 robustness experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from censure.estimation.calibration_storage import (
    calibration_chunk_count,
    calibration_chunk_repetition_indices,
)
from censure.estimation.robustness import (
    RobustnessCellSpec,
    RobustnessReplicateResult,
    RobustnessSummary,
)
from censure.storage import CorruptArtifactError, atomic_write_bytes, atomic_write_json


def _write(path: Path, value: Any) -> str:
    digest = atomic_write_json(path, value)
    atomic_write_bytes(path.with_suffix(".sha256"), f"{digest}\n".encode())
    return digest


def _read(path: Path) -> Any:
    digest_path = path.with_suffix(".sha256")
    if not path.is_file() or not digest_path.is_file():
        raise CorruptArtifactError(f"robustness artifact is absent: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest_path.read_text(encoding="utf-8").strip():
        raise CorruptArtifactError(f"robustness artifact checksum mismatch: {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptArtifactError(f"robustness artifact is invalid JSON: {path}") from exc


class RobustnessRunStore:
    def __init__(self, out_root: str | Path, experiment_id: str) -> None:
        self.root = (
            Path(out_root).expanduser().resolve()
            / experiment_id
            / "phase2"
            / "robustness"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _cell_root(self, spec: RobustnessCellSpec) -> Path:
        return self.root / "cells" / spec.cell_id

    def write_catalog(self, catalog: Any) -> str:
        payload = catalog.model_dump(mode="json")
        path = self.root / "catalog" / "resolved_catalog.json"
        if path.exists():
            existing = _read(path)
            if existing != payload:
                raise FileExistsError("the existing robustness catalog differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write(path, payload)

    def write_cell_spec(self, spec: RobustnessCellSpec) -> str:
        path = self._cell_root(spec) / "cell_spec.json"
        if path.exists():
            existing = RobustnessCellSpec.model_validate(_read(path))
            if existing != spec:
                raise FileExistsError("the existing robustness cell differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write(path, spec)

    def _chunk_path(self, spec: RobustnessCellSpec, chunk_index: int) -> Path:
        return self._cell_root(spec) / "chunks" / f"{chunk_index:04d}.json"

    def write_chunk(
        self,
        spec: RobustnessCellSpec,
        *,
        chunk_index: int,
        repetitions_per_chunk: int,
        rows: tuple[RobustnessReplicateResult, ...],
    ) -> str:
        expected = calibration_chunk_repetition_indices(
            repetitions=spec.repetitions,
            repetitions_per_chunk=repetitions_per_chunk,
            chunk_index=chunk_index,
        )
        if len(rows) != len(expected) or {row.repetition_index for row in rows} != set(expected):
            raise ValueError("robustness chunk has the wrong repetition set")
        if any(row.cell_id != spec.cell_id for row in rows):
            raise ValueError("robustness chunk contains the wrong cell")
        payload = {
            "schema_version": "censure.robustness-chunk.v1",
            "cell_id": spec.cell_id,
            "chunk_index": chunk_index,
            "repetitions_per_chunk": repetitions_per_chunk,
            "repetition_indices": list(expected),
            "rows": [row.model_dump(mode="json") for row in rows],
        }
        path = self._chunk_path(spec, chunk_index)
        if path.exists():
            existing = _read(path)
            if existing != payload:
                raise FileExistsError("the completed robustness chunk differs")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write(path, payload)

    def is_chunk_complete(
        self,
        spec: RobustnessCellSpec,
        *,
        chunk_index: int,
        repetitions_per_chunk: int,
    ) -> bool:
        expected = calibration_chunk_repetition_indices(
            repetitions=spec.repetitions,
            repetitions_per_chunk=repetitions_per_chunk,
            chunk_index=chunk_index,
        )
        try:
            raw = _read(self._chunk_path(spec, chunk_index))
            return bool(
                isinstance(raw, dict)
                and raw.get("schema_version") == "censure.robustness-chunk.v1"
                and raw.get("cell_id") == spec.cell_id
                and raw.get("chunk_index") == chunk_index
                and raw.get("repetition_indices") == list(expected)
                and isinstance(raw.get("rows"), list)
                and len(raw["rows"]) == len(expected)
            )
        except CorruptArtifactError:
            return False

    def read_chunk(
        self,
        spec: RobustnessCellSpec,
        *,
        chunk_index: int,
        repetitions_per_chunk: int,
    ) -> tuple[RobustnessReplicateResult, ...]:
        if not self.is_chunk_complete(
            spec,
            chunk_index=chunk_index,
            repetitions_per_chunk=repetitions_per_chunk,
        ):
            raise CorruptArtifactError("robustness chunk is absent, corrupt, or incomplete")
        raw = _read(self._chunk_path(spec, chunk_index))
        return tuple(RobustnessReplicateResult.model_validate(row) for row in raw["rows"])

    def read_completed_cell(
        self, spec: RobustnessCellSpec, *, repetitions_per_chunk: int
    ) -> tuple[RobustnessReplicateResult, ...]:
        rows: list[RobustnessReplicateResult] = []
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
        if missing:
            raise FileNotFoundError(
                f"robustness cell {spec.cell_id} is missing {len(missing)} chunks"
            )
        return tuple(rows)

    def write_summary(self, spec: RobustnessCellSpec, summary: RobustnessSummary) -> str:
        if summary.cell_id != spec.cell_id:
            raise ValueError("robustness summary contains the wrong cell")
        return _write(self._cell_root(spec) / "summary.json", summary)


__all__ = ["RobustnessRunStore"]
