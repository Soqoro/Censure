"""Atomic, resumable, oracle-separated persistence for Drive-backed runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from censure.serialization import canonical_json_bytes, canonical_sha256


class CorruptArtifactError(RuntimeError):
    pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write and rename within the destination directory, then fsync metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some Drive/FUSE implementations do not expose directory fsync.
            # Artifact checksums still detect partial persistence on resume.
            pass
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: Any) -> str:
    data = canonical_json_bytes(value) + b"\n"
    atomic_write_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def atomic_write_gzip_json(path: Path, value: Any) -> str:
    raw = canonical_json_bytes(value) + b"\n"
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    atomic_write_bytes(path, compressed)
    return hashlib.sha256(compressed).hexdigest()


def session_key(scientific_fields: Any) -> str:
    """Globally unique key over a complete scientifically relevant row."""

    return canonical_sha256({"key_schema": "censure.session.v1", "fields": scientific_fields})


class RunStore:
    """Public store view. Oracle reads are intentionally not exposed."""

    def __init__(self, out_root: str | Path, experiment_id: str) -> None:
        self.root = Path(out_root).expanduser().resolve() / experiment_id
        self.root.mkdir(parents=True, exist_ok=True)

    def _stage_dir(self, role: Literal["behavior", "target"]) -> Path:
        return self.root / ("behavior" if role == "behavior" else "oracle_private")

    def _paths(
        self, session_id: str, role: Literal["behavior", "target"]
    ) -> tuple[Path, Path, Path]:
        base = self._stage_dir(role) / session_id
        return base / "summary.json", base / "trace.json.gz", base / "complete.json"

    def write_manifest(self, manifest: Any) -> str:
        manifest_path = self.root / "manifest" / "frozen_manifest.json"
        digest = atomic_write_json(manifest_path, manifest)
        atomic_write_bytes(manifest_path.with_suffix(".sha256"), f"{digest}\n".encode())
        return digest

    def write_resolved_config(self, config: Any) -> str:
        return atomic_write_json(self.root / "provenance" / "resolved_config.json", config)

    def write_trajectory(
        self,
        *,
        session_id: str,
        role: Literal["behavior", "target"],
        summary: Any,
        trace: Any,
        force: bool = False,
    ) -> None:
        summary_path, trace_path, complete_path = self._paths(session_id, role)
        if self.is_complete(session_id=session_id, role=role) and not force:
            raise FileExistsError(
                f"valid {role} result already exists for {session_id}; use --force"
            )
        summary_hash = atomic_write_json(summary_path, summary)
        trace_hash = atomic_write_gzip_json(trace_path, trace)
        marker = {
            "schema_version": "censure.completion.v1",
            "session_id": session_id,
            "role": role,
            "summary_sha256": summary_hash,
            "trace_sha256": trace_hash,
        }
        atomic_write_json(complete_path, marker)

    def write_failure_record(
        self,
        *,
        session_id: str,
        role: Literal["behavior", "target"],
        record: Any,
    ) -> None:
        """Persist the latest failure provenance beside its protected trajectory."""

        path = self._stage_dir(role) / session_id / "failure.json"
        atomic_write_json(path, record)

    def is_complete(self, *, session_id: str, role: Literal["behavior", "target"]) -> bool:
        summary_path, trace_path, complete_path = self._paths(session_id, role)
        if not all(path.is_file() for path in (summary_path, trace_path, complete_path)):
            return False
        try:
            marker = json.loads(complete_path.read_text(encoding="utf-8"))
            return bool(
                marker.get("schema_version") == "censure.completion.v1"
                and marker.get("session_id") == session_id
                and marker.get("role") == role
                and marker.get("summary_sha256")
                == hashlib.sha256(summary_path.read_bytes()).hexdigest()
                and marker.get("trace_sha256")
                == hashlib.sha256(trace_path.read_bytes()).hexdigest()
            )
        except (OSError, json.JSONDecodeError):
            return False

    def read_behavior_summary(self, session_id: str) -> dict[str, Any]:
        if not self.is_complete(session_id=session_id, role="behavior"):
            raise CorruptArtifactError(f"behavior result is absent or corrupt: {session_id}")
        path, _, _ = self._paths(session_id, "behavior")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CorruptArtifactError(f"behavior summary is not an object: {session_id}")
        return value

    def read_behavior_trace(self, session_id: str) -> Any:
        if not self.is_complete(session_id=session_id, role="behavior"):
            raise CorruptArtifactError(f"behavior result is absent or corrupt: {session_id}")
        _, path, _ = self._paths(session_id, "behavior")
        try:
            return json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptArtifactError(f"behavior trace cannot be decoded: {session_id}") from exc

    def read_oracle_summary(self, session_id: str) -> dict[str, Any]:
        raise PermissionError(
            "oracle_private is evaluation-gated; request RunStore.evaluation_view(evaluation=True)"
        )

    def evaluation_view(self, *, evaluation: bool = False) -> EvaluationRunStore:
        if not evaluation:
            raise PermissionError("explicit evaluation=True is required for oracle access")
        return EvaluationRunStore(self)


class EvaluationRunStore:
    """Capability object used only by validate/analyze stages."""

    def __init__(self, public_store: RunStore) -> None:
        self._public = public_store

    @property
    def root(self) -> Path:
        return self._public.root

    def read_oracle_summary(self, session_id: str) -> dict[str, Any]:
        if not self._public.is_complete(session_id=session_id, role="target"):
            raise CorruptArtifactError(f"oracle result is absent or corrupt: {session_id}")
        path, _, _ = self._public._paths(session_id, "target")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CorruptArtifactError(f"oracle summary is not an object: {session_id}")
        return value

    def read_oracle_trace(self, session_id: str) -> Any:
        if not self._public.is_complete(session_id=session_id, role="target"):
            raise CorruptArtifactError(f"oracle result is absent or corrupt: {session_id}")
        _, path, _ = self._public._paths(session_id, "target")
        try:
            return json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptArtifactError(f"oracle trace cannot be decoded: {session_id}") from exc

    def write_paired_result(self, session_id: str, result: Any) -> None:
        atomic_write_json(self.root / "paired_private" / session_id / "paired.json", result)


def deterministic_shard(session_id: str, *, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % num_shards
