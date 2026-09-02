"""Append-only, outcome-firewalled persistence for Phase 2 audit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from censure.estimation.schemas import AuditLedger, CertificatePoint, FiniteCohortEnvelope
from censure.serialization import canonical_sha256
from censure.storage import CorruptArtifactError, atomic_write_bytes, atomic_write_json


def _write_checksummed_json(path: Path, value: Any) -> str:
    digest = atomic_write_json(path, value)
    atomic_write_bytes(path.with_suffix(".sha256"), f"{digest}\n".encode())
    return digest


def _read_checksummed_json(path: Path) -> Any:
    digest_path = path.with_suffix(".sha256")
    if not path.is_file() or not digest_path.is_file():
        raise CorruptArtifactError(f"checksummed artifact is missing: {path}")
    raw = path.read_bytes()
    expected = digest_path.read_text(encoding="utf-8").strip()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise CorruptArtifactError(f"artifact checksum mismatch: {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptArtifactError(f"artifact is not valid JSON: {path}") from exc


class AuditorRunStore:
    """Persistence capability containing no unselected target outcomes."""

    def __init__(self, out_root: str | Path, experiment_id: str) -> None:
        self.root = Path(out_root).expanduser().resolve() / experiment_id / "phase2" / "auditor"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def envelope_key(envelope: FiniteCohortEnvelope) -> str:
        return canonical_sha256(
            {
                "schema_version": "censure.envelope-key.v1",
                "protocol_id": envelope.protocol_id,
                "cohort_id": envelope.cohort_id,
            }
        )

    @staticmethod
    def ledger_key(ledger: AuditLedger) -> str:
        return canonical_sha256(
            {
                "schema_version": "censure.audit-ledger-key.v1",
                "protocol_id": ledger.protocol_id,
                "cohort_id": ledger.cohort_id,
                "policy": ledger.policy.value,
                "allocation_seed": ledger.allocation_seed,
                "alpha": ledger.alpha,
                "exploration_epsilon": ledger.exploration_epsilon,
            }
        )

    def _envelope_path(self, key: str) -> Path:
        return self.root / "envelopes" / key / "envelope.json"

    def _ledger_path(self, key: str) -> Path:
        return self.root / "ledgers" / key / "ledger.json"

    def write_envelope(self, envelope: FiniteCohortEnvelope) -> str:
        key = self.envelope_key(envelope)
        path = self._envelope_path(key)
        if path.exists():
            existing = FiniteCohortEnvelope.model_validate(_read_checksummed_json(path))
            if existing != envelope:
                raise FileExistsError(
                    "a different finite-cohort envelope already uses this protocol/cohort identity"
                )
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return _write_checksummed_json(path, envelope)

    def read_envelope(self, *, protocol_id: str, cohort_id: str) -> FiniteCohortEnvelope:
        key = canonical_sha256(
            {
                "schema_version": "censure.envelope-key.v1",
                "protocol_id": protocol_id,
                "cohort_id": cohort_id,
            }
        )
        return FiniteCohortEnvelope.model_validate(_read_checksummed_json(self._envelope_path(key)))

    def write_ledger(self, ledger: AuditLedger) -> str:
        key = self.ledger_key(ledger)
        path = self._ledger_path(key)
        if path.exists():
            existing = AuditLedger.model_validate(_read_checksummed_json(path))
            if len(ledger.disclosures) < len(existing.disclosures):
                raise ValueError("audit ledger persistence cannot truncate completed rounds")
            if ledger.disclosures[: len(existing.disclosures)] != existing.disclosures:
                raise ValueError("audit ledger persistence cannot rewrite disclosed history")
        return _write_checksummed_json(path, ledger)

    def read_ledger(self, template: AuditLedger) -> AuditLedger:
        key = self.ledger_key(template)
        return AuditLedger.model_validate(_read_checksummed_json(self._ledger_path(key)))

    def has_ledger(self, template: AuditLedger) -> bool:
        path = self._ledger_path(self.ledger_key(template))
        return path.is_file() or path.with_suffix(".sha256").is_file()

    def write_certificate_path(
        self, ledger: AuditLedger, points: tuple[CertificatePoint, ...]
    ) -> str:
        if len(points) != len(ledger.disclosures) + 1:
            raise ValueError("certificate path must contain zero-budget plus every audit round")
        if points[-1].round_index != len(ledger.disclosures):
            raise ValueError("certificate path does not terminate at the ledger round")
        key = self.ledger_key(ledger)
        path = self.root / "ledgers" / key / "certificates.json"
        return _write_checksummed_json(path, points)

    def read_certificate_path(self, ledger: AuditLedger) -> tuple[CertificatePoint, ...]:
        key = self.ledger_key(ledger)
        path = self.root / "ledgers" / key / "certificates.json"
        raw = _read_checksummed_json(path)
        if not isinstance(raw, list):
            raise CorruptArtifactError("certificate path is not a JSON array")
        return tuple(CertificatePoint.model_validate(item) for item in raw)


__all__ = ["AuditorRunStore"]
