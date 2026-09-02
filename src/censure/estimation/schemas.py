"""Outcome-separated contracts for Phase 2 frontier suffix auditing."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from censure.schemas import FrozenModel, Identifier, Probability, Sha256Hex

PositiveProbability = Annotated[float, Field(gt=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]

_FORBIDDEN_FEATURE_KEY_FRAGMENTS = (
    "oracle",
    "suffix_outcome",
    "target_harm",
    "terminal_harm",
    "target_final_state",
    "future_target_call",
    "safe_value",
)


def _assert_outcome_free(value: JsonValue, *, path: str = "behavior_features") -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = raw_key.casefold()
            if any(fragment in key for fragment in _FORBIDDEN_FEATURE_KEY_FRAGMENTS):
                raise ValueError(f"allocator-visible feature contains forbidden key at {path}.{raw_key}")
            _assert_outcome_free(item, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_outcome_free(item, path=f"{path}[{index}]")


class AllocationPolicyName(str, Enum):
    UNIFORM = "uniform"
    TARGET_MASS = "target_mass"
    GUARD_SCORE = "guard_score"
    UNCERTAINTY = "uncertainty"
    DOWNSTREAM_HARM = "downstream_harm"
    CENSURE_BOUND_TARGETED = "censure_bound_targeted"


class SuffixAuditStatus(str, Enum):
    COMPLETED = "completed"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    RESTORE_ERROR = "restore_error"
    EVALUATION_ERROR = "evaluation_error"


class FrontierCandidate(FrozenModel):
    """One outcome-free, first-support-exit suffix candidate.

    The allocator may inspect every field in this record. Consequently, target
    outcomes and future target trajectory content are structurally absent.
    """

    schema_version: Literal["censure.frontier-candidate.v1"] = "censure.frontier-candidate.v1"
    candidate_id: Identifier
    cohort_unit_id: Identifier
    source_session_id: Identifier
    target_mass: PositiveProbability
    auditable: bool
    checkpoint_sha256: Sha256Hex
    suffix_random_tape_id: Identifier
    declared_cost: PositiveFloat = 1.0
    guard_score: NonNegativeFloat = 0.0
    stratum: Identifier = "all"
    behavior_features: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_free_features(self) -> FrontierCandidate:
        _assert_outcome_free(self.behavior_features)
        return self


class FiniteCohortEnvelope(FrozenModel):
    """Known upper envelope and complete first-divergence frontier."""

    schema_version: Literal["censure.finite-cohort-envelope.v1"] = (
        "censure.finite-cohort-envelope.v1"
    )
    protocol_id: Identifier
    cohort_id: Identifier
    cohort_size: Annotated[int, Field(ge=1)]
    supported_harm_contribution: Probability
    candidates: tuple[FrontierCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_frontier(self) -> FiniteCohortEnvelope:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("frontier candidate IDs must be unique")
        unit_tapes = [
            (candidate.cohort_unit_id, candidate.suffix_random_tape_id)
            for candidate in self.candidates
        ]
        if len(set(unit_tapes)) != len(unit_tapes):
            raise ValueError("cohort unit/random-tape candidate identities must be unique")
        if self.theta_env > 1.0 + 1e-12:
            raise ValueError("supported harm plus target frontier mass exceeds one")
        return self

    @property
    def target_frontier_mass(self) -> float:
        return math.fsum(candidate.target_mass for candidate in self.candidates)

    @property
    def auditable_mass(self) -> float:
        return math.fsum(
            candidate.target_mass for candidate in self.candidates if candidate.auditable
        )

    @property
    def nonauditable_mass(self) -> float:
        return self.target_frontier_mass - self.auditable_mass

    @property
    def theta_env(self) -> float:
        return self.supported_harm_contribution + self.target_frontier_mass

    @property
    def auditable_candidates(self) -> tuple[FrontierCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.auditable)


class PrivateSuffixOutcome(FrozenModel):
    """Evaluation-only fixed potential outcome for one candidate."""

    schema_version: Literal["censure.private-suffix-outcome.v1"] = (
        "censure.private-suffix-outcome.v1"
    )
    candidate_id: Identifier
    status: SuffixAuditStatus
    terminal_harm: Probability | None = None
    suffix_tool_steps: NonNegativeInt = 0
    generation_tokens: NonNegativeInt = 0
    terminal_state_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> PrivateSuffixOutcome:
        if self.status is SuffixAuditStatus.COMPLETED and self.terminal_harm is None:
            raise ValueError("completed suffix outcomes require terminal_harm")
        if self.status is not SuffixAuditStatus.COMPLETED and self.terminal_harm is not None:
            raise ValueError("failed suffix outcomes cannot contribute terminal_harm")
        return self


class AuditDisclosure(FrozenModel):
    """The only selected-suffix information disclosed to the allocator."""

    schema_version: Literal["censure.audit-disclosure.v1"] = "censure.audit-disclosure.v1"
    round_index: Annotated[int, Field(ge=1)]
    candidate_id: Identifier
    selected_probability: PositiveProbability
    target_mass: PositiveProbability
    round_max_importance_weight: PositiveFloat
    propensity_vector_sha256: Sha256Hex
    status: SuffixAuditStatus
    safe_value: Probability
    outcome_sha256: Sha256Hex
    suffix_tool_steps: NonNegativeInt = 0
    generation_tokens: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_failure_rule(self) -> AuditDisclosure:
        if self.status is not SuffixAuditStatus.COMPLETED and self.safe_value != 0.0:
            raise ValueError("failed suffix audits must disclose zero safe mass")
        selected_weight = self.target_mass / self.selected_probability
        if selected_weight > self.round_max_importance_weight + 1e-12:
            raise ValueError("round maximum importance weight is below the selected weight")
        return self

    @property
    def importance_safe_value(self) -> float:
        return self.target_mass * self.safe_value / self.selected_probability


class AuditLedger(FrozenModel):
    """Append-only audit history sufficient for deterministic replay."""

    schema_version: Literal["censure.audit-ledger.v1"] = "censure.audit-ledger.v1"
    protocol_id: Identifier
    cohort_id: Identifier
    policy: AllocationPolicyName
    allocation_seed: NonNegativeInt
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)]
    exploration_epsilon: Probability
    disclosures: tuple[AuditDisclosure, ...] = ()

    @model_validator(mode="after")
    def validate_rounds(self) -> AuditLedger:
        rounds = tuple(disclosure.round_index for disclosure in self.disclosures)
        expected = tuple(range(1, len(self.disclosures) + 1))
        if rounds != expected:
            raise ValueError("audit disclosure rounds must be contiguous and one-indexed")
        return self


class CertificatePoint(FrozenModel):
    """One anytime-valid finite-cohort target-risk certificate."""

    schema_version: Literal["censure.certificate-point.v1"] = "censure.certificate-point.v1"
    protocol_id: Identifier
    cohort_id: Identifier
    policy: AllocationPolicyName
    round_index: NonNegativeInt
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)]
    theta_env: Probability
    target_frontier_mass: Probability
    auditable_mass: Probability
    nonauditable_mass: Probability
    cumulative_importance_safe_value: NonNegativeFloat
    cumulative_bound_squared: NonNegativeFloat
    stitched_boundary: NonNegativeFloat
    instantaneous_safe_mass_lcb: Probability
    safe_mass_lcb: Probability
    target_risk_ucb: Probability
    unique_audited_candidate_count: NonNegativeInt
    duplicate_draw_count: NonNegativeInt
    failed_audit_count: NonNegativeInt
__all__ = [
    "AllocationPolicyName",
    "AuditDisclosure",
    "AuditLedger",
    "CertificatePoint",
    "FiniteCohortEnvelope",
    "FrontierCandidate",
    "PrivateSuffixOutcome",
    "SuffixAuditStatus",
]
