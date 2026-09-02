"""Exactly evaluable finite-horizon cohorts for estimator conformance."""

from __future__ import annotations

import hashlib
import itertools
import math
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from censure.estimation.allocation import (
    allocation_probabilities,
    maximum_importance_weight,
    propensity_vector_sha256,
)
from censure.estimation.confidence import certificate_path
from censure.estimation.schemas import (
    AllocationPolicyName,
    AuditDisclosure,
    AuditLedger,
    FiniteCohortEnvelope,
    FrontierCandidate,
    PrivateSuffixOutcome,
    SuffixAuditStatus,
)
from censure.schemas import FrozenModel, Identifier, Probability
from censure.serialization import canonical_json_bytes, canonical_sha256


class SupportRegime(str, Enum):
    FULL_OVERLAP = "full_overlap"
    DETERMINISTIC_CLONEABLE_NONOVERLAP = "deterministic_cloneable_nonoverlap"
    MIXED_AUDITABILITY = "mixed_auditability"


class EnumerableUnit(FrozenModel):
    """One frozen unit with an exactly known target path and potential outcome."""

    schema_version: Literal["censure.enumerable-unit.v1"] = "censure.enumerable-unit.v1"
    unit_id: Identifier
    target_path_probability: Annotated[float, Field(gt=0.0, le=1.0)]
    first_support_exit: bool
    auditable: bool
    target_terminal_harm: Probability
    one_step_harm: Probability
    stratum: Literal["lower_risk", "higher_risk"]
    guard_score: Annotated[float, Field(ge=0.0)]
    suffix_cost: Annotated[float, Field(gt=0.0)]
    checkpoint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    random_tape_id: Identifier

    @model_validator(mode="after")
    def validate_path(self) -> EnumerableUnit:
        if not self.first_support_exit and self.auditable:
            raise ValueError("a shared-support unit cannot be a frontier audit candidate")
        if self.one_step_harm > self.target_terminal_harm:
            raise ValueError("the delayed-harm DGP cannot have one-step-only harm")
        return self


class EnumerableCohort(FrozenModel):
    """A finite cohort whose target risk and decomposition are exactly enumerable."""

    schema_version: Literal["censure.enumerable-cohort.v1"] = "censure.enumerable-cohort.v1"
    protocol_id: Identifier
    cohort_id: Identifier
    support_regime: SupportRegime
    generation_seed: Annotated[int, Field(ge=0)]
    units: tuple[EnumerableUnit, ...]

    @model_validator(mode="after")
    def validate_units(self) -> EnumerableCohort:
        if not self.units:
            raise ValueError("enumerable cohort must contain at least one unit")
        if len({unit.unit_id for unit in self.units}) != len(self.units):
            raise ValueError("enumerable unit IDs must be unique")
        if abs(math.fsum(unit.target_path_probability for unit in self.units) - 1.0) > 1e-12:
            raise ValueError("enumerable target path probabilities must sum to one")
        if self.support_regime is SupportRegime.FULL_OVERLAP and any(
            unit.first_support_exit for unit in self.units
        ):
            raise ValueError("full-overlap cohorts cannot contain a support exit")
        return self

    @property
    def exact_target_risk(self) -> float:
        return math.fsum(
            unit.target_path_probability * unit.target_terminal_harm for unit in self.units
        )

    @property
    def exact_one_step_risk(self) -> float:
        return math.fsum(unit.target_path_probability * unit.one_step_harm for unit in self.units)

    @property
    def delayed_harm_rate(self) -> float:
        return math.fsum(
            unit.target_path_probability
            for unit in self.units
            if unit.one_step_harm == 0.0 and unit.target_terminal_harm == 1.0
        )

    def envelope(self) -> FiniteCohortEnvelope:
        supported_harm = math.fsum(
            unit.target_path_probability * unit.target_terminal_harm
            for unit in self.units
            if not unit.first_support_exit
        )
        candidates = tuple(
            FrontierCandidate(
                candidate_id=f"candidate-{unit.unit_id}",
                cohort_unit_id=unit.unit_id,
                source_session_id=f"enumerable-{unit.unit_id}",
                target_mass=unit.target_path_probability,
                auditable=unit.auditable,
                checkpoint_sha256=unit.checkpoint_sha256,
                suffix_random_tape_id=unit.random_tape_id,
                declared_cost=unit.suffix_cost,
                guard_score=unit.guard_score,
                stratum=unit.stratum,
                behavior_features={
                    "support_regime": self.support_regime.value,
                    "risk_stratum": unit.stratum,
                    "guard_score": unit.guard_score,
                    "declared_suffix_cost": unit.suffix_cost,
                },
            )
            for unit in self.units
            if unit.first_support_exit
        )
        return FiniteCohortEnvelope(
            protocol_id=self.protocol_id,
            cohort_id=self.cohort_id,
            cohort_size=len(self.units),
            supported_harm_contribution=supported_harm,
            candidates=candidates,
        )

    def private_outcomes(self, *, auditable_only: bool = True) -> dict[str, PrivateSuffixOutcome]:
        return {
            f"candidate-{unit.unit_id}": PrivateSuffixOutcome(
                candidate_id=f"candidate-{unit.unit_id}",
                status=SuffixAuditStatus.COMPLETED,
                terminal_harm=unit.target_terminal_harm,
                suffix_tool_steps=max(1, math.ceil(unit.suffix_cost)),
            )
            for unit in self.units
            if unit.first_support_exit and (unit.auditable or not auditable_only)
        }

    def decomposition_error(self) -> float:
        envelope = self.envelope()
        safe_frontier_mass = math.fsum(
            unit.target_path_probability * (1.0 - unit.target_terminal_harm)
            for unit in self.units
            if unit.first_support_exit
        )
        return abs(self.exact_target_risk - (envelope.theta_env - safe_frontier_mass))


def _uniform_from_identity(payload: dict[str, object]) -> float:
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _unit_uniform(*, seed: int, unit_index: int, stream: str) -> float:
    return _uniform_from_identity(
        {
            "schema_version": "censure.enumerable-random.v1",
            "seed": seed,
            "unit_index": unit_index,
            "stream": stream,
        }
    )


def generate_enumerable_cohort(
    *,
    protocol_id: str,
    cohort_id: str,
    cohort_size: int,
    support_regime: SupportRegime | str,
    target_harm_prevalence: float,
    zero_support_mass: float,
    generation_seed: int,
    mixed_auditable_probability: float = 0.75,
    delayed_harm_probability: float = 0.60,
) -> EnumerableCohort:
    """Generate a deterministic, exactly scored calibration cohort."""

    regime = SupportRegime(support_regime)
    if cohort_size < 1:
        raise ValueError("cohort_size must be positive")
    for name, value in (
        ("target_harm_prevalence", target_harm_prevalence),
        ("zero_support_mass", zero_support_mass),
        ("mixed_auditable_probability", mixed_auditable_probability),
        ("delayed_harm_probability", delayed_harm_probability),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if generation_seed < 0:
        raise ValueError("generation_seed must be nonnegative")

    path_probability = 1.0 / cohort_size
    units: list[EnumerableUnit] = []
    for unit_index in range(cohort_size):
        unit_id = f"unit-{unit_index:06d}"
        higher_risk = _unit_uniform(
            seed=generation_seed, unit_index=unit_index, stream="stratum"
        ) < 0.5
        stratum: Literal["lower_risk", "higher_risk"] = (
            "higher_risk" if higher_risk else "lower_risk"
        )
        harm_probability = target_harm_prevalence * (1.5 if higher_risk else 0.5)
        harm_probability = min(1.0, harm_probability)
        target_harm = float(
            _unit_uniform(seed=generation_seed, unit_index=unit_index, stream="harm")
            < harm_probability
        )
        support_exit = regime is not SupportRegime.FULL_OVERLAP and (
            _unit_uniform(seed=generation_seed, unit_index=unit_index, stream="support")
            < zero_support_mass
        )
        if regime is SupportRegime.DETERMINISTIC_CLONEABLE_NONOVERLAP:
            auditable = support_exit
        elif regime is SupportRegime.MIXED_AUDITABILITY:
            auditable = support_exit and (
                _unit_uniform(seed=generation_seed, unit_index=unit_index, stream="auditability")
                < mixed_auditable_probability
            )
        else:
            auditable = False
        immediate_harm = target_harm == 1.0 and (
            _unit_uniform(seed=generation_seed, unit_index=unit_index, stream="delay")
            >= delayed_harm_probability
        )
        suffix_cost = 1.0 + math.floor(
            9.0 * _unit_uniform(seed=generation_seed, unit_index=unit_index, stream="cost")
        )
        units.append(
            EnumerableUnit(
                unit_id=unit_id,
                target_path_probability=path_probability,
                first_support_exit=support_exit,
                auditable=auditable,
                target_terminal_harm=target_harm,
                one_step_harm=float(immediate_harm),
                stratum=stratum,
                guard_score=1.0 - harm_probability,
                suffix_cost=suffix_cost,
                checkpoint_sha256=canonical_sha256(
                    {
                        "schema_version": "censure.enumerable-checkpoint.v1",
                        "cohort_id": cohort_id,
                        "unit_id": unit_id,
                        "generation_seed": generation_seed,
                    }
                ),
                random_tape_id=f"seed-{generation_seed}-unit-{unit_index}",
            )
        )
    return EnumerableCohort(
        protocol_id=protocol_id,
        cohort_id=cohort_id,
        support_regime=regime,
        generation_seed=generation_seed,
        units=tuple(units),
    )


def exact_simultaneous_coverage_probability(
    *,
    envelope: FiniteCohortEnvelope,
    all_private_outcomes: dict[str, PrivateSuffixOutcome],
    policy: AllocationPolicyName | str,
    max_rounds: int,
    alpha: float = 0.05,
    exploration_epsilon: float = 0.10,
) -> float:
    """Enumerate every audit path and return simultaneous upper coverage probability."""

    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")
    selected_policy = AllocationPolicyName(policy)
    candidates = envelope.auditable_candidates
    all_candidate_ids = {candidate.candidate_id for candidate in envelope.candidates}
    if set(all_private_outcomes) != all_candidate_ids:
        raise ValueError("exact coverage requires outcomes for every frontier candidate")
    exact_safe_mass = math.fsum(
        candidate.target_mass * (1.0 - (all_private_outcomes[candidate.candidate_id].terminal_harm or 0.0))
        for candidate in envelope.candidates
        if all_private_outcomes[candidate.candidate_id].status is SuffixAuditStatus.COMPLETED
    )
    exact_target_risk = envelope.theta_env - exact_safe_mass
    if not candidates or max_rounds == 0:
        return float(envelope.theta_env + 1e-12 >= exact_target_risk)

    covered_probability = 0.0

    def recurse(history: tuple[AuditDisclosure, ...], path_probability: float) -> None:
        nonlocal covered_probability
        ledger = AuditLedger(
            protocol_id=envelope.protocol_id,
            cohort_id=envelope.cohort_id,
            policy=selected_policy,
            allocation_seed=0,
            alpha=alpha,
            exploration_epsilon=exploration_epsilon,
            disclosures=history,
        )
        if any(
            point.target_risk_ucb + 1e-12 < exact_target_risk
            for point in certificate_path(envelope, ledger)
        ):
            return
        if len(history) == max_rounds:
            covered_probability += path_probability
            return

        probabilities = allocation_probabilities(
            selected_policy,
            candidates,
            history,
            exploration_epsilon=exploration_epsilon,
        )
        bound = maximum_importance_weight(candidates, probabilities)
        propensity_hash = propensity_vector_sha256(probabilities)
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for candidate_id, probability in probabilities.items():
            candidate = candidate_by_id[candidate_id]
            outcome = all_private_outcomes[candidate_id]
            safe_value = (
                1.0 - (outcome.terminal_harm or 0.0)
                if outcome.status is SuffixAuditStatus.COMPLETED
                else 0.0
            )
            disclosure = AuditDisclosure(
                round_index=len(history) + 1,
                candidate_id=candidate_id,
                selected_probability=probability,
                target_mass=candidate.target_mass,
                round_max_importance_weight=bound,
                propensity_vector_sha256=propensity_hash,
                status=outcome.status,
                safe_value=safe_value,
                outcome_sha256=canonical_sha256(outcome),
                suffix_tool_steps=outcome.suffix_tool_steps,
                generation_tokens=outcome.generation_tokens,
            )
            recurse((*history, disclosure), path_probability * probability)

    recurse((), 1.0)
    return covered_probability


def binary_outcome_maps(
    envelope: FiniteCohortEnvelope,
) -> tuple[dict[str, PrivateSuffixOutcome], ...]:
    """Enumerate all binary fixed potential-outcome maps for a tiny frontier."""

    candidate_ids = tuple(candidate.candidate_id for candidate in envelope.candidates)
    return tuple(
        {
            candidate_id: PrivateSuffixOutcome(
                candidate_id=candidate_id,
                status=SuffixAuditStatus.COMPLETED,
                terminal_harm=float(harm),
            )
            for candidate_id, harm in zip(candidate_ids, harms, strict=True)
        }
        for harms in itertools.product((0, 1), repeat=len(candidate_ids))
    )


__all__ = [
    "EnumerableCohort",
    "EnumerableUnit",
    "SupportRegime",
    "binary_outcome_maps",
    "exact_simultaneous_coverage_probability",
    "generate_enumerable_cohort",
]
