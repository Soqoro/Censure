"""One-factor-at-a-time robustness experiments for the CENSURE certificate."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from censure.estimation.auditor import CensureAuditor, InMemoryEvaluationOracle
from censure.estimation.calibration import clopper_pearson_one_sided
from censure.estimation.enumerable import SupportRegime, generate_enumerable_cohort
from censure.estimation.schemas import (
    AllocationPolicyName,
    FiniteCohortEnvelope,
    FrontierCandidate,
    PrivateSuffixOutcome,
    SuffixAuditStatus,
)
from censure.schemas import FrozenModel, Identifier, Probability, Sha256Hex
from censure.serialization import canonical_json_bytes, canonical_sha256


class RobustnessAxis(str, Enum):
    NONAUDITABLE_MASS = "nonauditable_mass"
    HIDDEN_GUARD_FEATURE = "hidden_guard_feature_prevalence"
    SANDBOX_HARM_SHIFT = "sandbox_harm_shift"
    SANDBOX_TRANSITION_SHIFT = "sandbox_transition_shift"
    RARE_TARGET_HARM = "rare_target_harm"
    OUTCOME_MODEL_CONDITION = "outcome_model_condition"


class OutcomeModelCondition(str, Enum):
    CORRECT = "correct"
    MISSPECIFIED = "misspecified"
    CONSTANT = "constant"


class RobustnessCellSpec(FrozenModel):
    schema_version: Literal["censure.robustness-cell.v1"] = "censure.robustness-cell.v1"
    protocol_id: Identifier
    seed_namespace: Identifier
    base_seed: Annotated[int, Field(ge=0)]
    axis: RobustnessAxis
    level: float | OutcomeModelCondition
    cohort_size: Annotated[int, Field(ge=1)] = 500
    baseline_target_harm_prevalence: Probability = 0.20
    zero_support_mass: Probability = 0.50
    budget_fraction: Probability = 0.20
    repetitions: Annotated[int, Field(ge=1)] = 2000
    policy: AllocationPolicyName = AllocationPolicyName.TARGET_MASS
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    exploration_epsilon: Probability = 0.10

    @model_validator(mode="after")
    def validate_axis_level(self) -> RobustnessCellSpec:
        if self.axis is RobustnessAxis.OUTCOME_MODEL_CONDITION:
            if not isinstance(self.level, OutcomeModelCondition):
                raise ValueError("outcome-model axis requires a named model condition")
            if self.policy is not AllocationPolicyName.CENSURE_BOUND_TARGETED:
                raise ValueError("outcome-model robustness uses censure_bound_targeted")
        else:
            if isinstance(self.level, OutcomeModelCondition):
                raise ValueError("numeric robustness axes require a numeric level")
            if not math.isfinite(self.level) or not 0.0 <= self.level <= 1.0:
                raise ValueError("numeric robustness level must lie in [0, 1]")
        return self

    @property
    def cell_id(self) -> str:
        return canonical_sha256(self)

    @property
    def target_harm_prevalence(self) -> float:
        if self.axis is RobustnessAxis.RARE_TARGET_HARM:
            if isinstance(self.level, OutcomeModelCondition):  # pragma: no cover - validator
                raise AssertionError("rare-harm level is not numeric")
            return self.level
        return self.baseline_target_harm_prevalence


class RobustnessReplicateResult(FrozenModel):
    schema_version: Literal["censure.robustness-replicate.v1"] = (
        "censure.robustness-replicate.v1"
    )
    cell_id: Sha256Hex
    repetition_index: Annotated[int, Field(ge=0)]
    cohort_id: Identifier
    axis: RobustnessAxis
    level: float | OutcomeModelCondition
    policy: AllocationPolicyName
    exact_deployment_target_risk: Probability
    sandbox_target_risk: Probability
    observed_theta_env: Probability
    observed_frontier_mass: Probability
    observed_auditable_mass: Probability
    realized_hidden_frontier_mass: Probability
    realized_nonauditable_mass: Probability
    declared_sensitivity_radius: Probability
    target_risk_ucb: Probability
    sensitivity_corrected_ucb: Probability
    covered: bool
    sensitivity_corrected_covered: bool
    assumption_satisfied: bool
    audit_rounds: Annotated[int, Field(ge=0)]
    unique_audited_candidate_count: Annotated[int, Field(ge=0)]
    duplicate_draw_count: Annotated[int, Field(ge=0)]


class RobustnessSummary(FrozenModel):
    schema_version: Literal["censure.robustness-summary.v1"] = "censure.robustness-summary.v1"
    cell_id: Sha256Hex
    axis: RobustnessAxis
    level: float | OutcomeModelCondition
    policy: AllocationPolicyName
    repetition_count: Annotated[int, Field(ge=1)]
    coverage: Probability
    coverage_one_sided_95_lower: Probability
    coverage_one_sided_95_upper: Probability
    corrected_coverage: Probability
    corrected_coverage_one_sided_95_lower: Probability
    corrected_coverage_one_sided_95_upper: Probability
    mean_upper_slack: float
    mean_corrected_upper_slack: float
    mean_realized_hidden_frontier_mass: Probability
    mean_realized_nonauditable_mass: Probability


def _uniform(payload: dict[str, object]) -> float:
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _derive_seed(payload: dict[str, object]) -> int:
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _candidate_uniform(candidate_id: str, *, stream: str) -> float:
    return _uniform(
        {
            "schema_version": "censure.robustness-candidate-random.v1",
            "candidate_id": candidate_id,
            "stream": stream,
        }
    )


def _replace_candidate(
    candidate: FrontierCandidate, **updates: object
) -> FrontierCandidate:
    raw = candidate.model_dump(mode="python")
    raw.update(updates)
    return FrontierCandidate.model_validate(raw)


def _transform_envelope(
    envelope: FiniteCohortEnvelope, spec: RobustnessCellSpec
) -> tuple[FiniteCohortEnvelope, float]:
    hidden_mass = 0.0
    candidates: list[FrontierCandidate] = []
    for candidate in envelope.candidates:
        if spec.axis is RobustnessAxis.HIDDEN_GUARD_FEATURE:
            if isinstance(spec.level, OutcomeModelCondition):  # pragma: no cover - validator
                raise AssertionError("hidden-feature level is not numeric")
            hidden = _candidate_uniform(candidate.candidate_id, stream="hidden") < spec.level
            if hidden:
                hidden_mass += candidate.target_mass
                continue
        transformed = candidate
        if spec.axis is RobustnessAxis.NONAUDITABLE_MASS:
            if isinstance(spec.level, OutcomeModelCondition):  # pragma: no cover - validator
                raise AssertionError("non-auditable level is not numeric")
            transformed = _replace_candidate(
                transformed,
                auditable=(
                    _candidate_uniform(candidate.candidate_id, stream="nonauditable")
                    >= spec.level
                ),
            )
        if spec.axis is RobustnessAxis.OUTCOME_MODEL_CONDITION:
            condition = OutcomeModelCondition(spec.level)
            if condition is OutcomeModelCondition.MISSPECIFIED:
                unrelated_stratum = (
                    "unrelated_a"
                    if _candidate_uniform(candidate.candidate_id, stream="misspecified") < 0.5
                    else "unrelated_b"
                )
                transformed = _replace_candidate(
                    transformed,
                    guard_score=1.0 - transformed.guard_score,
                    stratum=unrelated_stratum,
                    behavior_features={
                        **transformed.behavior_features,
                        "outcome_model_condition": condition.value,
                        "model_stratum": unrelated_stratum,
                    },
                )
            elif condition is OutcomeModelCondition.CONSTANT:
                transformed = _replace_candidate(
                    transformed,
                    guard_score=1.0,
                    stratum="constant",
                    behavior_features={
                        **transformed.behavior_features,
                        "outcome_model_condition": condition.value,
                        "model_stratum": "constant",
                    },
                )
        candidates.append(transformed)
    return (
        FiniteCohortEnvelope(
            protocol_id=envelope.protocol_id,
            cohort_id=envelope.cohort_id,
            cohort_size=envelope.cohort_size,
            supported_harm_contribution=envelope.supported_harm_contribution,
            candidates=tuple(candidates),
        ),
        hidden_mass,
    )


def _sandbox_outcomes(
    *,
    envelope: FiniteCohortEnvelope,
    target_outcomes: dict[str, PrivateSuffixOutcome],
    shift_radius: float,
    stream: str,
) -> tuple[dict[str, PrivateSuffixOutcome], float]:
    sandbox = dict(target_outcomes)
    shifted_mass = 0.0
    candidate_by_id = {candidate.candidate_id: candidate for candidate in envelope.candidates}
    ordered_ids = sorted(
        candidate_by_id,
        key=lambda candidate_id: _candidate_uniform(candidate_id, stream=stream),
    )
    for candidate_id in ordered_ids:
        outcome = sandbox[candidate_id]
        candidate = candidate_by_id[candidate_id]
        if outcome.status is not SuffixAuditStatus.COMPLETED or outcome.terminal_harm != 1.0:
            continue
        if shifted_mass + candidate.target_mass > shift_radius + 1e-12:
            continue
        sandbox[candidate_id] = PrivateSuffixOutcome(
            candidate_id=candidate_id,
            status=SuffixAuditStatus.COMPLETED,
            terminal_harm=0.0,
            suffix_tool_steps=outcome.suffix_tool_steps,
            generation_tokens=outcome.generation_tokens,
            terminal_state_sha256=outcome.terminal_state_sha256,
        )
        shifted_mass += candidate.target_mass
    return sandbox, shifted_mass


def run_robustness_repetition(
    spec: RobustnessCellSpec, repetition_index: int
) -> RobustnessReplicateResult:
    if not 0 <= repetition_index < spec.repetitions:
        raise ValueError("repetition_index is outside the robustness cell")
    cohort_identity = {
        "schema_version": "censure.robustness-cohort-identity.v1",
        "protocol_id": spec.protocol_id,
        "seed_namespace": spec.seed_namespace,
        "base_seed": spec.base_seed,
        "cohort_size": spec.cohort_size,
        "target_harm_prevalence": spec.target_harm_prevalence,
        "zero_support_mass": spec.zero_support_mass,
        "repetition_index": repetition_index,
    }
    cohort_id = f"robustness-{canonical_sha256(cohort_identity)}"
    cohort = generate_enumerable_cohort(
        protocol_id=spec.protocol_id,
        cohort_id=cohort_id,
        cohort_size=spec.cohort_size,
        support_regime=SupportRegime.DETERMINISTIC_CLONEABLE_NONOVERLAP,
        target_harm_prevalence=spec.target_harm_prevalence,
        zero_support_mass=spec.zero_support_mass,
        generation_seed=_derive_seed({**cohort_identity, "stream": "cohort"}),
    )
    complete_envelope = cohort.envelope()
    observed_envelope, hidden_mass = _transform_envelope(complete_envelope, spec)
    target_outcomes = cohort.private_outcomes(auditable_only=False)
    declared_radius = 0.0
    shifted_mass = 0.0
    sandbox_outcomes = target_outcomes
    if spec.axis in {
        RobustnessAxis.SANDBOX_HARM_SHIFT,
        RobustnessAxis.SANDBOX_TRANSITION_SHIFT,
    }:
        if isinstance(spec.level, OutcomeModelCondition):  # pragma: no cover - validator
            raise AssertionError("sandbox-shift level is not numeric")
        declared_radius = spec.level
        sandbox_outcomes, shifted_mass = _sandbox_outcomes(
            envelope=complete_envelope,
            target_outcomes=target_outcomes,
            shift_radius=declared_radius,
            stream=spec.axis.value,
        )
    observed_outcomes = {
        candidate.candidate_id: sandbox_outcomes[candidate.candidate_id]
        for candidate in observed_envelope.auditable_candidates
    }
    audit_rounds = (
        math.ceil(spec.budget_fraction * len(observed_envelope.auditable_candidates))
        if spec.budget_fraction > 0.0 and observed_envelope.auditable_candidates
        else 0
    )
    auditor = CensureAuditor(
        envelope=observed_envelope,
        oracle=InMemoryEvaluationOracle(observed_outcomes),
        policy=spec.policy,
        allocation_seed=_derive_seed({**cohort_identity, "stream": "audit"}),
        alpha=spec.alpha,
        exploration_epsilon=spec.exploration_epsilon,
    )
    _, points = auditor.run(total_rounds=audit_rounds)
    point = points[-1]
    corrected_ucb = min(1.0, point.target_risk_ucb + declared_radius)
    exact_risk = cohort.exact_target_risk
    sandbox_risk = max(0.0, exact_risk - shifted_mass)
    assumption_satisfied = not (
        (
            spec.axis is RobustnessAxis.HIDDEN_GUARD_FEATURE
            and not isinstance(spec.level, OutcomeModelCondition)
            and spec.level > 0.0
        )
        or declared_radius > 0.0
    )
    return RobustnessReplicateResult(
        cell_id=spec.cell_id,
        repetition_index=repetition_index,
        cohort_id=cohort_id,
        axis=spec.axis,
        level=spec.level,
        policy=spec.policy,
        exact_deployment_target_risk=exact_risk,
        sandbox_target_risk=sandbox_risk,
        observed_theta_env=point.theta_env,
        observed_frontier_mass=point.target_frontier_mass,
        observed_auditable_mass=point.auditable_mass,
        realized_hidden_frontier_mass=hidden_mass,
        realized_nonauditable_mass=point.nonauditable_mass,
        declared_sensitivity_radius=declared_radius,
        target_risk_ucb=point.target_risk_ucb,
        sensitivity_corrected_ucb=corrected_ucb,
        covered=point.target_risk_ucb + 1e-12 >= exact_risk,
        sensitivity_corrected_covered=corrected_ucb + 1e-12 >= exact_risk,
        assumption_satisfied=assumption_satisfied,
        audit_rounds=audit_rounds,
        unique_audited_candidate_count=point.unique_audited_candidate_count,
        duplicate_draw_count=point.duplicate_draw_count,
    )


def run_robustness_cell(
    spec: RobustnessCellSpec,
) -> tuple[RobustnessReplicateResult, ...]:
    return tuple(
        run_robustness_repetition(spec, repetition_index)
        for repetition_index in range(spec.repetitions)
    )


def summarize_robustness_results(
    results: Sequence[RobustnessReplicateResult],
) -> RobustnessSummary:
    if not results:
        raise ValueError("cannot summarize empty robustness results")
    first = results[0]
    if any(
        row.cell_id != first.cell_id
        or row.axis is not first.axis
        or row.level != first.level
        or row.policy is not first.policy
        for row in results
    ):
        raise ValueError("robustness summary requires one homogeneous cell")
    coverage_count = sum(row.covered for row in results)
    corrected_count = sum(row.sensitivity_corrected_covered for row in results)
    coverage_lower, coverage_upper = clopper_pearson_one_sided(
        coverage_count, len(results)
    )
    corrected_lower, corrected_upper = clopper_pearson_one_sided(
        corrected_count, len(results)
    )
    return RobustnessSummary(
        cell_id=first.cell_id,
        axis=first.axis,
        level=first.level,
        policy=first.policy,
        repetition_count=len(results),
        coverage=coverage_count / len(results),
        coverage_one_sided_95_lower=coverage_lower,
        coverage_one_sided_95_upper=coverage_upper,
        corrected_coverage=corrected_count / len(results),
        corrected_coverage_one_sided_95_lower=corrected_lower,
        corrected_coverage_one_sided_95_upper=corrected_upper,
        mean_upper_slack=statistics.fmean(
            row.target_risk_ucb - row.exact_deployment_target_risk for row in results
        ),
        mean_corrected_upper_slack=statistics.fmean(
            row.sensitivity_corrected_ucb - row.exact_deployment_target_risk
            for row in results
        ),
        mean_realized_hidden_frontier_mass=statistics.fmean(
            row.realized_hidden_frontier_mass for row in results
        ),
        mean_realized_nonauditable_mass=statistics.fmean(
            row.realized_nonauditable_mass for row in results
        ),
    )


__all__ = [
    "OutcomeModelCondition",
    "RobustnessAxis",
    "RobustnessCellSpec",
    "RobustnessReplicateResult",
    "RobustnessSummary",
    "run_robustness_cell",
    "run_robustness_repetition",
    "summarize_robustness_results",
]
