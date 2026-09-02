"""Prospective enumerable calibration and audit-efficiency experiment runner."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from censure.estimation.auditor import CensureAuditor, InMemoryEvaluationOracle
from censure.estimation.enumerable import SupportRegime, generate_enumerable_cohort
from censure.estimation.schemas import AllocationPolicyName, CertificatePoint
from censure.schemas import FrozenModel, Identifier, Probability, Sha256Hex
from censure.serialization import canonical_json_bytes, canonical_sha256


class CalibrationCellSpec(FrozenModel):
    schema_version: Literal["censure.calibration-cell.v1"] = "censure.calibration-cell.v1"
    protocol_id: Identifier
    seed_namespace: Identifier
    base_seed: Annotated[int, Field(ge=0)]
    support_regime: SupportRegime
    cohort_size: Annotated[int, Field(ge=1)]
    target_harm_prevalence: Probability
    zero_support_mass: Probability
    mixed_auditable_probability: Probability = 0.75
    delayed_harm_probability: Probability = 0.60
    policy: AllocationPolicyName
    budget_fractions: tuple[Probability, ...]
    repetitions: Annotated[int, Field(ge=1)]
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    exploration_epsilon: Probability = 0.10
    release_threshold_eta: Probability = 0.10

    @model_validator(mode="after")
    def validate_budgets(self) -> CalibrationCellSpec:
        if not self.budget_fractions:
            raise ValueError("calibration cell requires at least one audit budget")
        if tuple(sorted(set(self.budget_fractions))) != self.budget_fractions:
            raise ValueError("audit budget fractions must be unique and increasing")
        return self

    @property
    def cell_id(self) -> str:
        return canonical_sha256(self)


class CalibrationReplicateResult(FrozenModel):
    schema_version: Literal["censure.calibration-replicate.v1"] = (
        "censure.calibration-replicate.v1"
    )
    cell_id: Sha256Hex
    repetition_index: Annotated[int, Field(ge=0)]
    cohort_id: Identifier
    cohort_sha256: Sha256Hex
    policy: AllocationPolicyName
    budget_fraction: Probability
    audit_rounds: Annotated[int, Field(ge=0)]
    exact_target_risk: Probability
    exact_one_step_risk: Probability
    delayed_harm_rate: Probability
    theta_env: Probability
    auditable_mass: Probability
    nonauditable_mass: Probability
    target_risk_ucb: Probability
    identified_target_risk_lcb: Probability
    identified_interval_width: Probability
    upper_slack: float
    covered: bool
    released: bool
    false_release: bool
    unique_audited_candidate_count: Annotated[int, Field(ge=0)]
    duplicate_draw_count: Annotated[int, Field(ge=0)]
    failed_audit_count: Annotated[int, Field(ge=0)]
    suffix_tool_steps: Annotated[int, Field(ge=0)]
    generation_tokens: Annotated[int, Field(ge=0)]


class CalibrationBudgetSummary(FrozenModel):
    schema_version: Literal["censure.calibration-budget-summary.v1"] = (
        "censure.calibration-budget-summary.v1"
    )
    cell_id: Sha256Hex
    policy: AllocationPolicyName
    budget_fraction: Probability
    repetition_count: Annotated[int, Field(ge=1)]
    coverage: Probability
    coverage_one_sided_95_lower: Probability
    coverage_one_sided_95_upper: Probability
    nominal_coverage: Probability
    coverage_gate_passed: bool
    false_release_rate: Probability
    mean_upper_slack: float
    median_upper_slack: float
    mean_identified_interval_width: Probability
    mean_unique_audited_candidates: Annotated[float, Field(ge=0.0)]
    mean_duplicate_draws: Annotated[float, Field(ge=0.0)]
    mean_suffix_tool_steps: Annotated[float, Field(ge=0.0)]


def _derive_seed(payload: dict[str, object]) -> int:
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _cohort_identity(spec: CalibrationCellSpec, repetition_index: int) -> dict[str, object]:
    return {
        "schema_version": "censure.calibration-cohort-identity.v1",
        "protocol_id": spec.protocol_id,
        "seed_namespace": spec.seed_namespace,
        "base_seed": spec.base_seed,
        "support_regime": spec.support_regime.value,
        "cohort_size": spec.cohort_size,
        "target_harm_prevalence": spec.target_harm_prevalence,
        "zero_support_mass": spec.zero_support_mass,
        "mixed_auditable_probability": spec.mixed_auditable_probability,
        "delayed_harm_probability": spec.delayed_harm_probability,
        "repetition_index": repetition_index,
    }


def _rounds_for_budget(budget_fraction: float, auditable_candidate_count: int) -> int:
    if budget_fraction <= 0.0 or auditable_candidate_count == 0:
        return 0
    return math.ceil(budget_fraction * auditable_candidate_count)


def _result_from_point(
    *,
    spec: CalibrationCellSpec,
    repetition_index: int,
    cohort_id: str,
    cohort_sha256: str,
    exact_target_risk: float,
    exact_one_step_risk: float,
    delayed_harm_rate: float,
    budget_fraction: float,
    point: CertificatePoint,
    suffix_tool_steps: int,
    generation_tokens: int,
) -> CalibrationReplicateResult:
    covered = point.target_risk_ucb + 1e-12 >= exact_target_risk
    released = point.target_risk_ucb <= spec.release_threshold_eta
    return CalibrationReplicateResult(
        cell_id=spec.cell_id,
        repetition_index=repetition_index,
        cohort_id=cohort_id,
        cohort_sha256=cohort_sha256,
        policy=spec.policy,
        budget_fraction=budget_fraction,
        audit_rounds=point.round_index,
        exact_target_risk=exact_target_risk,
        exact_one_step_risk=exact_one_step_risk,
        delayed_harm_rate=delayed_harm_rate,
        theta_env=point.theta_env,
        auditable_mass=point.auditable_mass,
        nonauditable_mass=point.nonauditable_mass,
        target_risk_ucb=point.target_risk_ucb,
        identified_target_risk_lcb=point.identified_target_risk_lcb,
        identified_interval_width=point.identified_interval_width,
        upper_slack=point.target_risk_ucb - exact_target_risk,
        covered=covered,
        released=released,
        false_release=released and exact_target_risk > spec.release_threshold_eta,
        unique_audited_candidate_count=point.unique_audited_candidate_count,
        duplicate_draw_count=point.duplicate_draw_count,
        failed_audit_count=point.failed_audit_count,
        suffix_tool_steps=suffix_tool_steps,
        generation_tokens=generation_tokens,
    )


def run_calibration_cell(spec: CalibrationCellSpec) -> tuple[CalibrationReplicateResult, ...]:
    """Run one frozen cell, sharing cohorts and random tapes across policies."""

    results: list[CalibrationReplicateResult] = []
    for repetition_index in range(spec.repetitions):
        cohort_identity = _cohort_identity(spec, repetition_index)
        cohort_id = f"calibration-{canonical_sha256(cohort_identity)}"
        generation_seed = _derive_seed({**cohort_identity, "stream": "cohort"})
        allocation_seed = _derive_seed({**cohort_identity, "stream": "audit"})
        cohort = generate_enumerable_cohort(
            protocol_id=spec.protocol_id,
            cohort_id=cohort_id,
            cohort_size=spec.cohort_size,
            support_regime=spec.support_regime,
            target_harm_prevalence=spec.target_harm_prevalence,
            zero_support_mass=spec.zero_support_mass,
            generation_seed=generation_seed,
            mixed_auditable_probability=spec.mixed_auditable_probability,
            delayed_harm_probability=spec.delayed_harm_probability,
        )
        if cohort.decomposition_error() > 1e-12:
            raise AssertionError("enumerable cohort violates the frozen decomposition gate")
        envelope = cohort.envelope()
        max_rounds = max(
            _rounds_for_budget(fraction, len(envelope.auditable_candidates))
            for fraction in spec.budget_fractions
        )
        if max_rounds:
            auditor = CensureAuditor(
                envelope=envelope,
                oracle=InMemoryEvaluationOracle(cohort.private_outcomes()),
                policy=spec.policy,
                allocation_seed=allocation_seed,
                alpha=spec.alpha,
                exploration_epsilon=spec.exploration_epsilon,
            )
            ledger, points = auditor.run(total_rounds=max_rounds)
        else:
            auditor = CensureAuditor(
                envelope=envelope,
                oracle=InMemoryEvaluationOracle({}),
                policy=spec.policy,
                allocation_seed=allocation_seed,
                alpha=spec.alpha,
                exploration_epsilon=spec.exploration_epsilon,
            )
            ledger, points = auditor.run(total_rounds=0)
        cumulative_tool_steps = [0]
        cumulative_generation_tokens = [0]
        for disclosure in ledger.disclosures:
            cumulative_tool_steps.append(
                cumulative_tool_steps[-1] + disclosure.suffix_tool_steps
            )
            cumulative_generation_tokens.append(
                cumulative_generation_tokens[-1] + disclosure.generation_tokens
            )
        for budget_fraction in spec.budget_fractions:
            round_index = _rounds_for_budget(
                budget_fraction, len(envelope.auditable_candidates)
            )
            results.append(
                _result_from_point(
                    spec=spec,
                    repetition_index=repetition_index,
                    cohort_id=cohort_id,
                    cohort_sha256=canonical_sha256(cohort),
                    exact_target_risk=cohort.exact_target_risk,
                    exact_one_step_risk=cohort.exact_one_step_risk,
                    delayed_harm_rate=cohort.delayed_harm_rate,
                    budget_fraction=budget_fraction,
                    point=points[round_index],
                    suffix_tool_steps=cumulative_tool_steps[round_index],
                    generation_tokens=cumulative_generation_tokens[round_index],
                )
            )
    return tuple(results)


def _clopper_pearson_one_sided(
    successes: int, trials: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("binomial counts are invalid")
    try:
        from scipy.stats import beta
    except ImportError as exc:  # pragma: no cover - analysis extra contract
        raise RuntimeError("calibration aggregation requires the analysis extra (scipy)") from exc
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials - successes + 1))
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))
    )
    return lower, upper


def summarize_calibration_results(
    results: Sequence[CalibrationReplicateResult],
) -> tuple[CalibrationBudgetSummary, ...]:
    if not results:
        raise ValueError("cannot summarize an empty calibration result set")
    grouped: dict[tuple[str, AllocationPolicyName, float], list[CalibrationReplicateResult]] = {}
    for result in results:
        grouped.setdefault(
            (result.cell_id, result.policy, result.budget_fraction), []
        ).append(result)
    summaries: list[CalibrationBudgetSummary] = []
    for (cell_id, policy, budget_fraction), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])
    ):
        covered_count = sum(row.covered for row in rows)
        coverage = covered_count / len(rows)
        lower, upper = _clopper_pearson_one_sided(covered_count, len(rows))
        summaries.append(
            CalibrationBudgetSummary(
                cell_id=cell_id,
                policy=policy,
                budget_fraction=budget_fraction,
                repetition_count=len(rows),
                coverage=coverage,
                coverage_one_sided_95_lower=lower,
                coverage_one_sided_95_upper=upper,
                nominal_coverage=0.95,
                coverage_gate_passed=upper >= 0.95,
                false_release_rate=sum(row.false_release for row in rows) / len(rows),
                mean_upper_slack=statistics.fmean(row.upper_slack for row in rows),
                median_upper_slack=statistics.median(row.upper_slack for row in rows),
                mean_identified_interval_width=statistics.fmean(
                    row.identified_interval_width for row in rows
                ),
                mean_unique_audited_candidates=statistics.fmean(
                    row.unique_audited_candidate_count for row in rows
                ),
                mean_duplicate_draws=statistics.fmean(
                    row.duplicate_draw_count for row in rows
                ),
                mean_suffix_tool_steps=statistics.fmean(
                    row.suffix_tool_steps for row in rows
                ),
            )
        )
    return tuple(summaries)


__all__ = [
    "CalibrationBudgetSummary",
    "CalibrationCellSpec",
    "CalibrationReplicateResult",
    "run_calibration_cell",
    "summarize_calibration_results",
]
