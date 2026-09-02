from __future__ import annotations

import pytest

from censure.estimation.calibration import (
    CalibrationCellSpec,
    run_calibration_cell,
    summarize_calibration_results,
)
from censure.estimation.enumerable import SupportRegime
from censure.estimation.schemas import AllocationPolicyName


def _spec(policy: AllocationPolicyName, *, repetitions: int = 8) -> CalibrationCellSpec:
    return CalibrationCellSpec(
        protocol_id="censure-phase2-estimator-v1",
        seed_namespace="test-calibration",
        base_seed=42,
        support_regime=SupportRegime.DETERMINISTIC_CLONEABLE_NONOVERLAP,
        cohort_size=100,
        target_harm_prevalence=0.2,
        zero_support_mass=0.5,
        policy=policy,
        budget_fractions=(0.0, 0.1, 0.2),
        repetitions=repetitions,
    )


def test_calibration_cell_is_deterministic_and_reports_every_budget() -> None:
    spec = _spec(AllocationPolicyName.CENSURE_BOUND_TARGETED)
    first = run_calibration_cell(spec)
    second = run_calibration_cell(spec)

    assert first == second
    assert len(first) == spec.repetitions * len(spec.budget_fractions)
    assert {row.budget_fraction for row in first} == set(spec.budget_fractions)
    assert all(row.theta_env + 1e-12 >= row.exact_target_risk for row in first)
    assert all(row.audit_rounds == 0 for row in first if row.budget_fraction == 0.0)


def test_efficiency_policies_share_cohorts_and_random_tape_identity() -> None:
    uniform = run_calibration_cell(_spec(AllocationPolicyName.UNIFORM, repetitions=3))
    targeted = run_calibration_cell(
        _spec(AllocationPolicyName.CENSURE_BOUND_TARGETED, repetitions=3)
    )

    for left, right in zip(uniform, targeted, strict=True):
        assert left.repetition_index == right.repetition_index
        assert left.budget_fraction == right.budget_fraction
        assert left.cohort_id == right.cohort_id
        assert left.cohort_sha256 == right.cohort_sha256
        assert left.exact_target_risk == right.exact_target_risk


def test_calibration_summary_uses_frozen_coverage_gate() -> None:
    results = run_calibration_cell(_spec(AllocationPolicyName.TARGET_MASS, repetitions=20))
    summaries = summarize_calibration_results(results)

    assert len(summaries) == 3
    assert all(summary.repetition_count == 20 for summary in summaries)
    assert all(0.0 <= summary.coverage <= 1.0 for summary in summaries)
    assert all(
        summary.coverage_one_sided_95_lower
        <= summary.coverage
        <= summary.coverage_one_sided_95_upper
        for summary in summaries
    )
    assert all(
        summary.coverage_gate_passed
        == (summary.coverage_one_sided_95_upper >= summary.nominal_coverage)
        for summary in summaries
    )


def test_full_overlap_is_exact_without_audits() -> None:
    raw = _spec(AllocationPolicyName.UNIFORM, repetitions=3).model_dump(mode="python")
    raw["support_regime"] = SupportRegime.FULL_OVERLAP
    raw["zero_support_mass"] = 0.0
    spec = CalibrationCellSpec.model_validate(raw)
    results = run_calibration_cell(spec)

    assert all(row.audit_rounds == 0 for row in results)
    assert all(row.theta_env == pytest.approx(row.exact_target_risk) for row in results)
    assert all(row.upper_slack == pytest.approx(0.0) for row in results)
    assert all(row.covered for row in results)
