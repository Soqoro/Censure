from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from censure.estimation.cli import main as phase2_main
from censure.estimation.protocol import load_frozen_robustness_catalog
from censure.estimation.robustness import (
    OutcomeModelCondition,
    RobustnessAxis,
    RobustnessCellSpec,
    run_robustness_cell,
    summarize_robustness_results,
)
from censure.estimation.robustness_storage import RobustnessRunStore
from censure.estimation.schemas import AllocationPolicyName

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _numeric_spec(
    axis: RobustnessAxis, level: float, *, repetitions: int = 10
) -> RobustnessCellSpec:
    return RobustnessCellSpec(
        protocol_id="phase2-test",
        seed_namespace="robustness-test",
        base_seed=71,
        axis=axis,
        level=level,
        cohort_size=200,
        repetitions=repetitions,
    )


def test_robustness_cell_rejects_incompatible_axis_levels_and_policy() -> None:
    with pytest.raises(ValidationError, match="named model condition"):
        _numeric_spec(RobustnessAxis.OUTCOME_MODEL_CONDITION, 0.5)
    with pytest.raises(ValidationError, match="censure_bound_targeted"):
        RobustnessCellSpec(
            protocol_id="p",
            seed_namespace="s",
            base_seed=1,
            axis=RobustnessAxis.OUTCOME_MODEL_CONDITION,
            level=OutcomeModelCondition.CORRECT,
            policy=AllocationPolicyName.UNIFORM,
        )


@pytest.mark.parametrize(
    ("axis", "level"),
    [
        (RobustnessAxis.NONAUDITABLE_MASS, 0.25),
        (RobustnessAxis.HIDDEN_GUARD_FEATURE, 0.10),
        (RobustnessAxis.SANDBOX_HARM_SHIFT, 0.10),
        (RobustnessAxis.SANDBOX_TRANSITION_SHIFT, 0.10),
        (RobustnessAxis.RARE_TARGET_HARM, 0.05),
    ],
)
def test_numeric_robustness_cells_are_deterministic(
    axis: RobustnessAxis, level: float
) -> None:
    spec = _numeric_spec(axis, level)
    first = run_robustness_cell(spec)
    second = run_robustness_cell(spec)

    assert first == second
    assert len(first) == spec.repetitions
    assert all(row.sensitivity_corrected_ucb >= row.target_risk_ucb for row in first)


def test_nonauditable_mass_remains_in_the_valid_upper_envelope() -> None:
    results = run_robustness_cell(
        _numeric_spec(RobustnessAxis.NONAUDITABLE_MASS, 0.50, repetitions=30)
    )

    assert any(row.realized_nonauditable_mass > 0.0 for row in results)
    assert all(row.assumption_satisfied for row in results)
    assert all(row.covered for row in results)


def test_hidden_guard_features_are_labeled_as_identification_failure() -> None:
    results = run_robustness_cell(
        _numeric_spec(RobustnessAxis.HIDDEN_GUARD_FEATURE, 0.25, repetitions=10)
    )

    assert any(row.realized_hidden_frontier_mass > 0.0 for row in results)
    assert all(not row.assumption_satisfied for row in results)
    assert all(row.declared_sensitivity_radius == 0.0 for row in results)


@pytest.mark.parametrize(
    "axis",
    [RobustnessAxis.SANDBOX_HARM_SHIFT, RobustnessAxis.SANDBOX_TRANSITION_SHIFT],
)
def test_declared_sandbox_radius_covers_the_realized_shift(axis: RobustnessAxis) -> None:
    results = run_robustness_cell(_numeric_spec(axis, 0.20, repetitions=30))

    assert all(
        row.exact_deployment_target_risk - row.sandbox_target_risk
        <= row.declared_sensitivity_radius + 1e-12
        for row in results
    )
    assert all(row.sensitivity_corrected_covered for row in results)
    assert all(not row.assumption_satisfied for row in results)


@pytest.mark.parametrize("condition", list(OutcomeModelCondition))
def test_outcome_model_conditions_change_only_allocation_metadata(
    condition: OutcomeModelCondition,
) -> None:
    spec = RobustnessCellSpec(
        protocol_id="phase2-test",
        seed_namespace="robustness-test",
        base_seed=71,
        axis=RobustnessAxis.OUTCOME_MODEL_CONDITION,
        level=condition,
        cohort_size=200,
        repetitions=5,
        policy=AllocationPolicyName.CENSURE_BOUND_TARGETED,
    )
    results = run_robustness_cell(spec)

    assert len(results) == 5
    assert all(row.assumption_satisfied for row in results)
    assert all(row.policy is AllocationPolicyName.CENSURE_BOUND_TARGETED for row in results)


def test_robustness_summary_preserves_uncorrected_and_corrected_coverage() -> None:
    results = run_robustness_cell(
        _numeric_spec(RobustnessAxis.SANDBOX_HARM_SHIFT, 0.10, repetitions=20)
    )
    summary = summarize_robustness_results(results)

    assert summary.repetition_count == 20
    assert summary.corrected_coverage >= summary.coverage
    assert summary.mean_corrected_upper_slack >= summary.mean_upper_slack


def test_frozen_robustness_catalog_has_all_one_factor_cells() -> None:
    catalog = load_frozen_robustness_catalog(
        base_config_path=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "phase2_estimator_v1.yaml",
        amendment_3_path=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "phase2_estimator_v1_amendment_3.yaml",
    )

    assert len(catalog.specs) == 21
    assert catalog.repetitions_per_chunk == 25
    assert all(spec.repetitions == 2000 for spec in catalog.specs)
    assert len(catalog.catalog_sha256) == 64


def test_robustness_store_round_trips_atomic_chunk(tmp_path) -> None:
    spec = _numeric_spec(RobustnessAxis.NONAUDITABLE_MASS, 0.25, repetitions=2)
    rows = run_robustness_cell(spec)
    store = RobustnessRunStore(tmp_path, "robustness-test")

    store.write_cell_spec(spec)
    store.write_chunk(
        spec,
        chunk_index=0,
        repetitions_per_chunk=25,
        rows=rows,
    )

    assert store.is_chunk_complete(
        spec, chunk_index=0, repetitions_per_chunk=25
    )
    assert store.read_chunk(
        spec, chunk_index=0, repetitions_per_chunk=25
    ) == rows
    assert store.read_completed_cell(spec, repetitions_per_chunk=25) == rows


def test_phase2_cli_reports_robustness_catalog(capsys) -> None:
    exit_code = phase2_main(
        [
            "robustness-catalog",
            "--base-config",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "phase2_estimator_v1.yaml"
            ),
            "--amendment-3",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "phase2_estimator_v1_amendment_3.yaml"
            ),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"cell_count": 21' in output
    assert '"work_item_count": 1680' in output
