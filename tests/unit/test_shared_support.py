from __future__ import annotations

from pathlib import Path

import pytest

from censure.estimation.cli import main as phase2_main
from censure.estimation.protocol import load_frozen_shared_support_catalog
from censure.estimation.shared_support import (
    SharedSupportCellSpec,
    SharedSupportModelCondition,
    bounded_score_ucb,
    combine_supported_and_frontier_ucbs,
    exact_shared_support_target_risk,
    run_shared_support_cell,
    summarize_shared_support_results,
)
from censure.estimation.shared_support_storage import SharedSupportRunStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _spec(
    ratio: float,
    condition: SharedSupportModelCondition,
    *,
    repetitions: int = 20,
    cohort_size: int = 500,
) -> SharedSupportCellSpec:
    return SharedSupportCellSpec(
        protocol_id="phase2-test",
        seed_namespace="shared-support-test",
        base_seed=83,
        cohort_size=cohort_size,
        repetitions=repetitions,
        max_importance_ratio=ratio,
        model_condition=condition,
    )


@pytest.mark.parametrize("ratio", [1.0, 2.0, 5.0, 10.0])
@pytest.mark.parametrize("condition", list(SharedSupportModelCondition))
def test_shared_support_repetitions_are_deterministic_and_bounded(
    ratio: float, condition: SharedSupportModelCondition
) -> None:
    spec = _spec(ratio, condition, repetitions=3, cohort_size=200)
    first = run_shared_support_cell(spec)
    second = run_shared_support_cell(spec)

    assert first == second
    assert all(row.observed_max_importance_ratio <= ratio + 1e-12 for row in first)
    assert all(row.ips_ucb >= row.ips for row in first)
    assert all(row.ips_covered for row in first)


def test_correct_direct_and_dr_are_close_to_exact_risk_in_aggregate() -> None:
    spec = _spec(
        5.0,
        SharedSupportModelCondition.CORRECT,
        repetitions=200,
        cohort_size=500,
    )
    results = run_shared_support_cell(spec)
    summary = summarize_shared_support_results(
        results,
        max_importance_ratio=spec.max_importance_ratio,
        model_condition=spec.model_condition,
    )

    assert exact_shared_support_target_risk() == pytest.approx(0.4)
    assert abs(summary.ips_bias) < 0.03
    assert abs(summary.direct_method_bias) < 0.01
    assert abs(summary.sequential_doubly_robust_bias) < 0.02
    assert summary.ips_coverage >= 0.95


def test_misspecified_direct_method_is_biased_but_dr_remains_near_target() -> None:
    spec = _spec(
        2.0,
        SharedSupportModelCondition.MISSPECIFIED,
        repetitions=200,
        cohort_size=500,
    )
    summary = summarize_shared_support_results(
        run_shared_support_cell(spec),
        max_importance_ratio=spec.max_importance_ratio,
        model_condition=spec.model_condition,
    )

    assert abs(summary.direct_method_bias) > 0.10
    assert abs(summary.sequential_doubly_robust_bias) < 0.03


def test_hybrid_composition_adds_supported_uncertainty_and_frontier_envelope() -> None:
    assert combine_supported_and_frontier_ucbs(
        supported_harm_ucb=0.2,
        frontier_mass=0.4,
        audited_safe_mass_lcb=0.1,
    ) == pytest.approx(0.5)
    assert combine_supported_and_frontier_ucbs(
        supported_harm_ucb=0.9,
        frontier_mass=0.4,
        audited_safe_mass_lcb=0.0,
    ) == 1.0
    with pytest.raises(ValueError, match="exceeds frontier"):
        combine_supported_and_frontier_ucbs(
            supported_harm_ucb=0.1,
            frontier_mass=0.2,
            audited_safe_mass_lcb=0.3,
        )


def test_bounded_score_ucb_rejects_invalid_inputs() -> None:
    assert bounded_score_ucb(
        score_mean=0.2, score_bound=2.0, sample_size=1000, alpha=0.05
    ) > 0.2
    with pytest.raises(ValueError, match="score_bound"):
        bounded_score_ucb(score_mean=0.2, score_bound=0.0, sample_size=10, alpha=0.05)


def test_frozen_shared_support_catalog_has_ratio_model_cross_product() -> None:
    catalog = load_frozen_shared_support_catalog(
        base_config_path=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "phase2_estimator_v1.yaml",
        amendment_4_path=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "phase2_estimator_v1_amendment_4.yaml",
    )

    assert len(catalog.specs) == 12
    assert catalog.repetitions_per_chunk == 25
    assert {spec.max_importance_ratio for spec in catalog.specs} == {1.0, 2.0, 5.0, 10.0}
    assert {spec.model_condition for spec in catalog.specs} == set(SharedSupportModelCondition)
    assert all(spec.repetitions == 2000 for spec in catalog.specs)


def test_shared_support_store_round_trips_chunk(tmp_path) -> None:
    spec = _spec(2.0, SharedSupportModelCondition.CORRECT, repetitions=2, cohort_size=50)
    rows = run_shared_support_cell(spec)
    store = SharedSupportRunStore(tmp_path, "shared-support-test")

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


def test_phase2_cli_reports_shared_support_catalog(capsys) -> None:
    exit_code = phase2_main(
        [
            "shared-support-catalog",
            "--base-config",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "phase2_estimator_v1.yaml"
            ),
            "--amendment-4",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "phase2_estimator_v1_amendment_4.yaml"
            ),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"cell_count": 12' in output
    assert '"work_item_count": 960' in output
