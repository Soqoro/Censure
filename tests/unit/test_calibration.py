from __future__ import annotations

from pathlib import Path

import pytest

from censure.estimation.calibration import (
    CalibrationCellSpec,
    run_calibration_cell,
    summarize_calibration_results,
)
from censure.estimation.calibration_storage import (
    CalibrationRunStore,
    calibration_chunk_count,
    calibration_chunk_repetition_indices,
    calibration_chunk_shard,
    calibration_shard,
)
from censure.estimation.cli import main as phase2_main
from censure.estimation.enumerable import SupportRegime
from censure.estimation.protocol import load_frozen_calibration_catalog
from censure.estimation.schemas import AllocationPolicyName
from censure.storage import CorruptArtifactError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
    assert all(row.exact_population_target_risk == 0.2 for row in first)
    assert all(row.population_target_risk_ucb >= row.target_risk_ucb for row in first)
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
    assert all(0.0 <= summary.population_coverage <= 1.0 for summary in summaries)


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


def test_frozen_calibration_catalog_has_expected_primary_union() -> None:
    catalog = load_frozen_calibration_catalog(
        base_config_path=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "phase2_estimator_v1.yaml",
        amendment_1_path=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "phase2_estimator_v1_amendment_1.yaml",
        amendment_2_path=REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "phase2_estimator_v1_amendment_2.yaml",
    )

    assert len(catalog.entries) == 171
    assert sum("validity" in entry.purposes for entry in catalog.entries) == 81
    assert sum("efficiency" in entry.purposes for entry in catalog.entries) == 108
    assert sum(len(entry.purposes) == 2 for entry in catalog.entries) == 18
    assert all(entry.spec.repetitions == 2000 for entry in catalog.entries)
    assert catalog.repetitions_per_chunk == 25
    assert len(catalog.catalog_sha256) == 64


def test_calibration_store_resumes_per_repetition_and_detects_corruption(tmp_path) -> None:
    spec = _spec(AllocationPolicyName.UNIFORM, repetitions=2)
    rows = run_calibration_cell(spec)
    first_rows = tuple(row for row in rows if row.repetition_index == 0)
    store = CalibrationRunStore(tmp_path, "calibration-test")

    store.write_catalog((spec,))
    store.write_cell_spec(spec)
    store.write_repetition(spec, 0, first_rows)

    assert store.is_repetition_complete(spec, 0)
    assert not store.is_repetition_complete(spec, 1)
    assert store.read_repetition(spec, 0) == first_rows
    assert len(store.read_completed_cell(spec, require_all=False)) == len(spec.budget_fractions)
    with pytest.raises(FileNotFoundError, match="missing 1 repetitions"):
        store.read_completed_cell(spec)
    assert calibration_shard(cell_id=spec.cell_id, repetition_index=0, num_shards=7) == (
        calibration_shard(cell_id=spec.cell_id, repetition_index=0, num_shards=7)
    )

    repetition_path = next(store.root.glob("cells/*/repetitions/*.json"))
    repetition_path.write_text("{}", encoding="utf-8")
    assert not store.is_repetition_complete(spec, 0)
    with pytest.raises(CorruptArtifactError, match="checksum mismatch"):
        store.read_repetition(spec, 0)


def test_calibration_store_writes_atomic_repetition_chunks(tmp_path) -> None:
    spec = _spec(AllocationPolicyName.UNIFORM, repetitions=2)
    rows = run_calibration_cell(spec)
    store = CalibrationRunStore(tmp_path, "chunk-test")

    assert calibration_chunk_count(spec.repetitions, 25) == 1
    assert calibration_chunk_repetition_indices(
        repetitions=spec.repetitions, repetitions_per_chunk=25, chunk_index=0
    ) == (0, 1)
    store.write_chunk(spec, chunk_index=0, repetitions_per_chunk=25, rows=rows)

    assert store.is_chunk_complete(
        spec, chunk_index=0, repetitions_per_chunk=25
    )
    assert store.read_chunk(
        spec, chunk_index=0, repetitions_per_chunk=25
    ) == rows
    assert store.read_completed_cell_chunks(
        spec, repetitions_per_chunk=25
    ) == rows
    assert calibration_chunk_shard(cell_id=spec.cell_id, chunk_index=0, num_shards=3) == (
        calibration_chunk_shard(cell_id=spec.cell_id, chunk_index=0, num_shards=3)
    )


def test_phase2_cli_reports_frozen_catalog(capsys) -> None:
    exit_code = phase2_main(
        [
            "catalog",
            "--base-config",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "phase2_estimator_v1.yaml"
            ),
            "--amendment-1",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "phase2_estimator_v1_amendment_1.yaml"
            ),
            "--amendment-2",
            str(
                REPOSITORY_ROOT
                / "configs"
                / "experiments"
                / "phase2_estimator_v1_amendment_2.yaml"
            ),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"unique_cell_count": 171' in output
    assert '"repetition_count": 342000' in output
    assert '"work_item_count": 13680' in output
