"""Artifact-derived synthesis for the full CENSURE estimator paper."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from censure.config import load_yaml
from censure.estimation.calibration import (
    CalibrationReplicateResult,
    summarize_calibration_results,
)
from censure.estimation.calibration_storage import CalibrationRunStore
from censure.estimation.protocol import (
    FrozenCalibrationCatalog,
    FrozenRobustnessCatalog,
    FrozenSharedSupportCatalog,
)
from censure.estimation.robustness import summarize_robustness_results
from censure.estimation.robustness_storage import RobustnessRunStore
from censure.estimation.shared_support import summarize_shared_support_results
from censure.estimation.shared_support_storage import SharedSupportRunStore
from censure.provenance import collect_provenance
from censure.serialization import canonical_sha256
from censure.storage import CorruptArtifactError, atomic_write_bytes, atomic_write_json

PHASE2_SYNTHESIS_SCHEMA_VERSION = "censure.phase2-paper-evidence.v1"
EXPECTED_HELD_OUT_ACTORS = {
    "Qwen/Qwen3-8B",
    "google/gemma-3-12b-it",
    "mistralai/Ministral-3-14B-Instruct-2512-BF16",
}
EXPECTED_HELD_OUT_MANIFEST_SHA256 = (
    "e4a7b11680ed0d6181f24b3bf5c26420453503122b032fc302733fbb7bdfb96d"
)
EXPECTED_AUDIT_POLICIES = {
    "uniform",
    "target_mass",
    "guard_score",
    "uncertainty",
    "downstream_harm",
    "censure_bound_targeted",
}


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return statistics.fmean(values)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take the median of an empty sequence")
    return float(statistics.median(values))


def _trapezoid(points: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(points)
    if len(ordered) < 2 or len({x for x, _ in ordered}) != len(ordered):
        raise ValueError("AUC requires at least two unique budget points")
    return math.fsum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in pairwise(ordered)
    )


def _design_key(spec: Any) -> tuple[str, int, float, float]:
    return (
        spec.support_regime.value,
        spec.cohort_size,
        spec.target_harm_prevalence,
        spec.zero_support_mass,
    )


def _design_fields(design: tuple[str, int, float, float]) -> dict[str, Any]:
    regime, cohort_size, prevalence, zero_support_mass = design
    return {
        "support_regime": regime,
        "cohort_size": cohort_size,
        "target_harm_prevalence": prevalence,
        "zero_support_mass": zero_support_mass,
    }


def _equal_grid_median_contrast(
    pairs_by_design: Mapping[tuple[str, int, float, float], Sequence[tuple[float, float]]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 10_000,
    ci_level: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare methods by paired resampling and equal frozen-design weight."""

    if not pairs_by_design:
        raise ValueError("paired efficiency comparison has no designs")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie in (0, 1)")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - analysis extra contract
        raise RuntimeError("Phase 2 synthesis requires the analysis extra") from exc

    design_rows: list[dict[str, Any]] = []
    arrays: list[tuple[Any, Any]] = []
    for design in sorted(pairs_by_design):
        pairs = pairs_by_design[design]
        if not pairs:
            raise ValueError(f"paired efficiency design is empty: {design}")
        left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        arrays.append((left, right))
        paired_deltas = left - right
        design_rows.append(
            {
                **_design_fields(design),
                "repetition_count": len(pairs),
                "censure_median": float(np.median(left)),
                "uniform_median": float(np.median(right)),
                "median_contrast": float(np.median(left) - np.median(right)),
                "mean_paired_contrast": float(np.mean(paired_deltas)),
                "censure_lower_rate": float(np.mean(paired_deltas < 0.0)),
                "tie_rate": float(np.mean(paired_deltas == 0.0)),
            }
        )

    estimate = _mean([float(row["median_contrast"]) for row in design_rows])
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.zeros(bootstrap_samples, dtype=np.float64)
    batch_size = 250
    for left, right in arrays:
        for start in range(0, bootstrap_samples, batch_size):
            stop = min(bootstrap_samples, start + batch_size)
            indices = rng.integers(0, len(left), size=(stop - start, len(left)))
            draws[start:stop] += np.median(left[indices], axis=1) - np.median(
                right[indices], axis=1
            )
    draws /= len(arrays)
    tail = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(draws, [tail, 1.0 - tail], method="linear")
    return (
        {
            "design_count": len(arrays),
            "repetitions_per_design": sorted({len(left) for left, _right in arrays}),
            "estimand": ("equal_weight_design_mean_of_censure_median_minus_uniform_median"),
            "estimate": estimate,
            "ci_level": ci_level,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "bootstrap_method": "paired_within_design_percentile",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_samples": bootstrap_samples,
            "favorable_design_rate": sum(float(row["median_contrast"]) < 0.0 for row in design_rows)
            / len(design_rows),
            "tied_design_rate": sum(float(row["median_contrast"]) == 0.0 for row in design_rows)
            / len(design_rows),
        },
        design_rows,
    )


def _calibration_evidence(
    *,
    catalog: FrozenCalibrationCatalog,
    store: CalibrationRunStore,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    validity_rows: list[dict[str, Any]] = []
    efficiency_cell_rows: list[dict[str, Any]] = []
    efficiency_values: dict[tuple[str, float], list[CalibrationReplicateResult]] = defaultdict(list)
    replicate_curves: dict[
        tuple[tuple[str, int, float, float], int, str],
        list[tuple[float, float]],
    ] = defaultdict(list)
    longitudinality: list[dict[str, Any]] = []

    for entry in catalog.entries:
        rows = store.read_completed_cell_chunks(
            entry.spec,
            repetitions_per_chunk=catalog.repetitions_per_chunk,
            require_all=True,
        )
        if len(rows) != entry.spec.repetitions * len(entry.spec.budget_fractions):
            raise CorruptArtifactError("calibration cell row count differs from its freeze")
        summaries = summarize_calibration_results(rows)
        if "validity" in entry.purposes:
            for summary in summaries:
                validity_rows.append(
                    {
                        "cell_id": entry.spec.cell_id,
                        "support_regime": entry.spec.support_regime.value,
                        "cohort_size": entry.spec.cohort_size,
                        "target_harm_prevalence": entry.spec.target_harm_prevalence,
                        "zero_support_mass": entry.spec.zero_support_mass,
                        **summary.model_dump(mode="json"),
                    }
                )
            longitudinality.extend(
                {
                    "support_regime": entry.spec.support_regime.value,
                    "cohort_size": entry.spec.cohort_size,
                    "target_harm_prevalence": entry.spec.target_harm_prevalence,
                    "zero_support_mass": entry.spec.zero_support_mass,
                    "repetition_index": row.repetition_index,
                    "full_suffix_risk": row.exact_target_risk,
                    "one_step_risk": row.exact_one_step_risk,
                    "signed_one_step_bias": (row.exact_one_step_risk - row.exact_target_risk),
                    "delayed_harm_rate": row.delayed_harm_rate,
                }
                for row in rows
                if row.budget_fraction == 0.0
            )
        if "efficiency" in entry.purposes:
            policy = entry.spec.policy.value
            for summary in summaries:
                efficiency_cell_rows.append(
                    {
                        "cell_id": entry.spec.cell_id,
                        **_design_fields(_design_key(entry.spec)),
                        **summary.model_dump(mode="json"),
                    }
                )
            for row in rows:
                efficiency_values[(policy, row.budget_fraction)].append(row)
                replicate_curves[(_design_key(entry.spec), row.repetition_index, policy)].append(
                    (row.budget_fraction, row.upper_slack)
                )

    validity_by_regime_budget: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in validity_rows:
        validity_by_regime_budget[
            (str(row["support_regime"]), float(row["budget_fraction"]))
        ].append(row)
    validity_table = [
        {
            "support_regime": regime,
            "budget_fraction": budget,
            "cell_count": len(rows),
            "minimum_coverage": min(float(row["coverage"]) for row in rows),
            "mean_coverage": _mean([float(row["coverage"]) for row in rows]),
            "coverage_gate_failure_count": sum(
                not bool(row["coverage_gate_passed"]) for row in rows
            ),
            "minimum_population_coverage": min(float(row["population_coverage"]) for row in rows),
            "population_coverage_gate_failure_count": sum(
                not bool(row["population_coverage_gate_passed"]) for row in rows
            ),
            "maximum_false_release_rate": max(float(row["false_release_rate"]) for row in rows),
            "mean_upper_slack": _mean([float(row["mean_upper_slack"]) for row in rows]),
            "mean_identified_interval_width": _mean(
                [float(row["mean_identified_interval_width"]) for row in rows]
            ),
        }
        for (regime, budget), rows in sorted(validity_by_regime_budget.items())
    ]

    efficiency_table: list[dict[str, Any]] = []
    for (policy, budget), rows in sorted(efficiency_values.items()):
        slacks = [row.upper_slack for row in rows]
        costs = [row.suffix_tool_steps + row.generation_tokens for row in rows]
        efficiency_table.append(
            {
                "policy": policy,
                "budget_fraction": budget,
                "replicate_count": len(rows),
                "coverage": sum(row.covered for row in rows) / len(rows),
                "mean_upper_slack": _mean(slacks),
                "median_upper_slack": _median(slacks),
                "mean_identified_interval_width": _mean(
                    [row.identified_interval_width for row in rows]
                ),
                "mean_suffix_cost": _mean([float(value) for value in costs]),
                "mean_unique_audited_candidates": _mean(
                    [float(row.unique_audited_candidate_count) for row in rows]
                ),
                "mean_duplicate_draws": _mean([float(row.duplicate_draw_count) for row in rows]),
            }
        )

    paired: dict[tuple[tuple[str, int, float, float], int], dict[str, dict[str, float]]] = (
        defaultdict(dict)
    )
    for (design, repetition, policy), curve in replicate_curves.items():
        by_budget = dict(curve)
        if len(by_budget) != len(curve) or 0.20 not in by_budget:
            raise CorruptArtifactError("efficiency curve is missing a frozen budget")
        paired[(design, repetition)][policy] = {
            "slack_020": by_budget[0.20],
            "slack_auc_000_040": _trapezoid(curve),
        }
    slack_pairs_by_design: dict[tuple[str, int, float, float], list[tuple[float, float]]] = (
        defaultdict(list)
    )
    auc_pairs_by_design: dict[tuple[str, int, float, float], list[tuple[float, float]]] = (
        defaultdict(list)
    )
    for (design, repetition), values in sorted(paired.items()):
        try:
            censure = values["censure_bound_targeted"]
            uniform = values["uniform"]
        except KeyError as exc:
            raise CorruptArtifactError(
                f"primary policy pair is missing for efficiency unit {design}/{repetition}"
            ) from exc
        slack_pairs_by_design[design].append((censure["slack_020"], uniform["slack_020"]))
        auc_pairs_by_design[design].append(
            (censure["slack_auc_000_040"], uniform["slack_auc_000_040"])
        )

    slack_comparison, slack_design_rows = _equal_grid_median_contrast(
        slack_pairs_by_design,
        bootstrap_seed=20260907,
    )
    auc_comparison, auc_design_rows = _equal_grid_median_contrast(
        auc_pairs_by_design,
        bootstrap_seed=20260908,
    )
    efficiency_contrasts = [
        {
            **slack_row,
            "censure_median_slack_auc": auc_row["censure_median"],
            "uniform_median_slack_auc": auc_row["uniform_median"],
            "median_slack_auc_contrast": auc_row["median_contrast"],
            "mean_paired_slack_auc_contrast": auc_row["mean_paired_contrast"],
            "censure_lower_slack_auc_rate": auc_row["censure_lower_rate"],
            "slack_auc_tie_rate": auc_row["tie_rate"],
        }
        for slack_row, auc_row in zip(slack_design_rows, auc_design_rows, strict=True)
    ]
    censure_coverage = [
        row for row in efficiency_cell_rows if row["policy"] == "censure_bound_targeted"
    ]
    primary_censure_coverage = [
        row for row in censure_coverage if float(row["budget_fraction"]) == 0.20
    ]
    if not primary_censure_coverage:
        raise CorruptArtifactError("CENSURE primary-budget efficiency cells are missing")
    all_budget_coverage_failures = sum(
        not bool(row["coverage_gate_passed"]) for row in censure_coverage
    )
    primary_budget_coverage_failures = sum(
        not bool(row["coverage_gate_passed"]) for row in primary_censure_coverage
    )
    primary_comparison = {
        "slack_at_020": slack_comparison,
        "slack_auc_000_040": auc_comparison,
        "censure_all_budget_coverage_gate_failure_count": (all_budget_coverage_failures),
        "censure_primary_budget_coverage_gate_failure_count": (primary_budget_coverage_failures),
        "censure_primary_budget_minimum_coverage": min(
            float(row["coverage"]) for row in primary_censure_coverage
        ),
        "efficiency_claim_supported": (
            primary_budget_coverage_failures == 0 and float(slack_comparison["ci_high"]) < 0.0
        ),
        "interpretation": (
            "paired_monte_carlo_uncertainty_on_fixed_frozen_grid_not_"
            "deployment_superpopulation_inference"
        ),
    }
    evidence = {
        "catalog_sha256": catalog.catalog_sha256,
        "validity_cell_count": sum("validity" in entry.purposes for entry in catalog.entries),
        "efficiency_cell_count": sum("efficiency" in entry.purposes for entry in catalog.entries),
        "validity": {
            "summary_row_count": len(validity_rows),
            "minimum_coverage": min(float(row["coverage"]) for row in validity_rows),
            "coverage_gate_failure_count": sum(
                not bool(row["coverage_gate_passed"]) for row in validity_rows
            ),
            "minimum_population_coverage": min(
                float(row["population_coverage"]) for row in validity_rows
            ),
            "population_coverage_gate_failure_count": sum(
                not bool(row["population_coverage_gate_passed"]) for row in validity_rows
            ),
            "maximum_false_release_rate": max(
                float(row["false_release_rate"]) for row in validity_rows
            ),
        },
        "efficiency": {
            "primary_comparison": primary_comparison,
            "policy_budget_rows": efficiency_table,
        },
        "synthetic_longitudinality": {
            "cohort_replicate_count": len(longitudinality),
            "mean_one_step_risk": _mean([float(row["one_step_risk"]) for row in longitudinality]),
            "mean_full_suffix_risk": _mean(
                [float(row["full_suffix_risk"]) for row in longitudinality]
            ),
            "mean_signed_one_step_bias": _mean(
                [float(row["signed_one_step_bias"]) for row in longitudinality]
            ),
            "mean_delayed_harm_rate": _mean(
                [float(row["delayed_harm_rate"]) for row in longitudinality]
            ),
        },
    }
    return evidence, {
        "validity_cells": validity_rows,
        "validity_aggregate": validity_table,
        "efficiency_cells": efficiency_cell_rows,
        "efficiency_aggregate": efficiency_table,
        "efficiency_contrasts": efficiency_contrasts,
        "longitudinality_cells": longitudinality,
    }


def _robustness_evidence(
    *,
    catalog: FrozenRobustnessCatalog,
    store: RobustnessRunStore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table: list[dict[str, Any]] = []
    for spec in catalog.specs:
        rows = store.read_completed_cell(
            spec,
            repetitions_per_chunk=catalog.repetitions_per_chunk,
        )
        summary = summarize_robustness_results(rows)
        payload = summary.model_dump(mode="json")
        level = payload["level"]
        if payload["axis"] == "hidden_guard_feature_prevalence" and float(level) > 0.0:
            assumption_status = "unidentified"
        elif payload["axis"] in {"sandbox_harm_shift", "sandbox_transition_shift"} and (
            float(level) > 0.0
        ):
            assumption_status = "sensitivity_corrected"
        else:
            assumption_status = "identified"
        table.append({**payload, "assumption_status": assumption_status})
    identified = [row for row in table if row["assumption_status"] == "identified"]
    shifted = [row for row in table if row["assumption_status"] == "sensitivity_corrected"]
    unidentified = [row for row in table if row["assumption_status"] == "unidentified"]
    return (
        {
            "catalog_sha256": catalog.catalog_sha256,
            "cell_count": len(table),
            "minimum_raw_coverage": min(float(row["coverage"]) for row in table),
            "minimum_identified_coverage": min(float(row["coverage"]) for row in identified),
            "minimum_shift_corrected_coverage": min(
                float(row["corrected_coverage"]) for row in shifted
            ),
            "unidentified_cell_count": len(unidentified),
            "unidentified_minimum_raw_coverage": min(
                float(row["coverage"]) for row in unidentified
            ),
            "rows": table,
        },
        table,
    )


def _shared_support_evidence(
    *,
    catalog: FrozenSharedSupportCatalog,
    store: SharedSupportRunStore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table: list[dict[str, Any]] = []
    for spec in catalog.specs:
        rows = store.read_completed_cell(
            spec,
            repetitions_per_chunk=catalog.repetitions_per_chunk,
        )
        summary = summarize_shared_support_results(
            rows,
            max_importance_ratio=spec.max_importance_ratio,
            model_condition=spec.model_condition,
        )
        table.append(summary.model_dump(mode="json"))
    return (
        {
            "catalog_sha256": catalog.catalog_sha256,
            "cell_count": len(table),
            "minimum_ips_coverage": min(float(row["ips_coverage"]) for row in table),
            "maximum_sequential_dr_rmse": max(
                float(row["sequential_doubly_robust_rmse"]) for row in table
            ),
            "rows": table,
        },
        table,
    )


def _read_agent_summary(path: Path) -> dict[str, Any]:
    checksum = path.with_suffix(".sha256")
    if not path.is_file() or not checksum.is_file():
        raise FileNotFoundError(f"checksummed agent study summary is missing: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum.read_text(encoding="utf-8").strip():
        raise CorruptArtifactError("agent study summary checksum mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "censure.agent-audit-study-summary.v1"
    ):
        raise CorruptArtifactError("agent study summary schema is invalid")
    if payload.get("post_audit_full_oracle_revealed") is not True:
        raise CorruptArtifactError("agent study summary lacks post-seal full-oracle evaluation")
    if payload.get("protocol_id") != "censure-phase2-estimator-v1":
        raise CorruptArtifactError("agent study summary uses an unexpected protocol")
    if payload.get("source_manifest_sha256") != EXPECTED_HELD_OUT_MANIFEST_SHA256:
        raise CorruptArtifactError("agent study summary uses an unexpected held-out manifest")
    actor_rows = payload.get("actor_rows")
    audit_rows = payload.get("audit_rows")
    if not isinstance(actor_rows, list) or not isinstance(audit_rows, list):
        raise CorruptArtifactError("agent study summary rows are missing")
    actors = {str(row.get("actor_id")) for row in actor_rows if isinstance(row, dict)}
    if len(actor_rows) != 3 or actors != EXPECTED_HELD_OUT_ACTORS:
        raise CorruptArtifactError("agent study summary actor set differs from the freeze")
    expected_budgets = {0.0, 0.02, 0.05, 0.10, 0.20, 0.40}
    keys = {
        (
            str(row.get("actor_id")),
            str(row.get("policy")),
            float(row.get("budget_fraction", -1.0)),
        )
        for row in audit_rows
        if isinstance(row, dict)
    }
    if len(audit_rows) != 108 or len(keys) != 108:
        raise CorruptArtifactError("agent audit matrix is incomplete or duplicated")
    if (
        {key[0] for key in keys} != EXPECTED_HELD_OUT_ACTORS
        or {key[1] for key in keys} != EXPECTED_AUDIT_POLICIES
        or {key[2] for key in keys} != expected_budgets
    ):
        raise CorruptArtifactError("agent audit matrix differs from the frozen design")

    primary_rows: list[dict[str, Any]] = []
    actors_with_costs: list[dict[str, Any]] = []
    for actor in actor_rows:
        actor_id = str(actor["actor_id"])
        primary = next(
            row
            for row in audit_rows
            if row["actor_id"] == actor_id
            and row["policy"] == "censure_bound_targeted"
            and math.isclose(float(row["budget_fraction"]), 0.20)
        )
        logical_cost = int(primary["suffix_tool_steps"]) + int(primary["generation_tokens"])
        full_cost = int(actor["full_target_observed_final_trajectory_cost"]["combined_cost"])
        primary_rows.append(
            {
                **primary,
                "logical_combined_cost": logical_cost,
                "full_target_observed_final_combined_cost": full_cost,
                "logical_cost_fraction_of_full_target": (
                    None if full_cost == 0 else logical_cost / full_cost
                ),
            }
        )
        actors_with_costs.append(
            {
                **actor,
                "primary_censure_020": primary_rows[-1],
            }
        )
    payload["actor_rows"] = actors_with_costs
    payload["primary_censure_020_rows"] = primary_rows
    payload["audit_coverage_failure_count"] = sum(
        not bool(row["covers_target_identification_upper"]) for row in audit_rows
    )
    return payload


def _reporting_freeze_sha256(
    path: str | Path,
    *,
    calibration_catalog: FrozenCalibrationCatalog,
    robustness_catalog: FrozenRobustnessCatalog,
    shared_support_catalog: FrozenSharedSupportCatalog,
) -> str:
    amendment = load_yaml(path)
    if amendment.get("amendment_id") != "censure-phase2-estimator-v1-amendment-7":
        raise ValueError("paper synthesis requires Phase 2 amendment 7")
    if amendment.get("parent_amendment_id") != "censure-phase2-estimator-v1-amendment-6":
        raise ValueError("Phase 2 amendment 7 has the wrong parent")
    outcome_flags = (
        "frozen_primary_calibration_outcomes_inspected",
        "frozen_robustness_outcomes_inspected",
        "frozen_shared_support_outcomes_inspected",
        "held_out_agent_behavior_outcomes_inspected",
        "held_out_agent_suffix_outcomes_inspected",
        "held_out_agent_full_target_outcomes_inspected",
    )
    if any(amendment.get(flag) is not False for flag in outcome_flags):
        raise ValueError("paper synthesis reporting freeze is not outcome-blind")
    sources = amendment.get("source_commitments")
    if not isinstance(sources, dict):
        raise ValueError("paper synthesis source commitments are missing")
    expected = {
        "calibration_catalog_sha256": calibration_catalog.catalog_sha256,
        "robustness_catalog_sha256": robustness_catalog.catalog_sha256,
        "shared_support_catalog_sha256": shared_support_catalog.catalog_sha256,
        "held_out_manifest_sha256": EXPECTED_HELD_OUT_MANIFEST_SHA256,
    }
    if any(sources.get(key) != value for key, value in expected.items()):
        raise ValueError("paper synthesis source commitment differs from a loaded catalog")
    return canonical_sha256(amendment)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    columns = sorted({key for row in rows for key in row})
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    atomic_write_bytes(path, buffer.getvalue().encode())


def _actor_label(actor_id: str) -> str:
    lowered = actor_id.casefold()
    if "qwen" in lowered:
        return "Qwen"
    if "gemma" in lowered:
        return "Gemma"
    if "ministral" in lowered:
        return "Ministral"
    return "Actor" + canonical_sha256(actor_id)[:8]


def _write_tex_macros(path: Path, evidence: Mapping[str, Any]) -> None:
    calibration = evidence["calibration"]
    robustness = evidence["robustness"]
    shared = evidence["shared_support"]
    agents = evidence["held_out_agents"]
    primary = calibration["efficiency"]["primary_comparison"]
    slack = primary["slack_at_020"]
    longitudinality = calibration["synthetic_longitudinality"]
    lines = [
        "% Auto-generated by censure.estimation.synthesis; do not edit.",
        r"\newcommand{\PhaseTwoResultsAvailable}{1}",
        rf"\newcommand{{\PhaseTwoValidityCells}}{{{calibration['validity_cell_count']}}}",
        rf"\newcommand{{\PhaseTwoEfficiencyCells}}{{{calibration['efficiency_cell_count']}}}",
        rf"\newcommand{{\PhaseTwoMinCoverage}}{{{calibration['validity']['minimum_coverage']:.3f}}}",
        rf"\newcommand{{\PhaseTwoCoverageDefects}}{{{calibration['validity']['coverage_gate_failure_count']}}}",
        rf"\newcommand{{\PhaseTwoMinPopulationCoverage}}{{{calibration['validity']['minimum_population_coverage']:.3f}}}",
        rf"\newcommand{{\PhaseTwoPopulationCoverageDefects}}{{{calibration['validity']['population_coverage_gate_failure_count']}}}",
        rf"\newcommand{{\PhaseTwoMaxFalseRelease}}{{{calibration['validity']['maximum_false_release_rate']:.3f}}}",
        rf"\newcommand{{\CensureUniformSlackDelta}}{{{slack['estimate']:.3f}}}",
        rf"\newcommand{{\CensureUniformSlackDeltaLow}}{{{slack['ci_low']:.3f}}}",
        rf"\newcommand{{\CensureUniformSlackDeltaHigh}}{{{slack['ci_high']:.3f}}}",
        rf"\newcommand{{\CensureUniformWinRate}}{{{slack['favorable_design_rate']:.3f}}}",
        rf"\newcommand{{\PhaseTwoEfficiencyClaim}}{{{'supported' if primary['efficiency_claim_supported'] else 'not supported'}}}",
        rf"\newcommand{{\PhaseTwoMeanOneStepBias}}{{{longitudinality['mean_signed_one_step_bias']:.3f}}}",
        rf"\newcommand{{\PhaseTwoMeanDelayedHarm}}{{{longitudinality['mean_delayed_harm_rate']:.3f}}}",
        rf"\newcommand{{\PhaseTwoMinIdentifiedRobustnessCoverage}}{{{robustness['minimum_identified_coverage']:.3f}}}",
        rf"\newcommand{{\PhaseTwoMinCorrectedRobustnessCoverage}}{{{robustness['minimum_shift_corrected_coverage']:.3f}}}",
        rf"\newcommand{{\PhaseTwoUnidentifiedRobustnessCells}}{{{robustness['unidentified_cell_count']}}}",
        rf"\newcommand{{\PhaseTwoMinIPSCoverage}}{{{shared['minimum_ips_coverage']:.3f}}}",
        rf"\newcommand{{\PhaseTwoAgentCoverageDefects}}{{{agents['audit_coverage_failure_count']}}}",
    ]
    for actor in agents["actor_rows"]:
        actor_id = str(actor["actor_id"])
        label = _actor_label(actor_id)
        target = actor["target_risk"]
        lines.extend(
            [
                rf"\newcommand{{\PhaseTwo{label}TargetLower}}{{{float(target['risk_lower_endpoint']):.3f}}}",
                rf"\newcommand{{\PhaseTwo{label}TargetUpper}}{{{float(target['risk_upper_endpoint']):.3f}}}",
                rf"\newcommand{{\PhaseTwo{label}InvalidRate}}{{{float(target['invalid_rate']):.3f}}}",
            ]
        )
        selected = actor["primary_censure_020"]
        lines.append(
            rf"\newcommand{{\PhaseTwo{label}UCBTwenty}}{{{float(selected['target_risk_ucb']):.3f}}}"
        )
        cost_fraction = selected["logical_cost_fraction_of_full_target"]
        lines.append(
            rf"\newcommand{{\PhaseTwo{label}CostFraction}}{{{'N/A' if cost_fraction is None else f'{float(cost_fraction):.3f}'}}}"
        )
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode())


def _save_figure(fig: Any, base: Path) -> tuple[Path, Path]:
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "CENSURE"},
    )
    fig.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "CENSURE"})
    return pdf, png


def _write_figures(
    *,
    out_dir: Path,
    validity: Sequence[Mapping[str, Any]],
    efficiency: Sequence[Mapping[str, Any]],
    robustness: Sequence[Mapping[str, Any]],
    shared: Sequence[Mapping[str, Any]],
    agents: Mapping[str, Any],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - analysis extra contract
        raise RuntimeError("Phase 2 synthesis requires the analysis extra") from exc
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    regimes = sorted({str(row["support_regime"]) for row in validity})
    for regime in regimes:
        rows = sorted(
            (row for row in validity if row["support_regime"] == regime),
            key=lambda row: float(row["budget_fraction"]),
        )
        ax.plot(
            [100 * float(row["budget_fraction"]) for row in rows],
            [float(row["minimum_coverage"]) for row in rows],
            marker="o",
            label=regime.replace("_", " "),
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, label="nominal 0.95")
    ax.set(xlabel="Audit budget (% of candidates)", ylabel="Minimum cell coverage", ylim=(0, 1.01))
    ax.legend(frameon=False, fontsize=7)
    paths.extend(_save_figure(fig, out_dir / "figures" / "calibration_coverage"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    policies = sorted({str(row["policy"]) for row in efficiency})
    for policy in policies:
        rows = sorted(
            (row for row in efficiency if row["policy"] == policy),
            key=lambda row: float(row["budget_fraction"]),
        )
        ax.plot(
            [100 * float(row["budget_fraction"]) for row in rows],
            [float(row["median_upper_slack"]) for row in rows],
            marker="o",
            label=policy.replace("_", " "),
        )
    ax.set(xlabel="Audit budget (% of candidates)", ylabel="Median upper slack")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    paths.extend(_save_figure(fig, out_dir / "figures" / "audit_efficiency"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    numeric_axes = sorted(
        {str(row["axis"]) for row in robustness if isinstance(row["level"], (int, float))}
    )
    for axis in numeric_axes:
        rows = sorted(
            (row for row in robustness if row["axis"] == axis),
            key=lambda row: float(row["level"]),
        )
        ax.plot(
            [float(row["level"]) for row in rows],
            [
                float(
                    row[
                        "corrected_coverage"
                        if row["assumption_status"] == "sensitivity_corrected"
                        else "coverage"
                    ]
                )
                for row in rows
            ],
            marker="o",
            linestyle=("--" if axis == "hidden_guard_feature_prevalence" else "-"),
            label=axis.replace("_", " "),
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Robustness level",
        ylabel="Raw or declared-radius-corrected coverage",
        ylim=(0, 1.01),
    )
    ax.legend(frameon=False, fontsize=6, ncol=2)
    paths.extend(_save_figure(fig, out_dir / "figures" / "robustness_coverage"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    conditions = sorted({str(row["model_condition"]) for row in shared})
    for condition in conditions:
        rows = sorted(
            (row for row in shared if row["model_condition"] == condition),
            key=lambda row: float(row["max_importance_ratio"]),
        )
        ax.plot(
            [float(row["max_importance_ratio"]) for row in rows],
            [float(row["sequential_doubly_robust_rmse"]) for row in rows],
            marker="o",
            label=condition,
        )
    ax.set(xlabel="Maximum importance ratio", ylabel="Sequential DR RMSE")
    ax.legend(frameon=False)
    paths.extend(_save_figure(fig, out_dir / "figures" / "shared_support_rmse"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    audit_rows = agents["audit_rows"]
    for actor in agents["actor_rows"]:
        actor_id = str(actor["actor_id"])
        rows = sorted(
            (
                row
                for row in audit_rows
                if row["actor_id"] == actor_id and row["policy"] == "censure_bound_targeted"
            ),
            key=lambda row: float(row["budget_fraction"]),
        )
        ax.plot(
            [100 * float(row["budget_fraction"]) for row in rows],
            [float(row["target_risk_ucb"]) for row in rows],
            marker="o",
            label=_actor_label(actor_id),
        )
        ax.hlines(
            float(actor["target_risk"]["risk_upper_endpoint"]),
            xmin=0,
            xmax=40,
            linestyles="dotted",
            linewidth=1,
        )
    ax.set(
        xlabel="Audit budget (% of candidates)",
        ylabel="Target-risk upper certificate",
        ylim=(0, 1.01),
    )
    ax.legend(frameon=False)
    paths.extend(_save_figure(fig, out_dir / "figures" / "held_out_agent_bounds"))
    plt.close(fig)
    return paths


def synthesize_phase2_evidence(
    *,
    calibration_catalog: FrozenCalibrationCatalog,
    robustness_catalog: FrozenRobustnessCatalog,
    shared_support_catalog: FrozenSharedSupportCatalog,
    cpu_out_root: str | Path,
    cpu_experiment_id: str,
    agent_summary_path: str | Path,
    reporting_amendment_path: str | Path,
    out_dir: str | Path,
    repository: str | Path = ".",
) -> dict[str, Any]:
    """Validate complete frozen artifacts and emit one manuscript evidence bundle."""

    output = Path(out_dir).expanduser().resolve()
    provenance = collect_provenance(repository)
    if provenance["repository_git_sha"] is None:
        raise RuntimeError("paper synthesis requires a Git checkout")
    if provenance["repository_dirty"]:
        raise RuntimeError("paper synthesis requires a clean repository checkout")
    reporting_freeze_sha256 = _reporting_freeze_sha256(
        reporting_amendment_path,
        calibration_catalog=calibration_catalog,
        robustness_catalog=robustness_catalog,
        shared_support_catalog=shared_support_catalog,
    )
    calibration, calibration_tables = _calibration_evidence(
        catalog=calibration_catalog,
        store=CalibrationRunStore(cpu_out_root, cpu_experiment_id),
    )
    robustness, robustness_table = _robustness_evidence(
        catalog=robustness_catalog,
        store=RobustnessRunStore(cpu_out_root, cpu_experiment_id),
    )
    shared, shared_table = _shared_support_evidence(
        catalog=shared_support_catalog,
        store=SharedSupportRunStore(cpu_out_root, cpu_experiment_id),
    )
    agents = _read_agent_summary(Path(agent_summary_path).expanduser().resolve())
    evidence = {
        "schema_version": PHASE2_SYNTHESIS_SCHEMA_VERSION,
        "protocol_id": "censure-phase2-estimator-v1",
        "source_commit": provenance["repository_git_sha"],
        "reporting_amendment_id": "censure-phase2-estimator-v1-amendment-7",
        "reporting_freeze_sha256": reporting_freeze_sha256,
        "calibration": calibration,
        "robustness": robustness,
        "shared_support": shared,
        "held_out_agents": agents,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / "phase2_evidence.json"
    evidence_digest = atomic_write_json(evidence_path, evidence)
    atomic_write_bytes(evidence_path.with_suffix(".sha256"), f"{evidence_digest}\n".encode())
    table_payloads = {
        "calibration_validity_cells.csv": calibration_tables["validity_cells"],
        "calibration_validity_aggregate.csv": calibration_tables["validity_aggregate"],
        "audit_efficiency_cells.csv": calibration_tables["efficiency_cells"],
        "audit_efficiency_aggregate.csv": calibration_tables["efficiency_aggregate"],
        "audit_efficiency_contrasts.csv": calibration_tables["efficiency_contrasts"],
        "synthetic_longitudinality.csv": calibration_tables["longitudinality_cells"],
        "robustness.csv": robustness_table,
        "shared_support.csv": shared_table,
        "held_out_agents.csv": agents["actor_rows"],
        "held_out_agent_audits.csv": agents["audit_rows"],
    }
    table_paths: list[Path] = []
    for name, rows in table_payloads.items():
        path = output / "tables" / name
        _write_csv(path, rows)
        table_paths.append(path)
    tex_path = output / "phase2_results.tex"
    _write_tex_macros(tex_path, evidence)
    figure_paths = _write_figures(
        out_dir=output,
        validity=calibration_tables["validity_aggregate"],
        efficiency=calibration_tables["efficiency_aggregate"],
        robustness=robustness_table,
        shared=shared_table,
        agents=agents,
    )
    artifact_paths = sorted(
        [
            evidence_path,
            evidence_path.with_suffix(".sha256"),
            *table_paths,
            tex_path,
            *figure_paths,
        ]
    )
    artifact_manifest = {
        "schema_version": "censure.phase2-paper-artifacts.v1",
        "evidence_sha256": evidence["evidence_sha256"],
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    digest = atomic_write_json(output / "artifacts.json", artifact_manifest)
    atomic_write_bytes(output / "artifacts.sha256", f"{digest}\n".encode())
    return {
        "status": "complete",
        "out_dir": str(output),
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence["evidence_sha256"],
        "artifact_count": len(artifact_paths),
    }


__all__ = [
    "PHASE2_SYNTHESIS_SCHEMA_VERSION",
    "synthesize_phase2_evidence",
]
