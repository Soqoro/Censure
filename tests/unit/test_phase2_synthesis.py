from __future__ import annotations

import re
from pathlib import Path

import pytest

from censure.estimation.cli import build_parser
from censure.estimation.synthesis import (
    EXPECTED_AUDIT_POLICIES,
    EXPECTED_HELD_OUT_ACTORS,
    EXPECTED_HELD_OUT_MANIFEST_SHA256,
    _equal_grid_median_contrast,
    _read_agent_summary,
    _trapezoid,
    _write_tex_macros,
)
from censure.storage import CorruptArtifactError, atomic_write_bytes, atomic_write_json


def test_trapezoid_integrates_sorted_unique_budget_curve() -> None:
    assert _trapezoid(((0.4, 0.0), (0.0, 1.0), (0.2, 0.5))) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="unique budget"):
        _trapezoid(((0.0, 1.0), (0.0, 0.5)))


def test_equal_grid_median_contrast_is_paired_deterministic_and_equal_weighted() -> None:
    pairs = {
        ("a", 500, 0.2, 0.25): ((0.1, 0.4), (0.2, 0.5), (0.3, 0.6)),
        ("b", 500, 0.5, 0.75): ((0.7, 0.5), (0.8, 0.6), (0.9, 0.7)),
    }
    first, rows = _equal_grid_median_contrast(
        pairs,
        bootstrap_seed=17,
        bootstrap_samples=200,
    )
    second, _ = _equal_grid_median_contrast(
        pairs,
        bootstrap_seed=17,
        bootstrap_samples=200,
    )

    assert first == second
    assert first["design_count"] == 2
    assert first["estimate"] == pytest.approx(-0.05)
    assert first["favorable_design_rate"] == 0.5
    assert [row["median_contrast"] for row in rows] == pytest.approx([-0.3, 0.2])


def _agent_summary() -> dict[str, object]:
    actor_rows = []
    audit_rows = []
    for actor_index, actor_id in enumerate(sorted(EXPECTED_HELD_OUT_ACTORS)):
        actor_rows.append(
            {
                "actor_id": actor_id,
                "target_risk": {
                    "risk_lower_endpoint": 0.1,
                    "risk_upper_endpoint": 0.2,
                    "invalid_rate": 0.1,
                },
                "full_target_observed_final_trajectory_cost": {
                    "combined_cost": 100 + actor_index,
                },
            }
        )
        for policy in sorted(EXPECTED_AUDIT_POLICIES):
            for budget in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
                audit_rows.append(
                    {
                        "actor_id": actor_id,
                        "policy": policy,
                        "budget_fraction": budget,
                        "target_risk_ucb": 0.3,
                        "covers_target_identification_upper": not (
                            actor_index == 0 and policy == "uniform" and budget == 0.40
                        ),
                        "suffix_tool_steps": 10,
                        "generation_tokens": 20,
                    }
                )
    return {
        "schema_version": "censure.agent-audit-study-summary.v1",
        "protocol_id": "censure-phase2-estimator-v1",
        "source_manifest_sha256": EXPECTED_HELD_OUT_MANIFEST_SHA256,
        "post_audit_full_oracle_revealed": True,
        "actor_rows": actor_rows,
        "audit_rows": audit_rows,
    }


def _write_checksummed(path: Path, payload: object) -> None:
    digest = atomic_write_json(path, payload)
    atomic_write_bytes(path.with_suffix(".sha256"), f"{digest}\n".encode())


def test_agent_summary_requires_frozen_matrix_and_derives_costs(tmp_path: Path) -> None:
    path = tmp_path / "study_summary.json"
    _write_checksummed(path, _agent_summary())

    loaded = _read_agent_summary(path)

    assert loaded["audit_coverage_failure_count"] == 1
    assert len(loaded["primary_censure_020_rows"]) == 3
    assert loaded["primary_censure_020_rows"][0]["logical_combined_cost"] == 30
    assert loaded["primary_censure_020_rows"][0][
        "logical_cost_fraction_of_full_target"
    ] == pytest.approx(0.3)


def test_agent_summary_checksum_and_matrix_tampering_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "study_summary.json"
    payload = _agent_summary()
    _write_checksummed(path, payload)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CorruptArtifactError, match="checksum"):
        _read_agent_summary(path)

    payload = _agent_summary()
    assert isinstance(payload["audit_rows"], list)
    payload["audit_rows"].pop()
    _write_checksummed(path, payload)
    with pytest.raises(CorruptArtifactError, match="matrix"):
        _read_agent_summary(path)


def test_cli_exposes_checksumming_paper_synthesis() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "synthesize-paper",
            "--cpu-out-root",
            "/tmp/cpu",
            "--agent-summary",
            "/tmp/agents.json",
            "--out-dir",
            "/tmp/paper",
        ]
    )
    assert args.cpu_experiment_id == "phase2_estimator_v1"
    assert args.amendment_7.name == "phase2_estimator_v1_amendment_7.yaml"


def test_generated_and_placeholder_latex_macro_interfaces_match(tmp_path: Path) -> None:
    evidence = {
        "calibration": {
            "validity_cell_count": 81,
            "efficiency_cell_count": 108,
            "validity": {
                "minimum_coverage": 0.95,
                "coverage_gate_failure_count": 0,
                "minimum_population_coverage": 0.95,
                "population_coverage_gate_failure_count": 0,
                "maximum_false_release_rate": 0.01,
            },
            "efficiency": {
                "primary_comparison": {
                    "slack_at_020": {
                        "estimate": -0.1,
                        "ci_low": -0.2,
                        "ci_high": -0.01,
                        "favorable_design_rate": 0.8,
                    },
                    "efficiency_claim_supported": True,
                }
            },
            "synthetic_longitudinality": {
                "mean_signed_one_step_bias": -0.2,
                "mean_delayed_harm_rate": 0.2,
            },
        },
        "robustness": {
            "minimum_identified_coverage": 0.95,
            "minimum_shift_corrected_coverage": 0.95,
            "unidentified_cell_count": 2,
        },
        "shared_support": {"minimum_ips_coverage": 0.95},
        "held_out_agents": {
            "audit_coverage_failure_count": 0,
            "actor_rows": [
                {
                    "actor_id": actor_id,
                    "target_risk": {
                        "risk_lower_endpoint": 0.1,
                        "risk_upper_endpoint": 0.2,
                        "invalid_rate": 0.05,
                    },
                    "primary_censure_020": {
                        "target_risk_ucb": 0.3,
                        "logical_cost_fraction_of_full_target": 0.25,
                    },
                }
                for actor_id in sorted(EXPECTED_HELD_OUT_ACTORS)
            ],
        },
    }
    generated = tmp_path / "phase2_results.tex"
    _write_tex_macros(generated, evidence)
    placeholder = Path(__file__).resolve().parents[2] / "paper" / "phase2_results_placeholder.tex"
    pattern = re.compile(r"\\newcommand\{\\([^}]+)\}")
    generated_names = set(pattern.findall(generated.read_text(encoding="utf-8")))
    placeholder_names = set(pattern.findall(placeholder.read_text(encoding="utf-8")))

    assert generated_names - {"PhaseTwoResultsAvailable"} == placeholder_names
