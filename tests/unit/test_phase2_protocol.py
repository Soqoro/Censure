from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "phase2_estimator_v1.yaml"
AMENDMENT_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "phase2_estimator_v1_amendment_1.yaml"
)
PROTOCOL_PATH = REPOSITORY_ROOT / "docs" / "PHASE2_ESTIMATOR_PROTOCOL.md"


def _load_config() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_phase2_protocol_freezes_inferential_status_and_error_budget() -> None:
    config = _load_config()

    assert config["schema_version"] == "censure.phase2-experiment.v1"
    assert config["protocol_id"] == "censure-phase2-estimator-v1"
    assert config["phase2_outcomes_inspected_before_freeze"] is False
    assert config["experiment1_outcomes_known"] is True
    assert config["inferential_status"] == {
        "calibration": "prospective_estimator_validation",
        "held_out_agents": "prospective_estimator_validation",
        "experiment1_replay": "retrospective_software_and_external_validity_analysis",
    }

    confidence = config["confidence"]
    assert confidence["cohort_alpha_audit"] == 0.05
    assert confidence["population_alpha_audit"] + confidence["population_alpha_task"] == 0.05
    assert confidence["invalid_suffix_rule"] == "worst_case_harm"
    assert confidence["nonauditable_rule"] == "worst_case_harm"


def test_phase2_protocol_freezes_primary_audit_design() -> None:
    config = _load_config()
    auditing = config["auditing"]

    assert auditing["sampling"] == "predictable_with_replacement"
    assert auditing["exploration_epsilon"] == 0.10
    assert auditing["budgets_fraction"] == [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    assert auditing["duplicate_draws"] == "retain_and_report"
    assert auditing["policies"] == [
        "uniform",
        "target_mass",
        "guard_score",
        "uncertainty",
        "downstream_harm",
        "censure_bound_targeted",
    ]

    calibration = config["calibration"]
    assert calibration["repetitions"] == 2000
    assert calibration["cohort_sizes"] == [200, 500, 1000]
    assert calibration["target_harm_prevalence"] == [0.05, 0.20, 0.50]


def test_phase2_protocol_keeps_held_out_and_retrospective_evidence_distinct() -> None:
    config = _load_config()

    assert config["held_out_agents"]["selection_fields"] == "metadata_only"
    assert config["held_out_agents"]["require_unseen_experiment1_confirmatory_unit"] is True
    assert config["experiment1_replay"]["prohibit_relabel_as_phase2_confirmatory"] is True

    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "No Phase 2 result may be inspected before steps 1--4 pass locally." in protocol
    assert "General stochastic shared-support OPE is a\nsecondary analysis" in protocol


def test_phase2_amendment_freezes_outcome_blind_implementation_details() -> None:
    amendment = yaml.safe_load(AMENDMENT_PATH.read_text(encoding="utf-8"))

    assert amendment["schema_version"] == "censure.phase2-amendment.v1"
    assert amendment["parent_protocol_id"] == "censure-phase2-estimator-v1"
    assert amendment["parent_freeze_commit"] == (
        "c1c6d0d3c401ed02bef632b2c793cb4596e2fa98"
    )
    assert amendment["frozen_primary_calibration_outcomes_inspected"] is False
    assert amendment["held_out_agent_suffix_outcomes_inspected"] is False
    assert amendment["enumerable_dgp"]["mixed_auditable_probability"] == 0.75
    assert amendment["enumerable_dgp"]["delayed_harm_probability"] == 0.60
    assert amendment["budgeting"]["positive_budget_rounding"] == "ceiling"
    assert amendment["policy_comparison"]["common_uniform_random_tape_across_policies"] is True
    assert amendment["policy_comparison"]["completed_duplicate_suffix_cost"] == 0
