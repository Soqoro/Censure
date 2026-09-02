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
AMENDMENT_2_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "phase2_estimator_v1_amendment_2.yaml"
)
AMENDMENT_3_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "phase2_estimator_v1_amendment_3.yaml"
)
AMENDMENT_4_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "phase2_estimator_v1_amendment_4.yaml"
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


def test_phase2_second_amendment_freezes_validity_and_efficiency_grids() -> None:
    amendment = yaml.safe_load(AMENDMENT_2_PATH.read_text(encoding="utf-8"))

    assert amendment["amendment_id"] == "censure-phase2-estimator-v1-amendment-2"
    assert amendment["parent_freeze_commit"] == (
        "77f1784aae21c9e5043ac49faf0006bfe0b39ef1"
    )
    assert amendment["frozen_primary_calibration_outcomes_inspected"] is False
    assert amendment["validity_grid"]["policy"] == "target_mass"
    assert amendment["validity_grid"]["repetitions"] == 2000
    assert amendment["validity_grid"]["full_overlap_zero_support_mass"] == [0.0]
    assert len(amendment["efficiency_grid"]["policies"]) == 6
    assert amendment["efficiency_grid"]["cohort_sizes"] == [500]
    assert amendment["efficiency_grid"]["reuse_target_mass_from_validity"] is True
    assert amendment["execution"]["base_seed"] == 20260902
    assert amendment["execution"]["atomic_work_item"] == (
        "cell_id_and_repetition_chunk"
    )
    assert amendment["execution"]["repetitions_per_chunk"] == 25
    assert amendment["execution"]["checksummed_resume_required"] is True


def test_phase2_third_amendment_freezes_population_and_robustness() -> None:
    amendment = yaml.safe_load(AMENDMENT_3_PATH.read_text(encoding="utf-8"))

    assert amendment["parent_freeze_commit"] == (
        "86588e2017835a29edda21a8208f91c19d2c5ca2"
    )
    assert amendment["frozen_primary_calibration_outcomes_inspected"] is False
    assert amendment["frozen_robustness_outcomes_inspected"] is False
    assert amendment["population_calibration"]["audit_alpha"] == 0.025
    assert amendment["population_calibration"]["task_sampling_alpha"] == 0.025
    assert amendment["robustness_execution"]["repetitions"] == 2000
    assert amendment["robustness_execution"]["repetitions_per_chunk"] == 25
    assert amendment["robustness_axes"]["hidden_guard_feature_prevalence"][
        "assumption_status"
    ] == "unidentified_when_positive"
    assert amendment["robustness_axes"]["sandbox_harm_shift"]["correction"] == (
        "add_declared_radius"
    )


def test_phase2_fourth_amendment_freezes_shared_support_ope() -> None:
    amendment = yaml.safe_load(AMENDMENT_4_PATH.read_text(encoding="utf-8"))

    assert amendment["parent_freeze_commit"] == (
        "b20fae97a1551c31efab27592c61331571edf10d"
    )
    assert amendment["frozen_shared_support_outcomes_inspected"] is False
    execution = amendment["shared_support_execution"]
    assert execution["cohort_size"] == 1000
    assert execution["repetitions"] == 2000
    assert execution["repetitions_per_chunk"] == 25
    assert execution["max_importance_ratios"] == [1.0, 2.0, 5.0, 10.0]
    assert execution["model_conditions"] == ["correct", "misspecified", "constant"]
    assert execution["exact_target_risk"] == 0.40
    assert amendment["hybrid_certificate"]["supported_alpha"] == 0.025
    assert amendment["hybrid_certificate"]["audit_alpha"] == 0.025
