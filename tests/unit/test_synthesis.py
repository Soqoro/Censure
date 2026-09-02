from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from censure.analysis.synthesis import (
    SourceBundle,
    SynthesisInputError,
    SynthesisSpec,
    analyze_synthesis,
    load_source_bundles,
    load_synthesis_spec,
    write_synthesis_artifacts,
)
from censure.serialization import canonical_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_SPEC_PATH = REPOSITORY_ROOT / "configs" / "analysis" / "exp1_three_model_synthesis_v1.yaml"

ACTORS = (
    ("actor-qwen", "Qwen", "core"),
    ("actor-gemma", "Gemma", "core"),
    ("actor-ministral", "Ministral", "extensions"),
)


def synthesis_spec() -> SynthesisSpec:
    return SynthesisSpec.model_validate(
        {
            "schema_version": "censure.synthesis-spec.v1",
            "synthesis_id": "unit_synthesis",
            "inferential_status": "retrospective_cross_experiment_synthesis",
            "decision_date": "2026-09-02",
            "decision_timezone": "UTC",
            "source_outcomes_inspected_before_freeze": True,
            "complete_preregistered_actor_matrix": False,
            "model_collection_status": "closed_before_synthesis",
            "primary_reporting_policy": "actor_specific_no_pooled_primary_effect",
            "cross_model_contrast_status": "exploratory_task_paired_unadjusted",
            "mechanism_status": "descriptive_noncausal",
            "sources": [
                {
                    "source_id": "core",
                    "root_key": "core_root",
                    "experiment_id": "core_exp",
                    "manifest_relative_path": "core_exp/manifest/frozen_manifest.json",
                    "paired_rows_relative_path": "core_exp/private/paired_rows.json",
                    "validation_report_relative_path": "core_exp/private/validation.json",
                    "context_relative_path": "core_exp/results/analysis_scope.json",
                    "context_kind": "analysis_scope",
                    "context_id": "unit_scope",
                    "context_sha256": "a" * 64,
                    "source_inferential_status": "post_hoc_unit_analysis",
                    "expected_manifest_sha256": "b" * 64,
                    "expected_normalized_row_count": 10,
                    "actor_ids": ["actor-qwen", "actor-gemma"],
                },
                {
                    "source_id": "extensions",
                    "root_key": "extension_root",
                    "experiment_id": "extension_exp",
                    "manifest_relative_path": "extension_exp/manifest/frozen_manifest.json",
                    "paired_rows_relative_path": "extension_exp/private/paired_rows.json",
                    "validation_report_relative_path": "extension_exp/private/validation.json",
                    "context_relative_path": "extension_exp/results/extension_analysis.json",
                    "context_kind": "extension_protocol",
                    "context_id": "unit_protocol",
                    "context_sha256": "c" * 64,
                    "source_inferential_status": "prospective_unit_extension",
                    "expected_manifest_sha256": "d" * 64,
                    "expected_normalized_row_count": 5,
                    "actor_ids": ["actor-ministral"],
                },
            ],
            "actors": [
                {
                    "actor_id": actor_id,
                    "display_name": display,
                    "family": display,
                    "source_id": source,
                    "expected_pair_count": 5,
                    "expected_primary_pair_count": 2,
                }
                for actor_id, display, source in ACTORS
            ],
            "pairwise_contrasts": [
                {
                    "contrast_id": "qwen_minus_gemma",
                    "minuend_actor_id": "actor-qwen",
                    "subtrahend_actor_id": "actor-gemma",
                },
                {
                    "contrast_id": "ministral_minus_qwen",
                    "minuend_actor_id": "actor-ministral",
                    "subtrahend_actor_id": "actor-qwen",
                },
            ],
            "analysis": {
                "analysis_seed": 42,
                "bootstrap_samples": 50,
                "ci_level": 0.95,
                "invalid_behavior_rule": "harmful",
                "primary_split": "confirmatory",
                "primary_guard_pair": "strict_none",
                "unit_key_columns": ["scenario_id", "guard_pair_id"],
                "cluster_columns": ["environment_layer", "domain", "user_task_id"],
                "invariant_columns": [
                    "scenario_id",
                    "split",
                    "environment_layer",
                    "domain",
                    "user_task_id",
                    "injection_task_id",
                    "stratum",
                    "rendered_attack_id",
                    "rendered_attack_sha256",
                    "policy_sha256",
                    "initial_state_sha256",
                    "environment_seed",
                    "guard_pair_id",
                    "behavior_guard_id",
                    "target_guard_id",
                ],
                "require_matching_scenario_set_sha256": True,
                "expected_guard_pair_counts_per_actor": {
                    "strict_none": 2,
                    "strict_degraded_025": 2,
                    "same_guard_strict": 1,
                },
                "expected_confirmatory_degradation_scenarios_per_actor": 2,
                "expected_confirmatory_negative_control_scenarios_per_actor": 1,
                "degradation_guard_order": [
                    {
                        "guard_pair_id": "strict_degraded_025",
                        "display_label": "25% degraded",
                    },
                    {
                        "guard_pair_id": "strict_none",
                        "display_label": "No target guard",
                    },
                ],
                "negative_control_guard_pair": "same_guard_strict",
            },
            "limitations": ["Synthetic unit-test synthesis."],
        }
    )


def _row(
    *,
    actor_id: str,
    scenario_id: str,
    guard_pair_id: str,
    behavior_harm: int | None,
    target_harm: int | None,
    target_status: str = "completed",
) -> dict[str, object]:
    target_guard = {
        "strict_none": "none",
        "strict_degraded_025": "degraded_strict:0.25",
        "same_guard_strict": "strict",
    }[guard_pair_id]
    return {
        "pair_id": f"{actor_id}|{scenario_id}|{guard_pair_id}",
        "scenario_id": scenario_id,
        "split": "confirmatory",
        "environment_layer": "control",
        "domain": "payments",
        "suite_or_domain": "payments",
        "user_task_id": f"task-{scenario_id}",
        "injection_task_id": f"attack-{scenario_id}",
        "stratum": "untrusted_context",
        "rendered_attack_id": f"attack-id-{scenario_id}",
        "rendered_attack_sha256": "e" * 64,
        "policy_sha256": "f" * 64,
        "initial_state_sha256": "1" * 64,
        "environment_seed": int(scenario_id.removeprefix("scenario-")),
        "actor_id": actor_id,
        "guard_pair_id": guard_pair_id,
        "behavior_guard_id": "strict",
        "target_guard_id": target_guard,
        "behavior_status": "completed",
        "target_status": target_status,
        "behavior_harm": behavior_harm,
        "target_harm": target_harm,
        "behavior_user_utility": 0.5,
        "target_user_utility": 0.6,
        "behavior_attempted_unsafe_action": bool(behavior_harm),
        "target_attempted_unsafe_action": bool(target_harm) if target_harm is not None else None,
        "behavior_blocked_call_count": 1,
        "target_blocked_call_count": 0,
        "behavior_proposed_call_count": 2,
        "target_proposed_call_count": 2,
        "behavior_block_rate": 0.5,
        "target_block_rate": 0.0,
        "is_attack": True,
        "is_clean": False,
        "guard_dependent": True if target_status == "completed" else None,
    }


def synthesis_bundles() -> tuple[SourceBundle, SourceBundle]:
    rows_by_actor: dict[str, list[dict[str, object]]] = {}
    for actor_id, _, _ in ACTORS:
        if actor_id == "actor-gemma":
            strict_outcomes = [(0, 0, "completed"), (0, 0, "completed")]
        elif actor_id == "actor-qwen":
            strict_outcomes = [(0, 1, "completed"), (0, None, "timeout")]
        else:
            strict_outcomes = [(0, 1, "completed"), (0, 1, "completed")]
        rows: list[dict[str, object]] = []
        for index, (behavior, target, status) in enumerate(strict_outcomes, start=1):
            rows.append(
                _row(
                    actor_id=actor_id,
                    scenario_id=f"scenario-{index}",
                    guard_pair_id="strict_none",
                    behavior_harm=behavior,
                    target_harm=target,
                    target_status=status,
                )
            )
            rows.append(
                _row(
                    actor_id=actor_id,
                    scenario_id=f"scenario-{index}",
                    guard_pair_id="strict_degraded_025",
                    behavior_harm=0,
                    target_harm=0 if index == 1 else 1,
                )
            )
        rows.append(
            _row(
                actor_id=actor_id,
                scenario_id="scenario-1",
                guard_pair_id="same_guard_strict",
                behavior_harm=0,
                target_harm=0,
            )
        )
        rows_by_actor[actor_id] = rows

    scenario_hash = "9" * 64
    return (
        SourceBundle(
            rows=tuple(rows_by_actor["actor-qwen"] + rows_by_actor["actor-gemma"]),
            provenance={
                "source_id": "core",
                "scenario_set_sha256": scenario_hash,
            },
        ),
        SourceBundle(
            rows=tuple(rows_by_actor["actor-ministral"]),
            provenance={
                "source_id": "extensions",
                "scenario_set_sha256": scenario_hash,
            },
        ),
    )


class SynthesisTests(unittest.TestCase):
    def test_real_synthesis_spec_is_frozen_and_explicitly_retrospective(self) -> None:
        spec = load_synthesis_spec(REAL_SPEC_PATH)
        self.assertEqual(spec.synthesis_id, "qwen_gemma_ministral_v1")
        self.assertTrue(spec.source_outcomes_inspected_before_freeze)
        self.assertFalse(spec.complete_preregistered_actor_matrix)
        self.assertEqual(
            canonical_sha256(spec),
            "3e56c71a6acfc606794052d17bd10b7b00cf4da06d8bd0a0d77c637e435eda37",
        )

    def test_task_paired_actor_effects_and_contrast_bounds(self) -> None:
        result = analyze_synthesis(synthesis_bundles(), synthesis_spec())
        effects = result.actor_effects.set_index("actor_id")
        qwen = effects.loc["actor-qwen"]
        self.assertEqual(qwen["n_primary_pairs"], 2)
        self.assertEqual(qwen["n_complete_primary_pairs"], 1)
        self.assertEqual(qwen["masking_gap_lower_bound"], 0.5)
        self.assertEqual(qwen["masking_gap_upper_bound"], 1.0)

        contrasts = result.pairwise_contrasts.set_index("contrast_id")
        qwen_gemma = contrasts.loc["qwen_minus_gemma"]
        self.assertEqual(qwen_gemma["n_shared_primary_pairs"], 2)
        self.assertEqual(qwen_gemma["n_joint_complete_pairs"], 1)
        self.assertEqual(qwen_gemma["gap_contrast_lower_bound"], 0.5)
        self.assertEqual(qwen_gemma["gap_contrast_upper_bound"], 1.0)

        ministral_qwen = contrasts.loc["ministral_minus_qwen"]
        self.assertEqual(ministral_qwen["complete_case_gap_contrast"], 0.0)
        self.assertEqual(ministral_qwen["gap_contrast_lower_bound"], 0.0)
        self.assertEqual(ministral_qwen["gap_contrast_upper_bound"], 0.5)

    def test_cross_actor_invariant_mismatch_fails_closed(self) -> None:
        bundles = list(synthesis_bundles())
        extension_rows = [dict(row) for row in bundles[1].rows]
        extension_rows[0]["initial_state_sha256"] = "2" * 64
        bundles[1] = SourceBundle(
            rows=tuple(extension_rows),
            provenance=bundles[1].provenance,
        )
        with self.assertRaisesRegex(SynthesisInputError, "initial_state_sha256"):
            analyze_synthesis(tuple(bundles), synthesis_spec())

    def test_source_loader_verifies_manifest_validation_and_context(self) -> None:
        base = synthesis_spec().model_dump(mode="json")
        bundles = synthesis_bundles()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                "core_root": root / "core",
                "extension_root": root / "extensions",
            }
            source_rows = {
                "core": list(bundles[0].rows),
                "extensions": list(bundles[1].rows),
            }
            for source in base["sources"]:
                source_id = source["source_id"]
                manifest = {
                    "schema_version": "censure.exp1-manifest.v1",
                    "experiment_id": source["experiment_id"],
                    "scenario_set_sha256": "9" * 64,
                    "session_set_sha256": "8" * 64,
                }
                manifest_sha = canonical_sha256(manifest)
                source["expected_manifest_sha256"] = manifest_sha
                root_key = source["root_key"]
                artifacts = {
                    source["manifest_relative_path"]: manifest,
                    source["paired_rows_relative_path"]: source_rows[source_id],
                    source["validation_report_relative_path"]: {
                        "ok": True,
                        "issues": [],
                        "normalized_row_count": len(source_rows[source_id]),
                    },
                }
                if source["context_kind"] == "analysis_scope":
                    context = {
                        "scope_config": {
                            "scope_id": source["context_id"],
                            "source_experiment_id": source["experiment_id"],
                            "inferential_status": source["source_inferential_status"],
                        },
                        "scope_config_sha256": source["context_sha256"],
                        "source_manifest_sha256": manifest_sha,
                        "validation_report_sha256": canonical_sha256(
                            artifacts[source["validation_report_relative_path"]]
                        ),
                        "selected_session_count": len(source_rows[source_id]),
                        "included_actor_ids": source["actor_ids"],
                    }
                else:
                    context = {
                        "protocol_id": source["context_id"],
                        "protocol_sha256": source["context_sha256"],
                        "inferential_status": source["source_inferential_status"],
                        "experiment_id": source["experiment_id"],
                        "source_manifest_sha256": manifest_sha,
                        "validation_report_sha256": canonical_sha256(
                            artifacts[source["validation_report_relative_path"]]
                        ),
                        "selected_session_count": len(source_rows[source_id]),
                        "actor_ids": source["actor_ids"],
                    }
                artifacts[source["context_relative_path"]] = context
                for relative, value in artifacts.items():
                    path = roots[root_key] / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value), encoding="utf-8")
            spec = SynthesisSpec.model_validate(base)
            loaded = load_source_bundles(spec, roots)
            self.assertEqual([len(bundle.rows) for bundle in loaded], [10, 5])
            self.assertEqual(
                {bundle.provenance["scenario_set_sha256"] for bundle in loaded},
                {"9" * 64},
            )

    def test_writer_emits_publication_and_machine_readable_artifacts(self) -> None:
        result = analyze_synthesis(synthesis_bundles(), synthesis_spec())
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_synthesis_artifacts(result, temporary)
            for path in paths.values():
                self.assertTrue(path.is_file(), path)
            metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            self.assertEqual(metrics["counts"]["actors"], 3)
            self.assertEqual(len(metrics["negative_controls"]), 3)
            self.assertNotIn("NaN", paths["metrics"].read_text(encoding="utf-8"))
            report = paths["report"].read_text(encoding="utf-8")
            self.assertIn("Retrospective cross-experiment synthesis", report)
            self.assertIn("Actor-specific primary effects", report)
            self.assertIn("Identical-guard negative controls", report)
            self.assertNotIn("[nan", report.lower())
            self.assertIn(
                "Exploratory actor-specific domain effects",
                paths["table_domain_effects"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Descriptive, noncausal mechanism diagnostics",
                paths["table_mechanism_diagnostics"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Matched confirmatory degradation subset",
                paths["table_degradation"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Identical-strict negative controls",
                paths["table_negative_controls"].read_text(encoding="utf-8"),
            )
            self.assertEqual(len(pd.read_parquet(paths["combined_pairs"])), 15)
            write_synthesis_artifacts(result, temporary)
            result.source_provenance[0]["scenario_set_sha256"] = "0" * 64
            with self.assertRaisesRegex(SynthesisInputError, "different source artifacts"):
                write_synthesis_artifacts(result, temporary)


if __name__ == "__main__":
    unittest.main()
