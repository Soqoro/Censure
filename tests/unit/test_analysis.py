from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from censure.analysis.exp1 import (
    AnalysisConfig,
    AnalysisInputError,
    analyze_exp1,
    run_exp1_analysis,
)


def paired_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outcomes = {
        "actor-a": [(0, 1), (0, 1)],
        "actor-b": [(1, 0), (1, 0)],
    }
    for actor, actor_outcomes in outcomes.items():
        for index, (behavior_harm, target_harm) in enumerate(actor_outcomes, start=1):
            rows.append(
                {
                    "pair_id": f"{actor}-{index}",
                    "split": "confirmatory",
                    "domain": "payments" if index == 1 else "communication",
                    "actor_id": actor,
                    "guard_pair_id": "strict_none",
                    "user_task_id": f"task-{index}",
                    "injection_task_id": f"attack-{index}",
                    "behavior_status": "completed",
                    "target_status": "completed",
                    "behavior_harm": behavior_harm,
                    "target_harm": target_harm,
                    "behavior_user_utility": 0.5,
                    "target_user_utility": 0.75,
                    "behavior_unsafe_attempt_rate": float(behavior_harm),
                    "target_unsafe_attempt_rate": float(target_harm),
                    "behavior_block_rate": 0.5,
                    "target_block_rate": 0.0,
                    "guard_dependent": True,
                }
            )
    rows.append(
        {
            "pair_id": "development-only",
            "split": "development",
            "domain": "payments",
            "actor_id": "actor-a",
            "guard_pair_id": "strict_none",
            "user_task_id": "dev-task",
            "injection_task_id": None,
            "behavior_status": "completed",
            "target_status": "completed",
            "behavior_harm": 1,
            "target_harm": 1,
        }
    )
    return rows


class Exp1AnalysisTests(unittest.TestCase):
    def config(self, **overrides) -> AnalysisConfig:
        values = {
            "analysis_seed": 1234,
            "bootstrap_samples": 100,
            "cluster_key": "user_task_id",
            "invalid_behavior_rule": "harmful",
        }
        values.update(overrides)
        return AnalysisConfig(**values)

    def test_confirmatory_paired_metrics_are_signed_and_keep_reverse_events(self) -> None:
        result = analyze_exp1(paired_rows(), self.config())
        metrics = result.metrics["complete_case"]["overall"]["metrics"]

        self.assertEqual(len(result.confirmatory_pairs), 4)
        self.assertAlmostEqual(metrics["behavior_risk"]["value"], 0.5)
        self.assertAlmostEqual(metrics["oracle_target_risk"]["value"], 0.5)
        self.assertAlmostEqual(metrics["masking_gap"]["value"], 0.0)
        self.assertAlmostEqual(metrics["masking_event_rate"]["value"], 0.5)
        self.assertAlmostEqual(metrics["reverse_event_rate"]["value"], 0.5)
        self.assertAlmostEqual(metrics["kendall_tau_b"]["value"], -1.0)
        self.assertAlmostEqual(metrics["actor_ranking_accuracy"]["value"], 0.0)
        self.assertEqual(metrics["pairwise_actor_ranking_reversals"]["value"], 1.0)
        bounds = result.metrics["all_pair_bounds"]["overall"]["metrics"]
        self.assertEqual(bounds["behavior_risk_lower_bound"]["value"], 0.5)
        self.assertEqual(bounds["behavior_risk_upper_bound"]["value"], 0.5)
        self.assertEqual(bounds["oracle_target_risk_lower_bound"]["value"], 0.5)
        self.assertEqual(bounds["oracle_target_risk_upper_bound"]["value"], 0.5)
        self.assertEqual(bounds["masking_gap_lower_bound"]["value"], 0.0)
        self.assertEqual(bounds["masking_gap_upper_bound"]["value"], 0.0)
        self.assertEqual(
            sorted(result.confirmatory_pairs["realized_pair_difference"].unique()),
            [-1.0, 1.0],
        )

    def test_cluster_bootstrap_is_reproducible_for_fixed_seed(self) -> None:
        first = analyze_exp1(paired_rows(), self.config())
        second = analyze_exp1(paired_rows(), self.config())
        first_gap = first.metrics["complete_case"]["overall"]["metrics"]["masking_gap"]
        second_gap = second.metrics["complete_case"]["overall"]["metrics"]["masking_gap"]
        self.assertEqual(first_gap, second_gap)
        self.assertIsNotNone(first_gap["ci_low"])
        self.assertIsNotNone(first_gap["ci_high"])

    def test_secondary_guard_rows_do_not_reweight_primary_overall(self) -> None:
        rows = paired_rows()
        rows.extend(
            {
                **rows[0],
                "pair_id": f"secondary-{index}",
                "user_task_id": f"secondary-task-{index}",
                "guard_pair_id": "strict_degraded_100",
                "behavior_harm": 0,
                "target_harm": 1,
            }
            for index in range(10)
        )
        result = analyze_exp1(rows, self.config())
        overall = result.metrics["complete_case"]["overall"]["metrics"]
        assert overall["masking_gap"]["value"] == 0.0
        secondary = result.metrics["complete_case"]["by_guard_pair"]["strict_degraded_100"][
            "metrics"
        ]
        assert secondary["masking_gap"]["value"] == 1.0

    def test_invalid_sensitivity_policy_is_explicit_and_does_not_label_errors_safe(self) -> None:
        rows = paired_rows()[:2]
        rows.extend(
            [
                {
                    **rows[0],
                    "pair_id": "invalid-target",
                    "user_task_id": "task-invalid-target",
                    "target_status": "timeout",
                    "target_harm": None,
                    "behavior_harm": 0,
                },
                {
                    **rows[1],
                    "pair_id": "invalid-behavior",
                    "user_task_id": "task-invalid-behavior",
                    "behavior_status": "model_error",
                    "behavior_harm": None,
                    "target_harm": 0,
                },
            ]
        )
        result = analyze_exp1(rows, self.config())
        complete = result.metrics["complete_case"]["overall"]
        sensitivity = result.metrics["sensitivity"]["overall"]

        self.assertEqual(complete["n_pairs"], 2)
        self.assertEqual(sensitivity["n_pairs"], 4)
        self.assertEqual(result.metrics["sensitivity_policy"]["invalid_target"], "harmful")
        self.assertEqual(result.metrics["sensitivity_policy"]["invalid_behavior"], "harmful")
        self.assertAlmostEqual(
            complete["metrics"]["invalid_run_rate"]["value"],
            0.5,
        )
        invalid_target = result.all_pairs.set_index("pair_id").loc["invalid-target"]
        invalid_behavior = result.all_pairs.set_index("pair_id").loc["invalid-behavior"]
        self.assertEqual(invalid_target["sensitivity_target_harm"], 1.0)
        self.assertEqual(invalid_behavior["sensitivity_behavior_harm"], 1.0)
        self.assertTrue(pd.isna(invalid_target["realized_pair_difference"]))

        bounds = result.metrics["all_pair_bounds"]["overall"]["metrics"]
        self.assertEqual(bounds["behavior_risk_lower_bound"]["value"], 0.0)
        self.assertEqual(bounds["behavior_risk_upper_bound"]["value"], 0.25)
        self.assertEqual(bounds["oracle_target_risk_lower_bound"]["value"], 0.5)
        self.assertEqual(bounds["oracle_target_risk_upper_bound"]["value"], 0.75)
        self.assertEqual(bounds["masking_gap_lower_bound"]["value"], 0.25)
        self.assertEqual(bounds["masking_gap_upper_bound"]["value"], 0.75)
        self.assertEqual(bounds["behavior_harm_or_invalid_rate"]["value"], 0.25)
        self.assertEqual(bounds["target_harm_or_invalid_rate"]["value"], 0.75)
        overall_bounds = result.metrics["all_pair_bounds"]["overall"]
        self.assertEqual(overall_bounds["n_behavior_invalid"], 1)
        self.assertEqual(overall_bounds["n_target_invalid"], 1)
        self.assertEqual(overall_bounds["n_invalid_pairs"], 2)
        self.assertEqual(overall_bounds["invalid_pair_rate"], 0.5)
        sensitivity_gap = sensitivity["metrics"]["masking_gap"]["value"]
        self.assertLessEqual(bounds["masking_gap_lower_bound"]["value"], sensitivity_gap)
        self.assertLessEqual(sensitivity_gap, bounds["masking_gap_upper_bound"]["value"])

    def test_safe_invalid_behavior_rule_changes_only_exposed_sensitivity_imputation(self) -> None:
        row = paired_rows()[0]
        row = {
            **row,
            "behavior_status": "timeout",
            "behavior_harm": None,
            "target_harm": 1,
        }
        result = analyze_exp1([row], self.config(invalid_behavior_rule="safe"))
        sensitivity = result.metrics["sensitivity"]["overall"]["metrics"]
        self.assertEqual(result.metrics["sensitivity_policy"]["invalid_behavior"], "safe")
        self.assertEqual(sensitivity["behavior_risk"]["value"], 0.0)
        self.assertEqual(sensitivity["oracle_target_risk"]["value"], 1.0)
        self.assertEqual(sensitivity["masking_gap"]["value"], 1.0)

    def test_missing_ranking_cells_are_na_with_reason(self) -> None:
        result = analyze_exp1([paired_rows()[0]], self.config(bootstrap_samples=0))
        tau = result.metrics["complete_case"]["overall"]["metrics"]["kendall_tau_b"]
        self.assertIsNone(tau["value"])
        self.assertIn("requires", tau["reason"])

    def test_duplicate_pairs_and_unknown_status_fail_actionably(self) -> None:
        row = paired_rows()[0]
        with self.assertRaisesRegex(AnalysisInputError, "duplicate pair_id"):
            analyze_exp1([row, dict(row)], self.config())
        with self.assertRaisesRegex(AnalysisInputError, "unknown statuses"):
            analyze_exp1([{**row, "target_status": "probably_safe"}], self.config())

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow analysis extra is absent")
    def test_writer_emits_all_required_artifacts_without_json_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_exp1_analysis(paired_rows(), temporary, self.config(bootstrap_samples=20))
            root = Path(temporary)
            required = [
                "metrics.json",
                "paired_runs.parquet",
                "masking_by_domain.csv",
                "actor_rankings.csv",
                "guard_pair_summary.csv",
                "missing_harm_bounds.csv",
                "table_masking.tex",
                "report.md",
                "figures/behavior_vs_target_risk.png",
                "figures/masking_gap.png",
                "figures/ranking_reversals.png",
            ]
            for relative in required:
                self.assertTrue((root / relative).is_file(), relative)
            decoded = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(decoded["counts"]["confirmatory_pairs"], 4)
            self.assertNotIn("NaN", (root / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                len(pd.read_parquet(root / "paired_runs.parquet")), len(result.all_pairs)
            )
            latex = (root / "table_masking.tex").read_text(encoding="utf-8")
            self.assertIn("Domain & Behavior risk & Oracle target risk", latex)
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("All-pair missing-harm bounds", report)
            bounds_frame = pd.read_csv(root / "missing_harm_bounds.csv")
            self.assertIn("masking_gap_lower_bound", bounds_frame.columns)
            self.assertIn("masking_gap_upper_bound", bounds_frame.columns)


if __name__ == "__main__":
    unittest.main()
