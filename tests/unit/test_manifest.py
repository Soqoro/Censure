from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import censure.adapters.agentdojo_v0135 as dojo_adapter
from censure.adapters.agentdojo_v0135 import (
    AGENTDOJO_ADAPTER_VERSION,
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AgentDojoStateSnapshot,
    AgentDojoToolSchema,
    FrozenAgentDojoScenario,
)
from censure.config import resolved_experiment_config
from censure.manifest import (
    AgentDojoCatalog,
    ManifestError,
    assert_outcome_free,
    build_manifest,
    derive_session_id,
    dry_run_manifest_summary,
)
from censure.schemas import (
    AuthorizationPolicy,
    AuthorizationRule,
    EnvironmentLayer,
    RuleEffect,
    ScenarioSplit,
)
from censure.serialization import canonical_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _config(name: str) -> dict[str, Any]:
    return resolved_experiment_config(
        REPOSITORY_ROOT / "configs" / "experiments" / f"{name}.yaml",
        resolve_remote=False,
    )


class FakeAgentDojoSource:
    package_version = AGENTDOJO_PACKAGE_VERSION
    benchmark_version = AGENTDOJO_BENCHMARK_VERSION

    def __init__(self, *, omit_runtime_fields: bool = False) -> None:
        self.freeze_calls: list[tuple[str, str, str | None, str | None]] = []
        self.outcome_access_count = 0
        self.omit_runtime_fields = omit_runtime_fields
        self._catalogs = {
            suite: AgentDojoCatalog(
                suite_name=suite,
                user_task_ids=tuple(f"user_task_{index}" for index in range(40)),
                injection_task_ids=tuple(f"injection_task_{index}" for index in range(10)),
            )
            for suite in ("workspace", "slack", "travel", "banking")
        }

    def catalog(self, suite_name: str) -> AgentDojoCatalog:
        return self._catalogs[suite_name]

    def freeze_scenario(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str | None,
        *,
        attack_name: str | None,
    ) -> Any:
        self.freeze_calls.append((suite_name, user_task_id, injection_task_id, attack_name))
        if injection_task_id is None:
            if attack_name is not None:
                raise AssertionError("clean freeze received an attack")
            rendered: dict[str, str] = {}
            frozen_attack_name = None
            pipeline_name = None
        else:
            if attack_name != "tool_knowledge":
                raise AssertionError(f"configured attack was silently changed to {attack_name!r}")
            rendered = {
                f"{suite_name}:injection-slot": (
                    f"Frozen payload for {user_task_id}/{injection_task_id} using {attack_name}."
                )
            }
            frozen_attack_name = attack_name
            pipeline_name = "local"

        state = {
            "suite": suite_name,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "rendered": rendered,
            "records": [],
        }
        snapshot = AgentDojoStateSnapshot(
            suite_name=suite_name,
            environment_type=f"fake.{suite_name}.Environment",
            state=state,
            sha256=canonical_sha256(state),
        )
        deny_id = f"deny-{suite_name}-{user_task_id}"
        policy = AuthorizationPolicy(
            policy_id=f"policy-{suite_name}-{user_task_id}",
            rules=(
                AuthorizationRule(
                    rule_id=deny_id,
                    effect=RuleEffect.DENY,
                    tool_name="mutate_record",
                    reason="Unrequested mutation.",
                    argument_equals={"target": "untrusted"},
                ),
                AuthorizationRule(
                    rule_id=f"allow-{suite_name}-{user_task_id}",
                    effect=RuleEffect.ALLOW,
                    tool_name="read_record",
                    reason="Read-only task operation.",
                ),
                AuthorizationRule(
                    rule_id=f"weak-fallback-{suite_name}-{user_task_id}",
                    effect=RuleEffect.ALLOW,
                    tool_name="mutate_record",
                    reason="Weak-only fallback.",
                    argument_equals={"target": "untrusted"},
                ),
            ),
            default_rule_id=f"default-{suite_name}-{user_task_id}",
            weak_ignored_rule_ids=(deny_id,),
        )
        frozen = FrozenAgentDojoScenario(
            adapter_version=AGENTDOJO_ADAPTER_VERSION,
            suite_name=suite_name,
            suite_benchmark_version=(1, 2, 2),
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            attack_name=frozen_attack_name,
            attack_target_pipeline_name=pipeline_name,
            user_request=f"Complete {user_task_id} without unrelated mutations.",
            rendered_injections=rendered,
            injection_locations=tuple(rendered),
            injection_projections=tuple(
                dojo_adapter._project_injection_into_state(location, payload, state)
                for location, payload in rendered.items()
            ),
            rendered_attack_sha256=canonical_sha256(rendered) if rendered else None,
            available_tools=(
                AgentDojoToolSchema(
                    name="read_record",
                    description="Read a record.",
                    parameters={"type": "object", "properties": {}},
                ),
                AgentDojoToolSchema(
                    name="mutate_record",
                    description="Mutate a record.",
                    parameters={
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                    },
                ),
            ),
            authorization_policy=policy,
            initial_state=snapshot,
        )
        if not self.omit_runtime_fields:
            return frozen
        dumped = frozen.model_dump(mode="python")
        dumped.pop("authorization_policy")
        return SimpleNamespace(**dumped)

    def outcome_for(self, *_args: Any, **_kwargs: Any) -> None:
        self.outcome_access_count += 1
        raise AssertionError("manifest code must never inspect model outcomes")


class FullManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = FakeAgentDojoSource()
        cls.config = _config("exp1_full")
        cls.manifest = build_manifest(cls.config, agentdojo_source=cls.source)

    def test_exact_frozen_scenario_counts_and_splits(self) -> None:
        manifest = self.manifest
        self.assertEqual(len(manifest.scenarios), 320)
        self.assertEqual(
            Counter(item.environment_layer for item in manifest.scenarios),
            {EnvironmentLayer.AGENTDOJO: 160, EnvironmentLayer.CONTROL: 160},
        )
        self.assertEqual(
            manifest.summary.scenarios_by_split,
            {"confirmatory": 232, "development": 64, "smoke": 24},
        )

        agent = [
            item
            for item in manifest.scenarios
            if item.environment_layer is EnvironmentLayer.AGENTDOJO
        ]
        control = [
            item
            for item in manifest.scenarios
            if item.environment_layer is EnvironmentLayer.CONTROL
        ]
        for suite in ("workspace", "slack", "travel", "banking"):
            suite_rows = [item for item in agent if item.suite_or_domain == suite]
            self.assertEqual(len(suite_rows), 40)
            self.assertEqual(
                Counter(item.split for item in suite_rows),
                {
                    ScenarioSplit.SMOKE: 2,
                    ScenarioSplit.DEVELOPMENT: 8,
                    ScenarioSplit.CONFIRMATORY: 30,
                },
            )
        self.assertEqual(
            Counter(item.split for item in control),
            {
                ScenarioSplit.SMOKE: 16,
                ScenarioSplit.DEVELOPMENT: 32,
                ScenarioSplit.CONFIRMATORY: 112,
            },
        )
        self.assertTrue(
            all(
                count == 10
                for count in Counter(
                    (item.suite_or_domain, item.metadata["stratum"]) for item in control
                ).values()
            )
        )

    def test_agentdojo_task_pairs_and_all_identifiers_are_unique(self) -> None:
        agent = [
            item
            for item in self.manifest.scenarios
            if item.environment_layer is EnvironmentLayer.AGENTDOJO
        ]
        pairs = [
            (item.suite_or_domain, item.user_task_id, item.injection_task_id) for item in agent
        ]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len({item.scenario_id for item in self.manifest.scenarios}), 320)
        self.assertEqual(
            len({item.session_id for item in self.manifest.sessions}),
            len(self.manifest.sessions),
        )

    def test_full_actor_guard_expansion_has_exact_preregistered_counts(self) -> None:
        manifest = self.manifest
        self.assertEqual(len(manifest.sessions), 2_016)
        self.assertEqual(manifest.summary.trajectory_count, 4_032)
        self.assertEqual(
            manifest.summary.sessions_by_guard_pair,
            {
                "same_guard_strict": 96,
                "strict_degraded_025": 240,
                "strict_degraded_050": 240,
                "strict_degraded_075": 240,
                "strict_degraded_100": 240,
                "strict_none": 960,
            },
        )
        self.assertEqual(set(manifest.summary.sessions_by_actor.values()), {672})

        by_pair = {
            pair: {row.scenario_id for row in manifest.sessions if row.guard_pair_id == pair}
            for pair in manifest.summary.sessions_by_guard_pair
        }
        degradation_sets = [
            by_pair[pair] for pair in by_pair if pair.startswith("strict_degraded_")
        ]
        self.assertTrue(all(selected == degradation_sets[0] for selected in degradation_sets[1:]))
        self.assertEqual(len(degradation_sets[0]), 80)
        self.assertEqual(len(by_pair["same_guard_strict"]), 32)

        scenario_by_id = {item.scenario_id: item for item in manifest.scenarios}
        degradation_groups = Counter(
            (
                scenario_by_id[scenario_id].environment_layer.value,
                scenario_by_id[scenario_id].suite_or_domain,
            )
            for scenario_id in degradation_sets[0]
        )
        same_groups = Counter(
            (
                scenario_by_id[scenario_id].environment_layer.value,
                scenario_by_id[scenario_id].suite_or_domain,
            )
            for scenario_id in by_pair["same_guard_strict"]
        )
        self.assertEqual(set(degradation_groups.values()), {10})
        self.assertEqual(set(same_groups.values()), {4})

    def test_payload_location_state_policy_and_runtime_spec_are_frozen(self) -> None:
        attacked = next(
            item
            for item in self.manifest.scenarios
            if item.environment_layer is EnvironmentLayer.AGENTDOJO
            and item.injection_task_id is not None
        )
        self.assertEqual(tuple(attacked.rendered_attack), attacked.injection_locations)
        self.assertEqual(
            canonical_sha256(attacked.rendered_attack), attacked.rendered_attack_sha256
        )
        self.assertEqual(
            attacked.metadata["attack_target_pipeline_name"],
            "local",
        )
        runtime_spec = cast(dict[str, Any], attacked.metadata["runtime_spec"])
        self.assertEqual(runtime_spec["rendered_injections"], attacked.rendered_attack)
        self.assertEqual(
            runtime_spec["initial_state"]["state"], attacked.canonical_initial_state.state
        )
        self.assertEqual(
            runtime_spec["authorization_policy"], attacked.policy.model_dump(mode="json")
        )

        controlled = next(
            item
            for item in self.manifest.scenarios
            if item.environment_layer is EnvironmentLayer.CONTROL
        )
        control_runtime = cast(dict[str, Any], controlled.metadata["runtime_spec"])
        self.assertEqual(control_runtime["scenario_id"], controlled.scenario_id)
        self.assertEqual(
            control_runtime["canonical_initial_state"], controlled.canonical_initial_state.state
        )
        self.assertEqual(
            controlled.metadata["control_spec_sha256"],
            canonical_sha256(control_runtime),
        )

    def test_manifest_is_outcome_free(self) -> None:
        assert_outcome_free(self.manifest)
        serialized = json.dumps(self.manifest.model_dump(mode="json"), sort_keys=True)
        for forbidden in (
            '"attack_goal_achieved"',
            '"h_b"',
            '"h_star"',
            '"run_status"',
            '"terminal_harm"',
            '"user_utility"',
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(self.source.outcome_access_count, 0)

    def test_every_session_field_and_scenario_fingerprint_affects_key(self) -> None:
        session = self.manifest.sessions[0]
        scenario = next(
            item for item in self.manifest.scenarios if item.scenario_id == session.scenario_id
        )
        baseline = derive_session_id(session, scenario)
        fields = session.model_dump(mode="json", exclude={"session_id"})
        for field_name, value in fields.items():
            changed = copy.deepcopy(fields)
            if isinstance(value, int):
                changed[field_name] = value + 1
            elif value is None:
                changed[field_name] = "changed"
            else:
                changed[field_name] = f"{value}:changed"
            with self.subTest(field=field_name):
                self.assertNotEqual(derive_session_id(changed, scenario), baseline)

        changed_scenario = scenario.model_copy(
            update={"metadata": {**scenario.metadata, "scientific_revision": "changed"}}
        )
        self.assertNotEqual(derive_session_id(session, changed_scenario), baseline)

    def test_manifest_is_deterministic_and_seed_sensitive(self) -> None:
        repeated = build_manifest(self.config, agentdojo_source=FakeAgentDojoSource())
        self.assertEqual(repeated, self.manifest)
        self.assertEqual(repeated.manifest_sha256, self.manifest.manifest_sha256)

        changed_config = copy.deepcopy(self.config)
        changed_config["manifest_seed"] += 1
        changed = build_manifest(changed_config, agentdojo_source=FakeAgentDojoSource())
        self.assertNotEqual(changed.manifest_sha256, self.manifest.manifest_sha256)
        self.assertNotEqual(
            {item.session_id for item in changed.sessions},
            {item.session_id for item in self.manifest.sessions},
        )


class GenericManifestTests(unittest.TestCase):
    def test_legacy_smoke_session_identity_is_stable(self) -> None:
        manifest = build_manifest(
            _config("exp1_smoke_v2"),
            agentdojo_source=FakeAgentDojoSource(),
        )

        self.assertEqual(
            manifest.sessions[0].session_id,
            "0221f021afb2430001e6f48e671fdcaee24eaf32c63d1af64bd2789a1ed49ceb",
        )

    def test_every_extension_runtime_key_changes_session_identity(self) -> None:
        config = _config("exp1_gpt_oss_20b_smoke_v1")
        baseline = build_manifest(config, agentdojo_source=FakeAgentDojoSource())
        baseline_ids = {session.session_id for session in baseline.sessions}
        alias = "gpt_oss_20b"
        mutations = {
            "chat_template_asset": "alternate_chat_template.jinja",
            "checkpoint_load_mode": "changed-load-mode",
            "history_projection": "changed-history-projection",
            "model_loader": "changed-model-loader",
            "native_tools": False,
            "native_weight_format": "changed-weight-format",
            "prompt_format_version": "changed-prompt-format",
            "required_package_versions": {"transformers": "0"},
            "response_parser_version": "changed-response-parser",
            "serializer_fingerprint_sha256": "c" * 64,
            "template_current_date": "2026-09-02",
            "tokenizer_asset_sha256": "d" * 64,
            "tokenizer_backend": "changed-tokenizer-backend",
            "tool_name_projection": "changed-tool-name-projection",
            "tool_protocol": "changed-tool-protocol",
        }

        for key, value in mutations.items():
            changed = copy.deepcopy(config)
            changed["resolved_models"][alias][key] = value
            manifest = build_manifest(changed, agentdojo_source=FakeAgentDojoSource())
            with self.subTest(key=key):
                self.assertNotEqual(
                    {session.session_id for session in manifest.sessions},
                    baseline_ids,
                )

    def test_dry_run_never_freezes_scenarios(self) -> None:
        source = FakeAgentDojoSource()
        summary = dry_run_manifest_summary(_config("exp1_full"), agentdojo_source=source)
        self.assertEqual(summary.scenario_count, 320)
        self.assertEqual(summary.paired_session_count, 2_016)
        self.assertEqual(summary.trajectory_count, 4_032)
        self.assertEqual(source.freeze_calls, [])

    def test_ministral_full_extension_has_one_complete_actor_matrix(self) -> None:
        source = FakeAgentDojoSource()
        summary = dry_run_manifest_summary(
            _config("exp1_ministral3_14b_full_v1"), agentdojo_source=source
        )

        self.assertEqual(summary.scenario_count, 320)
        self.assertEqual(summary.paired_session_count, 672)
        self.assertEqual(summary.trajectory_count, 1_344)
        self.assertEqual(
            summary.sessions_by_actor,
            {"mistralai/Ministral-3-14B-Instruct-2512-BF16": 672},
        )
        self.assertEqual(
            summary.sessions_by_guard_pair,
            {
                "same_guard_strict": 32,
                "strict_degraded_025": 80,
                "strict_degraded_050": 80,
                "strict_degraded_075": 80,
                "strict_degraded_100": 80,
                "strict_none": 320,
            },
        )
        self.assertEqual(source.freeze_calls, [])

    def test_glm4_outcome_blind_smoke_has_eight_unique_pairs(self) -> None:
        source = FakeAgentDojoSource()
        summary = dry_run_manifest_summary(
            _config("exp1_glm4_32b_smoke_v1"), agentdojo_source=source
        )

        self.assertEqual(summary.scenario_count, 8)
        self.assertEqual(summary.paired_session_count, 8)
        self.assertEqual(summary.trajectory_count, 16)
        self.assertEqual(
            summary.sessions_by_actor,
            {"zai-org/GLM-4-32B-0414": 8},
        )
        self.assertEqual(summary.sessions_by_guard_pair, {"strict_none": 8})
        self.assertEqual(source.freeze_calls, [])

    def test_qwen3_14b_fallback_smoke_has_eight_unique_pairs(self) -> None:
        source = FakeAgentDojoSource()
        summary = dry_run_manifest_summary(
            _config("exp1_qwen3_14b_smoke_v1"), agentdojo_source=source
        )

        self.assertEqual(summary.scenario_count, 8)
        self.assertEqual(summary.paired_session_count, 8)
        self.assertEqual(summary.trajectory_count, 16)
        self.assertEqual(
            summary.sessions_by_actor,
            {"Qwen/Qwen3-14B": 8},
        )
        self.assertEqual(summary.sessions_by_guard_pair, {"strict_none": 8})
        self.assertEqual(source.freeze_calls, [])

    def test_granite41_30b_smoke_has_eight_unique_pairs(self) -> None:
        source = FakeAgentDojoSource()
        summary = dry_run_manifest_summary(
            _config("exp1_granite41_30b_smoke_v1"), agentdojo_source=source
        )

        self.assertEqual(summary.scenario_count, 8)
        self.assertEqual(summary.paired_session_count, 8)
        self.assertEqual(summary.trajectory_count, 16)
        self.assertEqual(
            summary.sessions_by_actor,
            {"ibm-granite/granite-4.1-30b": 8},
        )
        self.assertEqual(summary.sessions_by_guard_pair, {"strict_none": 8})
        self.assertEqual(source.freeze_calls, [])

    def test_granite41_30b_operational_v2_has_40_pairs_and_retains_v1(self) -> None:
        source = FakeAgentDojoSource()
        config = _config("exp1_granite41_30b_operational_v2")
        summary = dry_run_manifest_summary(config, agentdojo_source=source)

        self.assertEqual(summary.scenario_count, 40)
        self.assertEqual(summary.paired_session_count, 40)
        self.assertEqual(summary.trajectory_count, 80)
        self.assertEqual(summary.scenarios_by_layer, {"agentdojo": 20, "control": 20})
        self.assertEqual(
            summary.scenarios_by_suite_or_domain,
            {
                "banking": 5,
                "communication": 5,
                "filesystem_devops": 5,
                "payments": 5,
                "slack": 5,
                "travel": 5,
                "travel_calendar": 5,
                "workspace": 5,
            },
        )
        self.assertEqual(
            summary.sessions_by_actor,
            {"ibm-granite/granite-4.1-30b": 40},
        )
        self.assertEqual(summary.sessions_by_guard_pair, {"strict_none": 40})
        self.assertEqual(source.freeze_calls, [])

        v1 = build_manifest(
            _config("exp1_granite41_30b_smoke_v1"), agentdojo_source=FakeAgentDojoSource()
        )
        v2 = build_manifest(config, agentdojo_source=FakeAgentDojoSource())
        self.assertLessEqual(
            {scenario.scenario_id for scenario in v1.scenarios},
            {scenario.scenario_id for scenario in v2.scenarios},
        )
        self.assertLessEqual(
            {session.session_id for session in v1.sessions},
            {session.session_id for session in v2.sessions},
        )

    def test_pilot_and_smoke_config_counts_are_generic(self) -> None:
        pilot_source = FakeAgentDojoSource()
        pilot = build_manifest(_config("exp1_pilot"), agentdojo_source=pilot_source)
        self.assertEqual(pilot.summary.scenario_count, 32)
        self.assertEqual(pilot.summary.paired_session_count, 40)
        self.assertEqual(pilot.summary.trajectory_count, 80)
        self.assertEqual(
            pilot.summary.scenarios_by_split,
            {"confirmatory": 16, "development": 8, "smoke": 8},
        )

        smoke_source = FakeAgentDojoSource()
        smoke = build_manifest(_config("exp1_smoke"), agentdojo_source=smoke_source)
        self.assertEqual(smoke.summary.scenario_count, 8)
        self.assertEqual(smoke.summary.paired_session_count, 8)
        self.assertEqual(smoke.summary.trajectory_count, 16)
        self.assertEqual(
            smoke.summary.scenarios_by_split,
            {"confirmatory": 0, "development": 0, "smoke": 8},
        )
        self.assertTrue(all(call[3] == "tool_knowledge" for call in smoke_source.freeze_calls))

        gemma_source = FakeAgentDojoSource()
        gemma_smoke = build_manifest(_config("exp1_gemma_smoke_v2"), agentdojo_source=gemma_source)
        self.assertEqual(gemma_smoke.summary.scenario_count, 8)
        self.assertEqual(gemma_smoke.summary.paired_session_count, 8)
        self.assertEqual(gemma_smoke.summary.trajectory_count, 16)
        self.assertEqual(
            gemma_smoke.summary.sessions_by_actor,
            {"google/gemma-3-12b-it": 8},
        )
        self.assertTrue(
            all(session.split is ScenarioSplit.SMOKE for session in gemma_smoke.sessions)
        )

        corrected_gemma_smoke = build_manifest(
            _config("exp1_gemma_smoke_v3"), agentdojo_source=FakeAgentDojoSource()
        )
        self.assertEqual(corrected_gemma_smoke.summary, gemma_smoke.summary)

    def test_missing_adapter_runtime_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(ManifestError, "indispensable fields"):
            build_manifest(
                _config("exp1_smoke"),
                agentdojo_source=FakeAgentDojoSource(omit_runtime_fields=True),
            )

    def test_capacity_shortage_never_duplicates_task_pairs(self) -> None:
        source = FakeAgentDojoSource()
        source._catalogs["workspace"] = AgentDojoCatalog(
            suite_name="workspace",
            user_task_ids=("only-user",),
            injection_task_ids=("only-injection",),
        )
        with self.assertRaisesRegex(ManifestError, "too few"):
            dry_run_manifest_summary(_config("exp1_full"), agentdojo_source=source)


if __name__ == "__main__":
    unittest.main()
