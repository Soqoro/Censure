from __future__ import annotations

import unittest
from collections import Counter

from censure.environments import (
    CONTROL_DOMAINS,
    CONTROL_STRATA,
    ControlAttemptEvaluator,
    ControlEnvironment,
    ControlHarmValidator,
    ControlTerminalValidator,
    ControlUtilityValidator,
    InvalidToolCallError,
    SnapshotRestoreError,
    generate_control_scenarios,
    get_control_scenario,
)
from censure.guards import make_guard
from censure.schemas import GuardAction, RuleEffect, ToolCall
from censure.serialization import canonical_json, make_state_snapshot


def _policy_calls(spec, effect: RuleEffect) -> tuple[ToolCall, ...]:
    calls = []
    for index, rule in enumerate(spec.authorization_policy.rules):
        is_primary_allow = effect is not RuleEffect.ALLOW or ":allow:" in rule.rule_id
        if rule.effect is effect and is_primary_allow:
            calls.append(
                ToolCall(
                    id=f"test-{effect.value}-{index}",
                    name=rule.tool_name,
                    arguments=dict(rule.argument_equals),
                    index=index,
                )
            )
    return tuple(calls)


class ControlCatalogTests(unittest.TestCase):
    def test_catalog_is_exactly_four_by_four_by_ten(self) -> None:
        specs = generate_control_scenarios()
        counts = Counter((spec.domain, spec.stratum) for spec in specs)

        self.assertEqual(len(specs), 160)
        self.assertEqual(set(counts), set((d, s) for d in CONTROL_DOMAINS for s in CONTROL_STRATA))
        self.assertTrue(all(count == 10 for count in counts.values()))
        self.assertEqual(len({spec.scenario_id for spec in specs}), 160)
        self.assertEqual(len({spec.spec_sha256 for spec in specs}), 160)

    def test_seed_only_split_is_balanced_and_frozen(self) -> None:
        counts = Counter(spec.split for spec in generate_control_scenarios())
        self.assertEqual(counts, {"smoke": 16, "development": 32, "confirmatory": 112})
        for spec in generate_control_scenarios():
            expected = (
                "smoke" if spec.seed == 0 else "development" if spec.seed <= 2 else "confirmatory"
            )
            self.assertEqual(spec.split, expected)

    def test_generation_is_deterministic_and_records_every_required_field(self) -> None:
        first = generate_control_scenarios()
        second = generate_control_scenarios()
        self.assertEqual([spec.to_dict() for spec in first], [spec.to_dict() for spec in second])

        for spec in first:
            self.assertTrue(spec.user_request)
            self.assertTrue(spec.available_tools)
            self.assertTrue(spec.authorization_policy.rules)
            self.assertTrue(
                any(rule.effect is RuleEffect.ALLOW for rule in spec.authorization_policy.rules)
            )
            self.assertTrue(
                any(rule.effect is RuleEffect.DENY for rule in spec.authorization_policy.rules)
            )
            self.assertEqual(len(spec.initial_state_sha256), 64)
            self.assertEqual(spec.canonical_initial_state["scenario_id"], spec.scenario_id)
            self.assertIn("untrusted_content", spec.canonical_initial_state["resources"])
            self.assertEqual(
                spec.utility_predicate["schema_version"], "censure-control-predicate-v1"
            )
            self.assertEqual(
                spec.terminal_harm_predicate["schema_version"],
                "censure-control-predicate-v1",
            )
            self.assertEqual(
                spec.unsafe_attempt_predicate["schema_version"],
                "censure-control-predicate-v1",
            )
            # Durable records must be canonical JSON, never opaque runtime objects.
            canonical_json(spec.to_dict())

    def test_untrusted_payloads_are_declarative_and_in_the_frozen_state(self) -> None:
        for spec in generate_control_scenarios():
            state_payloads = spec.canonical_initial_state["resources"]["untrusted_content"]
            self.assertEqual(list(spec.untrusted_content), state_payloads)
            if spec.stratum in {"untrusted_context", "multi_step"}:
                self.assertGreater(len(spec.untrusted_content), 0)
                for payload in spec.untrusted_content:
                    self.assertEqual(payload["source_trust"], "untrusted")
                    self.assertTrue(payload["location"])
                    self.assertTrue(payload["content"])
            else:
                self.assertEqual(spec.untrusted_content, ())

    def test_filtered_generation_preserves_canonical_order(self) -> None:
        selected = generate_control_scenarios(
            domains=("filesystem_devops", "payments"),
            strata=("multi_step",),
            seeds=(9, 0),
        )
        self.assertEqual(
            [spec.scenario_id for spec in selected],
            [
                "control:payments:multi_step:00",
                "control:payments:multi_step:09",
                "control:filesystem_devops:multi_step:00",
                "control:filesystem_devops:multi_step:09",
            ],
        )


class ControlEnvironmentTests(unittest.TestCase):
    def test_snapshot_round_trip_for_every_scenario(self) -> None:
        for spec in generate_control_scenarios():
            environment = ControlEnvironment(spec)
            snapshot = environment.snapshot()
            before = environment.canonical_state()
            environment.restore(snapshot)
            self.assertEqual(environment.canonical_state(), before)
            self.assertEqual(environment.state_sha256, snapshot.sha256)

    def test_restored_environments_are_independent_copies(self) -> None:
        spec = get_control_scenario("payments", "clean", 0)
        initial = ControlEnvironment(spec).snapshot()
        behavior = ControlEnvironment(spec)
        target = ControlEnvironment(spec)
        behavior.restore(initial)
        target.restore(initial)

        behavior.execute(_policy_calls(spec, RuleEffect.ALLOW)[0])
        self.assertNotEqual(behavior.state_sha256, target.state_sha256)
        self.assertEqual(target.state_sha256, initial.sha256)

    def test_canonical_state_is_detached_from_runtime_state(self) -> None:
        spec = get_control_scenario("payments", "clean", 0)
        environment = ControlEnvironment(spec)
        exported = environment.canonical_state()
        exported["resources"]["accounts"]["checking"] = -1
        self.assertEqual(environment.canonical_state()["resources"]["accounts"]["checking"], 10_000)

    def test_authorized_plans_produce_utility_in_every_domain_and_stratum(self) -> None:
        utility = ControlUtilityValidator()
        harm = ControlHarmValidator()
        for domain in CONTROL_DOMAINS:
            for stratum in CONTROL_STRATA:
                spec = get_control_scenario(domain, stratum, 0)
                environment = ControlEnvironment(spec)
                for proposed_call in _policy_calls(spec, RuleEffect.ALLOW):
                    result = environment.execute(proposed_call)
                    self.assertTrue(result.ok, (domain, stratum, result.output))
                self.assertTrue(utility.evaluate(spec, environment.snapshot()).value)
                self.assertFalse(harm.evaluate(spec, environment.snapshot()).value)

    def test_unsafe_evaluator_and_terminal_harm_are_independent_of_enforcement(self) -> None:
        attempts = ControlAttemptEvaluator()
        validator = ControlTerminalValidator()
        for domain in CONTROL_DOMAINS:
            spec = get_control_scenario(domain, "untrusted_context", 0)
            environment = ControlEnvironment(spec)
            unsafe_call = _policy_calls(spec, RuleEffect.DENY)[0]

            pre = environment.snapshot()
            self.assertTrue(attempts.evaluate(spec, unsafe_call, pre).value)
            self.assertFalse(validator.evaluate(spec, pre).terminal_harm)

            # The environment is deliberately not a guard: if middleware supplies
            # the operation, it mutates state and the independent validator sees it.
            result = environment.execute(unsafe_call)
            self.assertTrue(result.ok)
            terminal = validator.evaluate(spec, environment.snapshot())
            self.assertTrue(terminal.terminal_harm)
            self.assertFalse(terminal.user_utility)

    def test_safe_calls_are_not_marked_as_unsafe_attempts(self) -> None:
        evaluator = ControlAttemptEvaluator()
        for domain in CONTROL_DOMAINS:
            spec = get_control_scenario(domain, "clean", 0)
            call = _policy_calls(spec, RuleEffect.ALLOW)[0]
            self.assertFalse(
                evaluator.evaluate(spec, call, ControlEnvironment(spec).snapshot()).value
            )

    def test_frozen_weak_relaxation_is_behaviorally_distinct(self) -> None:
        for domain in CONTROL_DOMAINS:
            spec = get_control_scenario(domain, "untrusted_context", 0)
            unsafe_call = _policy_calls(spec, RuleEffect.DENY)[0]
            state = ControlEnvironment(spec).canonical_state()
            strict = make_guard("strict").decide(
                user_request=spec.user_request,
                policy=spec.authorization_policy,
                observable_state=state,
                history=(),
                proposed_call=unsafe_call,
            )
            weak = make_guard("weak").decide(
                user_request=spec.user_request,
                policy=spec.authorization_policy,
                observable_state=state,
                history=(),
                proposed_call=unsafe_call,
            )
            self.assertEqual(strict.action, GuardAction.BLOCK)
            self.assertEqual(weak.action, GuardAction.ALLOW)

    def test_deterministic_execution_and_observations(self) -> None:
        spec = get_control_scenario("communication", "clean", 4)
        first = ControlEnvironment(spec)
        second = ControlEnvironment(spec)
        call = _policy_calls(spec, RuleEffect.ALLOW)[0]

        first_result = first.execute(call)
        second_result = second.execute(call)
        self.assertEqual(first_result, second_result)
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(first_result.observation.tool_call_id, call.id)

    def test_multiple_calls_execute_in_original_order(self) -> None:
        spec = get_control_scenario("payments", "multi_step", 0)
        environment = ControlEnvironment(spec)
        calls = _policy_calls(spec, RuleEffect.ALLOW)
        results = environment.execute_many(calls)

        self.assertEqual([result.tool_call_id for result in results], [call.id for call in calls])
        self.assertTrue(all(result.ok for result in results))
        state = environment.canonical_state()
        self.assertLess(
            state["resources"]["verified_invoices"][0]["logical_time"],
            state["resources"]["payments"][0]["logical_time"],
        )

    def test_valid_business_error_is_deterministic_and_does_not_mutate(self) -> None:
        spec = get_control_scenario("travel_calendar", "multi_step", 0)
        environment = ControlEnvironment(spec)
        booking = _policy_calls(spec, RuleEffect.ALLOW)[1]
        before = environment.snapshot()
        result = environment.execute(booking)

        self.assertFalse(result.ok)
        self.assertFalse(result.state_changed)
        self.assertTrue(result.observation.is_error)
        self.assertEqual(environment.snapshot(), before)

    def test_unknown_and_malformed_calls_are_invalid_and_do_not_mutate(self) -> None:
        spec = get_control_scenario("payments", "clean", 0)
        environment = ControlEnvironment(spec)
        initial = environment.snapshot()

        with self.assertRaisesRegex(InvalidToolCallError, "not available"):
            environment.execute(ToolCall(id="unknown", name="unknown.tool", arguments={}))
        with self.assertRaisesRegex(InvalidToolCallError, "missing required"):
            environment.execute(
                ToolCall(
                    id="malformed",
                    name="payments.create",
                    arguments={"recipient": "vendor"},
                )
            )
        self.assertEqual(environment.snapshot(), initial)

    def test_restore_rejects_corruption_and_another_scenario(self) -> None:
        spec = get_control_scenario("payments", "clean", 0)
        environment = ControlEnvironment(spec)
        corrupt = environment.snapshot().model_copy(update={"sha256": "0" * 64})
        with self.assertRaisesRegex(SnapshotRestoreError, "hash"):
            environment.restore(corrupt)

        other = get_control_scenario("payments", "clean", 1)
        with self.assertRaisesRegex(SnapshotRestoreError, "different"):
            environment.restore(make_state_snapshot(other.canonical_initial_state))


if __name__ == "__main__":
    unittest.main()
