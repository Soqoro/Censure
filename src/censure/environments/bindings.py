"""Execution bindings for the declarative controlled layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from censure.environments.control import (
    ControlAttemptEvaluator,
    ControlEnvironment,
    ControlScenarioSpec,
    ControlTerminalValidator,
)
from censure.execution import RuntimeBindings, TerminalEvaluation
from censure.schemas import StateSnapshot, ToolCall


def make_control_bindings(spec: ControlScenarioSpec) -> RuntimeBindings:
    environment = ControlEnvironment(spec)
    attempt = ControlAttemptEvaluator()
    terminal = ControlTerminalValidator()

    def attempt_evaluator(call: ToolCall, state: Mapping[str, Any]) -> bool:
        return attempt.evaluate(spec, call, state).value

    def terminal_validator(
        _answer: str,
        _initial: StateSnapshot,
        current: Any,
        _calls: Sequence[ToolCall],
    ) -> TerminalEvaluation:
        if not isinstance(current, ControlEnvironment):
            raise TypeError("control validator received a foreign environment")
        outcome = terminal.evaluate(spec, current.snapshot())
        return TerminalEvaluation(
            attack_goal_achieved=outcome.terminal_harm,
            terminal_harm=outcome.terminal_harm,
            user_utility=outcome.user_utility,
            validation_inputs={
                "state_sha256": outcome.state_sha256,
                "utility_predicate_sha256": outcome.utility_predicate_sha256,
                "harm_predicate_sha256": outcome.harm_predicate_sha256,
            },
        )

    return RuntimeBindings(
        environment=environment,
        initial_snapshot=environment.snapshot(),
        user_request=spec.user_request,
        policy=spec.authorization_policy,
        attempt_evaluator=attempt_evaluator,
        terminal_validator=terminal_validator,
        runtime_metadata={
            "control_scenario_version": spec.scenario_version,
            "control_spec_sha256": spec.spec_sha256,
            "domain": spec.domain,
            "stratum": spec.stratum,
            "seed": spec.seed,
        },
    )
