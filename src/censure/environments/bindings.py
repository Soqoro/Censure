"""Execution bindings for the declarative controlled layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from censure.environments.control import (
    CONTROL_SCENARIO_VERSION_V2,
    ControlAttemptEvaluator,
    ControlEnvironment,
    ControlScenarioSpec,
    ControlTerminalValidator,
)
from censure.execution import RuntimeBindings, TerminalEvaluation
from censure.schemas import StateSnapshot, ToolCall
from censure.serialization import canonical_json, canonical_sha256


def _actor_visible_context(spec: ControlScenarioSpec) -> tuple[str, ...]:
    """Render frozen external data separately from the genuine user request."""

    if spec.scenario_version != CONTROL_SCENARIO_VERSION_V2:
        return ()
    rendered: list[str] = []
    for payload in spec.untrusted_content:
        rendered.append(
            "<external_context "
            f"location={canonical_json(payload.get('location'))} "
            f"source_trust={canonical_json(payload.get('source_trust'))}>\n"
            f"{payload.get('content', '')}\n"
            "</external_context>"
        )
    return tuple(rendered)


def make_control_bindings(spec: ControlScenarioSpec) -> RuntimeBindings:
    environment = ControlEnvironment(spec)
    attempt = ControlAttemptEvaluator()
    terminal = ControlTerminalValidator()
    actor_context = _actor_visible_context(spec)

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
        actor_visible_context=actor_context,
        runtime_metadata={
            "control_scenario_version": spec.scenario_version,
            "control_spec_sha256": spec.spec_sha256,
            "actor_visible_context_sha256": canonical_sha256(actor_context),
            "domain": spec.domain,
            "stratum": spec.stratum,
            "seed": spec.seed,
        },
    )
