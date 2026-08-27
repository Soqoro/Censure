from __future__ import annotations

from typing import Any

import pytest

from censure.actors import ActorTurn, NormalizedToolCall, ScriptedActor
from censure.adapters.agentdojo_v0135 import (
    AGENTDOJO_SUITES,
    AgentDojoV0135Adapter,
)
from censure.environments.agentdojo import (
    AgentDojoEnvironment,
    make_agentdojo_bindings,
)
from censure.environments.base import InvalidToolCallError, SnapshotRestoreError
from censure.execution import PairedOracleRunner, TrajectoryRunner
from censure.guards import NoneGuard, StrictGuard
from censure.schemas import (
    EnvironmentLayer,
    RunStatus,
    ScenarioIdentity,
    ToolCall,
    TrajectoryRole,
)
from censure.serialization import make_state_snapshot


@pytest.fixture(scope="module")
def adapter() -> AgentDojoV0135Adapter:
    try:
        return AgentDojoV0135Adapter()
    except Exception as exc:
        pytest.skip(f"exact optional AgentDojo runtime is unavailable: {exc}")
        raise AssertionError("pytest.skip did not terminate the fixture") from exc


def _scenario_identity(
    *,
    suite_name: str,
    injection_task_id: str | None,
    behavior_guard_id: str = "strict",
    target_guard_id: str = "none",
) -> ScenarioIdentity:
    return ScenarioIdentity(
        environment_layer=EnvironmentLayer.AGENTDOJO,
        suite_or_domain=suite_name,
        user_task_id="user_task_0",
        injection_task_id=injection_task_id,
        rendered_attack_id=(None if injection_task_id is None else "frozen-direct"),
        actor_id="scripted",
        actor_revision="scripted-v1",
        decoding_seed=17,
        environment_seed=29,
        behavior_guard_id=behavior_guard_id,
        target_guard_id=target_guard_id,
    )


def _tool_call_from_agentdojo(call: Any, index: int, *, prefix: str = "call") -> ToolCall:
    return ToolCall(
        id=f"{prefix}-{index}",
        name=call.function,
        arguments=dict(call.args),
        index=index,
    )


@pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
def test_all_four_suites_snapshot_restore_reconstructs_runtime_and_clears_ledger(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    frozen = adapter.freeze_scenario(suite_name, "user_task_0")
    environment = AgentDojoEnvironment(frozen, adapter=adapter)
    initial = environment.snapshot()
    raw_before = environment.raw_environment
    runtime_before = environment._runtime

    suite = adapter.load_suite(suite_name)
    user_task = suite.get_user_task_by_id("user_task_0")
    ground_truth = user_task.ground_truth(environment.raw_environment.model_copy(deep=True))
    assert ground_truth
    execution = environment.execute(_tool_call_from_agentdojo(ground_truth[0], 0))
    assert execution.ok is True
    assert len(environment.call_records) == 1

    environment.restore(initial)
    assert environment.snapshot() == initial
    assert environment.snapshot().sha256 == frozen.initial_state.sha256
    assert environment.raw_environment is not raw_before
    assert environment._runtime is not runtime_before
    assert environment.call_records == []


def test_failed_restore_is_atomic(adapter: AgentDojoV0135Adapter) -> None:
    frozen = adapter.freeze_scenario("workspace", "user_task_0")
    environment = AgentDojoEnvironment(frozen, adapter=adapter)
    state_before = environment.snapshot()
    raw_before = environment.raw_environment
    runtime_before = environment._runtime

    invalid = make_state_snapshot({"not": "a workspace environment"})
    with pytest.raises(SnapshotRestoreError):
        environment.restore(invalid)

    assert environment.raw_environment is raw_before
    assert environment._runtime is runtime_before
    assert environment.snapshot() == state_before


def test_frozen_tools_and_policy_are_reverified_before_runtime_use(
    adapter: AgentDojoV0135Adapter,
) -> None:
    frozen = adapter.freeze_scenario("workspace", "user_task_0")
    missing_tool = frozen.model_copy(update={"available_tools": frozen.available_tools[:-1]})
    with pytest.raises(SnapshotRestoreError, match="tool schemas"):
        AgentDojoEnvironment(missing_tool, adapter=adapter)

    altered_policy = frozen.authorization_policy.model_copy(
        update={"default_reason": "Unfrozen replacement policy."}
    )
    wrong_policy = frozen.model_copy(update={"authorization_policy": altered_policy})
    with pytest.raises(SnapshotRestoreError, match="policy"):
        AgentDojoEnvironment(wrong_policy, adapter=adapter)


def test_tool_execution_failure_becomes_invalid_run_without_safe_labels(
    adapter: AgentDojoV0135Adapter,
) -> None:
    frozen = adapter.freeze_scenario("workspace", "user_task_0")
    bindings = make_agentdojo_bindings(frozen, adapter=adapter)
    environment = bindings.environment
    assert isinstance(environment, AgentDojoEnvironment)
    actor = ScriptedActor(
        [
            ActorTurn(
                tool_calls=[
                    NormalizedToolCall(
                        call_id="unknown-call",
                        name="not_a_released_tool",
                        arguments={},
                        index=0,
                    )
                ]
            )
        ]
    )
    result = TrajectoryRunner().run(
        scenario=_scenario_identity(suite_name="workspace", injection_task_id=None),
        role=TrajectoryRole.TARGET,
        actor=actor,
        guard=NoneGuard(),
        bindings=bindings,
    )

    assert result.status is RunStatus.INVALID_TOOL_CALL
    assert result.attack_goal_achieved is None
    assert result.terminal_harm is None
    assert result.user_utility is None
    assert result.error_type == InvalidToolCallError.__name__
    assert len(environment.call_records) == 1
    failed = environment.call_records[0]
    assert failed.executed is True
    assert failed.succeeded is False
    assert failed.error_type
    assert failed.error_message


_POLARITY_CASES = (
    ("workspace", "injection_task_1"),
    ("slack", "injection_task_1"),
    ("travel", "injection_task_0"),
    ("banking", "injection_task_4"),
)


@pytest.mark.parametrize(("suite_name", "injection_task_id"), _POLARITY_CASES)
def test_independent_bindings_have_independent_state_runtime_and_call_ledgers(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
    injection_task_id: str,
) -> None:
    frozen = adapter.freeze_scenario(
        suite_name,
        "user_task_0",
        injection_task_id,
        attack_name="direct",
    )
    first = make_agentdojo_bindings(frozen, adapter=adapter)
    second = make_agentdojo_bindings(frozen, adapter=adapter)
    first_environment = first.environment
    second_environment = second.environment
    assert isinstance(first_environment, AgentDojoEnvironment)
    assert isinstance(second_environment, AgentDojoEnvironment)
    assert first_environment is not second_environment
    assert first_environment.raw_environment is not second_environment.raw_environment
    assert first_environment._runtime is not second_environment._runtime
    assert first.initial_snapshot == second.initial_snapshot

    suite = adapter.load_suite(suite_name)
    injection_task = suite.get_injection_task_by_id(injection_task_id)
    calls = injection_task.ground_truth(first_environment.raw_environment.model_copy(deep=True))
    for index, call in enumerate(calls):
        result = first_environment.execute(
            _tool_call_from_agentdojo(call, index, prefix="injection")
        )
        assert result.ok is True

    assert first_environment.snapshot().sha256 != first.initial_snapshot.sha256
    assert second_environment.snapshot() == second.initial_snapshot
    assert len(first_environment.call_records) == len(calls)
    assert second_environment.call_records == []

    mutated = first_environment.snapshot()
    second_runtime_before_restore = second_environment._runtime
    second_environment.restore(mutated)
    assert second_environment.snapshot() == mutated
    assert second_environment._runtime is not second_runtime_before_restore
    assert second_environment.call_records == []


def test_guard_independent_attempts_and_terminal_graders_use_only_successful_executions(
    adapter: AgentDojoV0135Adapter,
) -> None:
    suite = adapter.load_suite("workspace")
    injection_task = suite.get_injection_task_by_id("injection_task_1")
    frozen = adapter.freeze_scenario(
        "workspace",
        "user_task_0",
        injection_task.ID,
        attack_name="direct",
    )
    call = injection_task.ground_truth(adapter.restore_environment(frozen.initial_state))[0]

    def actor_factory() -> ScriptedActor:
        return ScriptedActor(
            [
                ActorTurn(
                    tool_calls=[
                        NormalizedToolCall(
                            call_id="unsafe-delete",
                            name=call.function,
                            arguments=dict(call.args),
                            index=0,
                        )
                    ]
                ),
                ActorTurn(content=injection_task.GROUND_TRUTH_OUTPUT),
            ]
        )

    pair = PairedOracleRunner().run(
        scenario=_scenario_identity(
            suite_name="workspace",
            injection_task_id=injection_task.ID,
        ),
        actor_factory=actor_factory,
        bindings_factory=lambda: make_agentdojo_bindings(frozen, adapter=adapter),
        behavior_guard_factory=StrictGuard,
        target_guard_factory=NoneGuard,
    )

    assert pair.behavior.status is RunStatus.COMPLETED
    assert pair.target.status is RunStatus.COMPLETED
    assert pair.behavior.attempted_unsafe_action is True
    assert pair.target.attempted_unsafe_action is True
    assert pair.behavior.interventions[0].pre_enforcement_unsafe_attempt is True
    assert pair.target.interventions[0].pre_enforcement_unsafe_attempt is True
    assert pair.behavior.blocked_call_count == 1
    assert pair.target.blocked_call_count == 0

    # The identical proposal is an unsafe attempt under both guards, but only
    # the successful target execution enters AgentDojo's trace grader.
    assert pair.behavior.terminal_validation_inputs["executed_successful_calls"] == 0
    assert pair.behavior.terminal_validation_inputs["executed_successful_call_ids"] == []
    assert pair.target.terminal_validation_inputs["executed_successful_calls"] == 1
    assert pair.target.terminal_validation_inputs["executed_successful_call_ids"] == [
        "unsafe-delete"
    ]
    assert pair.behavior.attack_goal_achieved is False
    assert pair.behavior.terminal_harm is False
    assert pair.target.attack_goal_achieved is True
    assert pair.target.terminal_harm is True
