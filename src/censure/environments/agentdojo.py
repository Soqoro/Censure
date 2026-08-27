"""StatefulEnvironment facade over the pinned AgentDojo adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from censure.adapters.agentdojo_v0135 import (
    AgentDojoCallExecution,
    AgentDojoEvaluationError,
    AgentDojoStateSnapshot,
    AgentDojoV0135Adapter,
    FrozenAgentDojoScenario,
)
from censure.environments.base import (
    InvalidToolCallError,
    SnapshotRestoreError,
    ToolDefinition,
    ToolExecutionResult,
)
from censure.execution import RuntimeBindings, TerminalEvaluation
from censure.schemas import ObservationSource, StateSnapshot, ToolCall, ToolObservation
from censure.serialization import canonical_json, make_state_snapshot, verify_state_snapshot


class AgentDojoEnvironment:
    """Rebuild suite runtime/tools while storing only canonical environment state."""

    def __init__(
        self,
        frozen: FrozenAgentDojoScenario,
        *,
        adapter: AgentDojoV0135Adapter | None = None,
    ) -> None:
        self.frozen = frozen
        self.adapter = adapter or AgentDojoV0135Adapter()
        self._environment = self.adapter.restore_environment(frozen.initial_state)
        expected_tools = self.adapter.available_tool_schemas(frozen.suite_name)
        if frozen.available_tools != expected_tools:
            raise SnapshotRestoreError("frozen AgentDojo tool schemas differ from the pinned suite")
        expected_policy = self.adapter.build_authorization_policy(
            frozen.suite_name,
            frozen.user_task_id,
            self._environment,
        )
        if frozen.authorization_policy != expected_policy:
            raise SnapshotRestoreError(
                "frozen AgentDojo policy differs from user-task ground truth"
            )
        self._runtime = self.adapter.make_runtime(frozen.suite_name)
        self.call_records: list[AgentDojoCallExecution] = []
        self._tools = tuple(
            ToolDefinition(tool.name, tool.description, tool.parameters) for tool in expected_tools
        )

    @property
    def environment_id(self) -> str:
        return f"agentdojo:{self.frozen.suite_name}:{self.frozen.user_task_id}"

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        return self._tools

    @property
    def raw_environment(self) -> Any:
        return self._environment

    def canonical_state(self) -> dict[str, Any]:
        snapshot = self.adapter.snapshot_environment(self.frozen.suite_name, self._environment)
        if not isinstance(snapshot.state, dict):
            raise SnapshotRestoreError("AgentDojo environment did not serialize to an object")
        return snapshot.state

    def snapshot(self) -> StateSnapshot:
        return make_state_snapshot(self.canonical_state())

    def restore(self, snapshot: StateSnapshot) -> None:
        if not verify_state_snapshot(snapshot) or not isinstance(snapshot.state, dict):
            raise SnapshotRestoreError("AgentDojo snapshot is invalid")
        raw = AgentDojoStateSnapshot(
            suite_name=self.frozen.suite_name,
            environment_type=self.frozen.initial_state.environment_type,
            state=snapshot.state,
            sha256=snapshot.sha256,
        )
        try:
            restored_environment = self.adapter.restore_environment(raw)
            restored_runtime = self.adapter.make_runtime(self.frozen.suite_name)
        except Exception as exc:
            raise SnapshotRestoreError(str(exc)) from exc
        # Commit only after both reconstruction steps succeed.  A failed
        # restore cannot leave a new state paired with an old runtime.
        self._environment = restored_environment
        self._runtime = restored_runtime
        self.call_records.clear()

    def execute(self, proposed_call: ToolCall) -> ToolExecutionResult:
        before = self.snapshot().sha256
        execution = self.adapter.execute_tool_call(
            self.frozen.suite_name,
            self._environment,
            function=proposed_call.name,
            arguments=proposed_call.arguments,
            call_id=proposed_call.id,
            index=proposed_call.index,
            runtime=self._runtime,
        )
        self.call_records.append(execution.record)
        if execution.record.succeeded:
            content = canonical_json({"ok": True, "result": execution.result})
        else:
            content = canonical_json(
                {
                    "ok": False,
                    "error_type": execution.record.error_type,
                    "error": execution.record.error_message,
                }
            )
        observation = ToolObservation(
            tool_call_id=proposed_call.id,
            content=content,
            source=ObservationSource.TOOL,
            is_error=not execution.record.succeeded,
        )
        return ToolExecutionResult(
            tool_call_id=proposed_call.id,
            tool_name=proposed_call.name,
            ok=execution.record.succeeded,
            output={"result": content},
            state_changed=before != self.snapshot().sha256,
            observation=observation,
        )


class AgentDojoAttemptEvaluator:
    """Hidden attempt evaluator; its output is never passed into a guard."""

    def __init__(self, frozen: FrozenAgentDojoScenario, adapter: AgentDojoV0135Adapter) -> None:
        self._frozen = frozen
        suite = adapter.load_suite(frozen.suite_name)
        self._injection_task = (
            None
            if frozen.injection_task_id is None
            else suite.get_injection_task_by_id(frozen.injection_task_id)
        )
        if self._injection_task is None:
            self._ground_truth: tuple[Any, ...] = ()
        else:
            pre = adapter.restore_environment(frozen.initial_state)
            self._ground_truth = tuple(self._injection_task.ground_truth(pre))

    def __call__(self, proposed_call: ToolCall, _state: Mapping[str, Any]) -> bool:
        if self._injection_task is None:
            return False
        # Exact state-aware ground truth is the conservative, reproducible
        # pre-enforcement attempt definition for this phase.
        for expected in self._ground_truth:
            if expected.function != proposed_call.name:
                continue
            expected_args = dict(expected.args)
            if all(
                key in proposed_call.arguments
                and canonical_json(proposed_call.arguments[key]) == canonical_json(value)
                for key, value in expected_args.items()
            ):
                return True
        return False


def _validate_successful_call_alignment(
    environment: AgentDojoEnvironment,
    executed_calls: Sequence[ToolCall],
) -> tuple[AgentDojoCallExecution, ...]:
    """Prove that terminal traces contain exactly successful mutations."""

    failures = [
        record for record in environment.call_records if record.executed and not record.succeeded
    ]
    if failures:
        first = failures[0]
        raise InvalidToolCallError(
            f"AgentDojo call {first.index} ({first.function}) failed: "
            f"{first.error_type or 'ToolExecutionError'}: {first.error_message}"
        )
    successful = tuple(
        record
        for record in environment.call_records
        if record.executed
        and record.succeeded
        and record.error_type is None
        and record.error_message is None
    )
    if len(successful) != len(executed_calls):
        raise AgentDojoEvaluationError(
            "runner executed-call ledger differs from AgentDojo successful-call ledger"
        )
    for record, call in zip(successful, executed_calls, strict=True):
        if (
            record.call_id != call.id
            or record.function != call.name
            or record.index != call.index
            or canonical_json(record.arguments) != canonical_json(call.arguments)
        ):
            raise AgentDojoEvaluationError(
                "runner executed-call ledger does not match AgentDojo call provenance"
            )
    return successful


def make_agentdojo_bindings(
    frozen: FrozenAgentDojoScenario,
    *,
    adapter: AgentDojoV0135Adapter | None = None,
) -> RuntimeBindings:
    """Create fresh state, runtime, independent attempt evaluator, and validators."""

    active_adapter = adapter or AgentDojoV0135Adapter()
    environment = AgentDojoEnvironment(frozen, adapter=active_adapter)
    initial = environment.snapshot()
    if initial.sha256 != frozen.initial_state.sha256:
        raise SnapshotRestoreError("generic and AgentDojo initial-state hashes differ")
    suite = active_adapter.load_suite(frozen.suite_name)
    user_task = suite.get_user_task_by_id(frozen.user_task_id)
    injection_task = (
        None
        if frozen.injection_task_id is None
        else suite.get_injection_task_by_id(frozen.injection_task_id)
    )
    attempt_evaluator = AgentDojoAttemptEvaluator(frozen, active_adapter)

    def terminal_validator(
        model_output: str,
        frozen_initial: StateSnapshot,
        current: Any,
        executed_calls: Sequence[ToolCall],
    ) -> TerminalEvaluation:
        if not isinstance(current, AgentDojoEnvironment):
            raise TypeError("AgentDojo terminal validator received a foreign environment")
        if current.frozen != frozen:
            raise AgentDojoEvaluationError(
                "AgentDojo terminal validator received a different frozen scenario"
            )
        if frozen_initial.sha256 != initial.sha256:
            raise AgentDojoEvaluationError(
                "AgentDojo terminal validator received a different initial checkpoint"
            )
        successful_records = _validate_successful_call_alignment(current, executed_calls)
        live_state_before = current.snapshot().sha256

        # Each released evaluator gets independent state objects.  This makes
        # accidental evaluator mutation unable to affect the other label or the
        # live environment used to compute the trajectory's final-state hash.
        utility = active_adapter.evaluate_utility(
            user_task,
            model_output=model_output,
            pre_environment=active_adapter.restore_environment(frozen.initial_state),
            post_environment=current.raw_environment.model_copy(deep=True),
            call_records=successful_records,
        )
        security = active_adapter.evaluate_security(
            injection_task,
            model_output=model_output,
            pre_environment=active_adapter.restore_environment(frozen.initial_state),
            post_environment=current.raw_environment.model_copy(deep=True),
            call_records=successful_records,
        )
        if current.snapshot().sha256 != live_state_before:
            raise AgentDojoEvaluationError("terminal evaluation mutated the live environment")
        if utility.user_utility is None or security.terminal_harm is None:
            raise AgentDojoEvaluationError(
                "terminal evaluator produced an invalid outcome: "
                f"utility={utility.run_status.value}, security={security.run_status.value}"
            )
        return TerminalEvaluation(
            attack_goal_achieved=bool(security.attack_goal_achieved),
            terminal_harm=bool(security.terminal_harm),
            user_utility=bool(utility.user_utility),
            validation_inputs={
                "utility": utility.model_dump(mode="json"),
                "security": security.model_dump(mode="json"),
                "executed_successful_calls": len(successful_records),
                "executed_successful_call_ids": [record.call_id for record in successful_records],
            },
        )

    return RuntimeBindings(
        environment=environment,
        initial_snapshot=initial,
        user_request=frozen.user_request,
        policy=frozen.authorization_policy,
        attempt_evaluator=attempt_evaluator,
        terminal_validator=terminal_validator,
        runtime_metadata={
            "adapter_version": frozen.adapter_version,
            "agentdojo_package_version": frozen.package_version,
            "agentdojo_benchmark_version": frozen.benchmark_version,
            "suite_benchmark_version": list(frozen.suite_benchmark_version),
            "rendered_attack_sha256": frozen.rendered_attack_sha256,
        },
    )
