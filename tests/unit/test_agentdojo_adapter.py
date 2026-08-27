from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from censure.adapters.agentdojo_v0135 import (
    ADAPTER_LIMITATIONS,
    AGENTDOJO_BENCHMARK_VERSION,
    AGENTDOJO_PACKAGE_VERSION,
    AGENTDOJO_SOURCE_REVISION,
    AGENTDOJO_SUITES,
    AgentDojoCallExecution,
    AgentDojoRestoreError,
    AgentDojoV0135Adapter,
    build_authorization_policy,
    compatibility_report,
    evaluate_security,
    evaluate_utility,
    execute_tool_call,
    executed_successful_function_calls,
    normalize_security,
    restore_environment,
    snapshot_environment,
)
from censure.guards import make_guard
from censure.schemas import GuardAction, RuleEffect, RunStatus, ToolCall
from censure.serialization import canonical_sha256


@pytest.fixture(scope="module")
def adapter() -> Iterator[AgentDojoV0135Adapter]:
    try:
        yield AgentDojoV0135Adapter()
    except Exception as exc:
        pytest.skip(f"exact optional AgentDojo runtime is unavailable: {exc}")


def test_security_polarity_normalization_is_explicit_and_fail_closed() -> None:
    positive = normalize_security(
        injection_task_id="injection_task_0",
        raw_security=True,
        raw_security_source="security",
    )
    negative = normalize_security(
        injection_task_id="injection_task_0",
        raw_security=False,
        raw_security_source="security",
    )
    clean_sentinel = normalize_security(injection_task_id=None, raw_security=True)
    malformed = normalize_security(injection_task_id="injection_task_0", raw_security=1)
    missing = normalize_security(injection_task_id="injection_task_0")

    assert positive.attack_goal_achieved is True
    assert positive.terminal_harm is True
    assert negative.attack_goal_achieved is False
    assert negative.terminal_harm is False

    # AgentDojo returns security=True as a clean-run sentinel.  It is retained
    # raw but is never reinterpreted as an attacker success.
    assert clean_sentinel.raw_security is True
    assert clean_sentinel.raw_security_source == "task_suite_clean_sentinel"
    assert clean_sentinel.attack_goal_achieved is False
    assert clean_sentinel.terminal_harm is False

    for invalid in (malformed, missing):
        assert invalid.run_status is RunStatus.VALIDATOR_ERROR
        assert invalid.attack_goal_achieved is None
        assert invalid.terminal_harm is None
    assert malformed.raw_security_repr == "1"


def test_existing_error_status_never_acquires_a_safe_label() -> None:
    outcome = normalize_security(
        injection_task_id="injection_task_0",
        run_status=RunStatus.TIMEOUT,
    )
    assert outcome.run_status is RunStatus.TIMEOUT
    assert outcome.attack_goal_achieved is None
    assert outcome.terminal_harm is None


def test_exact_release_metadata_and_all_four_suite_counts(
    adapter: AgentDojoV0135Adapter,
) -> None:
    report = compatibility_report()
    assert report.package_version == AGENTDOJO_PACKAGE_VERSION
    assert report.benchmark_version == AGENTDOJO_BENCHMARK_VERSION
    assert report.expected_source_revision == AGENTDOJO_SOURCE_REVISION
    assert report.archive_bytes_verified is False
    assert tuple(suite.name for suite in report.suites) == AGENTDOJO_SUITES
    assert {
        suite.name: (
            suite.benchmark_version,
            len(suite.user_task_ids),
            len(suite.injection_task_ids),
            len(suite.tool_names),
            len(suite.injection_vector_ids),
        )
        for suite in report.suites
    } == {
        "workspace": ((1, 2, 2), 40, 14, 24, 16),
        "slack": ((1, 2, 0), 21, 5, 11, 6),
        "travel": ((1, 2, 0), 20, 7, 28, 13),
        "banking": ((1, 2, 2), 16, 9, 11, 4),
    }
    assert report.limitations == ADAPTER_LIMITATIONS
    assert adapter.compatibility_report() == report


@pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
def test_environment_snapshot_round_trip_is_canonical_and_independent(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    suite = adapter.load_suite(suite_name)
    original = suite.load_and_inject_default_environment({})
    snapshot = snapshot_environment(suite_name, original)
    first = restore_environment(snapshot)
    second = restore_environment(snapshot.model_dump(mode="json"))

    assert first == original
    assert second == original
    assert first is not original
    assert second is not first
    assert snapshot.sha256 == canonical_sha256(snapshot.state)
    assert snapshot_environment(suite_name, first).sha256 == snapshot.sha256


def test_tampered_state_snapshot_fails_before_restore(
    adapter: AgentDojoV0135Adapter,
) -> None:
    suite = adapter.load_suite("banking")
    snapshot = snapshot_environment("banking", suite.load_and_inject_default_environment({}))
    tampered = snapshot.model_copy(update={"state": {"not": "the frozen state"}})
    with pytest.raises(AgentDojoRestoreError, match="hash"):
        restore_environment(tampered)


@pytest.mark.parametrize("suite_name", ("workspace", "travel"))
def test_mutated_validator_reset_collections_round_trip_exactly(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    suite = adapter.load_suite(suite_name)
    environment = suite.load_and_inject_default_environment({})
    runtime = adapter.make_runtime(suite_name)
    first_email_id: str | None = None
    calls: list[tuple[str, dict[str, Any]]] = [
        (
            "send_email",
            {
                "recipients": ["checkpoint@example.com"],
                "subject": "Checkpoint",
                "body": "Persist this sent message.",
            },
        ),
        (
            "create_calendar_event",
            {
                "title": "Checkpoint event",
                "start_time": "2025-01-20 10:00",
                "end_time": "2025-01-20 11:00",
                "description": "Persist this created event.",
                "participants": ["checkpoint@example.com"],
            },
        ),
    ]
    if suite_name == "workspace":
        first_email_id = next(iter(environment.inbox.emails))
        calls.extend(
            (
                ("delete_email", {"email_id": first_email_id}),
                (
                    "create_file",
                    {
                        "filename": "checkpoint.txt",
                        "content": "Persist this created file.",
                    },
                ),
                ("delete_file", {"file_id": "13"}),
            )
        )

    for index, (function, arguments) in enumerate(calls):
        execution = execute_tool_call(
            suite_name,
            environment,
            function=function,
            arguments=arguments,
            call_id=f"mutation-{index}",
            index=index,
            runtime=runtime,
        )
        assert execution.record.succeeded, execution.record.error_message

    snapshot = snapshot_environment(suite_name, environment)
    restored = restore_environment(snapshot)
    assert snapshot_environment(suite_name, restored) == snapshot
    assert any(email.subject == "Checkpoint" for email in restored.inbox.emails.values())
    assert any(event.title == "Checkpoint event" for event in restored.calendar.events.values())

    # Re-establishing the released aliases preserves later in-place mutations:
    # unchanged live objects are the corresponding initial objects.
    initial_email_by_id = {email.id_: email for email in restored.inbox.initial_emails}
    assert all(
        email is initial_email_by_id[email_id]
        for email_id, email in restored.inbox.emails.items()
        if email_id in initial_email_by_id
    )
    initial_event_by_id = {event.id_: event for event in restored.calendar.initial_events}
    assert all(
        event is initial_event_by_id[event_id]
        for event_id, event in restored.calendar.events.items()
        if event_id in initial_event_by_id
    )
    if suite_name == "workspace":
        assert first_email_id is not None
        assert "13" not in restored.cloud_drive.files
        assert any(
            file.filename == "checkpoint.txt" for file in restored.cloud_drive.files.values()
        )
        assert first_email_id not in restored.inbox.emails
        assert first_email_id in restored.inbox.trash
        assert restored.inbox.trash[first_email_id] is initial_email_by_id[first_email_id]
        initial_file_by_id = {file.id_: file for file in restored.cloud_drive.initial_files}
        assert all(
            file is initial_file_by_id[file_id]
            for file_id, file in restored.cloud_drive.files.items()
            if file_id in initial_file_by_id
        )


@pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
def test_direct_attack_freeze_preserves_payload_locations_and_state(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    metadata_for_suite = next(
        suite for suite in adapter.compatibility_report().suites if suite.name == suite_name
    )
    frozen = adapter.freeze_scenario(
        suite_name,
        "user_task_0",
        metadata_for_suite.injection_task_ids[0],
        attack_name="direct",
    )
    assert frozen.attack_name == "direct"
    assert frozen.rendered_injections
    assert frozen.injection_locations == tuple(frozen.rendered_injections)
    assert tuple(item.injection_location for item in frozen.injection_projections) == (
        frozen.injection_locations
    )
    assert frozen.rendered_attack_sha256 == canonical_sha256(frozen.rendered_injections)
    assert len(frozen.available_tools) == len(metadata_for_suite.tool_names)
    assert tuple(tool.name for tool in frozen.available_tools) == metadata_for_suite.tool_names
    assert all(tool.parameters.get("type") == "object" for tool in frozen.available_tools)
    assert restore_environment(frozen.initial_state) is not None


def test_clean_freeze_contains_no_attack_sentinel(
    adapter: AgentDojoV0135Adapter,
) -> None:
    frozen = adapter.freeze_scenario("workspace", "user_task_0")
    assert frozen.injection_task_id is None
    assert frozen.attack_name is None
    assert frozen.attack_target_pipeline_name is None
    assert frozen.rendered_injections == {}
    assert frozen.injection_projections == ()
    assert frozen.rendered_attack_sha256 is None


@pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
def test_tool_knowledge_freeze_uses_deterministic_local_identity_without_query(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    metadata_for_suite = next(
        suite for suite in adapter.compatibility_report().suites if suite.name == suite_name
    )
    frozen = adapter.freeze_scenario(
        suite_name,
        "user_task_0",
        metadata_for_suite.injection_task_ids[0],
        attack_name="tool_knowledge",
    )
    assert frozen.attack_name == "tool_knowledge"
    assert frozen.attack_target_pipeline_name == "local"
    assert any("Local model" in payload for payload in frozen.rendered_injections.values())
    assert frozen.rendered_attack_sha256 == canonical_sha256(frozen.rendered_injections)
    assert tuple(item.injection_location for item in frozen.injection_projections) == (
        frozen.injection_locations
    )
    for projection in frozen.injection_projections:
        payload = frozen.rendered_injections[projection.injection_location]
        normalized_payload = " ".join(payload.split())
        assert projection.rendered_payload_sha256 == canonical_sha256(payload)
        assert projection.normalized_payload_sha256 == canonical_sha256(normalized_payload)
        for match in projection.state_matches:
            state_value: Any = frozen.initial_state.state
            for path_part in match.state_path:
                state_value = state_value[path_part]
            assert isinstance(state_value, str)
            assert match.state_value_sha256 == canonical_sha256(state_value)
            assert normalized_payload in " ".join(state_value.split())


@pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
def test_policy_uses_only_user_ground_truth_and_has_strict_and_weak_semantics(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    suite = adapter.load_suite(suite_name)
    environment = suite.load_and_inject_default_environment({})
    policy = build_authorization_policy(suite_name, "user_task_0", environment)
    user_task = suite.get_user_task_by_id("user_task_0")
    ground_truth = user_task.ground_truth(environment.model_copy(deep=True))

    # Every exact user-authorized call passes strict.
    for index, call in enumerate(ground_truth):
        arguments = call.model_dump(mode="json", round_trip=True)["args"]
        proposed = ToolCall(
            id=f"ground-truth-{index}",
            name=call.function,
            arguments=arguments,
            index=index,
        )
        decision = make_guard("strict").decide(
            user_request=user_task.PROMPT,
            policy=policy,
            observable_state={},
            history=(),
            proposed_call=proposed,
        )
        assert decision.action is GuardAction.ALLOW
        assert decision.policy_rule_id.startswith("allow-user-ground-truth-")

    # Every explicit unrequested mutation has a paired fallback: strict denies
    # and weak permits after ignoring only the preregistered deny rule.
    deny_rule = next(
        (rule for rule in policy.rules if rule.effect is RuleEffect.DENY),
        None,
    )
    assert deny_rule is not None
    proposed = ToolCall(id="unrequested", name=deny_rule.tool_name, arguments={}, index=0)
    strict = make_guard("strict").decide(
        user_request=user_task.PROMPT,
        policy=policy,
        observable_state={},
        history=(),
        proposed_call=proposed,
    )
    weak = make_guard("weak").decide(
        user_request=user_task.PROMPT,
        policy=policy,
        observable_state={},
        history=(),
        proposed_call=proposed,
    )
    assert strict.action is GuardAction.BLOCK
    assert strict.policy_rule_id == deny_rule.rule_id
    assert deny_rule.rule_id in policy.weak_ignored_rule_ids
    assert weak.action is GuardAction.ALLOW
    assert weak.policy_rule_id == f"allow-weak-fallback-{deny_rule.tool_name}"


@pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
def test_attack_choice_does_not_enter_user_authorization_policy(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    metadata_for_suite = next(
        suite for suite in adapter.compatibility_report().suites if suite.name == suite_name
    )
    clean = adapter.freeze_scenario(suite_name, "user_task_0")
    attacked = adapter.freeze_scenario(
        suite_name,
        "user_task_0",
        metadata_for_suite.injection_task_ids[0],
        attack_name="tool_knowledge",
    )
    assert attacked.authorization_policy == clean.authorization_policy


_POLARITY_CASES = (
    ("workspace", "injection_task_1"),
    ("slack", "injection_task_1"),
    ("travel", "injection_task_0"),
    ("banking", "injection_task_4"),
)


@pytest.mark.parametrize(("suite_name", "injection_task_id"), _POLARITY_CASES)
def test_released_ground_truth_is_positive_and_unchanged_state_is_negative(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
    injection_task_id: str,
) -> None:
    suite = adapter.load_suite(suite_name)
    injection_task = suite.get_injection_task_by_id(injection_task_id)
    pre_environment = suite.load_and_inject_default_environment({})
    post_environment = pre_environment.model_copy(deep=True)
    runtime = adapter.make_runtime(suite_name)
    records: list[AgentDojoCallExecution] = []
    for index, call in enumerate(injection_task.ground_truth(post_environment)):
        execution = execute_tool_call(
            suite_name,
            post_environment,
            function=call.function,
            arguments=call.args,
            call_id=call.id,
            index=index,
            runtime=runtime,
        )
        assert execution.record.succeeded, execution.record.error_message
        records.append(execution.record)

    # Every harmful mutation used to establish polarity must also survive a
    # durable canonical checkpoint.  Workspace's delete case is particularly
    # important because the released CloudDrive validator rebuilds ``files``
    # from ``initial_files`` unless the adapter explicitly rehydrates it.
    post_snapshot = snapshot_environment(suite_name, post_environment)
    restored_post = restore_environment(post_snapshot)
    assert snapshot_environment(suite_name, restored_post) == post_snapshot

    positive = evaluate_security(
        injection_task,
        model_output=injection_task.GROUND_TRUTH_OUTPUT,
        pre_environment=pre_environment,
        post_environment=restored_post,
        call_records=records,
    )
    negative = evaluate_security(
        injection_task,
        model_output="",
        pre_environment=pre_environment,
        post_environment=pre_environment.model_copy(deep=True),
        call_records=(),
    )
    assert positive.raw_security is True
    assert positive.attack_goal_achieved is True
    assert positive.terminal_harm is True
    assert negative.raw_security is False
    assert negative.attack_goal_achieved is False
    assert negative.terminal_harm is False


def _record_from_call(
    call: Any,
    index: int,
    *,
    executed: bool,
    succeeded: bool,
    error: str | None = None,
) -> AgentDojoCallExecution:
    return AgentDojoCallExecution(
        function=call.function,
        arguments=call.args,
        call_id=call.id,
        index=index,
        executed=executed,
        succeeded=succeeded,
        error_type="SyntheticToolError" if error is not None else None,
        error_message=error,
    )


def test_trace_security_receives_only_executed_successful_calls(
    adapter: AgentDojoV0135Adapter,
) -> None:
    suite = adapter.load_suite("slack")
    task = suite.get_injection_task_by_id("injection_task_5")
    environment = suite.load_and_inject_default_environment({})
    calls = task.ground_truth(environment)
    blocked = tuple(
        _record_from_call(call, index, executed=False, succeeded=False)
        for index, call in enumerate(calls)
    )
    successful = tuple(
        _record_from_call(call, index, executed=True, succeeded=True)
        for index, call in enumerate(calls)
    )

    assert executed_successful_function_calls(blocked) == ()
    assert len(executed_successful_function_calls(successful)) == len(calls)
    blocked_outcome = evaluate_security(
        task,
        model_output="",
        pre_environment=environment,
        post_environment=environment.model_copy(deep=True),
        call_records=blocked,
    )
    successful_outcome = evaluate_security(
        task,
        model_output="",
        pre_environment=environment,
        post_environment=environment.model_copy(deep=True),
        call_records=successful,
    )
    assert blocked_outcome.raw_security is False
    assert blocked_outcome.terminal_harm is False
    assert successful_outcome.raw_security is True
    assert successful_outcome.terminal_harm is True


def test_failed_execution_is_invalid_not_safe(
    adapter: AgentDojoV0135Adapter,
) -> None:
    suite = adapter.load_suite("workspace")
    environment = suite.load_and_inject_default_environment({})
    failed = execute_tool_call(
        "workspace",
        environment,
        function="delete_file",
        arguments={},
    ).record
    assert failed.executed is True
    assert failed.succeeded is False
    task = suite.get_injection_task_by_id("injection_task_1")
    outcome = evaluate_security(
        task,
        model_output="",
        pre_environment=environment,
        post_environment=environment,
        call_records=(failed,),
    )
    assert outcome.run_status is RunStatus.INVALID_TOOL_CALL
    assert outcome.attack_goal_achieved is None
    assert outcome.terminal_harm is None
    assert outcome.error_message is not None


class _BrokenSecurityTask:
    ID = "injection_task_broken"

    def security_from_traces(self, *args: Any) -> bool:
        raise RuntimeError("validator exploded")


def test_validator_exception_is_invalid_not_negative(
    adapter: AgentDojoV0135Adapter,
) -> None:
    suite = adapter.load_suite("banking")
    environment = suite.load_and_inject_default_environment({})
    outcome = evaluate_security(
        _BrokenSecurityTask(),
        model_output="",
        pre_environment=environment,
        post_environment=environment,
        call_records=(),
    )
    assert outcome.run_status is RunStatus.VALIDATOR_ERROR
    assert outcome.raw_security is None
    assert outcome.attack_goal_achieved is None
    assert outcome.terminal_harm is None
    assert outcome.error_type == "RuntimeError"


@pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
def test_released_user_task_ground_truth_has_positive_utility(
    adapter: AgentDojoV0135Adapter,
    suite_name: str,
) -> None:
    suite = adapter.load_suite(suite_name)
    user_task = suite.get_user_task_by_id("user_task_0")
    pre_environment = suite.load_and_inject_default_environment({})
    post_environment = pre_environment.model_copy(deep=True)
    runtime = adapter.make_runtime(suite_name)
    records = []
    for index, call in enumerate(user_task.ground_truth(post_environment)):
        execution = execute_tool_call(
            suite_name,
            post_environment,
            function=call.function,
            arguments=call.args,
            call_id=call.id,
            index=index,
            runtime=runtime,
        )
        assert execution.record.succeeded, execution.record.error_message
        records.append(execution.record)
    utility = evaluate_utility(
        user_task,
        model_output=user_task.GROUND_TRUTH_OUTPUT,
        pre_environment=pre_environment,
        post_environment=post_environment,
        call_records=records,
    )
    assert utility.run_status is RunStatus.COMPLETED
    assert utility.raw_utility is True
    assert utility.user_utility is True
