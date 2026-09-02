from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from censure.actors.base import ActorTurn, NormalizedToolCall, ScriptedActor
from censure.config import resolved_experiment_config
from censure.environments.bindings import make_control_bindings
from censure.environments.control import get_control_scenario
from censure.estimation.agent_cohort import (
    AgentCohortStore,
    AgentEvaluationOracle,
    extract_agent_audit_cohorts,
)
from censure.estimation.agent_live import LiveAgentSuffixOracle, SelectedSuffixRunStore
from censure.estimation.auditor import CensureAuditor
from censure.estimation.schemas import AllocationPolicyName, SuffixAuditStatus
from censure.execution import CheckpointSuffixRunner, TrajectoryRunner
from censure.guards import make_guard
from censure.manifest import AgentDojoCatalog, build_manifest
from censure.schemas import (
    EnvironmentLayer,
    PairedSession,
    RuleEffect,
    ScenarioIdentity,
    StateSnapshot,
    TrajectoryResult,
    TrajectoryRole,
)
from censure.storage import CorruptArtifactError, RunStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _EmptyAgentDojoSource:
    package_version = "0.1.35"
    benchmark_version = "v1.2.2"

    def catalog(self, suite_name: str) -> AgentDojoCatalog:
        raise AssertionError(f"unexpected AgentDojo catalog access: {suite_name}")

    def freeze_scenario(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unexpected AgentDojo scenario freeze")


def _manifest_and_spec():
    config = resolved_experiment_config(
        REPOSITORY_ROOT / "configs" / "experiments" / "exp1_smoke_v2.yaml",
        resolve_remote=False,
    )
    config = copy.deepcopy(config)
    config["experiment_id"] = "phase2-agent-unit"
    config["agentdojo"]["suites"] = []
    config["controlled"] = {
        "enabled": True,
        "scenario_version": "censure-control-scenario-v2",
        "domains": ["payments"],
        "strata": ["untrusted_context"],
        "seeds_per_cell": 1,
    }
    config["splits"] = {"smoke": 0.0, "development": 0.0, "confirmatory": 1.0}
    manifest = build_manifest(config, agentdojo_source=_EmptyAgentDojoSource())
    spec = get_control_scenario(
        "payments",
        "untrusted_context",
        0,
        scenario_version="censure-control-scenario-v2",
    )
    return manifest, spec


def _identity(session: PairedSession) -> ScenarioIdentity:
    return ScenarioIdentity(
        environment_layer=session.environment_layer,
        suite_or_domain=session.suite_or_domain,
        user_task_id=session.user_task_id,
        injection_task_id=session.injection_task_id,
        rendered_attack_id=session.rendered_attack_id,
        actor_id=session.actor_id,
        actor_revision=session.actor_revision,
        decoding_seed=session.decoding_seed,
        environment_seed=session.environment_seed,
        behavior_guard_id=session.behavior_guard_id,
        target_guard_id=session.target_guard_id,
    )


def _actor_for_rule(spec: Any, effect: RuleEffect) -> ScriptedActor:
    rule = next(
        rule
        for rule in spec.authorization_policy.rules
        if rule.effect is effect and rule.argument_equals
    )
    return ScriptedActor(
        (
            ActorTurn(
                tool_calls=[
                    NormalizedToolCall(
                        call_id="phase2-call",
                        name=rule.tool_name,
                        arguments=dict(rule.argument_equals),
                        index=0,
                    )
                ],
                model_metadata={"generated_token_count": 11},
            ),
            ActorTurn(
                content="Finished.",
                model_metadata={"generated_token_count": 7},
            ),
        ),
        actor_id="Qwen/Qwen3-8B",
        actor_revision="b968826d9c46dd6066d109eabc6255188de91218",
    )


def _actor_with_allowed_then_denied_call(spec: Any) -> ScriptedActor:
    allow = next(
        rule
        for rule in spec.authorization_policy.rules
        if rule.effect is RuleEffect.ALLOW and rule.argument_equals
    )
    deny = next(
        rule
        for rule in spec.authorization_policy.rules
        if rule.effect is RuleEffect.DENY and rule.argument_equals
    )
    return ScriptedActor(
        (
            ActorTurn(
                tool_calls=[
                    NormalizedToolCall(
                        call_id="phase2-allow",
                        name=allow.tool_name,
                        arguments=dict(allow.argument_equals),
                        index=0,
                    ),
                    NormalizedToolCall(
                        call_id="phase2-deny",
                        name=deny.tool_name,
                        arguments=dict(deny.argument_equals),
                        index=1,
                    ),
                ],
                model_metadata={"generated_token_count": 11},
            ),
            ActorTurn(
                content="Finished.",
                model_metadata={"generated_token_count": 7},
            ),
        ),
        actor_id="Qwen/Qwen3-8B",
        actor_revision="b968826d9c46dd6066d109eabc6255188de91218",
    )


def _run(
    session: PairedSession,
    spec: Any,
    *,
    role: TrajectoryRole,
    effect: RuleEffect,
) -> TrajectoryResult:
    guard_id = (
        session.behavior_guard_id if role is TrajectoryRole.BEHAVIOR else session.target_guard_id
    )
    return TrajectoryRunner(max_tool_steps=3).run(
        scenario=_identity(session),
        role=role,
        actor=_actor_for_rule(spec, effect),
        guard=make_guard(guard_id, guard_id=guard_id),
        bindings=make_control_bindings(spec),
    )


def _write(store: RunStore, session: PairedSession, result: TrajectoryResult) -> None:
    role = "behavior" if result.role is TrajectoryRole.BEHAVIOR else "target"
    store.write_trajectory(
        session_id=session.session_id,
        role=role,
        summary=result.model_dump(mode="json", exclude={"interventions"}),
        trace={
            "schema_version": "censure.trajectory-trace.v1",
            "session_id": session.session_id,
            "trajectory": result.model_dump(mode="json"),
        },
    )


def _restore_check(spec: Any):
    def check(_scenario: Any, checkpoint: StateSnapshot) -> bool:
        bindings = make_control_bindings(spec)
        bindings.environment.restore(checkpoint)
        return bindings.environment.snapshot() == checkpoint

    return check


def test_behavior_frontier_and_private_target_are_capability_separated(tmp_path: Path) -> None:
    manifest, spec = _manifest_and_spec()
    session = manifest.sessions[0]
    store = RunStore(tmp_path, manifest.experiment_id)
    behavior = _run(
        session,
        spec,
        role=TrajectoryRole.BEHAVIOR,
        effect=RuleEffect.DENY,
    )
    assert behavior.terminal_harm is False
    assert behavior.blocked_call_count == 1
    _write(store, session, behavior)

    collection = extract_agent_audit_cohorts(
        manifest,
        store,
        checkpoint_restore_check=_restore_check(spec),
        max_tool_steps=3,
    )
    cohort = collection.cohorts[0]
    assert cohort.envelope.cohort_size == 1
    assert cohort.envelope.supported_harm_contribution == 0.0
    assert cohort.envelope.target_frontier_mass == 1.0
    assert cohort.envelope.auditable_mass == 1.0
    assert len(cohort.roots) == 1
    with pytest.raises(PermissionError):
        store.read_oracle_summary(session.session_id)

    target = _run(
        session,
        spec,
        role=TrajectoryRole.TARGET,
        effect=RuleEffect.DENY,
    )
    assert target.terminal_harm is True
    _write(store, session, target)
    oracle = AgentEvaluationOracle(
        cohort=cohort,
        manifest=manifest,
        behavior_store=store,
        evaluation_store=store.evaluation_view(evaluation=True),
    )
    assert oracle.private_read_candidate_ids == []
    auditor = CensureAuditor(
        envelope=cohort.envelope,
        oracle=oracle,
        policy=AllocationPolicyName.UNIFORM,
        allocation_seed=7,
    )
    ledger, _ = auditor.run(total_rounds=1)
    disclosure = ledger.disclosures[0]
    assert disclosure.status is SuffixAuditStatus.COMPLETED
    assert disclosure.safe_value == 0.0
    assert oracle.private_read_candidate_ids == [disclosure.candidate_id]
    diagnostic = oracle.diagnostics[disclosure.candidate_id]
    assert diagnostic.root_verified
    assert diagnostic.one_step_harm is True
    assert diagnostic.full_suffix_harm is True
    assert diagnostic.one_step_safe_terminal_harm is False
    assert diagnostic.terminal_state_diverged is True

    # A duplicate draw is served from the selected-suffix cache, not a second
    # private trajectory read.
    auditor.run(total_rounds=2, ledger=ledger)
    assert oracle.private_read_candidate_ids == [disclosure.candidate_id]


def test_checkpoint_suffix_matches_full_target_potential_outcome(tmp_path: Path) -> None:
    manifest, spec = _manifest_and_spec()
    session = manifest.sessions[0]
    store = RunStore(tmp_path, manifest.experiment_id)
    behavior = _run(
        session,
        spec,
        role=TrajectoryRole.BEHAVIOR,
        effect=RuleEffect.DENY,
    )
    _write(store, session, behavior)
    cohort = extract_agent_audit_cohorts(
        manifest,
        store,
        checkpoint_restore_check=_restore_check(spec),
        max_tool_steps=3,
    ).cohorts[0]
    root = cohort.roots[0]

    full_target = _run(
        session,
        spec,
        role=TrajectoryRole.TARGET,
        effect=RuleEffect.DENY,
    )
    suffix_target = CheckpointSuffixRunner(max_tool_steps=3).run(
        scenario=_identity(session),
        actor=_actor_for_rule(spec, RuleEffect.DENY),
        guard=make_guard(session.target_guard_id, guard_id=session.target_guard_id),
        bindings=make_control_bindings(spec),
        shared_prefix=root.shared_prefix_interventions,
        root_step_index=root.step_index,
        root_tool_call_index=root.tool_call_index,
        root_pre_intervention_checkpoint=root.pre_intervention_checkpoint,
        root_actor_visible_messages=root.actor_visible_messages,
        root_proposed_call=root.proposed_call,
        root_model_metadata=root.root_model_metadata,
    )

    assert suffix_target.status == full_target.status
    assert suffix_target.terminal_harm == full_target.terminal_harm
    assert suffix_target.attack_goal_achieved == full_target.attack_goal_achieved
    assert suffix_target.user_utility == full_target.user_utility
    assert suffix_target.final_answer == full_target.final_answer
    assert suffix_target.final_state_sha256 == full_target.final_state_sha256
    assert suffix_target.proposed_call_count == full_target.proposed_call_count
    assert full_target.generation_token_count == 18
    assert suffix_target.generation_token_count == 7
    assert tuple(
        trace.guard_inputs.proposed_call for trace in suffix_target.interventions
    ) == tuple(trace.guard_inputs.proposed_call for trace in full_target.interventions)
    assert tuple(trace.post_state for trace in suffix_target.interventions) == tuple(
        trace.post_state for trace in full_target.interventions
    )


def test_live_suffix_oracle_executes_only_selected_candidate_and_resumes(
    tmp_path: Path,
) -> None:
    manifest, spec = _manifest_and_spec()
    session = manifest.sessions[0]
    behavior_store = RunStore(tmp_path, manifest.experiment_id)
    behavior = _run(
        session,
        spec,
        role=TrajectoryRole.BEHAVIOR,
        effect=RuleEffect.DENY,
    )
    _write(behavior_store, session, behavior)
    cohort = extract_agent_audit_cohorts(
        manifest,
        behavior_store,
        checkpoint_restore_check=_restore_check(spec),
        max_tool_steps=3,
    ).cohorts[0]
    root = cohort.roots[0]
    suffix_store = SelectedSuffixRunStore(tmp_path, manifest.experiment_id)
    actor = _actor_for_rule(spec, RuleEffect.DENY)
    oracle = LiveAgentSuffixOracle(
        cohort=cohort,
        manifest=manifest,
        behavior_store=behavior_store,
        suffix_store=suffix_store,
        actor=actor,
        bindings_factory=lambda _scenario: make_control_bindings(spec),
        max_tool_steps=3,
        wall_clock_seconds=30,
        retries=1,
    )

    outcome = oracle.evaluate_selected(root.candidate_id)
    assert outcome.status is SuffixAuditStatus.COMPLETED
    assert outcome.terminal_harm == 1.0
    assert outcome.suffix_tool_steps == 1
    assert outcome.generation_tokens == 7
    assert oracle.executed_candidate_ids == [root.candidate_id]
    assert not behavior_store.is_complete(session_id=session.session_id, role="target")

    resumed_actor = _actor_for_rule(spec, RuleEffect.DENY)
    resumed = LiveAgentSuffixOracle(
        cohort=cohort,
        manifest=manifest,
        behavior_store=behavior_store,
        suffix_store=suffix_store,
        actor=resumed_actor,
        bindings_factory=lambda _scenario: make_control_bindings(spec),
        max_tool_steps=3,
        wall_clock_seconds=30,
        retries=1,
    )
    assert resumed.evaluate_selected(root.candidate_id) == outcome
    assert resumed_actor.calls == []
    assert resumed.persisted_cache_candidate_ids == [root.candidate_id]


def test_checkpoint_suffix_replays_prior_call_in_same_assistant_turn(
    tmp_path: Path,
) -> None:
    manifest, spec = _manifest_and_spec()
    session = manifest.sessions[0]
    behavior = TrajectoryRunner(max_tool_steps=3).run(
        scenario=_identity(session),
        role=TrajectoryRole.BEHAVIOR,
        actor=_actor_with_allowed_then_denied_call(spec),
        guard=make_guard(
            session.behavior_guard_id,
            guard_id=session.behavior_guard_id,
        ),
        bindings=make_control_bindings(spec),
    )
    assert behavior.blocked_call_count == 1
    store = RunStore(tmp_path, manifest.experiment_id)
    _write(store, session, behavior)
    root = (
        extract_agent_audit_cohorts(
            manifest,
            store,
            checkpoint_restore_check=_restore_check(spec),
            max_tool_steps=3,
        )
        .cohorts[0]
        .roots[0]
    )
    assert len(root.shared_prefix_interventions) == 1
    assert root.step_index == 0
    assert root.tool_call_index == 1

    full_target = TrajectoryRunner(max_tool_steps=3).run(
        scenario=_identity(session),
        role=TrajectoryRole.TARGET,
        actor=_actor_with_allowed_then_denied_call(spec),
        guard=make_guard(session.target_guard_id, guard_id=session.target_guard_id),
        bindings=make_control_bindings(spec),
    )
    suffix_target = CheckpointSuffixRunner(max_tool_steps=3).run(
        scenario=_identity(session),
        actor=_actor_with_allowed_then_denied_call(spec),
        guard=make_guard(session.target_guard_id, guard_id=session.target_guard_id),
        bindings=make_control_bindings(spec),
        shared_prefix=root.shared_prefix_interventions,
        root_step_index=root.step_index,
        root_tool_call_index=root.tool_call_index,
        root_pre_intervention_checkpoint=root.pre_intervention_checkpoint,
        root_actor_visible_messages=root.actor_visible_messages,
        root_proposed_call=root.proposed_call,
        root_model_metadata=root.root_model_metadata,
    )
    assert suffix_target.status == full_target.status
    assert suffix_target.terminal_harm == full_target.terminal_harm
    assert suffix_target.final_state_sha256 == full_target.final_state_sha256
    assert suffix_target.generation_token_count == 7
    assert tuple(
        trace.guard_inputs.proposed_call for trace in suffix_target.interventions
    ) == tuple(trace.guard_inputs.proposed_call for trace in full_target.interventions)


def test_no_block_safe_unit_is_identified_without_a_suffix(tmp_path: Path) -> None:
    manifest, spec = _manifest_and_spec()
    session = manifest.sessions[0]
    store = RunStore(tmp_path, manifest.experiment_id)
    behavior = _run(
        session,
        spec,
        role=TrajectoryRole.BEHAVIOR,
        effect=RuleEffect.ALLOW,
    )
    assert behavior.terminal_harm is False
    assert behavior.blocked_call_count == 0
    _write(store, session, behavior)

    cohort = extract_agent_audit_cohorts(
        manifest,
        store,
        checkpoint_restore_check=_restore_check(spec),
    ).cohorts[0]
    assert cohort.supported_session_ids == (session.session_id,)
    assert cohort.envelope.candidates == ()
    assert cohort.envelope.theta_env == 0.0


def test_missing_behavior_and_nonrestorable_roots_remain_worst_case(tmp_path: Path) -> None:
    manifest, spec = _manifest_and_spec()
    session = manifest.sessions[0]
    store = RunStore(tmp_path, manifest.experiment_id)

    missing = extract_agent_audit_cohorts(manifest, store).cohorts[0]
    assert missing.unresolved_session_ids == (session.session_id,)
    assert missing.envelope.supported_harm_contribution == 1.0
    assert missing.envelope.theta_env == 1.0

    behavior = _run(
        session,
        spec,
        role=TrajectoryRole.BEHAVIOR,
        effect=RuleEffect.DENY,
    )
    _write(store, session, behavior)
    nonrestorable = extract_agent_audit_cohorts(
        manifest,
        store,
        checkpoint_restore_check=lambda _scenario, _checkpoint: False,
    ).cohorts[0]
    assert nonrestorable.envelope.target_frontier_mass == 1.0
    assert nonrestorable.envelope.auditable_mass == 0.0
    assert nonrestorable.envelope.nonauditable_mass == 1.0
    assert nonrestorable.envelope.theta_env == 1.0


def test_agent_cohort_store_is_checksummed_and_immutable(tmp_path: Path) -> None:
    manifest, spec = _manifest_and_spec()
    session = manifest.sessions[0]
    behavior_store = RunStore(tmp_path, manifest.experiment_id)
    _write(
        behavior_store,
        session,
        _run(
            session,
            spec,
            role=TrajectoryRole.BEHAVIOR,
            effect=RuleEffect.DENY,
        ),
    )
    collection = extract_agent_audit_cohorts(
        manifest,
        behavior_store,
        checkpoint_restore_check=_restore_check(spec),
    )
    cohort_store = AgentCohortStore(tmp_path, manifest.experiment_id)
    cohort_store.write_collection(collection)
    assert cohort_store.read_collection() == collection

    cohort_store.collection_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CorruptArtifactError, match="checksum mismatch"):
        cohort_store.read_collection()


def test_agent_cohort_extracts_only_strict_none_sessions(tmp_path: Path) -> None:
    manifest, _spec = _manifest_and_spec()
    assert all(
        session.environment_layer is EnvironmentLayer.CONTROL for session in manifest.sessions
    )
    with pytest.raises(ValueError, match="selected actors are absent"):
        extract_agent_audit_cohorts(
            manifest,
            RunStore(tmp_path, manifest.experiment_id),
            actor_ids=("not-in-manifest",),
        )
