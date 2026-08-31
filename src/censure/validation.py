"""Independent integrity validation for paired Experiment 1 trajectories.

This module is deliberately separate from execution and statistical analysis.
It validates frozen scientific identity, checkpoint provenance, and trace
alignment before producing one normalized analysis row. Invalid trajectories
remain rows with their raw status and null harm; they are never dropped or
silently converted to safe outcomes.
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from censure.execution import PairingError, find_first_divergence
from censure.manifest import derive_session_id
from censure.schemas import (
    FirstDivergence,
    FrozenScenario,
    PairedSession,
    RunStatus,
    ScenarioIdentity,
    StateSnapshot,
    TrajectoryResult,
    TrajectoryRole,
)
from censure.serialization import canonical_sha256, verify_state_snapshot

SUCCESS_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.NO_DIVERGENCE})
RestoreCheckResult = bool | str | StateSnapshot
# New callbacks receive (scenario, checkpoint) for every unique saved state.
# One-argument callbacks remain supported as legacy initial-state-only checks.
CheckpointRestoreCheck = Callable[..., RestoreCheckResult]
CHECKPOINT_FAILURE_CODES = frozenset(
    {
        "checkpoint_not_restorable",
        "checkpoint_restore_error",
        "final_checkpoint_mismatch",
        "frozen_checkpoint_corrupt",
        "initial_checkpoint_mismatch",
        "trace_checkpoint_chain_mismatch",
        "trace_checkpoint_corrupt",
    }
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable structural validation failure."""

    session_id: str
    code: str
    message: str
    remediation: str

    @property
    def actionable(self) -> str:
        return f"[{self.code}] {self.message} Action: {self.remediation}"

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
        }


class PairValidationError(ValueError):
    """Raised when one pair cannot safely enter analysis."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("PairValidationError requires at least one issue")
        super().__init__("; ".join(issue.actionable for issue in self.issues))


class ValidationAggregateError(RuntimeError):
    """Raised on request when an aggregate report contains structural errors."""


@dataclass(frozen=True, slots=True)
class PairValidationInput:
    """Frozen material and zero, one, or both persisted trajectory summaries."""

    scenario: FrozenScenario
    session: PairedSession
    behavior: TrajectoryResult | None
    oracle: TrajectoryResult | None


@dataclass(frozen=True, slots=True)
class ValidatedPair:
    """A structurally sound pair and its analysis-compatible normalized row."""

    scenario: FrozenScenario
    session: PairedSession
    behavior: TrajectoryResult
    oracle: TrajectoryResult
    alignment: Literal["diverged", "no_divergence", "invalid"]
    first_divergence: FirstDivergence | None
    normalized_row: dict[str, Any]
    checkpoint_restorable: bool
    runtime_restore_checked: bool
    unique_checkpoint_count: int


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate integrity, missingness, failure, and restorability accounting."""

    total_sessions: int
    normalized_row_count: int
    valid_pair_count: int
    invalid_pair_count: int
    missing_behavior_count: int
    missing_oracle_count: int
    invalid_behavior_count: int
    invalid_oracle_count: int
    checkpoint_restorable_count: int
    checkpoint_restore_checked_count: int
    checkpoint_restore_failure_count: int
    runtime_restore_unchecked_count: int
    saved_checkpoint_count: int
    unique_checkpoint_count: int
    diverged_pair_count: int
    no_divergence_pair_count: int
    structural_error_count: int
    normalized_rows: tuple[dict[str, Any], ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        """Whether every expected pair was present and structurally sound."""

        return not self.issues

    @property
    def actionable_errors(self) -> tuple[str, ...]:
        return tuple(issue.actionable for issue in self.issues)

    def raise_for_errors(self) -> None:
        if self.issues:
            raise ValidationAggregateError("\n".join(self.actionable_errors))

    def to_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "censure.validation-report.v1",
            "ok": self.ok,
            "total_sessions": self.total_sessions,
            "normalized_row_count": self.normalized_row_count,
            "valid_pair_count": self.valid_pair_count,
            "invalid_pair_count": self.invalid_pair_count,
            "missing_behavior_count": self.missing_behavior_count,
            "missing_oracle_count": self.missing_oracle_count,
            "invalid_behavior_count": self.invalid_behavior_count,
            "invalid_oracle_count": self.invalid_oracle_count,
            "checkpoint_restorable_count": self.checkpoint_restorable_count,
            "checkpoint_restore_checked_count": self.checkpoint_restore_checked_count,
            "checkpoint_restore_failure_count": self.checkpoint_restore_failure_count,
            "runtime_restore_unchecked_count": self.runtime_restore_unchecked_count,
            "saved_checkpoint_count": self.saved_checkpoint_count,
            "unique_checkpoint_count": self.unique_checkpoint_count,
            "diverged_pair_count": self.diverged_pair_count,
            "no_divergence_pair_count": self.no_divergence_pair_count,
            "structural_error_count": self.structural_error_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if include_rows:
            result["normalized_rows"] = list(self.normalized_rows)
        return result


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Outcome-blind execution, persistence, and restoration diagnostics.

    This report deliberately has no row-level payload and no fields derived
    from terminal labels, user utility, guard decisions, or paired trajectory
    disagreement.  It is safe to inspect while scientific outcomes remain
    blinded.
    """

    total_sessions: int
    complete_pair_count: int
    successful_pair_count: int
    invalid_pair_count: int
    structurally_valid_pair_count: int
    missing_behavior_count: int
    missing_oracle_count: int
    invalid_behavior_count: int
    invalid_oracle_count: int
    behavior_status_counts: dict[str, int]
    oracle_status_counts: dict[str, int]
    behavior_error_class_counts: dict[str, int]
    oracle_error_class_counts: dict[str, int]
    checkpoint_restorable_count: int
    checkpoint_restore_checked_count: int
    checkpoint_restore_failure_count: int
    runtime_restore_unchecked_count: int
    saved_checkpoint_count: int
    proposal_coverage: dict[str, Any]
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def actionable_errors(self) -> tuple[str, ...]:
        return tuple(issue.actionable for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return only the outcome-blind feasibility contract."""

        return {
            "schema_version": "censure.feasibility-report.v1",
            "ok": self.ok,
            "technical_run_validity": {
                "selected_pair_count": self.total_sessions,
                "complete_pair_count": self.complete_pair_count,
                "successful_pair_count": self.successful_pair_count,
                "invalid_pair_count": self.invalid_pair_count,
                "structurally_valid_pair_count": self.structurally_valid_pair_count,
                "missing_behavior_count": self.missing_behavior_count,
                "missing_oracle_count": self.missing_oracle_count,
                "invalid_behavior_count": self.invalid_behavior_count,
                "invalid_oracle_count": self.invalid_oracle_count,
                "behavior_status_counts": dict(sorted(self.behavior_status_counts.items())),
                "oracle_status_counts": dict(sorted(self.oracle_status_counts.items())),
                "structural_error_count": len(self.issues),
                "issues": [issue.to_dict() for issue in self.issues],
            },
            "error_classes": {
                "behavior": dict(sorted(self.behavior_error_class_counts.items())),
                "oracle": dict(sorted(self.oracle_error_class_counts.items())),
            },
            "checkpoint_restoration": {
                "restorable_pair_count": self.checkpoint_restorable_count,
                "checked_pair_count": self.checkpoint_restore_checked_count,
                "failure_pair_count": self.checkpoint_restore_failure_count,
                "unchecked_pair_count": self.runtime_restore_unchecked_count,
                "saved_checkpoint_count": self.saved_checkpoint_count,
            },
            "proposal_coverage": self.proposal_coverage,
        }


def _issue(
    session: PairedSession,
    code: str,
    message: str,
    remediation: str,
) -> ValidationIssue:
    return ValidationIssue(
        session_id=session.session_id,
        code=code,
        message=message,
        remediation=remediation,
    )


def _expected_identity(session: PairedSession) -> ScenarioIdentity:
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


def _projection_issues(
    scenario: FrozenScenario,
    session: PairedSession,
    behavior: TrajectoryResult,
    oracle: TrajectoryResult,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    projection = {
        "scenario_id": scenario.scenario_id,
        "environment_layer": scenario.environment_layer,
        "suite_or_domain": scenario.suite_or_domain,
        "user_task_id": scenario.user_task_id,
        "injection_task_id": scenario.injection_task_id,
        "rendered_attack_id": scenario.rendered_attack_id,
        "rendered_attack_sha256": scenario.rendered_attack_sha256,
        "initial_state_sha256": scenario.canonical_initial_state.sha256,
        "policy_sha256": scenario.policy_sha256,
        "environment_seed": scenario.environment_seed,
        "split": scenario.split,
        "agentdojo_package_version": scenario.agentdojo_package_version,
        "agentdojo_benchmark_version": scenario.agentdojo_benchmark_version,
    }
    for field_name, expected in projection.items():
        observed = getattr(session, field_name)
        if observed != expected:
            issues.append(
                _issue(
                    session,
                    "session_scenario_identity_mismatch",
                    f"session {field_name}={observed!r} differs from frozen value {expected!r}",
                    "discard this result and regenerate the manifest/session expansion",
                )
            )

    expected_session_id = derive_session_id(session, scenario)
    if session.session_id != expected_session_id:
        issues.append(
            _issue(
                session,
                "session_identity_hash_mismatch",
                "session_id is not the hash of the frozen scenario and scientific session fields",
                "regenerate the session key; never reuse cached traces under a changed identity",
            )
        )

    expected_identity = _expected_identity(session)
    for label, trajectory, expected_role in (
        ("behavior", behavior, TrajectoryRole.BEHAVIOR),
        ("oracle", oracle, TrajectoryRole.TARGET),
    ):
        if trajectory.role is not expected_role:
            issues.append(
                _issue(
                    session,
                    "trajectory_role_mismatch",
                    f"{label} trajectory declares role {trajectory.role.value!r}",
                    f"load the persisted {expected_role.value} trajectory for this session",
                )
            )
        if trajectory.scenario != expected_identity:
            issues.append(
                _issue(
                    session,
                    "trajectory_identity_mismatch",
                    f"{label} trajectory scientific identity differs from PairedSession",
                    "discard the mismatched summary and rerun this exact session ID",
                )
            )
    return issues


def _checkpoint_roundtrip_ok(snapshot: StateSnapshot) -> bool:
    if not verify_state_snapshot(snapshot):
        return False
    try:
        restored = StateSnapshot.model_validate_json(snapshot.model_dump_json())
    except ValueError:
        return False
    return restored == snapshot and verify_state_snapshot(restored)


def _trace_checkpoint_issues(
    session: PairedSession,
    scenario: FrozenScenario,
    trajectory: TrajectoryResult,
    label: str,
    *,
    expected_guard_id: str,
    expected_guard_configuration_hash: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    expected_pre_hash = scenario.canonical_initial_state.sha256
    previous_position: tuple[int, int] | None = None
    for index, trace in enumerate(trajectory.interventions):
        position = (trace.step_index, trace.tool_call_index)
        if trace.guard_decision.guard_id != expected_guard_id:
            issues.append(
                _issue(
                    session,
                    "trace_guard_id_mismatch",
                    f"{label} trace {index} guard ID "
                    f"{trace.guard_decision.guard_id!r} differs from frozen "
                    f"{expected_guard_id!r}",
                    "discard the trace and rerun it with the role-specific frozen guard",
                )
            )
        if trace.guard_decision.guard_configuration_hash != expected_guard_configuration_hash:
            issues.append(
                _issue(
                    session,
                    "trace_guard_configuration_hash_mismatch",
                    f"{label} trace {index} guard configuration hash differs from "
                    "the role-specific frozen hash",
                    "discard the trace and rerun it with the frozen guard configuration",
                )
            )
        if previous_position is not None and position <= previous_position:
            issues.append(
                _issue(
                    session,
                    "trace_order_error",
                    f"{label} intervention positions are not strictly ordered at trace {index}",
                    "rerun with deterministic tool-call indexing and preserve original call order",
                )
            )
        previous_position = position
        for boundary, snapshot in (("pre", trace.pre_state), ("post", trace.post_state)):
            if not _checkpoint_roundtrip_ok(snapshot):
                issues.append(
                    _issue(
                        session,
                        "trace_checkpoint_corrupt",
                        f"{label} trace {index} {boundary}-state hash or JSON round trip is invalid",
                        "restore the scenario from the frozen manifest and rerun this trajectory",
                    )
                )
        if trace.pre_state.sha256 != expected_pre_hash:
            issues.append(
                _issue(
                    session,
                    "trace_checkpoint_chain_mismatch",
                    f"{label} trace {index} pre-state does not follow the previous checkpoint",
                    "discard the trace; environment state changed outside instrumented tool execution",
                )
            )
        expected_pre_hash = trace.post_state.sha256
        if canonical_sha256(trace.guard_inputs.observable_state) != trace.pre_state.sha256:
            issues.append(
                _issue(
                    session,
                    "guard_observable_state_mismatch",
                    f"{label} trace {index} guard state differs from its pre-state checkpoint",
                    "capture guard inputs immediately before enforcement and rerun",
                )
            )
        if trace.guard_inputs.user_request != scenario.user_request:
            issues.append(
                _issue(
                    session,
                    "guard_user_request_mismatch",
                    f"{label} trace {index} guard received a non-frozen user request",
                    "use the manifest user request verbatim for every guard decision",
                )
            )
        if canonical_sha256(trace.guard_inputs.policy) != scenario.policy_sha256:
            issues.append(
                _issue(
                    session,
                    "trace_policy_hash_mismatch",
                    f"{label} trace {index} guard policy differs from the frozen policy hash",
                    "discard the trace and rerun with the manifest authorization policy",
                )
            )

    if (
        trajectory.final_state_sha256 is not None
        and trajectory.final_state_sha256 != expected_pre_hash
    ):
        issues.append(
            _issue(
                session,
                "final_checkpoint_mismatch",
                f"{label} final-state hash does not match the last instrumented checkpoint",
                "rerun and checkpoint every environment mutation before terminal validation",
            )
        )
    return issues


def _checkpoint_issues(
    scenario: FrozenScenario,
    session: PairedSession,
    behavior: TrajectoryResult,
    oracle: TrajectoryResult,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    frozen_hash = scenario.canonical_initial_state.sha256
    if not _checkpoint_roundtrip_ok(scenario.canonical_initial_state):
        issues.append(
            _issue(
                session,
                "frozen_checkpoint_corrupt",
                "canonical initial state fails its SHA-256 or durable JSON round trip",
                "refreeze the scenario before inspecting or reusing any model outcome",
            )
        )
    observed = {
        "session": session.initial_state_sha256,
        "behavior": behavior.initial_state_sha256,
        "oracle": oracle.initial_state_sha256,
    }
    for label, state_hash in observed.items():
        if state_hash != frozen_hash:
            issues.append(
                _issue(
                    session,
                    "initial_checkpoint_mismatch",
                    f"{label} initial hash {state_hash} differs from frozen hash {frozen_hash}",
                    "discard both paired runs and independently restore each from the frozen state",
                )
            )
    issues.extend(
        _trace_checkpoint_issues(
            session,
            scenario,
            behavior,
            "behavior",
            expected_guard_id=session.behavior_guard_id,
            expected_guard_configuration_hash=session.behavior_guard_config_sha256,
        )
    )
    issues.extend(
        _trace_checkpoint_issues(
            session,
            scenario,
            oracle,
            "oracle",
            expected_guard_id=session.target_guard_id,
            expected_guard_configuration_hash=session.target_guard_config_sha256,
        )
    )
    return issues


def _hash_issues(scenario: FrozenScenario, session: PairedSession) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    rendered_hash = canonical_sha256(scenario.rendered_attack)
    if rendered_hash != scenario.rendered_attack_sha256:
        issues.append(
            _issue(
                session,
                "rendered_attack_hash_mismatch",
                "frozen rendered attack payload does not match rendered_attack_sha256",
                "refreeze the payload and rerun both trajectories from the new session key",
            )
        )
    if session.rendered_attack_sha256 != rendered_hash:
        issues.append(
            _issue(
                session,
                "session_attack_hash_mismatch",
                "PairedSession attack hash differs from the frozen rendered payload",
                "discard this session and regenerate manifest expansion",
            )
        )
    policy_hash = canonical_sha256(scenario.policy)
    if policy_hash != scenario.policy_sha256:
        issues.append(
            _issue(
                session,
                "frozen_policy_hash_mismatch",
                "frozen authorization policy does not match policy_sha256",
                "refreeze policy rules before any actor execution",
            )
        )
    if session.policy_sha256 != policy_hash:
        issues.append(
            _issue(
                session,
                "session_policy_hash_mismatch",
                "PairedSession policy hash differs from the frozen authorization policy",
                "discard this session and regenerate manifest expansion",
            )
        )
    return issues


def _failure_provenance_issues(
    session: PairedSession,
    trajectory: TrajectoryResult,
    label: str,
) -> list[ValidationIssue]:
    if trajectory.status in SUCCESS_STATUSES:
        return []
    if trajectory.terminal_harm is not None:
        return [
            _issue(
                session,
                "invalid_trajectory_has_harm",
                f"{label} invalid trajectory carries a terminal harm label",
                "store null harm for failures and retain the raw error status separately",
            )
        ]
    if trajectory.error_type or trajectory.error_message:
        return []
    return [
        _issue(
            session,
            "missing_error_provenance",
            f"{label} trajectory status {trajectory.status.value!r} has no error provenance",
            "persist the original error type/message and rerun only with bounded --retry-failed",
        )
    ]


def _unique_checkpoints(
    scenario: FrozenScenario,
    behavior: TrajectoryResult | None,
    oracle: TrajectoryResult | None,
) -> tuple[StateSnapshot, ...]:
    """Return initial/pre/post snapshots once each, in first-seen order."""

    checkpoints: dict[str, StateSnapshot] = {
        scenario.canonical_initial_state.sha256: scenario.canonical_initial_state
    }
    for trajectory in (behavior, oracle):
        if trajectory is None:
            continue
        for trace in trajectory.interventions:
            checkpoints.setdefault(trace.pre_state.sha256, trace.pre_state)
            checkpoints.setdefault(trace.post_state.sha256, trace.post_state)
    return tuple(checkpoints.values())


def _saved_checkpoint_count(
    behavior: TrajectoryResult | None,
    oracle: TrajectoryResult | None,
) -> int:
    """Count the manifest initial snapshot and every persisted trace boundary."""

    return 1 + sum(
        2 * len(trajectory.interventions)
        for trajectory in (behavior, oracle)
        if trajectory is not None
    )


def _accepts_snapshot_argument(
    restore_check: CheckpointRestoreCheck,
    scenario: FrozenScenario,
    checkpoint: StateSnapshot,
) -> bool:
    try:
        inspect.signature(restore_check).bind(scenario, checkpoint)
    except (TypeError, ValueError):
        return False
    return True


def _restore_result_matches(result: RestoreCheckResult, checkpoint: StateSnapshot) -> bool:
    if isinstance(result, StateSnapshot):
        return _checkpoint_roundtrip_ok(result) and result.sha256 == checkpoint.sha256
    if isinstance(result, str):
        return result == checkpoint.sha256
    return result if isinstance(result, bool) else False


def _run_restore_check(
    scenario: FrozenScenario,
    session: PairedSession,
    behavior: TrajectoryResult,
    oracle: TrajectoryResult,
    restore_check: CheckpointRestoreCheck | None,
) -> tuple[bool, bool, int, list[ValidationIssue]]:
    checkpoints = _unique_checkpoints(scenario, behavior, oracle)
    if restore_check is None:
        return True, False, len(checkpoints), []

    supports_snapshots = _accepts_snapshot_argument(
        restore_check,
        scenario,
        checkpoints[0],
    )
    checked = checkpoints if supports_snapshots else checkpoints[:1]
    issues: list[ValidationIssue] = []
    for index, checkpoint in enumerate(checked):
        try:
            restored = (
                restore_check(scenario, checkpoint)
                if supports_snapshots
                else restore_check(scenario)
            )
        except Exception as exc:
            issues.append(
                _issue(
                    session,
                    "checkpoint_restore_error",
                    "runtime restore raised "
                    f"{type(exc).__name__} for checkpoint {index} ({checkpoint.sha256}): {exc}",
                    "verify the pinned environment adapter and rerun every saved checkpoint round trip",
                )
            )
            continue
        if not _restore_result_matches(restored, checkpoint):
            issues.append(
                _issue(
                    session,
                    "checkpoint_not_restorable",
                    f"runtime restore did not reproduce checkpoint {index} ({checkpoint.sha256})",
                    "repair the version-pinned adapter before running or analyzing later-stage suffixes",
                )
            )

    full_pair_checked = supports_snapshots or len(checkpoints) == 1
    return not issues, full_pair_checked, len(checkpoints), issues


def _technical_failure_provenance_issues(
    session: PairedSession,
    trajectory: TrajectoryResult,
    label: str,
) -> list[ValidationIssue]:
    """Check failure diagnostics without reading any scientific outcome field."""

    if trajectory.status in SUCCESS_STATUSES:
        return []
    if trajectory.error_type or trajectory.error_message:
        return []
    return [
        _issue(
            session,
            "missing_error_provenance",
            f"{label} trajectory status {trajectory.status.value!r} has no error provenance",
            "persist the original error type/message before any feasibility decision",
        )
    ]


def _feasibility_status(status: RunStatus) -> str:
    """Collapse every successful execution status to a technical completion label."""

    return "completed" if status in SUCCESS_STATUSES else status.value


def _guard_configuration_issues(
    session: PairedSession,
) -> list[ValidationIssue]:
    """Validate frozen same-guard configuration without comparing outcomes or traces."""

    issues: list[ValidationIssue] = []
    declared_same_guard = session.guard_pair_id.startswith("same_guard")
    identical_guard_ids = session.behavior_guard_id == session.target_guard_id
    if declared_same_guard and not identical_guard_ids:
        issues.append(
            _issue(
                session,
                "same_guard_configuration_mismatch",
                "negative-control guard pair declares different behavior and target guard IDs",
                "regenerate the negative-control session with identical guard IDs and hashes",
            )
        )
    if identical_guard_ids and (
        session.behavior_guard_config_sha256 != session.target_guard_config_sha256
    ):
        issues.append(
            _issue(
                session,
                "same_guard_hash_mismatch",
                "identical guard IDs have different frozen configuration hashes",
                "freeze one guard configuration and reuse it for both trajectories",
            )
        )
    return issues


def _proposal_coverage_report(
    records: Iterable[PairValidationInput],
) -> dict[str, Any]:
    """Summarize captured pre-guard proposals without inspecting their semantics."""

    groups: dict[tuple[str, str], Counter[str]] = {}
    overall: Counter[str] = Counter()
    for record in records:
        key = (
            record.session.environment_layer.value,
            record.session.suite_or_domain,
        )
        counts = groups.setdefault(key, Counter())
        counts["selected_pair_count"] += 1
        overall["selected_pair_count"] += 1
        behavior_has_proposal = bool(record.behavior is not None and record.behavior.interventions)
        oracle_has_proposal = bool(record.oracle is not None and record.oracle.interventions)
        if behavior_has_proposal:
            counts["behavior_pair_count"] += 1
            overall["behavior_pair_count"] += 1
        if oracle_has_proposal:
            counts["oracle_pair_count"] += 1
            overall["oracle_pair_count"] += 1
        if behavior_has_proposal and oracle_has_proposal:
            counts["both_roles_pair_count"] += 1
            overall["both_roles_pair_count"] += 1

    by_environment_domain: list[dict[str, Any]] = []
    agentdojo_missing: list[str] = []
    for (environment_layer, suite_or_domain), counts in sorted(groups.items()):
        row = {
            "environment_layer": environment_layer,
            "suite_or_domain": suite_or_domain,
            "selected_pair_count": counts["selected_pair_count"],
            "behavior_pair_count": counts["behavior_pair_count"],
            "oracle_pair_count": counts["oracle_pair_count"],
            "both_roles_pair_count": counts["both_roles_pair_count"],
        }
        by_environment_domain.append(row)
        if environment_layer == "agentdojo" and counts["both_roles_pair_count"] == 0:
            agentdojo_missing.append(suite_or_domain)

    return {
        "overall": {
            "selected_pair_count": overall["selected_pair_count"],
            "behavior_pair_count": overall["behavior_pair_count"],
            "oracle_pair_count": overall["oracle_pair_count"],
            "both_roles_pair_count": overall["both_roles_pair_count"],
        },
        "by_environment_domain": by_environment_domain,
        "agentdojo_suites_missing_both_role_proposal": agentdojo_missing,
        "all_agentdojo_suites_have_both_role_proposal": not agentdojo_missing,
    }


def _block_rate(trajectory: TrajectoryResult) -> float | None:
    if trajectory.proposed_call_count == 0:
        return None
    return trajectory.blocked_call_count / trajectory.proposed_call_count


def normalized_pair_row(
    scenario: FrozenScenario,
    session: PairedSession,
    behavior: TrajectoryResult,
    oracle: TrajectoryResult,
    *,
    alignment: Literal["diverged", "no_divergence", "invalid"],
    first_divergence: FirstDivergence | None,
    checkpoint_restorable: bool,
    unique_checkpoint_count: int,
) -> dict[str, Any]:
    """Project a validated pair into the ``analysis.exp1`` row contract."""

    pair_successful = behavior.status in SUCCESS_STATUSES and oracle.status in SUCCESS_STATUSES
    raw_stratum = scenario.metadata.get("stratum")
    stratum = raw_stratum if isinstance(raw_stratum, str) else None
    is_clean = (
        stratum == "clean"
        if scenario.environment_layer.value == "control"
        else session.injection_task_id is None
    )
    return {
        "pair_id": session.session_id,
        "session_id": session.session_id,
        "scenario_id": scenario.scenario_id,
        "split": session.split.value,
        "environment_layer": session.environment_layer.value,
        "domain": session.suite_or_domain,
        "suite_or_domain": session.suite_or_domain,
        "user_task_id": session.user_task_id,
        "injection_task_id": session.injection_task_id,
        "stratum": stratum,
        "rendered_attack_id": session.rendered_attack_id,
        "rendered_attack_sha256": session.rendered_attack_sha256,
        "policy_sha256": session.policy_sha256,
        "initial_state_sha256": session.initial_state_sha256,
        "behavior_final_state_sha256": behavior.final_state_sha256,
        "target_final_state_sha256": oracle.final_state_sha256,
        "actor_id": session.actor_id,
        "actor_revision": session.actor_revision,
        "decoding_seed": session.decoding_seed,
        "environment_seed": session.environment_seed,
        "guard_pair_id": session.guard_pair_id,
        "behavior_guard_id": session.behavior_guard_id,
        "target_guard_id": session.target_guard_id,
        "behavior_status": behavior.status.value,
        "target_status": oracle.status.value,
        "behavior_run_status": behavior.run_status.value,
        "target_run_status": oracle.run_status.value,
        "behavior_harm": behavior.terminal_harm if behavior.status in SUCCESS_STATUSES else None,
        "target_harm": oracle.terminal_harm if oracle.status in SUCCESS_STATUSES else None,
        "behavior_attack_goal_achieved": behavior.attack_goal_achieved,
        "target_attack_goal_achieved": oracle.attack_goal_achieved,
        "behavior_user_utility": behavior.user_utility,
        "target_user_utility": oracle.user_utility,
        "behavior_attempted_unsafe_action": behavior.attempted_unsafe_action,
        "target_attempted_unsafe_action": oracle.attempted_unsafe_action,
        "behavior_blocked_call_count": behavior.blocked_call_count,
        "target_blocked_call_count": oracle.blocked_call_count,
        "behavior_proposed_call_count": behavior.proposed_call_count,
        "target_proposed_call_count": oracle.proposed_call_count,
        "behavior_block_rate": _block_rate(behavior),
        "target_block_rate": _block_rate(oracle),
        "behavior_error_type": behavior.error_type,
        "target_error_type": oracle.error_type,
        "behavior_error_message": behavior.error_message,
        "target_error_message": oracle.error_message,
        "is_attack": session.injection_task_id is not None,
        "is_clean": is_clean,
        "alignment": alignment,
        "guard_dependent": first_divergence is not None if pair_successful else None,
        "first_divergence_step": (
            first_divergence.step_index if first_divergence is not None else None
        ),
        "first_divergence_shared_prefix_id": (
            first_divergence.shared_prefix_id if first_divergence is not None else None
        ),
        "checkpoint_restorable": checkpoint_restorable,
        "behavior_saved_checkpoint_count": 1 + 2 * len(behavior.interventions),
        "target_saved_checkpoint_count": 1 + 2 * len(oracle.interventions),
        "total_saved_checkpoint_count": _saved_checkpoint_count(behavior, oracle),
        "unique_checkpoint_count": unique_checkpoint_count,
    }


def validate_pair(
    scenario: FrozenScenario,
    session: PairedSession,
    behavior: TrajectoryResult,
    oracle: TrajectoryResult,
    *,
    checkpoint_restore_check: CheckpointRestoreCheck | None = None,
) -> ValidatedPair:
    """Validate one complete persisted pair and return its normalized row.

    A complete persisted pair may contain failed trajectories. Those statuses are
    valid data and are returned with null realized harm. Structural mismatches
    instead raise ``PairValidationError`` and must never enter analysis.
    """

    issues = _projection_issues(scenario, session, behavior, oracle)
    issues.extend(_hash_issues(scenario, session))
    issues.extend(_checkpoint_issues(scenario, session, behavior, oracle))
    issues.extend(_failure_provenance_issues(session, behavior, "behavior"))
    issues.extend(_failure_provenance_issues(session, oracle, "oracle"))
    checkpoint_restorable, runtime_checked, checkpoint_count, restore_issues = _run_restore_check(
        scenario,
        session,
        behavior,
        oracle,
        checkpoint_restore_check,
    )
    issues.extend(restore_issues)
    if issues:
        raise PairValidationError(issues)

    try:
        divergence = find_first_divergence(behavior, oracle)
    except PairingError as exc:
        raise PairValidationError(
            [
                _issue(
                    session,
                    "pre_divergence_alignment_error",
                    str(exc),
                    "discard both traces and rerun from independent copies of the frozen checkpoint",
                )
            ]
        ) from exc

    pair_successful = behavior.status in SUCCESS_STATUSES and oracle.status in SUCCESS_STATUSES
    if (
        pair_successful
        and divergence is None
        and len(behavior.interventions) != len(oracle.interventions)
    ):
        raise PairValidationError(
            [
                _issue(
                    session,
                    "trace_length_mismatch_without_divergence",
                    "successful traces have different lengths without an intervention disagreement",
                    "rerun with deterministic decoding and identical initial actor-visible histories",
                )
            ]
        )

    declared_same_guard = session.guard_pair_id.startswith("same_guard")
    identical_guard_ids = session.behavior_guard_id == session.target_guard_id
    same_guard = declared_same_guard or identical_guard_ids
    same_guard_issues: list[ValidationIssue] = []
    if declared_same_guard and not identical_guard_ids:
        same_guard_issues.append(
            _issue(
                session,
                "same_guard_configuration_mismatch",
                "negative-control guard pair declares different behavior and target guard IDs",
                "regenerate the negative-control session with identical guard IDs and hashes",
            )
        )
    if identical_guard_ids and (
        session.behavior_guard_config_sha256 != session.target_guard_config_sha256
    ):
        same_guard_issues.append(
            _issue(
                session,
                "same_guard_hash_mismatch",
                "identical guard IDs have different frozen configuration hashes",
                "freeze one guard configuration and reuse it for both negative-control trajectories",
            )
        )
    if pair_successful and same_guard:
        if behavior.terminal_harm != oracle.terminal_harm:
            same_guard_issues.append(
                _issue(
                    session,
                    "same_guard_nonzero_masking_gap",
                    "valid same-guard pair has different realized harms and a nonzero masking gap",
                    "discard the pair and audit reset, model determinism, and evaluator determinism",
                )
            )
        if divergence is not None:
            same_guard_issues.append(
                _issue(
                    session,
                    "same_guard_intervention_divergence",
                    "same-guard negative-control trajectories disagree at an intervention",
                    "reuse the identical guard configuration and deterministic RNG stream",
                )
            )
    if same_guard_issues:
        raise PairValidationError(same_guard_issues)

    alignment: Literal["diverged", "no_divergence", "invalid"]
    if not pair_successful:
        alignment = "invalid"
    elif divergence is None:
        alignment = "no_divergence"
    else:
        alignment = "diverged"
    row = normalized_pair_row(
        scenario,
        session,
        behavior,
        oracle,
        alignment=alignment,
        first_divergence=divergence,
        checkpoint_restorable=checkpoint_restorable,
        unique_checkpoint_count=checkpoint_count,
    )
    return ValidatedPair(
        scenario=scenario,
        session=session,
        behavior=behavior,
        oracle=oracle,
        alignment=alignment,
        first_divergence=divergence,
        normalized_row=row,
        checkpoint_restorable=checkpoint_restorable,
        runtime_restore_checked=runtime_checked,
        unique_checkpoint_count=checkpoint_count,
    )


def aggregate_validation_report(
    records: Iterable[PairValidationInput],
    *,
    checkpoint_restore_check: CheckpointRestoreCheck | None = None,
) -> ValidationReport:
    """Validate all expected sessions without dropping persisted invalid runs."""

    materialized = tuple(records)
    rows: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    seen_session_ids: set[str] = set()
    missing_behavior = 0
    missing_oracle = 0
    invalid_behavior = 0
    invalid_oracle = 0
    invalid_pairs = 0
    valid_pairs = 0
    checkpoint_restorable = 0
    checkpoint_checked = 0
    checkpoint_failures = 0
    runtime_unchecked = 0
    saved_checkpoints = 0
    unique_checkpoints = 0
    diverged = 0
    no_divergence = 0

    for record in materialized:
        session = record.session
        if session.session_id in seen_session_ids:
            issues.append(
                _issue(
                    session,
                    "duplicate_validation_session",
                    "the validation input repeats a session_id",
                    "deduplicate completion records by the frozen session key",
                )
            )
            continue
        seen_session_ids.add(session.session_id)

        saved_checkpoints += _saved_checkpoint_count(record.behavior, record.oracle)
        pair_checkpoints = _unique_checkpoints(
            record.scenario,
            record.behavior,
            record.oracle,
        )
        unique_checkpoints += len(pair_checkpoints)
        behavior_missing = record.behavior is None
        oracle_missing = record.oracle is None
        missing_behavior += int(behavior_missing)
        missing_oracle += int(oracle_missing)
        if behavior_missing:
            issues.append(
                _issue(
                    session,
                    "missing_behavior_trajectory",
                    "behavior trajectory completion record is missing",
                    "resume the behavior stage for this session ID",
                )
            )
        if oracle_missing:
            issues.append(
                _issue(
                    session,
                    "missing_oracle_trajectory",
                    "oracle trajectory completion record is missing",
                    "resume the evaluation-gated oracle stage for this session ID",
                )
            )
        if behavior_missing or oracle_missing:
            continue
        behavior = record.behavior
        oracle = record.oracle
        if behavior is None or oracle is None:  # static narrowing for Python 3.10.
            raise AssertionError("missing trajectories were handled above")

        behavior_invalid = behavior.status not in SUCCESS_STATUSES
        oracle_invalid = oracle.status not in SUCCESS_STATUSES
        invalid_behavior += int(behavior_invalid)
        invalid_oracle += int(oracle_invalid)
        pair_invalid = behavior_invalid or oracle_invalid
        invalid_pairs += int(pair_invalid)

        try:
            validated = validate_pair(
                record.scenario,
                session,
                behavior,
                oracle,
                checkpoint_restore_check=checkpoint_restore_check,
            )
        except PairValidationError as exc:
            issues.extend(exc.issues)
            restore_failed = any(issue.code in CHECKPOINT_FAILURE_CODES for issue in exc.issues)
            if restore_failed:
                checkpoint_failures += 1
                full_runtime_check = checkpoint_restore_check is not None and (
                    len(pair_checkpoints) == 1
                    or _accepts_snapshot_argument(
                        checkpoint_restore_check,
                        record.scenario,
                        pair_checkpoints[0],
                    )
                )
                if full_runtime_check:
                    checkpoint_checked += 1
            continue

        rows.append(validated.normalized_row)
        checkpoint_restorable += int(validated.checkpoint_restorable)
        checkpoint_checked += int(validated.runtime_restore_checked)
        runtime_unchecked += int(not validated.runtime_restore_checked)
        if not pair_invalid:
            valid_pairs += 1
        if validated.alignment == "diverged":
            diverged += 1
        elif validated.alignment == "no_divergence":
            no_divergence += 1

    return ValidationReport(
        total_sessions=len(materialized),
        normalized_row_count=len(rows),
        valid_pair_count=valid_pairs,
        invalid_pair_count=invalid_pairs,
        missing_behavior_count=missing_behavior,
        missing_oracle_count=missing_oracle,
        invalid_behavior_count=invalid_behavior,
        invalid_oracle_count=invalid_oracle,
        checkpoint_restorable_count=checkpoint_restorable,
        checkpoint_restore_checked_count=checkpoint_checked,
        checkpoint_restore_failure_count=checkpoint_failures,
        runtime_restore_unchecked_count=runtime_unchecked,
        saved_checkpoint_count=saved_checkpoints,
        unique_checkpoint_count=unique_checkpoints,
        diverged_pair_count=diverged,
        no_divergence_pair_count=no_divergence,
        structural_error_count=len(issues),
        normalized_rows=tuple(rows),
        issues=tuple(issues),
    )


def aggregate_feasibility_report(
    records: Iterable[PairValidationInput],
    *,
    checkpoint_restore_check: CheckpointRestoreCheck | None = None,
) -> FeasibilityReport:
    """Aggregate technical readiness without deriving or persisting outcomes.

    Unlike :func:`aggregate_validation_report`, this path never normalizes
    terminal labels and never attempts paired disagreement detection.  It is
    intended for outcome-blind adapter smoke and development gates.
    """

    materialized = tuple(records)
    issues: list[ValidationIssue] = []
    seen_session_ids: set[str] = set()
    complete_pairs = 0
    successful_pairs = 0
    invalid_pairs = 0
    structurally_valid_pairs = 0
    missing_behavior = 0
    missing_oracle = 0
    invalid_behavior = 0
    invalid_oracle = 0
    behavior_statuses: Counter[str] = Counter()
    oracle_statuses: Counter[str] = Counter()
    behavior_errors: Counter[str] = Counter()
    oracle_errors: Counter[str] = Counter()
    checkpoint_restorable = 0
    checkpoint_checked = 0
    checkpoint_failures = 0
    runtime_unchecked = 0
    saved_checkpoints = 0

    for record in materialized:
        session = record.session
        if session.session_id in seen_session_ids:
            issues.append(
                _issue(
                    session,
                    "duplicate_feasibility_session",
                    "the feasibility input repeats a session_id",
                    "deduplicate completion records by the frozen session key",
                )
            )
            continue
        seen_session_ids.add(session.session_id)

        saved_checkpoints += _saved_checkpoint_count(record.behavior, record.oracle)

        behavior_missing = record.behavior is None
        oracle_missing = record.oracle is None
        missing_behavior += int(behavior_missing)
        missing_oracle += int(oracle_missing)
        if behavior_missing:
            issues.append(
                _issue(
                    session,
                    "missing_behavior_trajectory",
                    "behavior trajectory completion record is missing",
                    "resume the behavior stage for this session ID",
                )
            )
        if oracle_missing:
            issues.append(
                _issue(
                    session,
                    "missing_oracle_trajectory",
                    "oracle trajectory completion record is missing",
                    "resume the evaluation-gated oracle stage for this session ID",
                )
            )
        if behavior_missing or oracle_missing:
            continue

        behavior = record.behavior
        oracle = record.oracle
        if behavior is None or oracle is None:  # static narrowing for Python 3.10.
            raise AssertionError("missing trajectories were handled above")
        complete_pairs += 1
        behavior_statuses[_feasibility_status(behavior.status)] += 1
        oracle_statuses[_feasibility_status(oracle.status)] += 1
        behavior_failed = behavior.status not in SUCCESS_STATUSES
        oracle_failed = oracle.status not in SUCCESS_STATUSES
        invalid_behavior += int(behavior_failed)
        invalid_oracle += int(oracle_failed)
        pair_failed = behavior_failed or oracle_failed
        invalid_pairs += int(pair_failed)
        successful_pairs += int(not pair_failed)
        if behavior_failed:
            behavior_errors[behavior.error_type or "UnspecifiedError"] += 1
        if oracle_failed:
            oracle_errors[oracle.error_type or "UnspecifiedError"] += 1

        pair_issues = _projection_issues(record.scenario, session, behavior, oracle)
        pair_issues.extend(_hash_issues(record.scenario, session))
        pair_issues.extend(_checkpoint_issues(record.scenario, session, behavior, oracle))
        pair_issues.extend(_technical_failure_provenance_issues(session, behavior, "behavior"))
        pair_issues.extend(_technical_failure_provenance_issues(session, oracle, "oracle"))
        pair_issues.extend(_guard_configuration_issues(session))
        restore_ok, runtime_checked, _, restore_issues = _run_restore_check(
            record.scenario,
            session,
            behavior,
            oracle,
            checkpoint_restore_check,
        )
        pair_issues.extend(restore_issues)
        has_checkpoint_failure = any(
            issue.code in CHECKPOINT_FAILURE_CODES for issue in pair_issues
        )
        checkpoint_checked += int(runtime_checked)
        runtime_unchecked += int(not runtime_checked)
        checkpoint_failures += int(has_checkpoint_failure)
        checkpoint_restorable += int(runtime_checked and restore_ok and not has_checkpoint_failure)
        if not pair_issues:
            structurally_valid_pairs += 1
        issues.extend(pair_issues)

    return FeasibilityReport(
        total_sessions=len(materialized),
        complete_pair_count=complete_pairs,
        successful_pair_count=successful_pairs,
        invalid_pair_count=invalid_pairs,
        structurally_valid_pair_count=structurally_valid_pairs,
        missing_behavior_count=missing_behavior,
        missing_oracle_count=missing_oracle,
        invalid_behavior_count=invalid_behavior,
        invalid_oracle_count=invalid_oracle,
        behavior_status_counts=dict(behavior_statuses),
        oracle_status_counts=dict(oracle_statuses),
        behavior_error_class_counts=dict(behavior_errors),
        oracle_error_class_counts=dict(oracle_errors),
        checkpoint_restorable_count=checkpoint_restorable,
        checkpoint_restore_checked_count=checkpoint_checked,
        checkpoint_restore_failure_count=checkpoint_failures,
        runtime_restore_unchecked_count=runtime_unchecked,
        saved_checkpoint_count=saved_checkpoints,
        proposal_coverage=_proposal_coverage_report(materialized),
        issues=tuple(issues),
    )


# Descriptive aliases for CLI and notebook callers.
validate_paired_session = validate_pair
validate_experiment = aggregate_validation_report


__all__ = [
    "CheckpointRestoreCheck",
    "FeasibilityReport",
    "PairValidationError",
    "PairValidationInput",
    "ValidatedPair",
    "ValidationAggregateError",
    "ValidationIssue",
    "ValidationReport",
    "aggregate_feasibility_report",
    "aggregate_validation_report",
    "normalized_pair_row",
    "validate_experiment",
    "validate_pair",
    "validate_paired_session",
]
