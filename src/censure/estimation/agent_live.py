"""Live, selected-only checkpoint suffix execution for held-out agents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from censure.actors.base import Actor
from censure.estimation.agent_cohort import (
    AgentAuditCohort,
    AgentSuffixDiagnostics,
    AgentSuffixRoot,
    _trajectory_from_store_trace,
    evaluate_agent_suffix_trajectory,
)
from censure.estimation.schemas import PrivateSuffixOutcome, SuffixAuditStatus
from censure.execution import CheckpointSuffixRunner, RuntimeBindings, seeded_guard_rng
from censure.guards import make_guard
from censure.manifest import ExperimentManifest
from censure.schemas import (
    FrozenModel,
    FrozenScenario,
    PairedSession,
    ScenarioIdentity,
    Sha256Hex,
    TrajectoryResult,
)
from censure.serialization import canonical_sha256
from censure.storage import CorruptArtifactError, RunStore, atomic_write_bytes, atomic_write_json

SELECTED_SUFFIX_RUN_SCHEMA_VERSION = "censure.selected-suffix-run.v1"

BindingsFactory = Callable[[FrozenScenario], RuntimeBindings]


class SelectedSuffixRun(FrozenModel):
    """One immutable selected suffix, including all configured retry attempts."""

    schema_version: Literal["censure.selected-suffix-run.v1"] = SELECTED_SUFFIX_RUN_SCHEMA_VERSION
    protocol_id: str
    source_manifest_sha256: Sha256Hex
    cohort_id: str
    cohort_sha256: Sha256Hex
    root_sha256: Sha256Hex
    candidate_id: Sha256Hex
    source_session_id: Sha256Hex
    actor_id: str
    attempts: tuple[TrajectoryResult, ...] = Field(min_length=1)
    outcome: PrivateSuffixOutcome
    diagnostics: AgentSuffixDiagnostics

    @model_validator(mode="after")
    def validate_projection(self) -> SelectedSuffixRun:
        if self.outcome.candidate_id != self.candidate_id:
            raise ValueError("selected suffix outcome belongs to another candidate")
        if self.diagnostics.candidate_id != self.candidate_id:
            raise ValueError("selected suffix diagnostics belong to another candidate")
        if self.diagnostics.source_session_id != self.source_session_id:
            raise ValueError("selected suffix diagnostics belong to another session")
        if any(attempt.scenario.actor_id != self.actor_id for attempt in self.attempts):
            raise ValueError("selected suffix attempts belong to another actor")
        return self


class SelectedSuffixRunStore:
    """Checksummed private cache containing selected candidates only."""

    def __init__(self, out_root: str | Path, experiment_id: str) -> None:
        self.root = (
            Path(out_root).expanduser().resolve()
            / experiment_id
            / "phase2"
            / "suffix_oracle_private"
        )

    def _path(self, *, cohort_id: str, candidate_id: str) -> Path:
        return self.root / cohort_id / candidate_id / "selected_suffix_run.json"

    def has_run(self, *, cohort_id: str, candidate_id: str) -> bool:
        path = self._path(cohort_id=cohort_id, candidate_id=candidate_id)
        return path.is_file() or path.with_suffix(".sha256").is_file()

    def write_run(self, run: SelectedSuffixRun) -> str:
        path = self._path(cohort_id=run.cohort_id, candidate_id=run.candidate_id)
        if path.is_file() or path.with_suffix(".sha256").is_file():
            existing = self.read_run(
                cohort_id=run.cohort_id,
                candidate_id=run.candidate_id,
            )
            if existing != run:
                raise FileExistsError("a selected suffix outcome cannot be rewritten")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        digest = atomic_write_json(path, run)
        atomic_write_bytes(path.with_suffix(".sha256"), f"{digest}\n".encode())
        return digest

    def read_run(self, *, cohort_id: str, candidate_id: str) -> SelectedSuffixRun:
        path = self._path(cohort_id=cohort_id, candidate_id=candidate_id)
        checksum = path.with_suffix(".sha256")
        if not path.is_file() or not checksum.is_file():
            raise CorruptArtifactError("selected suffix run or checksum is missing")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum.read_text(encoding="utf-8").strip():
            raise CorruptArtifactError("selected suffix run checksum mismatch")
        try:
            return SelectedSuffixRun.model_validate(json.loads(raw))
        except Exception as exc:
            raise CorruptArtifactError("selected suffix run is invalid") from exc


def _scenario_identity(session: PairedSession) -> ScenarioIdentity:
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


def _target_guard(session: PairedSession):
    guard_id = session.target_guard_id
    if guard_id.startswith("degraded_strict:"):
        _, _, raw = guard_id.partition(":")
        return make_guard(
            "degraded_strict",
            rho=float(raw),
            rng=seeded_guard_rng(session.session_id, "target"),
            guard_id=guard_id,
        )
    return make_guard(guard_id, guard_id=guard_id)


class LiveAgentSuffixOracle:
    """Execute and persist only suffixes explicitly selected by an auditor."""

    def __init__(
        self,
        *,
        cohort: AgentAuditCohort,
        manifest: ExperimentManifest,
        behavior_store: RunStore,
        suffix_store: SelectedSuffixRunStore,
        actor: Actor,
        bindings_factory: BindingsFactory,
        max_tool_steps: int,
        wall_clock_seconds: float,
        retries: int,
    ) -> None:
        if cohort.source_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("agent cohort and live suffix manifest differ")
        if retries < 0:
            raise ValueError("live suffix retries must be nonnegative")
        self._cohort = cohort
        self._manifest = manifest
        self._behavior_store = behavior_store
        self._suffix_store = suffix_store
        self._actor = actor
        self._bindings_factory = bindings_factory
        self._runner = CheckpointSuffixRunner(
            max_tool_steps=max_tool_steps,
            wall_clock_seconds=wall_clock_seconds,
        )
        self._retries = retries
        self._roots = {root.candidate_id: root for root in cohort.roots}
        self._sessions = {session.session_id: session for session in manifest.sessions}
        self._scenarios = {scenario.scenario_id: scenario for scenario in manifest.scenarios}
        self._cache: dict[str, PrivateSuffixOutcome] = {}
        self._diagnostics: dict[str, AgentSuffixDiagnostics] = {}
        self.requested_candidate_ids: list[str] = []
        self.executed_candidate_ids: list[str] = []
        self.persisted_cache_candidate_ids: list[str] = []

    @property
    def diagnostics(self) -> Mapping[str, AgentSuffixDiagnostics]:
        return dict(self._diagnostics)

    def _validate_persisted(self, root: AgentSuffixRoot, run: SelectedSuffixRun) -> None:
        expected = {
            "protocol_id": self._cohort.protocol_id,
            "source_manifest_sha256": self._manifest.manifest_sha256,
            "cohort_id": self._cohort.cohort_id,
            "cohort_sha256": self._cohort.cohort_sha256,
            "root_sha256": canonical_sha256(root),
            "candidate_id": root.candidate_id,
            "source_session_id": root.source_session_id,
            "actor_id": root.actor_id,
        }
        for field, value in expected.items():
            if getattr(run, field) != value:
                raise CorruptArtifactError(
                    f"selected suffix {field} differs from its frozen cohort"
                )

    def _failed_evaluation(
        self,
        root: AgentSuffixRoot,
        exc: BaseException,
        *,
        suffix_tool_steps: int,
        generation_tokens: int,
    ) -> tuple[PrivateSuffixOutcome, AgentSuffixDiagnostics]:
        return (
            PrivateSuffixOutcome(
                candidate_id=root.candidate_id,
                status=SuffixAuditStatus.EVALUATION_ERROR,
                suffix_tool_steps=suffix_tool_steps,
                generation_tokens=generation_tokens,
            ),
            AgentSuffixDiagnostics(
                candidate_id=root.candidate_id,
                source_session_id=root.source_session_id,
                status=SuffixAuditStatus.EVALUATION_ERROR,
                root_verified=False,
                suffix_tool_steps=suffix_tool_steps,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )

    def evaluate_selected(self, candidate_id: str) -> PrivateSuffixOutcome:
        self.requested_candidate_ids.append(candidate_id)
        if candidate_id not in self._roots:
            raise KeyError(f"candidate is outside the frozen agent cohort: {candidate_id}")
        cached = self._cache.get(candidate_id)
        if cached is not None:
            return cached
        root = self._roots[candidate_id]
        if self._suffix_store.has_run(
            cohort_id=self._cohort.cohort_id,
            candidate_id=candidate_id,
        ):
            run = self._suffix_store.read_run(
                cohort_id=self._cohort.cohort_id,
                candidate_id=candidate_id,
            )
            self._validate_persisted(root, run)
            self.persisted_cache_candidate_ids.append(candidate_id)
            self._cache[candidate_id] = run.outcome
            self._diagnostics[candidate_id] = run.diagnostics
            return run.outcome

        session = self._sessions[root.source_session_id]
        scenario = self._scenarios[root.scenario_id]
        behavior = _trajectory_from_store_trace(
            self._behavior_store.read_behavior_trace(root.source_session_id),
            session_id=root.source_session_id,
        )
        attempts: list[TrajectoryResult] = []
        total_tool_steps = 0
        total_generation_tokens = 0
        for _attempt_index in range(self._retries + 1):
            reset = getattr(self._actor, "reset", None)
            if callable(reset):
                reset()
            guard = _target_guard(session)
            if guard.configuration_hash != session.target_guard_config_sha256:
                raise ValueError("live suffix target guard differs from the frozen session")
            target = self._runner.run(
                scenario=_scenario_identity(session),
                actor=self._actor,
                guard=guard,
                bindings=self._bindings_factory(scenario),
                shared_prefix=root.shared_prefix_interventions,
                root_step_index=root.step_index,
                root_tool_call_index=root.tool_call_index,
                root_pre_intervention_checkpoint=root.pre_intervention_checkpoint,
                root_actor_visible_messages=root.actor_visible_messages,
                root_proposed_call=root.proposed_call,
                root_model_metadata=root.root_model_metadata,
            )
            attempts.append(target)
            total_tool_steps += max(
                0,
                len(target.interventions) - len(root.shared_prefix_interventions),
            )
            total_generation_tokens += target.generation_token_count
            if target.status.value in {"completed", "no_divergence"}:
                break
        final_target = attempts[-1]
        try:
            outcome, diagnostics = evaluate_agent_suffix_trajectory(
                root=root,
                target=final_target,
                behavior=behavior,
                session=session,
                scenario=scenario,
                suffix_tool_steps=total_tool_steps,
                generation_tokens=total_generation_tokens,
            )
        except Exception as exc:
            outcome, diagnostics = self._failed_evaluation(
                root,
                exc,
                suffix_tool_steps=total_tool_steps,
                generation_tokens=total_generation_tokens,
            )
        run = SelectedSuffixRun(
            protocol_id=self._cohort.protocol_id,
            source_manifest_sha256=self._manifest.manifest_sha256,
            cohort_id=self._cohort.cohort_id,
            cohort_sha256=self._cohort.cohort_sha256,
            root_sha256=canonical_sha256(root),
            candidate_id=root.candidate_id,
            source_session_id=root.source_session_id,
            actor_id=root.actor_id,
            attempts=tuple(attempts),
            outcome=outcome,
            diagnostics=diagnostics,
        )
        self._suffix_store.write_run(run)
        self.executed_candidate_ids.append(candidate_id)
        self._cache[candidate_id] = outcome
        self._diagnostics[candidate_id] = diagnostics
        return outcome


__all__ = [
    "SELECTED_SUFFIX_RUN_SCHEMA_VERSION",
    "LiveAgentSuffixOracle",
    "SelectedSuffixRun",
    "SelectedSuffixRunStore",
]
