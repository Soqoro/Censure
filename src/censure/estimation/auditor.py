"""Capability-separated audit execution with deterministic resume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from censure.estimation.allocation import (
    allocation_probabilities,
    deterministic_draw_uniform,
    maximum_importance_weight,
    propensity_vector_sha256,
    select_candidate,
)
from censure.estimation.confidence import certificate_path, current_certificate
from censure.estimation.schemas import (
    AllocationPolicyName,
    AuditDisclosure,
    AuditLedger,
    CertificatePoint,
    FiniteCohortEnvelope,
    PrivateSuffixOutcome,
    SuffixAuditStatus,
)
from censure.serialization import canonical_sha256


class SelectedSuffixOracle(Protocol):
    """Narrow evaluation capability: disclose one explicitly selected suffix."""

    def evaluate_selected(self, candidate_id: str) -> PrivateSuffixOutcome: ...


class InMemoryEvaluationOracle:
    """Evaluation-only fixed outcome map used by exact and simulated studies."""

    def __init__(self, outcomes: Mapping[str, PrivateSuffixOutcome]) -> None:
        self._outcomes = dict(outcomes)
        if set(self._outcomes) != {
            outcome.candidate_id for outcome in self._outcomes.values()
        }:
            raise ValueError("private outcome mapping keys must equal their candidate IDs")
        self.requested_candidate_ids: list[str] = []

    def evaluate_selected(self, candidate_id: str) -> PrivateSuffixOutcome:
        self.requested_candidate_ids.append(candidate_id)
        try:
            return self._outcomes[candidate_id]
        except KeyError as exc:
            raise KeyError(f"no private suffix outcome for candidate {candidate_id!r}") from exc


class CensureAuditor:
    """Run one frozen allocation policy without exposing unselected outcomes."""

    def __init__(
        self,
        *,
        envelope: FiniteCohortEnvelope,
        oracle: SelectedSuffixOracle,
        policy: AllocationPolicyName | str,
        allocation_seed: int,
        alpha: float = 0.05,
        exploration_epsilon: float = 0.10,
    ) -> None:
        if allocation_seed < 0:
            raise ValueError("allocation_seed must be nonnegative")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 <= exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must lie in [0, 1]")
        self.envelope = envelope
        self._oracle = oracle
        self.policy = AllocationPolicyName(policy)
        self.allocation_seed = allocation_seed
        self.alpha = alpha
        self.exploration_epsilon = exploration_epsilon
        self._candidates = envelope.auditable_candidates
        self._candidates_by_id = {
            candidate.candidate_id: candidate for candidate in self._candidates
        }

    def initial_ledger(self) -> AuditLedger:
        return AuditLedger(
            protocol_id=self.envelope.protocol_id,
            cohort_id=self.envelope.cohort_id,
            policy=self.policy,
            allocation_seed=self.allocation_seed,
            alpha=self.alpha,
            exploration_epsilon=self.exploration_epsilon,
        )

    def _probabilities(self, disclosures: Sequence[AuditDisclosure]) -> dict[str, float]:
        return allocation_probabilities(
            self.policy,
            self._candidates,
            disclosures,
            exploration_epsilon=self.exploration_epsilon,
        )

    def _selected_id(self, probabilities: Mapping[str, float], *, round_index: int) -> str:
        uniform_draw = deterministic_draw_uniform(
            protocol_id=self.envelope.protocol_id,
            cohort_id=self.envelope.cohort_id,
            policy=self.policy,
            seed=self.allocation_seed,
            round_index=round_index,
        )
        return select_candidate(probabilities, uniform_draw=uniform_draw)

    def validate_ledger(self, ledger: AuditLedger) -> None:
        expected_header = self.initial_ledger()
        for field in (
            "protocol_id",
            "cohort_id",
            "policy",
            "allocation_seed",
            "alpha",
            "exploration_epsilon",
        ):
            if getattr(ledger, field) != getattr(expected_header, field):
                raise ValueError(f"audit ledger {field} does not match the configured auditor")
        if ledger.disclosures and not self._candidates:
            raise ValueError("audit ledger has disclosures but the cohort has no auditable candidates")

        prefix: tuple[AuditDisclosure, ...] = ()
        for disclosure in ledger.disclosures:
            probabilities = self._probabilities(prefix)
            selected_id = self._selected_id(
                probabilities, round_index=disclosure.round_index
            )
            if disclosure.candidate_id != selected_id:
                raise ValueError("audit ledger candidate draw does not replay deterministically")
            candidate = self._candidates_by_id[selected_id]
            if abs(disclosure.selected_probability - probabilities[selected_id]) > 1e-12:
                raise ValueError("audit ledger selected propensity does not replay")
            if abs(disclosure.target_mass - candidate.target_mass) > 1e-12:
                raise ValueError("audit ledger target mass differs from the frozen candidate")
            expected_bound = maximum_importance_weight(self._candidates, probabilities)
            if abs(disclosure.round_max_importance_weight - expected_bound) > 1e-12:
                raise ValueError("audit ledger importance bound does not replay")
            if disclosure.propensity_vector_sha256 != propensity_vector_sha256(probabilities):
                raise ValueError("audit ledger propensity-vector digest does not replay")
            prefix = (*prefix, disclosure)

    def _next_disclosure(
        self, disclosures: Sequence[AuditDisclosure]
    ) -> AuditDisclosure:
        if not self._candidates:
            raise ValueError("cohort has no auditable suffix candidates")
        round_index = len(disclosures) + 1
        probabilities = self._probabilities(disclosures)
        selected_id = self._selected_id(probabilities, round_index=round_index)
        candidate = self._candidates_by_id[selected_id]
        try:
            outcome = self._oracle.evaluate_selected(selected_id)
        except Exception:
            outcome = PrivateSuffixOutcome(
                candidate_id=selected_id,
                status=SuffixAuditStatus.EVALUATION_ERROR,
            )
        if outcome.candidate_id != selected_id:
            raise ValueError("evaluation oracle returned a different candidate")

        if outcome.status is SuffixAuditStatus.COMPLETED:
            if outcome.terminal_harm is None:  # pragma: no cover - schema invariant
                raise AssertionError("completed private outcome has no terminal harm")
            safe_value = 1.0 - outcome.terminal_harm
        else:
            safe_value = 0.0
        cached_completed_suffix = any(
            previous.candidate_id == selected_id
            and previous.status is SuffixAuditStatus.COMPLETED
            for previous in disclosures
        )
        return AuditDisclosure(
            round_index=round_index,
            candidate_id=selected_id,
            selected_probability=probabilities[selected_id],
            target_mass=candidate.target_mass,
            round_max_importance_weight=maximum_importance_weight(
                self._candidates, probabilities
            ),
            propensity_vector_sha256=propensity_vector_sha256(probabilities),
            status=outcome.status,
            safe_value=safe_value,
            outcome_sha256=canonical_sha256(outcome),
            suffix_tool_steps=0 if cached_completed_suffix else outcome.suffix_tool_steps,
            generation_tokens=0 if cached_completed_suffix else outcome.generation_tokens,
        )

    def _append_next_disclosure(self, ledger: AuditLedger) -> AuditLedger:
        disclosure = self._next_disclosure(ledger.disclosures)
        return AuditLedger(
            protocol_id=ledger.protocol_id,
            cohort_id=ledger.cohort_id,
            policy=ledger.policy,
            allocation_seed=ledger.allocation_seed,
            alpha=ledger.alpha,
            exploration_epsilon=ledger.exploration_epsilon,
            disclosures=(*ledger.disclosures, disclosure),
        )

    def step(self, ledger: AuditLedger) -> tuple[AuditLedger, CertificatePoint]:
        self.validate_ledger(ledger)
        updated = self._append_next_disclosure(ledger)
        return updated, current_certificate(self.envelope, updated)

    def run(
        self, *, total_rounds: int, ledger: AuditLedger | None = None
    ) -> tuple[AuditLedger, tuple[CertificatePoint, ...]]:
        if total_rounds < 0:
            raise ValueError("total_rounds must be nonnegative")
        current = self.initial_ledger() if ledger is None else ledger
        self.validate_ledger(current)
        if len(current.disclosures) > total_rounds:
            raise ValueError("resume ledger already exceeds total_rounds")
        if total_rounds > len(current.disclosures) and not self._candidates:
            raise ValueError("cohort has no auditable suffix candidates")
        disclosures = list(current.disclosures)
        while len(disclosures) < total_rounds:
            disclosures.append(self._next_disclosure(disclosures))
        current = AuditLedger(
            protocol_id=current.protocol_id,
            cohort_id=current.cohort_id,
            policy=current.policy,
            allocation_seed=current.allocation_seed,
            alpha=current.alpha,
            exploration_epsilon=current.exploration_epsilon,
            disclosures=tuple(disclosures),
        )
        return current, certificate_path(self.envelope, current)


__all__ = ["CensureAuditor", "InMemoryEvaluationOracle", "SelectedSuffixOracle"]
