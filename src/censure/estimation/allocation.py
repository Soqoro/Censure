"""Predictable, outcome-firewalled allocation policies for suffix audits."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

from censure.estimation.schemas import (
    AllocationPolicyName,
    AuditDisclosure,
    FrontierCandidate,
)
from censure.serialization import canonical_json_bytes, canonical_sha256

_ADAPTIVE_POLICIES = frozenset(
    {
        AllocationPolicyName.GUARD_SCORE,
        AllocationPolicyName.UNCERTAINTY,
        AllocationPolicyName.DOWNSTREAM_HARM,
        AllocationPolicyName.CENSURE_BOUND_TARGETED,
    }
)


def _ordered(candidates: Sequence[FrontierCandidate]) -> tuple[FrontierCandidate, ...]:
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))
    if not ordered:
        raise ValueError("allocation requires at least one auditable candidate")
    if any(not candidate.auditable for candidate in ordered):
        raise ValueError("allocation received a non-auditable candidate")
    if len({candidate.candidate_id for candidate in ordered}) != len(ordered):
        raise ValueError("allocation candidate IDs must be unique")
    return ordered


def _normalize_scores(
    candidates: Sequence[FrontierCandidate], scores: Sequence[float]
) -> dict[str, float]:
    if len(candidates) != len(scores):
        raise ValueError("candidate and score lengths differ")
    if any(not math.isfinite(score) or score < 0.0 for score in scores):
        raise ValueError("allocation scores must be finite and nonnegative")
    total = math.fsum(scores)
    if total <= 0.0:
        scores = tuple(candidate.target_mass for candidate in candidates)
        total = math.fsum(scores)
    probabilities = {
        candidate.candidate_id: score / total
        for candidate, score in zip(candidates, scores, strict=True)
    }
    validate_probability_vector(probabilities)
    return probabilities


def _prequential_safe_probability(
    candidate: FrontierCandidate,
    *,
    candidates_by_id: Mapping[str, FrontierCandidate],
    history: Sequence[AuditDisclosure],
) -> float:
    safe_sum = 1.0
    count = 2.0
    for disclosure in history:
        previous = candidates_by_id.get(disclosure.candidate_id)
        if previous is None:
            raise ValueError(f"audit history references unknown candidate {disclosure.candidate_id!r}")
        if previous.stratum == candidate.stratum:
            safe_sum += disclosure.safe_value
            count += 1.0
    return safe_sum / count


def allocation_probabilities(
    policy: AllocationPolicyName | str,
    candidates: Sequence[FrontierCandidate],
    history: Sequence[AuditDisclosure] = (),
    *,
    exploration_epsilon: float = 0.10,
) -> dict[str, float]:
    """Return the predictable propensity vector for the next audit round.

    Only outcome-free candidate records and already-disclosed selected outcomes
    are accepted. The function has no evaluation-store or oracle parameter.
    """

    selected_policy = AllocationPolicyName(policy)
    ordered = _ordered(candidates)
    if not math.isfinite(exploration_epsilon) or not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("exploration_epsilon must lie in [0, 1]")

    if selected_policy is AllocationPolicyName.UNIFORM:
        return _normalize_scores(ordered, (1.0,) * len(ordered))

    mass_probabilities = _normalize_scores(
        ordered, tuple(candidate.target_mass for candidate in ordered)
    )
    if selected_policy is AllocationPolicyName.TARGET_MASS:
        return mass_probabilities

    candidates_by_id = {candidate.candidate_id: candidate for candidate in ordered}
    if selected_policy is AllocationPolicyName.GUARD_SCORE:
        raw_scores = tuple(
            candidate.target_mass * candidate.guard_score for candidate in ordered
        )
    else:
        predicted_safe = tuple(
            _prequential_safe_probability(
                candidate,
                candidates_by_id=candidates_by_id,
                history=history,
            )
            for candidate in ordered
        )
        if selected_policy is AllocationPolicyName.UNCERTAINTY:
            raw_scores = tuple(
                candidate.target_mass * math.sqrt(probability * (1.0 - probability))
                for candidate, probability in zip(ordered, predicted_safe, strict=True)
            )
        elif selected_policy is AllocationPolicyName.DOWNSTREAM_HARM:
            raw_scores = tuple(
                candidate.target_mass * math.sqrt(probability)
                for candidate, probability in zip(ordered, predicted_safe, strict=True)
            )
        elif selected_policy is AllocationPolicyName.CENSURE_BOUND_TARGETED:
            raw_scores = tuple(
                candidate.target_mass * math.sqrt(probability / candidate.declared_cost)
                for candidate, probability in zip(ordered, predicted_safe, strict=True)
            )
        else:  # pragma: no cover - exhaustive enum protection
            raise AssertionError(f"unhandled allocation policy: {selected_policy}")

    score_probabilities = _normalize_scores(ordered, raw_scores)
    probabilities = {
        candidate.candidate_id: (
            exploration_epsilon * mass_probabilities[candidate.candidate_id]
            + (1.0 - exploration_epsilon) * score_probabilities[candidate.candidate_id]
        )
        for candidate in ordered
    }
    if selected_policy not in _ADAPTIVE_POLICIES:  # pragma: no cover - invariant guard
        raise AssertionError(f"policy unexpectedly bypassed the frozen adaptive set: {policy}")
    validate_probability_vector(probabilities, require_positive=True)
    return probabilities


def validate_probability_vector(
    probabilities: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
    require_positive: bool = True,
) -> None:
    if not probabilities:
        raise ValueError("propensity vector is empty")
    for candidate_id, probability in probabilities.items():
        if not candidate_id:
            raise ValueError("propensity vector contains an empty candidate ID")
        if not math.isfinite(probability):
            raise ValueError("propensity vector contains a non-finite value")
        if probability < 0.0 or (require_positive and probability <= 0.0):
            raise ValueError("propensity vector violates positive support")
    if abs(math.fsum(probabilities.values()) - 1.0) > tolerance:
        raise ValueError("propensity vector does not sum to one")


def propensity_vector_sha256(probabilities: Mapping[str, float]) -> str:
    validate_probability_vector(probabilities)
    return canonical_sha256(
        [
            {"candidate_id": candidate_id, "probability": probabilities[candidate_id]}
            for candidate_id in sorted(probabilities)
        ]
    )


def deterministic_draw_uniform(
    *, protocol_id: str, cohort_id: str, policy: AllocationPolicyName, seed: int, round_index: int
) -> float:
    if seed < 0 or round_index < 1:
        raise ValueError("allocation seed must be nonnegative and round_index must be positive")
    payload = {
        "schema_version": "censure.audit-random-draw.v1",
        "protocol_id": protocol_id,
        "cohort_id": cohort_id,
        "policy": policy.value,
        "seed": seed,
        "round_index": round_index,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def select_candidate(probabilities: Mapping[str, float], *, uniform_draw: float) -> str:
    validate_probability_vector(probabilities)
    if not math.isfinite(uniform_draw) or not 0.0 <= uniform_draw < 1.0:
        raise ValueError("uniform_draw must lie in [0, 1)")
    cumulative = 0.0
    ordered_ids = sorted(probabilities)
    for candidate_id in ordered_ids:
        cumulative += probabilities[candidate_id]
        if uniform_draw < cumulative:
            return candidate_id
    return ordered_ids[-1]


def maximum_importance_weight(
    candidates: Sequence[FrontierCandidate], probabilities: Mapping[str, float]
) -> float:
    ordered = _ordered(candidates)
    validate_probability_vector(probabilities)
    if set(probabilities) != {candidate.candidate_id for candidate in ordered}:
        raise ValueError("propensity vector does not match the auditable candidate set")
    return max(
        candidate.target_mass / probabilities[candidate.candidate_id] for candidate in ordered
    )


__all__ = [
    "allocation_probabilities",
    "deterministic_draw_uniform",
    "maximum_importance_weight",
    "propensity_vector_sha256",
    "select_candidate",
    "validate_probability_vector",
]
