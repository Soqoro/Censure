"""Anytime Hoeffding certificate for auditable frontier safe mass."""

from __future__ import annotations

import math

from censure.estimation.schemas import (
    AuditLedger,
    CertificatePoint,
    FiniteCohortEnvelope,
    SuffixAuditStatus,
)


def stitched_hoeffding_boundary(*, round_index: int, cumulative_bound_squared: float, alpha: float) -> float:
    """Return the frozen union-stitched one-sided Hoeffding boundary."""

    if round_index < 1:
        raise ValueError("round_index must be positive")
    if not math.isfinite(cumulative_bound_squared) or cumulative_bound_squared < 0.0:
        raise ValueError("cumulative_bound_squared must be finite and nonnegative")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    log_term = math.log((math.pi**2 * round_index**2) / (6.0 * alpha))
    return math.sqrt(0.5 * cumulative_bound_squared * log_term)


def certificate_path(
    envelope: FiniteCohortEnvelope, ledger: AuditLedger
) -> tuple[CertificatePoint, ...]:
    """Compute the zero-budget and every post-disclosure certificate point."""

    if ledger.protocol_id != envelope.protocol_id:
        raise ValueError("audit ledger protocol does not match the cohort envelope")
    if ledger.cohort_id != envelope.cohort_id:
        raise ValueError("audit ledger cohort does not match the cohort envelope")

    auditable = {candidate.candidate_id: candidate for candidate in envelope.auditable_candidates}
    theta_env = min(1.0, max(0.0, envelope.theta_env))
    target_frontier_mass = min(1.0, max(0.0, envelope.target_frontier_mass))
    auditable_mass = min(1.0, max(0.0, envelope.auditable_mass))
    nonauditable_mass = min(1.0, max(0.0, envelope.nonauditable_mass))

    def make_point(
        *,
        round_index: int,
        cumulative_value: float,
        cumulative_bound_squared: float,
        boundary: float,
        instantaneous_lcb: float,
        safe_mass_lcb: float,
        seen: set[str],
        identified_harm_by_candidate: dict[str, float],
        failed_count: int,
    ) -> CertificatePoint:
        target_risk_ucb = min(1.0, max(0.0, theta_env - safe_mass_lcb))
        identified_target_risk_lcb = min(
            1.0,
            max(
                0.0,
                envelope.supported_harm_contribution
                + math.fsum(identified_harm_by_candidate.values()),
            ),
        )
        return CertificatePoint(
            protocol_id=envelope.protocol_id,
            cohort_id=envelope.cohort_id,
            policy=ledger.policy,
            round_index=round_index,
            alpha=ledger.alpha,
            theta_env=theta_env,
            target_frontier_mass=target_frontier_mass,
            auditable_mass=auditable_mass,
            nonauditable_mass=nonauditable_mass,
            cumulative_importance_safe_value=cumulative_value,
            cumulative_bound_squared=cumulative_bound_squared,
            stitched_boundary=boundary,
            instantaneous_safe_mass_lcb=instantaneous_lcb,
            safe_mass_lcb=safe_mass_lcb,
            target_risk_ucb=target_risk_ucb,
            identified_target_risk_lcb=identified_target_risk_lcb,
            identified_interval_width=max(
                0.0, target_risk_ucb - identified_target_risk_lcb
            ),
            unique_audited_candidate_count=len(seen),
            duplicate_draw_count=round_index - len(seen),
            failed_audit_count=failed_count,
        )

    points = [
        make_point(
            round_index=0,
            cumulative_value=0.0,
            cumulative_bound_squared=0.0,
            boundary=0.0,
            instantaneous_lcb=0.0,
            safe_mass_lcb=0.0,
            seen=set(),
            identified_harm_by_candidate={},
            failed_count=0,
        )
    ]
    cumulative_value = 0.0
    cumulative_bound_squared = 0.0
    running_lcb = 0.0
    seen: set[str] = set()
    identified_harm_by_candidate: dict[str, float] = {}
    failed_count = 0
    for disclosure in ledger.disclosures:
        candidate = auditable.get(disclosure.candidate_id)
        if candidate is None:
            raise ValueError(
                f"audit disclosure references unknown/non-auditable candidate "
                f"{disclosure.candidate_id!r}"
            )
        if abs(candidate.target_mass - disclosure.target_mass) > 1e-12:
            raise ValueError("audit disclosure target mass differs from the frozen candidate")
        cumulative_value += disclosure.importance_safe_value
        cumulative_bound_squared += disclosure.round_max_importance_weight**2
        boundary = stitched_hoeffding_boundary(
            round_index=disclosure.round_index,
            cumulative_bound_squared=cumulative_bound_squared,
            alpha=ledger.alpha,
        )
        instantaneous_lcb = min(
            auditable_mass,
            max(0.0, (cumulative_value - boundary) / disclosure.round_index),
        )
        running_lcb = max(running_lcb, instantaneous_lcb)
        seen.add(disclosure.candidate_id)
        if (
            disclosure.status is SuffixAuditStatus.COMPLETED
            and disclosure.candidate_id not in identified_harm_by_candidate
        ):
            identified_harm_by_candidate[disclosure.candidate_id] = (
                candidate.target_mass * (1.0 - disclosure.safe_value)
            )
        if disclosure.status is not SuffixAuditStatus.COMPLETED:
            failed_count += 1
        points.append(
            make_point(
                round_index=disclosure.round_index,
                cumulative_value=cumulative_value,
                cumulative_bound_squared=cumulative_bound_squared,
                boundary=boundary,
                instantaneous_lcb=instantaneous_lcb,
                safe_mass_lcb=running_lcb,
                seen=seen,
                identified_harm_by_candidate=identified_harm_by_candidate,
                failed_count=failed_count,
            )
        )
    return tuple(points)


def current_certificate(
    envelope: FiniteCohortEnvelope, ledger: AuditLedger
) -> CertificatePoint:
    return certificate_path(envelope, ledger)[-1]


__all__ = ["certificate_path", "current_certificate", "stitched_hoeffding_boundary"]
