from __future__ import annotations

from inspect import signature

import pytest
from pydantic import JsonValue, ValidationError

from censure.estimation.allocation import (
    allocation_probabilities,
    deterministic_draw_uniform,
    select_candidate,
)
from censure.estimation.auditor import CensureAuditor, InMemoryEvaluationOracle
from censure.estimation.confidence import stitched_hoeffding_boundary
from censure.estimation.schemas import (
    AllocationPolicyName,
    AuditDisclosure,
    AuditLedger,
    FiniteCohortEnvelope,
    FrontierCandidate,
    PrivateSuffixOutcome,
    SuffixAuditStatus,
)
from censure.estimation.storage import AuditorRunStore
from censure.storage import CorruptArtifactError


def _candidate(
    candidate_id: str,
    *,
    mass: float,
    auditable: bool = True,
    guard_score: float = 1.0,
    cost: float = 1.0,
    stratum: str = "all",
    features: dict[str, JsonValue] | None = None,
) -> FrontierCandidate:
    return FrontierCandidate(
        candidate_id=candidate_id,
        cohort_unit_id=f"unit-{candidate_id}",
        source_session_id=f"session-{candidate_id}",
        target_mass=mass,
        auditable=auditable,
        checkpoint_sha256=(candidate_id[0] * 64),
        suffix_random_tape_id=f"tape-{candidate_id}",
        declared_cost=cost,
        guard_score=guard_score,
        stratum=stratum,
        behavior_features={} if features is None else features,
    )


def _envelope() -> FiniteCohortEnvelope:
    return FiniteCohortEnvelope(
        protocol_id="protocol-v1",
        cohort_id="cohort-v1",
        cohort_size=10,
        supported_harm_contribution=0.1,
        candidates=(
            _candidate("a", mass=0.2, guard_score=1.0, cost=1.0, stratum="x"),
            _candidate("b", mass=0.3, guard_score=3.0, cost=4.0, stratum="y"),
            _candidate("c", mass=0.1, auditable=False),
        ),
    )


def _outcomes() -> dict[str, PrivateSuffixOutcome]:
    return {
        "a": PrivateSuffixOutcome(
            candidate_id="a", status=SuffixAuditStatus.COMPLETED, terminal_harm=0.0
        ),
        "b": PrivateSuffixOutcome(
            candidate_id="b", status=SuffixAuditStatus.COMPLETED, terminal_harm=1.0
        ),
    }


def test_frontier_contract_rejects_outcome_leakage_and_invalid_envelope() -> None:
    with pytest.raises(ValidationError, match="forbidden key"):
        _candidate("a", mass=0.2, features={"nested": {"target_harm": 1}})

    with pytest.raises(ValidationError, match="must be unique"):
        FiniteCohortEnvelope(
            protocol_id="p",
            cohort_id="c",
            cohort_size=1,
            supported_harm_contribution=0.0,
            candidates=(_candidate("a", mass=0.2), _candidate("a", mass=0.2)),
        )

    with pytest.raises(ValidationError, match="exceeds one"):
        FiniteCohortEnvelope(
            protocol_id="p",
            cohort_id="c",
            cohort_size=1,
            supported_harm_contribution=0.8,
            candidates=(_candidate("a", mass=0.3),),
        )


@pytest.mark.parametrize("policy", list(AllocationPolicyName))
def test_allocation_policies_have_full_positive_support(policy: AllocationPolicyName) -> None:
    candidates = _envelope().auditable_candidates
    probabilities = allocation_probabilities(policy, candidates, exploration_epsilon=0.10)

    assert set(probabilities) == {"a", "b"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in probabilities.values())


def test_target_mass_propensity_and_deterministic_draw_are_exactly_replayable() -> None:
    envelope = _envelope()
    probabilities = allocation_probabilities(
        AllocationPolicyName.TARGET_MASS, envelope.auditable_candidates
    )
    assert probabilities == pytest.approx({"a": 0.4, "b": 0.6})

    draw = deterministic_draw_uniform(
        protocol_id=envelope.protocol_id,
        cohort_id=envelope.cohort_id,
        policy=AllocationPolicyName.TARGET_MASS,
        seed=17,
        round_index=1,
    )
    assert draw == deterministic_draw_uniform(
        protocol_id=envelope.protocol_id,
        cohort_id=envelope.cohort_id,
        policy=AllocationPolicyName.TARGET_MASS,
        seed=17,
        round_index=1,
    )
    assert select_candidate(probabilities, uniform_draw=draw) in {"a", "b"}


def test_adaptive_allocation_uses_only_past_disclosures() -> None:
    candidates = _envelope().auditable_candidates
    before = allocation_probabilities(
        AllocationPolicyName.CENSURE_BOUND_TARGETED, candidates
    )
    disclosure = AuditDisclosure(
        round_index=1,
        candidate_id="a",
        selected_probability=0.5,
        target_mass=0.2,
        round_max_importance_weight=0.6,
        propensity_vector_sha256="d" * 64,
        status=SuffixAuditStatus.COMPLETED,
        safe_value=1.0,
        outcome_sha256="e" * 64,
    )
    after = allocation_probabilities(
        AllocationPolicyName.CENSURE_BOUND_TARGETED, candidates, (disclosure,)
    )

    assert before != after
    assert "oracle" not in signature(allocation_probabilities).parameters
    assert "outcomes" not in signature(allocation_probabilities).parameters


def test_zero_budget_and_failed_audits_remain_worst_case() -> None:
    envelope = _envelope()
    failures = {
        candidate.candidate_id: PrivateSuffixOutcome(
            candidate_id=candidate.candidate_id,
            status=SuffixAuditStatus.RESTORE_ERROR,
        )
        for candidate in envelope.auditable_candidates
    }
    auditor = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(failures),
        policy=AllocationPolicyName.UNIFORM,
        allocation_seed=4,
    )
    ledger, points = auditor.run(total_rounds=20)

    assert points[0].target_risk_ucb == pytest.approx(envelope.theta_env)
    assert points[-1].safe_mass_lcb == 0.0
    assert points[-1].target_risk_ucb == pytest.approx(envelope.theta_env)
    assert points[-1].failed_audit_count == 20
    assert all(disclosure.safe_value == 0.0 for disclosure in ledger.disclosures)


def test_certificate_is_anytime_and_nonincreasing() -> None:
    envelope = _envelope()
    auditor = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(_outcomes()),
        policy=AllocationPolicyName.TARGET_MASS,
        allocation_seed=19,
    )
    ledger, points = auditor.run(total_rounds=200)

    upper_bounds = [point.target_risk_ucb for point in points]
    assert upper_bounds == sorted(upper_bounds, reverse=True)
    assert points[-1].safe_mass_lcb > 0.0
    assert points[-1].safe_mass_lcb <= envelope.auditable_mass
    assert points[-1].duplicate_draw_count == 200 - len(
        {disclosure.candidate_id for disclosure in ledger.disclosures}
    )
    assert stitched_hoeffding_boundary(
        round_index=1, cumulative_bound_squared=0.25, alpha=0.05
    ) > 0.0


def test_interrupted_audit_resumes_to_byte_identical_ledger() -> None:
    envelope = _envelope()
    full_auditor = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(_outcomes()),
        policy=AllocationPolicyName.UNCERTAINTY,
        allocation_seed=101,
    )
    full_ledger, full_points = full_auditor.run(total_rounds=40)

    first_auditor = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(_outcomes()),
        policy=AllocationPolicyName.UNCERTAINTY,
        allocation_seed=101,
    )
    partial_ledger, _ = first_auditor.run(total_rounds=13)
    resumed_auditor = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(_outcomes()),
        policy=AllocationPolicyName.UNCERTAINTY,
        allocation_seed=101,
    )
    resumed_ledger, resumed_points = resumed_auditor.run(
        total_rounds=40, ledger=partial_ledger
    )

    assert resumed_ledger == full_ledger
    assert resumed_points == full_points


def test_unselected_private_outcome_cannot_change_first_disclosure() -> None:
    envelope = _envelope()
    auditor_a = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(_outcomes()),
        policy=AllocationPolicyName.UNIFORM,
        allocation_seed=9,
    )
    first_ledger, _ = auditor_a.run(total_rounds=1)
    selected = first_ledger.disclosures[0].candidate_id
    unselected = ({"a", "b"} - {selected}).pop()

    changed = _outcomes()
    changed[unselected] = PrivateSuffixOutcome(
        candidate_id=unselected,
        status=SuffixAuditStatus.COMPLETED,
        terminal_harm=1.0 - (changed[unselected].terminal_harm or 0.0),
    )
    oracle_b = InMemoryEvaluationOracle(changed)
    auditor_b = CensureAuditor(
        envelope=envelope,
        oracle=oracle_b,
        policy=AllocationPolicyName.UNIFORM,
        allocation_seed=9,
    )
    second_ledger, _ = auditor_b.run(total_rounds=1)

    assert second_ledger == first_ledger
    assert oracle_b.requested_candidate_ids == [selected]


def test_tampered_resume_propensity_is_rejected() -> None:
    envelope = _envelope()
    auditor = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(_outcomes()),
        policy=AllocationPolicyName.UNIFORM,
        allocation_seed=3,
    )
    ledger, _ = auditor.run(total_rounds=1)
    raw = ledger.model_dump(mode="python")
    raw["disclosures"][0]["selected_probability"] = 0.6
    raw["disclosures"][0]["round_max_importance_weight"] = 0.6
    tampered = AuditLedger.model_validate(raw)

    with pytest.raises(ValueError, match="propensity does not replay"):
        auditor.validate_ledger(tampered)


def test_auditor_store_is_append_only_checksummed_and_has_no_oracle_reader(tmp_path) -> None:
    envelope = _envelope()
    auditor = CensureAuditor(
        envelope=envelope,
        oracle=InMemoryEvaluationOracle(_outcomes()),
        policy=AllocationPolicyName.TARGET_MASS,
        allocation_seed=55,
    )
    ledger_2, points_2 = auditor.run(total_rounds=2)
    ledger_4, points_4 = auditor.run(total_rounds=4, ledger=ledger_2)
    store = AuditorRunStore(tmp_path, "phase2-test")

    store.write_envelope(envelope)
    store.write_ledger(ledger_2)
    store.write_certificate_path(ledger_2, points_2)
    store.write_ledger(ledger_4)
    store.write_certificate_path(ledger_4, points_4)

    assert store.read_envelope(
        protocol_id=envelope.protocol_id, cohort_id=envelope.cohort_id
    ) == envelope
    assert store.read_ledger(ledger_4) == ledger_4
    assert store.read_certificate_path(ledger_4) == points_4
    assert not hasattr(store, "read_oracle_summary")
    assert not hasattr(store, "evaluation_view")

    with pytest.raises(ValueError, match="cannot truncate"):
        store.write_ledger(ledger_2)

    ledger_path = next(store.root.glob("ledgers/*/ledger.json"))
    ledger_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CorruptArtifactError, match="checksum mismatch"):
        store.read_ledger(ledger_4)
