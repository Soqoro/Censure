from __future__ import annotations

import itertools

import pytest

from censure.estimation.enumerable import (
    SupportRegime,
    binary_outcome_maps,
    exact_simultaneous_coverage_probability,
    generate_enumerable_cohort,
)
from censure.estimation.schemas import (
    AllocationPolicyName,
    FiniteCohortEnvelope,
    FrontierCandidate,
)


def _tiny_envelope(*, nonauditable: bool = False) -> FiniteCohortEnvelope:
    return FiniteCohortEnvelope(
        protocol_id="phase2-test",
        cohort_id="tiny",
        cohort_size=4,
        supported_harm_contribution=0.25,
        candidates=tuple(
            FrontierCandidate(
                candidate_id=f"candidate-{index}",
                cohort_unit_id=f"unit-{index}",
                source_session_id=f"session-{index}",
                target_mass=0.25,
                auditable=not (nonauditable and index == 1),
                checkpoint_sha256=str(index + 1) * 64,
                suffix_random_tape_id=f"tape-{index}",
                declared_cost=float(index + 1),
                guard_score=float(2 - index),
                stratum=f"stratum-{index}",
            )
            for index in range(2)
        ),
    )


@pytest.mark.parametrize("regime", list(SupportRegime))
@pytest.mark.parametrize("seed", range(10))
def test_enumerable_decomposition_is_exact_and_generation_is_deterministic(
    regime: SupportRegime, seed: int
) -> None:
    kwargs = {
        "protocol_id": "phase2-test",
        "cohort_id": f"cohort-{regime.value}-{seed}",
        "cohort_size": 200,
        "support_regime": regime,
        "target_harm_prevalence": 0.2,
        "zero_support_mass": 0.5,
        "generation_seed": seed,
    }
    first = generate_enumerable_cohort(**kwargs)
    second = generate_enumerable_cohort(**kwargs)

    assert first == second
    assert first.decomposition_error() <= 1e-12
    assert first.envelope().theta_env + 1e-12 >= first.exact_target_risk
    if regime is SupportRegime.FULL_OVERLAP:
        assert not first.envelope().candidates
        assert first.envelope().theta_env == pytest.approx(first.exact_target_risk)
    elif regime is SupportRegime.DETERMINISTIC_CLONEABLE_NONOVERLAP:
        assert all(candidate.auditable for candidate in first.envelope().candidates)
    else:
        assert any(not candidate.auditable for candidate in first.envelope().candidates)


def test_enumerable_longitudinality_has_delayed_terminal_harm() -> None:
    cohort = generate_enumerable_cohort(
        protocol_id="phase2-test",
        cohort_id="longitudinality",
        cohort_size=1000,
        support_regime=SupportRegime.DETERMINISTIC_CLONEABLE_NONOVERLAP,
        target_harm_prevalence=0.5,
        zero_support_mass=0.75,
        generation_seed=91,
    )

    assert cohort.delayed_harm_rate > 0.0
    assert cohort.exact_one_step_risk < cohort.exact_target_risk


@pytest.mark.parametrize("policy", list(AllocationPolicyName))
@pytest.mark.parametrize("outcome_index", range(4))
def test_exact_short_path_enumeration_satisfies_simultaneous_coverage(
    policy: AllocationPolicyName, outcome_index: int
) -> None:
    envelope = _tiny_envelope()
    outcomes = binary_outcome_maps(envelope)[outcome_index]

    coverage = exact_simultaneous_coverage_probability(
        envelope=envelope,
        all_private_outcomes=outcomes,
        policy=policy,
        max_rounds=6,
        alpha=0.05,
    )

    assert 0.95 <= coverage <= 1.0 + 1e-12


def test_exact_enumeration_keeps_nonauditable_safe_mass_in_envelope() -> None:
    envelope = _tiny_envelope(nonauditable=True)
    outcomes = next(
        outcome_map
        for outcome_map in binary_outcome_maps(envelope)
        if all(outcome.terminal_harm == 0.0 for outcome in outcome_map.values())
    )

    coverage = exact_simultaneous_coverage_probability(
        envelope=envelope,
        all_private_outcomes=outcomes,
        policy=AllocationPolicyName.TARGET_MASS,
        max_rounds=6,
    )

    assert coverage >= 0.95
    assert envelope.nonauditable_mass == pytest.approx(0.25)


def test_every_binary_map_is_enumerated_once() -> None:
    envelope = _tiny_envelope()
    maps = binary_outcome_maps(envelope)
    observed = {
        tuple(outcomes[candidate_id].terminal_harm for candidate_id in sorted(outcomes))
        for outcomes in maps
    }
    assert observed == set(itertools.product((0.0, 1.0), repeat=2))
