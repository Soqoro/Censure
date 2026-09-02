"""Stochastic shared-support OPE diagnostics and hybrid risk composition."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field

from censure.estimation.calibration import clopper_pearson_one_sided
from censure.schemas import FrozenModel, Identifier, Probability, Sha256Hex
from censure.serialization import canonical_json_bytes, canonical_sha256


class SharedSupportModelCondition(str, Enum):
    CORRECT = "correct"
    MISSPECIFIED = "misspecified"
    CONSTANT = "constant"


class SharedSupportCellSpec(FrozenModel):
    schema_version: Literal["censure.shared-support-cell.v1"] = (
        "censure.shared-support-cell.v1"
    )
    protocol_id: Identifier
    seed_namespace: Identifier
    base_seed: Annotated[int, Field(ge=0)]
    cohort_size: Annotated[int, Field(ge=1)] = 1000
    repetitions: Annotated[int, Field(ge=1)] = 2000
    max_importance_ratio: Annotated[float, Field(ge=1.0)]
    model_condition: SharedSupportModelCondition
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05

    @property
    def cell_id(self) -> str:
        return canonical_sha256(self)


class SharedSupportReplicateResult(FrozenModel):
    schema_version: Literal["censure.shared-support-replicate.v1"] = (
        "censure.shared-support-replicate.v1"
    )
    cell_id: Sha256Hex
    repetition_index: Annotated[int, Field(ge=0)]
    exact_target_risk: Probability
    behavior_risk: Probability
    ips: Annotated[float, Field(ge=0.0)]
    snips: Probability
    direct_method: Probability
    sequential_doubly_robust: float
    ips_ucb: Probability
    ips_covered: bool
    theoretical_max_importance_ratio: Annotated[float, Field(ge=1.0)]
    observed_max_importance_ratio: Annotated[float, Field(ge=0.0)]
    effective_sample_size: Annotated[float, Field(ge=0.0)]


class SharedSupportSummary(FrozenModel):
    schema_version: Literal["censure.shared-support-summary.v1"] = (
        "censure.shared-support-summary.v1"
    )
    cell_id: Sha256Hex
    max_importance_ratio: Annotated[float, Field(ge=1.0)]
    model_condition: SharedSupportModelCondition
    repetition_count: Annotated[int, Field(ge=1)]
    ips_bias: float
    ips_rmse: Annotated[float, Field(ge=0.0)]
    snips_bias: float
    snips_rmse: Annotated[float, Field(ge=0.0)]
    direct_method_bias: float
    direct_method_rmse: Annotated[float, Field(ge=0.0)]
    sequential_doubly_robust_bias: float
    sequential_doubly_robust_rmse: Annotated[float, Field(ge=0.0)]
    ips_coverage: Probability
    ips_coverage_one_sided_95_lower: Probability
    ips_coverage_one_sided_95_upper: Probability
    mean_effective_sample_size: Annotated[float, Field(ge=0.0)]


def _uniform(payload: dict[str, object]) -> float:
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _draw(
    spec: SharedSupportCellSpec, repetition_index: int, unit_index: int, stream: str
) -> float:
    return _uniform(
        {
            "schema_version": "censure.shared-support-random.v1",
            "seed_namespace": spec.seed_namespace,
            "base_seed": spec.base_seed,
            "repetition_index": repetition_index,
            "unit_index": unit_index,
            "stream": stream,
        }
    )


def _target_action_one_probability(x: int, action_zero: int) -> float:
    state_one = x ^ action_zero
    return 0.75 if state_one else 0.25


def _terminal_harm_probability(x: int, action_zero: int, action_one: int) -> float:
    return 0.05 + 0.10 * x + 0.05 * action_zero + 0.55 * action_one


def _outcome_model(
    condition: SharedSupportModelCondition, x: int, action_zero: int, action_one: int
) -> float:
    if condition is SharedSupportModelCondition.CORRECT:
        return _terminal_harm_probability(x, action_zero, action_one)
    if condition is SharedSupportModelCondition.MISSPECIFIED:
        return min(1.0, 0.15 + 0.10 * x + 0.05 * action_zero)
    return 0.25


def exact_shared_support_target_risk() -> float:
    return math.fsum(
        0.5
        * 0.5
        * (
            target_probability * _terminal_harm_probability(x, action_zero, 1)
            + (1.0 - target_probability)
            * _terminal_harm_probability(x, action_zero, 0)
        )
        for x in (0, 1)
        for action_zero in (0, 1)
        for target_probability in (_target_action_one_probability(x, action_zero),)
    )


def _direct_value(condition: SharedSupportModelCondition, x: int) -> float:
    return math.fsum(
        0.5
        * (
            target_probability * _outcome_model(condition, x, action_zero, 1)
            + (1.0 - target_probability)
            * _outcome_model(condition, x, action_zero, 0)
        )
        for action_zero in (0, 1)
        for target_probability in (_target_action_one_probability(x, action_zero),)
    )


def bounded_score_ucb(*, score_mean: float, score_bound: float, sample_size: int, alpha: float) -> float:
    if not math.isfinite(score_mean) or score_mean < 0.0:
        raise ValueError("score_mean must be finite and nonnegative")
    if not math.isfinite(score_bound) or score_bound <= 0.0:
        raise ValueError("score_bound must be finite and positive")
    if sample_size < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("sample_size/alpha are invalid")
    radius = score_bound * math.sqrt(math.log(1.0 / alpha) / (2.0 * sample_size))
    return min(1.0, score_mean + radius)


def combine_supported_and_frontier_ucbs(
    *, supported_harm_ucb: float, frontier_mass: float, audited_safe_mass_lcb: float
) -> float:
    for name, value in (
        ("supported_harm_ucb", supported_harm_ucb),
        ("frontier_mass", frontier_mass),
        ("audited_safe_mass_lcb", audited_safe_mass_lcb),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if audited_safe_mass_lcb > frontier_mass + 1e-12:
        raise ValueError("audited safe-mass lower bound exceeds frontier mass")
    return min(1.0, supported_harm_ucb + frontier_mass - audited_safe_mass_lcb)


def run_shared_support_repetition(
    spec: SharedSupportCellSpec, repetition_index: int
) -> SharedSupportReplicateResult:
    if not 0 <= repetition_index < spec.repetitions:
        raise ValueError("repetition_index is outside the shared-support cell")
    weighted_outcomes: list[float] = []
    weights: list[float] = []
    outcomes: list[float] = []
    direct_values: list[float] = []
    dr_values: list[float] = []
    for unit_index in range(spec.cohort_size):
        x = int(_draw(spec, repetition_index, unit_index, "context") < 0.5)
        action_zero = int(_draw(spec, repetition_index, unit_index, "action_zero") < 0.5)
        target_probability = _target_action_one_probability(x, action_zero)
        behavior_probability = target_probability / spec.max_importance_ratio
        action_one = int(
            _draw(spec, repetition_index, unit_index, "action_one") < behavior_probability
        )
        outcome = float(
            _draw(spec, repetition_index, unit_index, "outcome")
            < _terminal_harm_probability(x, action_zero, action_one)
        )
        weight = (
            target_probability / behavior_probability
            if action_one
            else (1.0 - target_probability) / (1.0 - behavior_probability)
        )
        model_value = _outcome_model(spec.model_condition, x, action_zero, action_one)
        direct_value = _direct_value(spec.model_condition, x)
        outcomes.append(outcome)
        weights.append(weight)
        weighted_outcomes.append(weight * outcome)
        direct_values.append(direct_value)
        dr_values.append(direct_value + weight * (outcome - model_value))

    ips = statistics.fmean(weighted_outcomes)
    weight_sum = math.fsum(weights)
    snips = math.fsum(weight * outcome for weight, outcome in zip(weights, outcomes, strict=True)) / weight_sum
    direct_method = statistics.fmean(direct_values)
    sequential_doubly_robust = statistics.fmean(dr_values)
    exact_risk = exact_shared_support_target_risk()
    ips_ucb = bounded_score_ucb(
        score_mean=ips,
        score_bound=spec.max_importance_ratio,
        sample_size=spec.cohort_size,
        alpha=spec.alpha,
    )
    effective_sample_size = weight_sum**2 / math.fsum(weight**2 for weight in weights)
    return SharedSupportReplicateResult(
        cell_id=spec.cell_id,
        repetition_index=repetition_index,
        exact_target_risk=exact_risk,
        behavior_risk=statistics.fmean(outcomes),
        ips=ips,
        snips=min(1.0, max(0.0, snips)),
        direct_method=min(1.0, max(0.0, direct_method)),
        sequential_doubly_robust=sequential_doubly_robust,
        ips_ucb=ips_ucb,
        ips_covered=ips_ucb + 1e-12 >= exact_risk,
        theoretical_max_importance_ratio=spec.max_importance_ratio,
        observed_max_importance_ratio=max(weights),
        effective_sample_size=effective_sample_size,
    )


def run_shared_support_cell(
    spec: SharedSupportCellSpec,
) -> tuple[SharedSupportReplicateResult, ...]:
    return tuple(
        run_shared_support_repetition(spec, repetition_index)
        for repetition_index in range(spec.repetitions)
    )


def _bias_rmse(values: Sequence[float], truth: float) -> tuple[float, float]:
    errors = tuple(value - truth for value in values)
    return statistics.fmean(errors), math.sqrt(statistics.fmean(error**2 for error in errors))


def summarize_shared_support_results(
    results: Sequence[SharedSupportReplicateResult],
    *,
    max_importance_ratio: float,
    model_condition: SharedSupportModelCondition,
) -> SharedSupportSummary:
    if not results:
        raise ValueError("cannot summarize empty shared-support results")
    first = results[0]
    if any(row.cell_id != first.cell_id for row in results):
        raise ValueError("shared-support summary requires one homogeneous cell")
    truth = first.exact_target_risk
    ips_bias, ips_rmse = _bias_rmse([row.ips for row in results], truth)
    snips_bias, snips_rmse = _bias_rmse([row.snips for row in results], truth)
    direct_bias, direct_rmse = _bias_rmse(
        [row.direct_method for row in results], truth
    )
    dr_bias, dr_rmse = _bias_rmse(
        [row.sequential_doubly_robust for row in results], truth
    )
    covered = sum(row.ips_covered for row in results)
    coverage_lower, coverage_upper = clopper_pearson_one_sided(covered, len(results))
    return SharedSupportSummary(
        cell_id=first.cell_id,
        max_importance_ratio=max_importance_ratio,
        model_condition=model_condition,
        repetition_count=len(results),
        ips_bias=ips_bias,
        ips_rmse=ips_rmse,
        snips_bias=snips_bias,
        snips_rmse=snips_rmse,
        direct_method_bias=direct_bias,
        direct_method_rmse=direct_rmse,
        sequential_doubly_robust_bias=dr_bias,
        sequential_doubly_robust_rmse=dr_rmse,
        ips_coverage=covered / len(results),
        ips_coverage_one_sided_95_lower=coverage_lower,
        ips_coverage_one_sided_95_upper=coverage_upper,
        mean_effective_sample_size=statistics.fmean(
            row.effective_sample_size for row in results
        ),
    )


__all__ = [
    "SharedSupportCellSpec",
    "SharedSupportModelCondition",
    "SharedSupportReplicateResult",
    "SharedSupportSummary",
    "bounded_score_ucb",
    "combine_supported_and_frontier_ucbs",
    "exact_shared_support_target_risk",
    "run_shared_support_cell",
    "run_shared_support_repetition",
    "summarize_shared_support_results",
]
