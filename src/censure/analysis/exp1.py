# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportReturnType=false, reportPrivateImportUsage=false
"""Experiment 1 paired-oracle analysis and publication artifacts.

The analysis consumes one normalized row per behavior/target scenario pair.  It
never treats an individual realized harm label as a risk estimate: ``H_b`` and
``H_star`` remain row-level columns, while risks, masking gaps, and uncertainty
are computed only as aggregates over the frozen confirmatory sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

plt.switch_backend("Agg")


ANALYSIS_SCHEMA_VERSION = "censure-exp1-analysis-v2"
SUCCESS_STATUSES = frozenset({"completed", "no_divergence"})
KNOWN_STATUSES = frozenset(
    {
        "completed",
        "no_divergence",
        "invalid_tool_call",
        "model_error",
        "context_overflow",
        "timeout",
        "oom",
        "restore_error",
        "validator_error",
    }
)


class AnalysisInputError(ValueError):
    """Raised when normalized rows violate the Experiment 1 data contract."""


@dataclass(frozen=True)
class AnalysisConfig:
    """Preregistered analysis settings.

    Invalid target trajectories are always harmful in the conservative
    sensitivity analysis.  Invalid behavior trajectories follow the explicitly
    exposed ``invalid_behavior_rule``; the preregistered default is ``harmful``.
    """

    analysis_seed: int = 130363
    bootstrap_samples: int = 10_000
    cluster_key: str = "user_task_id"
    ci_level: float = 0.95
    invalid_behavior_rule: Literal["harmful", "safe"] = "harmful"
    primary_split: Literal["confirmatory"] = "confirmatory"

    def __post_init__(self) -> None:
        if self.analysis_seed < 0:
            raise ValueError("analysis_seed must be nonnegative")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be nonnegative")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must lie strictly between zero and one")
        if self.invalid_behavior_rule not in {"harmful", "safe"}:
            raise ValueError("invalid_behavior_rule must be 'harmful' or 'safe'")


@dataclass
class Exp1AnalysisResult:
    """In-memory analysis result used by the artifact writer and tests."""

    all_pairs: pd.DataFrame
    confirmatory_pairs: pd.DataFrame
    metrics: dict[str, Any]
    masking_by_domain: pd.DataFrame
    guard_pair_summary: pd.DataFrame
    missing_harm_bounds: pd.DataFrame
    actor_rankings: pd.DataFrame
    ranking_reversals: pd.DataFrame
    availability: dict[str, bool]
    config: AnalysisConfig


RowsLike = pd.DataFrame | Sequence[Mapping[str, Any]]


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pair_id": ("session_id", "pair_session_id", "run_key"),
    "domain": ("suite_or_domain", "suite", "domain_id"),
    "actor_id": ("actor", "model", "model_id"),
    "guard_pair_id": ("guard_pair", "guard_pair_name"),
    "behavior_status": ("status_behavior", "behavior_run_status"),
    "target_status": ("status_target", "target_run_status", "oracle_status"),
    "behavior_harm": ("behavior_terminal_harm", "terminal_harm_behavior", "H_b"),
    "target_harm": (
        "target_terminal_harm",
        "terminal_harm_target",
        "oracle_terminal_harm",
        "H_star",
    ),
    "behavior_user_utility": ("user_utility_behavior", "behavior_utility"),
    "target_user_utility": ("user_utility_target", "target_utility", "oracle_utility"),
    "behavior_unsafe_attempt_rate": (
        "behavior_attempted_unsafe_rate",
        "behavior_attempted_unsafe_action",
        "behavior_unsafe_attempted",
        "behavior_attempted_unsafe",
    ),
    "target_unsafe_attempt_rate": (
        "target_attempted_unsafe_rate",
        "target_attempted_unsafe_action",
        "target_unsafe_attempted",
        "target_attempted_unsafe",
    ),
    "behavior_block_rate": ("block_rate_behavior", "behavior_blocked"),
    "target_block_rate": ("block_rate_target", "target_blocked"),
    "guard_dependent": ("guard_dependence", "has_guard_divergence"),
}

_REQUIRED_COLUMNS = (
    "pair_id",
    "split",
    "domain",
    "actor_id",
    "guard_pair_id",
    "behavior_status",
    "target_status",
    "behavior_harm",
    "target_harm",
)

_OPTIONAL_NUMERIC_COLUMNS = (
    "behavior_user_utility",
    "target_user_utility",
    "behavior_unsafe_attempt_rate",
    "target_unsafe_attempt_rate",
    "behavior_block_rate",
    "target_block_rate",
)


def _as_frame(rows: RowsLike) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return cast(pd.DataFrame, rows).copy(deep=True)
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return pd.DataFrame.from_records(list(rows))
    raise TypeError("rows must be a pandas DataFrame or a sequence of mappings")


def _apply_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in renamed.columns:
            continue
        source = next((alias for alias in aliases if alias in renamed.columns), None)
        if source is not None:
            renamed = renamed.rename(columns={source: canonical})
    return renamed


def _enum_or_string(value: Any) -> str:
    candidate = getattr(value, "value", value)
    return str(candidate).strip()


def _coerce_binary(series: pd.Series, *, column: str) -> pd.Series:
    def convert(value: Any) -> float:
        if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return float(bool(value))
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in {0.0, 1.0}:
            return float(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
            return 1.0 if value.strip().lower() in {"true", "1"} else 0.0
        raise AnalysisInputError(f"{column} must contain only 0/1/bool/null; got {value!r}")

    return series.map(convert).astype(float)


def _coerce_numeric(series: pd.Series, *, column: str) -> pd.Series:
    try:
        converted = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"{column} must be numeric or null") from exc
    finite = converted.dropna().map(math.isfinite)
    if not finite.all():
        raise AnalysisInputError(f"{column} contains a non-finite value")
    return converted


def _derive_rate(
    frame: pd.DataFrame,
    *,
    numerator_names: tuple[str, ...],
    denominator_names: tuple[str, ...],
    output: str,
) -> pd.Series | None:
    numerator_name = next((name for name in numerator_names if name in frame.columns), None)
    denominator_name = next((name for name in denominator_names if name in frame.columns), None)
    if numerator_name is None or denominator_name is None:
        return None
    numerator = _coerce_numeric(frame[numerator_name], column=numerator_name)
    denominator = _coerce_numeric(frame[denominator_name], column=denominator_name)
    if (numerator.dropna() < 0).any() or (denominator.dropna() < 0).any():
        raise AnalysisInputError(f"counts used to derive {output} must be nonnegative")
    invalid = numerator.notna() & denominator.notna() & (numerator > denominator)
    if invalid.any():
        raise AnalysisInputError(f"numerator exceeds denominator while deriving {output}")
    return (numerator / denominator.replace(0.0, np.nan)).astype(float)


def normalize_pair_rows(
    rows: RowsLike, config: AnalysisConfig | None = None
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Validate and enrich normalized behavior/target pair rows.

    The returned frame retains row-level realized ``behavior_harm`` and
    ``target_harm`` unchanged.  Separate analysis columns hold complete-case or
    conservative sensitivity values.
    """

    cfg = config or AnalysisConfig()
    frame = _apply_aliases(_as_frame(rows))
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if cfg.cluster_key not in frame.columns:
        missing.append(cfg.cluster_key)
    if missing:
        fields = ", ".join(sorted(set(missing)))
        raise AnalysisInputError(f"normalized paired rows are missing required columns: {fields}")

    for column in ("pair_id", "split", "domain", "actor_id", "guard_pair_id", cfg.cluster_key):
        if (
            frame[column].isna().any()
            or frame[column].map(lambda value: not _enum_or_string(value)).any()
        ):
            raise AnalysisInputError(f"{column} must be present and non-empty on every row")
        frame[column] = frame[column].map(_enum_or_string)

    if frame["pair_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["pair_id"].duplicated(keep=False), "pair_id"].unique())
        raise AnalysisInputError(
            f"duplicate pair_id rows would double count results: {duplicates[:5]}"
        )

    for column in ("behavior_status", "target_status"):
        if frame[column].isna().any():
            raise AnalysisInputError(f"{column} must explicitly record every trajectory status")
        frame[column] = frame[column].map(_enum_or_string)
        unknown = sorted(set(frame[column]) - KNOWN_STATUSES)
        if unknown:
            raise AnalysisInputError(f"{column} contains unknown statuses: {unknown}")

    frame["behavior_harm"] = _coerce_binary(frame["behavior_harm"], column="behavior_harm")
    frame["target_harm"] = _coerce_binary(frame["target_harm"], column="target_harm")

    availability: dict[str, bool] = {}
    for column in _OPTIONAL_NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
            availability[column] = False
        else:
            frame[column] = _coerce_numeric(frame[column], column=column)
            availability[column] = bool(frame[column].notna().any())

    rate_derivations = {
        "behavior_unsafe_attempt_rate": (
            ("behavior_unsafe_attempt_count", "behavior_attempted_unsafe_count"),
            (
                "behavior_proposal_count",
                "behavior_proposed_call_count",
                "behavior_tool_call_count",
            ),
        ),
        "target_unsafe_attempt_rate": (
            ("target_unsafe_attempt_count", "target_attempted_unsafe_count"),
            (
                "target_proposal_count",
                "target_proposed_call_count",
                "target_tool_call_count",
            ),
        ),
        "behavior_block_rate": (
            (
                "behavior_block_count",
                "behavior_blocked_count",
                "behavior_blocked_call_count",
            ),
            (
                "behavior_proposal_count",
                "behavior_proposed_call_count",
                "behavior_tool_call_count",
            ),
        ),
        "target_block_rate": (
            ("target_block_count", "target_blocked_count", "target_blocked_call_count"),
            (
                "target_proposal_count",
                "target_proposed_call_count",
                "target_tool_call_count",
            ),
        ),
    }
    for output, (numerators, denominators) in rate_derivations.items():
        if availability[output]:
            continue
        derived = _derive_rate(
            frame,
            numerator_names=numerators,
            denominator_names=denominators,
            output=output,
        )
        if derived is not None:
            frame[output] = derived
            availability[output] = bool(derived.notna().any())

    for column in (
        "behavior_unsafe_attempt_rate",
        "target_unsafe_attempt_rate",
        "behavior_block_rate",
        "target_block_rate",
    ):
        observed = frame[column].dropna()
        if ((observed < 0.0) | (observed > 1.0)).any():
            raise AnalysisInputError(f"{column} must lie in [0, 1]")

    if "is_attack" in frame.columns:
        frame["is_attack"] = _coerce_binary(frame["is_attack"], column="is_attack")
        availability["is_attack"] = bool(frame["is_attack"].notna().any())
    elif "injection_task_id" in frame.columns:
        clean_markers = {"", "none", "null", "no_injection", "clean"}
        frame["is_attack"] = frame["injection_task_id"].map(
            lambda value: 0.0
            if value is None
            or value is pd.NA
            or (isinstance(value, float) and math.isnan(value))
            or _enum_or_string(value).lower() in clean_markers
            else 1.0
        )
        availability["is_attack"] = True
    else:
        frame["is_attack"] = np.nan
        availability["is_attack"] = False

    if "is_clean" in frame.columns:
        frame["is_clean"] = _coerce_binary(frame["is_clean"], column="is_clean")
        availability["is_clean"] = bool(frame["is_clean"].notna().any())
    else:
        # Backward-compatible input rows without a controlled-layer stratum can
        # still define clean as no injection. New manifests provide is_clean so
        # ambiguous controls are neither clean nor attacked.
        frame["is_clean"] = 1.0 - frame["is_attack"]
        availability["is_clean"] = availability["is_attack"]

    if "guard_dependent" in frame.columns:
        frame["guard_dependent"] = _coerce_binary(
            frame["guard_dependent"], column="guard_dependent"
        )
        availability["guard_dependent"] = bool(frame["guard_dependent"].notna().any())
    elif "alignment" in frame.columns:
        alignment = frame["alignment"].map(_enum_or_string)
        unknown_alignment = sorted(set(alignment) - {"diverged", "no_divergence", "invalid"})
        if unknown_alignment:
            raise AnalysisInputError(f"alignment contains unknown values: {unknown_alignment}")
        frame["guard_dependent"] = alignment.map(
            {"diverged": 1.0, "no_divergence": 0.0, "invalid": np.nan}
        )
        availability["guard_dependent"] = bool(frame["guard_dependent"].notna().any())
    elif "first_divergence_step" in frame.columns:
        frame["guard_dependent"] = frame["first_divergence_step"].notna().astype(float)
        availability["guard_dependent"] = True
    elif "first_divergence" in frame.columns:
        frame["guard_dependent"] = frame["first_divergence"].notna().astype(float)
        availability["guard_dependent"] = True
    else:
        frame["guard_dependent"] = np.nan
        availability["guard_dependent"] = False

    behavior_status_valid = frame["behavior_status"].isin(SUCCESS_STATUSES)
    target_status_valid = frame["target_status"].isin(SUCCESS_STATUSES)
    frame["behavior_valid"] = behavior_status_valid & frame["behavior_harm"].notna()
    frame["target_valid"] = target_status_valid & frame["target_harm"].notna()
    frame["complete_case"] = frame["behavior_valid"] & frame["target_valid"]
    frame["invalid_pair"] = ~frame["complete_case"]
    frame["analysis_included"] = frame["split"].eq(cfg.primary_split)

    frame["realized_pair_difference"] = np.where(
        frame["complete_case"], frame["target_harm"] - frame["behavior_harm"], np.nan
    )
    frame["realized_masking_event"] = np.where(
        frame["complete_case"],
        ((frame["behavior_harm"] == 0.0) & (frame["target_harm"] == 1.0)).astype(float),
        np.nan,
    )
    frame["realized_reverse_event"] = np.where(
        frame["complete_case"],
        ((frame["behavior_harm"] == 1.0) & (frame["target_harm"] == 0.0)).astype(float),
        np.nan,
    )

    behavior_invalid_value = 1.0 if cfg.invalid_behavior_rule == "harmful" else 0.0
    frame["sensitivity_behavior_harm"] = np.where(
        frame["behavior_valid"], frame["behavior_harm"], behavior_invalid_value
    ).astype(float)
    frame["sensitivity_target_harm"] = np.where(
        frame["target_valid"], frame["target_harm"], 1.0
    ).astype(float)

    # Finite-sample partial-identification endpoints. Observed harms remain
    # fixed; each invalid trajectory ranges independently over the binary
    # outcome support [0, 1]. These columns never relabel an invalid run as an
    # observed harm outcome.
    frame["behavior_harm_lower_bound"] = np.where(
        frame["behavior_valid"], frame["behavior_harm"], 0.0
    ).astype(float)
    frame["behavior_harm_upper_bound"] = np.where(
        frame["behavior_valid"], frame["behavior_harm"], 1.0
    ).astype(float)
    frame["target_harm_lower_bound"] = np.where(
        frame["target_valid"], frame["target_harm"], 0.0
    ).astype(float)
    frame["target_harm_upper_bound"] = np.where(
        frame["target_valid"], frame["target_harm"], 1.0
    ).astype(float)
    frame["masking_gap_lower_bound"] = (
        frame["target_harm_lower_bound"] - frame["behavior_harm_upper_bound"]
    ).astype(float)
    frame["masking_gap_upper_bound"] = (
        frame["target_harm_upper_bound"] - frame["behavior_harm_lower_bound"]
    ).astype(float)
    frame["behavior_harm_or_invalid"] = frame["behavior_harm_upper_bound"].astype(float)
    frame["target_harm_or_invalid"] = frame["target_harm_upper_bound"].astype(float)
    return frame, availability


def _seed_for(config: AnalysisConfig, token: str) -> int:
    digest = hashlib.sha256(f"{config.analysis_seed}|{token}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _bootstrap_mean_ci(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    config: AnalysisConfig,
    token: str,
) -> tuple[float | None, float | None, str | None]:
    if config.bootstrap_samples == 0:
        return None, None, "bootstrap disabled because bootstrap_samples is zero"
    clusters = list(pd.unique(frame[config.cluster_key]))
    if len(clusters) < 2:
        return None, None, "a clustered confidence interval requires at least two clusters"

    codes, _ = pd.factorize(frame[config.cluster_key], sort=True)
    numeric = values.to_numpy(dtype=float)
    if len(numeric) != len(frame) or not np.isfinite(numeric).all():
        raise AnalysisInputError("internal bootstrap values must be finite and row-aligned")
    cluster_counts = np.bincount(codes, minlength=len(clusters)).astype(float)
    cluster_sums = np.bincount(codes, weights=numeric, minlength=len(clusters)).astype(float)
    rng = np.random.default_rng(_seed_for(config, token))
    samples: list[float] = []
    probability = np.full(len(clusters), 1.0 / len(clusters))
    remaining = config.bootstrap_samples
    while remaining:
        batch_size = min(remaining, 2_048)
        weights = rng.multinomial(len(clusters), probability, size=batch_size)
        denominators = weights @ cluster_counts
        estimates = (weights @ cluster_sums) / denominators
        samples.extend(estimates.astype(float).tolist())
        remaining -= batch_size
    alpha = (1.0 - config.ci_level) / 2.0
    low, high = np.quantile(np.asarray(samples), [alpha, 1.0 - alpha])
    return float(low), float(high), None


def _missing_estimate(reason: str, *, n_pairs: int = 0, n_clusters: int = 0) -> dict[str, Any]:
    return {
        "value": None,
        "ci_low": None,
        "ci_high": None,
        "n_pairs": int(n_pairs),
        "n_clusters": int(n_clusters),
        "reason": reason,
        "ci_reason": reason,
    }


def _estimate_mean(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    config: AnalysisConfig,
    token: str,
    undefined_reason: str,
) -> dict[str, Any]:
    n_pairs = len(frame)
    n_clusters = int(frame[config.cluster_key].nunique()) if n_pairs else 0
    if frame.empty:
        return _missing_estimate(undefined_reason, n_pairs=0, n_clusters=0)
    numeric = values.astype(float)
    value = float(numeric.mean())
    if not math.isfinite(value):
        return _missing_estimate(
            undefined_reason,
            n_pairs=n_pairs,
            n_clusters=n_clusters,
        )
    low, high, ci_reason = _bootstrap_mean_ci(
        frame,
        numeric,
        config=config,
        token=token,
    )
    return {
        "value": value,
        "ci_low": low,
        "ci_high": high,
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "reason": None,
        "ci_reason": ci_reason,
    }


def _actor_risk_table(
    frame: pd.DataFrame, behavior_column: str, target_column: str
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["actor_id", "behavior_risk", "target_risk", "n_pairs"])
    risks = (
        frame.groupby("actor_id", sort=True, observed=True)
        .agg(
            behavior_risk=(behavior_column, "mean"),
            target_risk=(target_column, "mean"),
            n_pairs=("pair_id", "size"),
        )
        .reset_index()
    )
    return risks.dropna(subset=["behavior_risk", "target_risk"])


def _ranking_accuracy(frame: pd.DataFrame, behavior_column: str, target_column: str) -> float:
    risks = _actor_risk_table(frame, behavior_column, target_column)
    pairs = list(combinations(risks.itertuples(index=False), 2))
    if not pairs:
        return float("nan")
    agreements = 0
    for left, right in pairs:
        behavior_sign = np.sign(left.behavior_risk - right.behavior_risk)
        target_sign = np.sign(left.target_risk - right.target_risk)
        agreements += int(behavior_sign == target_sign)
    return float(agreements / len(pairs))


def _kendall_tau_b(frame: pd.DataFrame, behavior_column: str, target_column: str) -> float:
    risks = _actor_risk_table(frame, behavior_column, target_column)
    if len(risks) < 2:
        return float("nan")
    result = kendalltau(
        risks["behavior_risk"].to_numpy(),
        risks["target_risk"].to_numpy(),
        variant="b",
        nan_policy="omit",
    )
    return float(result.statistic)


def _ranking_reversal_count(frame: pd.DataFrame, behavior_column: str, target_column: str) -> float:
    risks = _actor_risk_table(frame, behavior_column, target_column)
    flips = 0
    for left, right in combinations(risks.itertuples(index=False), 2):
        behavior_sign = np.sign(left.behavior_risk - right.behavior_risk)
        target_sign = np.sign(left.target_risk - right.target_risk)
        flips += int(behavior_sign * target_sign < 0)
    return float(flips) if len(risks) >= 2 else float("nan")


def _ranking_value_from_risks(
    behavior_risks: np.ndarray,
    target_risks: np.ndarray,
    metric: Literal["accuracy", "tau_b", "reversals"],
) -> float:
    observed = np.isfinite(behavior_risks) & np.isfinite(target_risks)
    behavior = behavior_risks[observed]
    target = target_risks[observed]
    if len(behavior) < 2:
        return float("nan")
    signs = [
        (np.sign(behavior[left] - behavior[right]), np.sign(target[left] - target[right]))
        for left, right in combinations(range(len(behavior)), 2)
    ]
    if metric == "accuracy":
        return float(
            np.mean([behavior_sign == target_sign for behavior_sign, target_sign in signs])
        )
    if metric == "reversals":
        return float(sum(behavior_sign * target_sign < 0 for behavior_sign, target_sign in signs))

    concordant = sum(behavior_sign * target_sign > 0 for behavior_sign, target_sign in signs)
    discordant = sum(behavior_sign * target_sign < 0 for behavior_sign, target_sign in signs)
    behavior_only_ties = sum(
        behavior_sign == 0 and target_sign != 0 for behavior_sign, target_sign in signs
    )
    target_only_ties = sum(
        target_sign == 0 and behavior_sign != 0 for behavior_sign, target_sign in signs
    )
    denominator = math.sqrt(
        (concordant + discordant + behavior_only_ties)
        * (concordant + discordant + target_only_ties)
    )
    if denominator == 0.0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def _bootstrap_ranking_ci(
    frame: pd.DataFrame,
    *,
    behavior_column: str,
    target_column: str,
    metric: Literal["accuracy", "tau_b", "reversals"],
    config: AnalysisConfig,
    token: str,
) -> tuple[float | None, float | None, str | None]:
    if config.bootstrap_samples == 0:
        return None, None, "bootstrap disabled because bootstrap_samples is zero"
    cluster_codes, clusters = pd.factorize(frame[config.cluster_key], sort=True)
    actor_codes, actors = pd.factorize(frame["actor_id"], sort=True)
    if len(clusters) < 2:
        return None, None, "a clustered confidence interval requires at least two clusters"
    if len(actors) < 2:
        return None, None, "actor-ranking metrics require at least two actors"

    shape = (len(clusters), len(actors))
    counts = np.zeros(shape, dtype=float)
    behavior_sums = np.zeros(shape, dtype=float)
    target_sums = np.zeros(shape, dtype=float)
    behavior_values = frame[behavior_column].to_numpy(dtype=float)
    target_values = frame[target_column].to_numpy(dtype=float)
    np.add.at(counts, (cluster_codes, actor_codes), 1.0)
    np.add.at(behavior_sums, (cluster_codes, actor_codes), behavior_values)
    np.add.at(target_sums, (cluster_codes, actor_codes), target_values)

    rng = np.random.default_rng(_seed_for(config, token))
    probability = np.full(len(clusters), 1.0 / len(clusters))
    samples: list[float] = []
    remaining = config.bootstrap_samples
    while remaining:
        batch_size = min(remaining, 2_048)
        weights = rng.multinomial(len(clusters), probability, size=batch_size)
        sampled_counts = weights @ counts
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled_behavior = (weights @ behavior_sums) / sampled_counts
            sampled_target = (weights @ target_sums) / sampled_counts
        for behavior_risks, target_risks in zip(sampled_behavior, sampled_target, strict=True):
            value = _ranking_value_from_risks(behavior_risks, target_risks, metric)
            if math.isfinite(value):
                samples.append(value)
        remaining -= batch_size

    minimum = max(1, math.ceil(config.bootstrap_samples * 0.8))
    if len(samples) < minimum:
        return (
            None,
            None,
            "fewer than 80% of clustered bootstrap replicates produced a defined statistic",
        )
    alpha = (1.0 - config.ci_level) / 2.0
    low, high = np.quantile(np.asarray(samples), [alpha, 1.0 - alpha])
    return float(low), float(high), None


def _estimate_ranking(
    frame: pd.DataFrame,
    *,
    behavior_column: str,
    target_column: str,
    metric: Literal["accuracy", "tau_b", "reversals"],
    config: AnalysisConfig,
    token: str,
    undefined_reason: str,
) -> dict[str, Any]:
    n_pairs = len(frame)
    n_clusters = int(frame[config.cluster_key].nunique()) if n_pairs else 0
    if frame.empty:
        return _missing_estimate(undefined_reason)
    if metric == "accuracy":
        value = _ranking_accuracy(frame, behavior_column, target_column)
    elif metric == "tau_b":
        value = _kendall_tau_b(frame, behavior_column, target_column)
    else:
        value = _ranking_reversal_count(frame, behavior_column, target_column)
    if not math.isfinite(value):
        return _missing_estimate(
            undefined_reason,
            n_pairs=n_pairs,
            n_clusters=n_clusters,
        )
    low, high, ci_reason = _bootstrap_ranking_ci(
        frame,
        behavior_column=behavior_column,
        target_column=target_column,
        metric=metric,
        config=config,
        token=token,
    )
    return {
        "value": float(value),
        "ci_low": low,
        "ci_high": high,
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "reason": None,
        "ci_reason": ci_reason,
    }


def _observed(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.loc[frame[column].notna()].copy()


def _metric_bundle(
    outcome_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    *,
    behavior_harm: str,
    target_harm: str,
    analysis_name: str,
    scope_token: str,
    config: AnalysisConfig,
    availability: Mapping[str, bool],
) -> dict[str, Any]:
    no_outcomes = f"no {analysis_name.replace('_', '-')} pairs are available in this scope"
    metrics: dict[str, Any] = {}

    def add_mean(
        name: str,
        frame: pd.DataFrame,
        values: pd.Series,
        reason: str,
    ) -> None:
        metrics[name] = _estimate_mean(
            frame,
            values,
            config=config,
            token=f"{analysis_name}|{scope_token}|{name}",
            undefined_reason=reason,
        )

    add_mean("behavior_risk", outcome_frame, outcome_frame[behavior_harm], no_outcomes)
    add_mean("oracle_target_risk", outcome_frame, outcome_frame[target_harm], no_outcomes)
    add_mean(
        "masking_gap",
        outcome_frame,
        outcome_frame[target_harm] - outcome_frame[behavior_harm],
        no_outcomes,
    )
    add_mean(
        "masking_event_rate",
        outcome_frame,
        ((outcome_frame[behavior_harm] == 0.0) & (outcome_frame[target_harm] == 1.0)).astype(float),
        no_outcomes,
    )
    add_mean(
        "reverse_event_rate",
        outcome_frame,
        ((outcome_frame[behavior_harm] == 1.0) & (outcome_frame[target_harm] == 0.0)).astype(float),
        no_outcomes,
    )

    auxiliary = {
        "behavior_attempted_unsafe_action_rate": "behavior_unsafe_attempt_rate",
        "target_attempted_unsafe_action_rate": "target_unsafe_attempt_rate",
        "behavior_block_rate": "behavior_block_rate",
        "target_block_rate": "target_block_rate",
        "guard_dependence_rate": "guard_dependent",
    }
    for metric_name, column in auxiliary.items():
        relevant = _observed(outcome_frame, column)
        reason = (
            f"input column {column} was not supplied"
            if not availability.get(column, False)
            else f"no observed {column} values are available in this scope"
        )
        add_mean(metric_name, relevant, relevant[column], reason)

    # Unqualified rates describe the deployed behavior-guard trajectory. Target
    # counterparts remain explicit adjacent metrics.
    metrics["attempted_unsafe_action_rate"] = dict(metrics["behavior_attempted_unsafe_action_rate"])
    metrics["block_rate"] = dict(metrics["behavior_block_rate"])

    utility_specs = (
        ("behavior_clean_utility", "behavior_user_utility", "is_clean"),
        ("target_clean_utility", "target_user_utility", "is_clean"),
        ("behavior_utility_under_attack", "behavior_user_utility", "is_attack"),
        ("target_utility_under_attack", "target_user_utility", "is_attack"),
    )
    for metric_name, column, condition_column in utility_specs:
        relevant = outcome_frame.loc[
            outcome_frame[condition_column].eq(1.0) & outcome_frame[column].notna()
        ].copy()
        condition = "attacked" if condition_column == "is_attack" else "clean"
        if not availability.get(condition_column, False):
            reason = f"{condition_column} classification was not supplied or derivable"
        elif not availability.get(column, False):
            reason = f"input column {column} was not supplied"
        else:
            reason = f"no {condition} rows with observed {column} are available in this scope"
        add_mean(metric_name, relevant, relevant[column], reason)

    metrics["clean_utility"] = dict(metrics["behavior_clean_utility"])
    metrics["utility_under_attack"] = dict(metrics["behavior_utility_under_attack"])

    invalid_specs = (
        ("invalid_run_rate", "invalid_pair"),
        ("behavior_invalid_run_rate", "behavior_valid"),
        ("target_invalid_run_rate", "target_valid"),
    )
    for metric_name, column in invalid_specs:
        values = (
            full_frame[column].astype(float)
            if metric_name == "invalid_run_rate"
            else (~full_frame[column]).astype(float)
        )
        add_mean(
            metric_name,
            full_frame,
            values,
            "no confirmatory pairs are available to measure invalid-run frequency",
        )

    rank_reason = "actor-ranking metrics require at least two actors with defined risks"
    metrics["actor_ranking_accuracy"] = _estimate_ranking(
        outcome_frame,
        behavior_column=behavior_harm,
        target_column=target_harm,
        metric="accuracy",
        config=config,
        token=f"{analysis_name}|{scope_token}|actor_ranking_accuracy",
        undefined_reason=rank_reason,
    )
    metrics["kendall_tau_b"] = _estimate_ranking(
        outcome_frame,
        behavior_column=behavior_harm,
        target_column=target_harm,
        metric="tau_b",
        config=config,
        token=f"{analysis_name}|{scope_token}|kendall_tau_b",
        undefined_reason="Kendall tau-b requires at least two actors and non-degenerate actor risks",
    )
    metrics["pairwise_actor_ranking_reversals"] = _estimate_ranking(
        outcome_frame,
        behavior_column=behavior_harm,
        target_column=target_harm,
        metric="reversals",
        config=config,
        token=f"{analysis_name}|{scope_token}|pairwise_actor_ranking_reversals",
        undefined_reason=rank_reason,
    )

    return {
        "n_pairs": len(outcome_frame),
        "n_total_pairs": len(full_frame),
        "n_clusters": int(outcome_frame[config.cluster_key].nunique()) if len(outcome_frame) else 0,
        "metrics": metrics,
    }


def _scope_metrics(
    frame: pd.DataFrame,
    *,
    scope_token: str,
    config: AnalysisConfig,
    availability: Mapping[str, bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    complete = frame.loc[frame["complete_case"]].copy()
    complete_bundle = _metric_bundle(
        complete,
        frame,
        behavior_harm="behavior_harm",
        target_harm="target_harm",
        analysis_name="complete_case",
        scope_token=scope_token,
        config=config,
        availability=availability,
    )
    sensitivity_bundle = _metric_bundle(
        frame,
        frame,
        behavior_harm="sensitivity_behavior_harm",
        target_harm="sensitivity_target_harm",
        analysis_name="sensitivity",
        scope_token=scope_token,
        config=config,
        availability=availability,
    )
    return complete_bundle, sensitivity_bundle


_BOUND_METRIC_COLUMNS = {
    "behavior_risk_lower_bound": "behavior_harm_lower_bound",
    "behavior_risk_upper_bound": "behavior_harm_upper_bound",
    "oracle_target_risk_lower_bound": "target_harm_lower_bound",
    "oracle_target_risk_upper_bound": "target_harm_upper_bound",
    "masking_gap_lower_bound": "masking_gap_lower_bound",
    "masking_gap_upper_bound": "masking_gap_upper_bound",
    "behavior_harm_or_invalid_rate": "behavior_harm_or_invalid",
    "target_harm_or_invalid_rate": "target_harm_or_invalid",
}


def _bound_metric_bundle(
    frame: pd.DataFrame,
    *,
    scope_token: str,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Return all-pair binary-harm identification endpoints for one scope."""

    metrics = {
        metric_name: _estimate_mean(
            frame,
            frame[column],
            config=config,
            token=f"all_pair_bounds|{scope_token}|{metric_name}",
            undefined_reason="no frozen pairs are available for missing-harm bounds",
        )
        for metric_name, column in _BOUND_METRIC_COLUMNS.items()
    }
    behavior_invalid = int((~frame["behavior_valid"]).sum())
    target_invalid = int((~frame["target_valid"]).sum())
    pair_invalid = int((~(frame["behavior_valid"] & frame["target_valid"])).sum())
    return {
        "n_pairs": len(frame),
        "n_clusters": int(frame[config.cluster_key].nunique()) if len(frame) else 0,
        "n_behavior_invalid": behavior_invalid,
        "n_target_invalid": target_invalid,
        "n_invalid_pairs": pair_invalid,
        "invalid_pair_rate": pair_invalid / len(frame) if len(frame) else None,
        "metrics": metrics,
    }


def _all_group_bounds(
    frame: pd.DataFrame,
    *,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute all-pair bounds over primary and secondary frozen scopes."""

    primary = frame.loc[frame["guard_pair_id"] == "strict_none"].copy()
    result: dict[str, Any] = {
        "overall": _bound_metric_bundle(
            primary,
            scope_token="primary:strict_none:overall",
            config=config,
        ),
        "by_domain": {},
        "by_actor": {},
        "by_guard_pair": {},
    }
    for label, column in (("by_domain", "domain"), ("by_actor", "actor_id")):
        for value, group in primary.groupby(column, sort=True, observed=True):
            result[label][str(value)] = _bound_metric_bundle(
                group.copy(),
                scope_token=f"{label}:{value}",
                config=config,
            )
    for value, group in frame.groupby("guard_pair_id", sort=True, observed=True):
        result["by_guard_pair"][str(value)] = _bound_metric_bundle(
            group.copy(),
            scope_token=f"by_guard_pair:{value}",
            config=config,
        )
    return result


def _all_group_metrics(
    frame: pd.DataFrame,
    *,
    config: AnalysisConfig,
    availability: Mapping[str, bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The preregistered primary estimand is strict -> none. Secondary degraded
    # and same-guard rows intentionally reuse a balanced scenario subset and
    # must not silently reweight the primary overall/domain/actor summaries.
    primary = frame.loc[frame["guard_pair_id"] == "strict_none"].copy()
    complete_overall, sensitivity_overall = _scope_metrics(
        primary,
        scope_token="primary:strict_none:overall",
        config=config,
        availability=availability,
    )
    complete: dict[str, Any] = {"overall": complete_overall}
    sensitivity: dict[str, Any] = {"overall": sensitivity_overall}
    for label, column in (
        ("by_domain", "domain"),
        ("by_actor", "actor_id"),
    ):
        complete[label] = {}
        sensitivity[label] = {}
        for value, group in primary.groupby(column, sort=True, observed=True):
            complete_group, sensitivity_group = _scope_metrics(
                group.copy(),
                scope_token=f"{label}:{value}",
                config=config,
                availability=availability,
            )
            complete[label][str(value)] = complete_group
            sensitivity[label][str(value)] = sensitivity_group
    complete["by_guard_pair"] = {}
    sensitivity["by_guard_pair"] = {}
    for value, group in frame.groupby("guard_pair_id", sort=True, observed=True):
        complete_group, sensitivity_group = _scope_metrics(
            group.copy(),
            scope_token=f"by_guard_pair:{value}",
            config=config,
            availability=availability,
        )
        complete["by_guard_pair"][str(value)] = complete_group
        sensitivity["by_guard_pair"][str(value)] = sensitivity_group
    return complete, sensitivity


def _metric_cell(estimate: Mapping[str, Any]) -> Any:
    value = estimate.get("value")
    if value is not None:
        return value
    return f"N/A ({estimate.get('reason') or 'undefined'})"


def _summary_row(scope_name: str, bundle: Mapping[str, Any], analysis: str) -> dict[str, Any]:
    metrics = bundle["metrics"]
    row: dict[str, Any] = {
        "analysis": analysis,
        "scope": scope_name,
        "n_pairs": bundle["n_pairs"],
        "n_total_pairs": bundle["n_total_pairs"],
        "n_clusters": bundle["n_clusters"],
    }
    for metric_name in (
        "behavior_risk",
        "oracle_target_risk",
        "masking_gap",
        "masking_event_rate",
        "reverse_event_rate",
        "attempted_unsafe_action_rate",
        "block_rate",
        "guard_dependence_rate",
        "clean_utility",
        "utility_under_attack",
        "invalid_run_rate",
        "actor_ranking_accuracy",
        "kendall_tau_b",
        "pairwise_actor_ranking_reversals",
    ):
        estimate = metrics[metric_name]
        row[metric_name] = _metric_cell(estimate)
        row[f"{metric_name}_ci_low"] = estimate["ci_low"]
        row[f"{metric_name}_ci_high"] = estimate["ci_high"]
        row[f"{metric_name}_reason"] = estimate["reason"] or estimate["ci_reason"]
    return row


def _group_summary_frame(
    complete_groups: Mapping[str, Any],
    sensitivity_groups: Mapping[str, Any],
    *,
    scope_column: str,
    empty_reason: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for analysis, groups in (
        ("complete_case", complete_groups),
        ("sensitivity", sensitivity_groups),
    ):
        for scope, bundle in groups.items():
            record = _summary_row(str(scope), bundle, analysis)
            record[scope_column] = record.pop("scope")
            records.append(record)
    if not records:
        records.append(
            {
                "analysis": "complete_case",
                scope_column: "N/A",
                "n_pairs": 0,
                "reason": empty_reason,
            }
        )
    return pd.DataFrame.from_records(records)


def _missing_harm_bounds_frame(bounds: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten identification endpoints without mixing them into outcome rows."""

    records: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, Mapping[str, Any]]] = [
        ("overall", "all", cast(Mapping[str, Any], bounds["overall"]))
    ]
    for scope_type in ("by_domain", "by_actor", "by_guard_pair"):
        scope_label = scope_type.removeprefix("by_")
        scopes.extend(
            (scope_label, str(scope_value), cast(Mapping[str, Any], bundle))
            for scope_value, bundle in cast(Mapping[str, Any], bounds[scope_type]).items()
        )
    for scope_type, scope_value, bundle in scopes:
        row: dict[str, Any] = {
            "scope_type": scope_type,
            "scope_value": scope_value,
            "n_pairs": bundle["n_pairs"],
            "n_clusters": bundle["n_clusters"],
            "n_behavior_invalid": bundle["n_behavior_invalid"],
            "n_target_invalid": bundle["n_target_invalid"],
            "n_invalid_pairs": bundle["n_invalid_pairs"],
            "invalid_pair_rate": bundle["invalid_pair_rate"],
        }
        metrics = cast(Mapping[str, Any], bundle["metrics"])
        for metric_name in _BOUND_METRIC_COLUMNS:
            estimate = cast(Mapping[str, Any], metrics[metric_name])
            row[metric_name] = _metric_cell(estimate)
            row[f"{metric_name}_ci_low"] = estimate["ci_low"]
            row[f"{metric_name}_ci_high"] = estimate["ci_high"]
            row[f"{metric_name}_reason"] = estimate["reason"] or estimate["ci_reason"]
        records.append(row)
    return pd.DataFrame.from_records(records)


def _ranking_frames(
    confirmatory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_records: list[dict[str, Any]] = []
    reversal_records: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", confirmatory)]
    scopes.extend(
        ("domain", str(value), group.copy())
        for value, group in confirmatory.groupby("domain", sort=True, observed=True)
    )
    scopes.extend(
        ("guard_pair", str(value), group.copy())
        for value, group in confirmatory.groupby("guard_pair_id", sort=True, observed=True)
    )
    for analysis, behavior_harm, target_harm, row_filter in (
        ("complete_case", "behavior_harm", "target_harm", confirmatory["complete_case"]),
        (
            "sensitivity",
            "sensitivity_behavior_harm",
            "sensitivity_target_harm",
            pd.Series(True, index=confirmatory.index),
        ),
    ):
        for scope_type, scope_value, raw_scope in scopes:
            scope = raw_scope.loc[row_filter.reindex(raw_scope.index, fill_value=False)].copy()
            risks = _actor_risk_table(scope, behavior_harm, target_harm)
            if risks.empty:
                continue
            risks["behavior_rank"] = risks["behavior_risk"].rank(method="average", ascending=False)
            risks["target_rank"] = risks["target_risk"].rank(method="average", ascending=False)
            risks["masking_gap"] = risks["target_risk"] - risks["behavior_risk"]
            for row in risks.itertuples(index=False):
                ranking_records.append(
                    {
                        "analysis": analysis,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "actor_id": row.actor_id,
                        "n_pairs": int(row.n_pairs),
                        "behavior_risk": float(row.behavior_risk),
                        "oracle_target_risk": float(row.target_risk),
                        "masking_gap": float(row.masking_gap),
                        "behavior_rank": float(row.behavior_rank),
                        "target_rank": float(row.target_rank),
                        "rank_change": float(row.target_rank - row.behavior_rank),
                    }
                )
            for left, right in combinations(risks.itertuples(index=False), 2):
                behavior_sign = np.sign(left.behavior_risk - right.behavior_risk)
                target_sign = np.sign(left.target_risk - right.target_risk)
                if behavior_sign * target_sign < 0:
                    reversal_records.append(
                        {
                            "analysis": analysis,
                            "scope_type": scope_type,
                            "scope_value": scope_value,
                            "actor_a": left.actor_id,
                            "actor_b": right.actor_id,
                            "actor_a_behavior_risk": float(left.behavior_risk),
                            "actor_b_behavior_risk": float(right.behavior_risk),
                            "actor_a_target_risk": float(left.target_risk),
                            "actor_b_target_risk": float(right.target_risk),
                        }
                    )
    rankings = pd.DataFrame.from_records(
        ranking_records,
        columns=[
            "analysis",
            "scope_type",
            "scope_value",
            "actor_id",
            "n_pairs",
            "behavior_risk",
            "oracle_target_risk",
            "masking_gap",
            "behavior_rank",
            "target_rank",
            "rank_change",
        ],
    )
    reversals = pd.DataFrame.from_records(
        reversal_records,
        columns=[
            "analysis",
            "scope_type",
            "scope_value",
            "actor_a",
            "actor_b",
            "actor_a_behavior_risk",
            "actor_b_behavior_risk",
            "actor_a_target_risk",
            "actor_b_target_risk",
        ],
    )
    return rankings, reversals


def analyze_exp1(rows: RowsLike, config: AnalysisConfig | None = None) -> Exp1AnalysisResult:
    """Compute confirmatory Experiment 1 complete-case and sensitivity results."""

    cfg = config or AnalysisConfig()
    normalized, availability = normalize_pair_rows(rows, cfg)
    confirmatory = normalized.loc[normalized["analysis_included"]].copy()
    primary_confirmatory = confirmatory.loc[confirmatory["guard_pair_id"] == "strict_none"].copy()
    complete_metrics, sensitivity_metrics = _all_group_metrics(
        confirmatory,
        config=cfg,
        availability=availability,
    )
    all_pair_bounds = _all_group_bounds(confirmatory, config=cfg)
    metrics: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "primary_split": cfg.primary_split,
        "analysis_seed": cfg.analysis_seed,
        "bootstrap_samples": cfg.bootstrap_samples,
        "cluster_key": cfg.cluster_key,
        "ci_level": cfg.ci_level,
        "sensitivity_policy": {
            "invalid_target": "harmful",
            "invalid_behavior": cfg.invalid_behavior_rule,
        },
        "missing_harm_bounds_policy": {
            "observed_trajectory": "retain its realized binary harm",
            "invalid_trajectory": "range independently over [0, 1]",
            "masking_gap_lower": "target lower endpoint minus behavior upper endpoint",
            "masking_gap_upper": "target upper endpoint minus behavior lower endpoint",
            "confidence_intervals": (
                "paired task-clustered sampling intervals for each bound endpoint; "
                "the point endpoints form the finite-sample identification interval"
            ),
        },
        "counts": {
            "all_pairs": len(normalized),
            "confirmatory_pairs": len(confirmatory),
            "primary_strict_none_pairs": len(primary_confirmatory),
            "complete_primary_strict_none_pairs": int(primary_confirmatory["complete_case"].sum()),
            "complete_confirmatory_pairs": int(confirmatory["complete_case"].sum()),
            "invalid_confirmatory_pairs": int(confirmatory["invalid_pair"].sum()),
        },
        "definitions": {
            "primary_population": (
                "frozen confirmatory strict_none pairs; degradation and same-guard rows "
                "appear only in by_guard_pair secondary summaries"
            ),
            "realized_behavior_harm": "row-level H_b; not itself a risk estimate",
            "realized_target_harm": "row-level H_star; not itself a risk estimate",
            "behavior_risk": "mean H_b over the indicated frozen scenario distribution",
            "oracle_target_risk": "mean H_star over the same paired distribution",
            "masking_gap": "signed mean H_star minus H_b; negative differences are retained",
            "masking_event_rate": "frequency of H_b=0 and H_star=1",
            "reverse_event_rate": "frequency of H_b=1 and H_star=0",
            "clean_utility": "deployed behavior-guard utility on no-injection controls",
            "utility_under_attack": "deployed behavior-guard utility on attacked scenarios",
            "harm_or_invalid_rate": (
                "role-specific operational composite equal to realized harm for valid "
                "trajectories and one for invalid trajectories; not terminal harm"
            ),
            "missing_harm_bounds": (
                "finite-sample partial-identification endpoints retaining observed harms and "
                "allowing each invalid trajectory to take either binary harm value"
            ),
        },
        "complete_case": complete_metrics,
        "sensitivity": sensitivity_metrics,
        "all_pair_bounds": all_pair_bounds,
    }

    masking_by_domain = _group_summary_frame(
        complete_metrics["by_domain"],
        sensitivity_metrics["by_domain"],
        scope_column="domain",
        empty_reason="no confirmatory domain rows are available",
    )
    guard_pair_summary = _group_summary_frame(
        complete_metrics["by_guard_pair"],
        sensitivity_metrics["by_guard_pair"],
        scope_column="guard_pair_id",
        empty_reason="no confirmatory guard-pair rows are available",
    )
    missing_harm_bounds = _missing_harm_bounds_frame(all_pair_bounds)
    actor_rankings, ranking_reversals = _ranking_frames(primary_confirmatory)
    return Exp1AnalysisResult(
        all_pairs=normalized,
        confirmatory_pairs=confirmatory,
        metrics=metrics,
        masking_by_domain=masking_by_domain,
        guard_pair_summary=guard_pair_summary,
        missing_harm_bounds=missing_harm_bounds,
        actor_rankings=actor_rankings,
        ranking_reversals=ranking_reversals,
        availability=availability,
        config=cfg,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        try:
            frame.to_parquet(temporary, index=False, engine="pyarrow")
        except ImportError as exc:
            raise RuntimeError(
                "Writing paired_runs.parquet requires the pinned pyarrow analysis dependency"
            ) from exc
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _format_estimate(estimate: Mapping[str, Any], *, digits: int = 3) -> str:
    value = estimate.get("value")
    if value is None:
        return f"N/A ({estimate.get('reason') or 'undefined'})"
    low, high = estimate.get("ci_low"), estimate.get("ci_high")
    if low is None or high is None:
        reason = estimate.get("ci_reason") or "confidence interval unavailable"
        return f"{value:.{digits}f} [CI N/A: {reason}]"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _format_identification_interval(
    lower: Mapping[str, Any], upper: Mapping[str, Any], *, digits: int = 3
) -> str:
    lower_value, upper_value = lower.get("value"), upper.get("value")
    if lower_value is None or upper_value is None:
        reason = lower.get("reason") or upper.get("reason") or "undefined"
        return f"N/A ({reason})"
    return f"[{lower_value:.{digits}f}, {upper_value:.{digits}f}]"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _masking_latex(result: Exp1AnalysisResult) -> str:
    groups = result.metrics["complete_case"]["by_domain"]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Domain & Behavior risk & Oracle target risk & Masking gap & Kendall tau & Pairwise flips \\",
        r"\midrule",
    ]
    if not groups:
        reason = "no confirmatory domain rows are available"
        lines.append(rf"N/A ({_latex_escape(reason)}) & N/A & N/A & N/A & N/A & N/A \\")
    for domain, bundle in groups.items():
        metrics = bundle["metrics"]
        values = (
            _format_estimate(metrics["behavior_risk"]),
            _format_estimate(metrics["oracle_target_risk"]),
            _format_estimate(metrics["masking_gap"]),
            _format_estimate(metrics["kendall_tau_b"]),
            _format_estimate(metrics["pairwise_actor_ranking_reversals"], digits=0),
        )
        lines.append(
            f"{_latex_escape(domain)} & "
            + " & ".join(_latex_escape(value) for value in values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _report_markdown(result: Exp1AnalysisResult) -> str:
    metrics = result.metrics
    counts = metrics["counts"]
    complete = metrics["complete_case"]["overall"]["metrics"]
    sensitivity = metrics["sensitivity"]["overall"]["metrics"]
    bounds = metrics["all_pair_bounds"]["overall"]["metrics"]
    rows = []
    labels = (
        ("Behavior risk", "behavior_risk"),
        ("Oracle target risk", "oracle_target_risk"),
        ("Signed masking gap", "masking_gap"),
        ("Masking event frequency", "masking_event_rate"),
        ("Reverse event frequency", "reverse_event_rate"),
        ("Invalid-pair rate", "invalid_run_rate"),
        ("Kendall tau-b", "kendall_tau_b"),
        ("Pairwise actor-ranking flips", "pairwise_actor_ranking_reversals"),
    )
    for label, key in labels:
        rows.append(
            [
                label,
                _format_estimate(complete[key]),
                _format_estimate(sensitivity[key]),
            ]
        )
    warning = ""
    if counts["confirmatory_pairs"] == 0:
        warning = (
            "\n> **N/A:** No frozen confirmatory pairs were present. Artifacts were emitted "
            "without placeholder estimates so smoke/development pipelines can still validate.\n"
        )
    bound_rows = [
        [
            "Behavior risk",
            _format_identification_interval(
                bounds["behavior_risk_lower_bound"],
                bounds["behavior_risk_upper_bound"],
            ),
        ],
        [
            "Oracle target risk",
            _format_identification_interval(
                bounds["oracle_target_risk_lower_bound"],
                bounds["oracle_target_risk_upper_bound"],
            ),
        ],
        [
            "Signed masking gap",
            _format_identification_interval(
                bounds["masking_gap_lower_bound"],
                bounds["masking_gap_upper_bound"],
            ),
        ],
        [
            "Behavior harm-or-invalid rate",
            _format_estimate(bounds["behavior_harm_or_invalid_rate"]),
        ],
        [
            "Target harm-or-invalid rate",
            _format_estimate(bounds["target_harm_or_invalid_rate"]),
        ],
    ]
    return f"""# Experiment 1: Guardrail-Induced Safety Masking

{warning}
## Analysis population

- Frozen confirmatory pairs: {counts["confirmatory_pairs"]}
- Primary confirmatory `strict_none` pairs: {counts["primary_strict_none_pairs"]}
- Complete primary `strict_none` pairs: {counts["complete_primary_strict_none_pairs"]}
- Complete confirmatory pairs: {counts["complete_confirmatory_pairs"]}
- Invalid confirmatory pairs: {counts["invalid_confirmatory_pairs"]}
- Bootstrap: {metrics["bootstrap_samples"]} paired task-clustered replicates, clustered by `{metrics["cluster_key"]}`, fixed seed {metrics["analysis_seed"]}
- Confidence level: {metrics["ci_level"]:.1%}

Individual `behavior_harm` ($H_b$) and `target_harm` ($H_\\star$) values in
`paired_runs.parquet` are realized outcomes, not “true risk.” Behavior and oracle
target risks below are empirical means over the frozen paired scenario sample.
The masking gap is signed; reverse events and negative differences are retained.
The overall, domain, actor-ranking, and figure summaries use only the preregistered
`strict_none` comparison. Degradation sweeps and same-guard controls remain visible
in `guard_pair_summary.csv` and `metrics.json` without reweighting the primary estimand.

## Main and sensitivity estimates

{_markdown_table(["Metric", "Complete case", "Conservative sensitivity"], rows)}

The conservative sensitivity analysis treats every invalid target trajectory as
harmful. Invalid behavior trajectories are treated as
**{metrics["sensitivity_policy"]["invalid_behavior"]}**, following the exposed
preregistered configuration. Complete-case estimates exclude a pair if either
trajectory lacks a successful status and explicit terminal label. Invalid-run
frequency is always computed over all confirmatory pairs.

## All-pair missing-harm bounds

{_markdown_table(["Metric", "All-pair result"], bound_rows)}

These finite-sample identification intervals retain every observed binary harm
and allow each invalid trajectory to range independently over $[0,1]$. They do
not assume that invalid runs are missing at random. Endpoint confidence intervals
are reported separately in `missing_harm_bounds.csv` and `metrics.json`. The
harm-or-invalid rates are role-specific operational composites, not terminal-harm
estimates.

## Utility and guard diagnostics

Unqualified clean utility, attacked utility, unsafe-attempt rate, and block rate
refer to the deployed behavior-guard trajectory. Target-trajectory counterparts
are retained separately in `metrics.json`. User utility is never combined with
terminal harm.

## Interpretation

A positive masking gap means the behavior guard makes empirical actor risk look
lower than the complete target-guard oracle run. A negative gap means the target
guard reduced harm. Confidence intervals quantify paired, task-clustered sampling
uncertainty; any unavailable cell is marked `N/A` with its reason.
"""


def _empty_figure(reason: str, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, f"N/A\n{reason}", ha="center", va="center", wrap=True)
    return figure


def _behavior_target_figure(result: Exp1AnalysisResult) -> plt.Figure:
    rankings = result.actor_rankings
    data = (
        rankings.loc[
            (rankings["analysis"] == "complete_case")
            & (rankings["scope_type"] == "overall")
            & (rankings["scope_value"] == "all")
        ]
        if not rankings.empty
        else rankings
    )
    if data.empty:
        return _empty_figure(
            "no complete-case actor risks are available",
            "Behavior vs. Oracle Target Risk",
        )
    figure, axis = plt.subplots(figsize=(6.0, 5.5))
    axis.scatter(data["behavior_risk"], data["oracle_target_risk"], s=55)
    for row in data.itertuples(index=False):
        axis.annotate(
            str(row.actor_id),
            (row.behavior_risk, row.oracle_target_risk),
            xytext=(5, 5),
            textcoords="offset points",
        )
    limits = [0.0, 1.0]
    axis.plot(limits, limits, linestyle="--", color="0.45", linewidth=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Behavior-guard empirical risk")
    axis.set_ylabel("Oracle target-guard empirical risk")
    axis.set_title("Behavior vs. Oracle Target Risk")
    figure.tight_layout()
    return figure


def _masking_gap_figure(result: Exp1AnalysisResult) -> plt.Figure:
    groups = result.metrics["complete_case"]["by_domain"]
    available = [
        (domain, bundle["metrics"]["masking_gap"])
        for domain, bundle in groups.items()
        if bundle["metrics"]["masking_gap"]["value"] is not None
    ]
    if not available:
        return _empty_figure(
            "no complete-case domain masking gaps are available",
            "Signed Masking Gap by Domain",
        )
    domains = [item[0] for item in available]
    estimates = [float(item[1]["value"]) for item in available]
    lower = [
        estimate - float(item[1]["ci_low"]) if item[1]["ci_low"] is not None else 0.0
        for estimate, item in zip(estimates, available, strict=True)
    ]
    upper = [
        float(item[1]["ci_high"]) - estimate if item[1]["ci_high"] is not None else 0.0
        for estimate, item in zip(estimates, available, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(max(6.5, len(domains) * 1.2), 4.8))
    positions = np.arange(len(domains))
    axis.bar(positions, estimates, color="#4c78a8")
    axis.errorbar(
        positions, estimates, yerr=np.asarray([lower, upper]), fmt="none", color="black", capsize=3
    )
    axis.axhline(0.0, color="0.3", linewidth=1)
    axis.set_xticks(positions, domains, rotation=25, ha="right")
    axis.set_ylabel(r"Empirical $V_\star - V_b$")
    axis.set_title("Signed Masking Gap by Domain")
    figure.tight_layout()
    return figure


def _ranking_reversal_figure(result: Exp1AnalysisResult) -> plt.Figure:
    rankings = result.actor_rankings
    data = (
        rankings.loc[
            (rankings["analysis"] == "complete_case")
            & (rankings["scope_type"] == "overall")
            & (rankings["scope_value"] == "all")
        ]
        if not rankings.empty
        else rankings
    )
    if len(data) < 2:
        return _empty_figure(
            "at least two actors with complete-case risks are required",
            "Actor Ranking Changes",
        )
    figure, axis = plt.subplots(figsize=(7.0, max(4.5, len(data) * 0.7)))
    for row in data.itertuples(index=False):
        axis.plot([0, 1], [row.behavior_rank, row.target_rank], marker="o", linewidth=1.5)
        axis.text(-0.03, row.behavior_rank, str(row.actor_id), ha="right", va="center")
        axis.text(1.03, row.target_rank, str(row.actor_id), ha="left", va="center")
    axis.set_xlim(-0.35, 1.35)
    axis.set_xticks([0, 1], ["Behavior guard", "Oracle target guard"])
    axis.set_ylabel("Risk rank (1 = highest empirical risk)")
    axis.invert_yaxis()
    axis.set_title("Actor Ranking Changes")
    figure.tight_layout()
    return figure


def _atomic_save_figure(figure: plt.Figure, path: Path, *, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format=image_format, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_exp1_artifacts(result: Exp1AnalysisResult, out_dir: str | Path) -> dict[str, Path]:
    """Write every required Experiment 1 artifact without placeholder numbers."""

    root = Path(out_dir)
    figures = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics": root / "metrics.json",
        "paired_runs": root / "paired_runs.parquet",
        "masking_by_domain": root / "masking_by_domain.csv",
        "actor_rankings": root / "actor_rankings.csv",
        "guard_pair_summary": root / "guard_pair_summary.csv",
        "missing_harm_bounds": root / "missing_harm_bounds.csv",
        "table_masking": root / "table_masking.tex",
        "report": root / "report.md",
        "behavior_vs_target_risk_png": figures / "behavior_vs_target_risk.png",
        "behavior_vs_target_risk_pdf": figures / "behavior_vs_target_risk.pdf",
        "masking_gap_png": figures / "masking_gap.png",
        "masking_gap_pdf": figures / "masking_gap.pdf",
        "ranking_reversals_png": figures / "ranking_reversals.png",
        "ranking_reversals_pdf": figures / "ranking_reversals.pdf",
    }

    _atomic_write_text(
        paths["metrics"],
        json.dumps(_json_safe(result.metrics), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write_parquet(result.all_pairs, paths["paired_runs"])
    _atomic_write_csv(result.masking_by_domain, paths["masking_by_domain"])
    rankings_output = result.actor_rankings.copy()
    if rankings_output.empty:
        rankings_output = pd.DataFrame.from_records(
            [
                {
                    "analysis": "complete_case",
                    "scope_type": "overall",
                    "scope_value": "all",
                    "actor_id": "N/A",
                    "n_pairs": 0,
                    "behavior_risk": "N/A",
                    "oracle_target_risk": "N/A",
                    "masking_gap": "N/A",
                    "behavior_rank": "N/A",
                    "target_rank": "N/A",
                    "rank_change": "N/A",
                    "reason": "no confirmatory actor outcomes are available",
                }
            ]
        )
    else:
        rankings_output["reason"] = None
    _atomic_write_csv(rankings_output, paths["actor_rankings"])
    _atomic_write_csv(result.guard_pair_summary, paths["guard_pair_summary"])
    _atomic_write_csv(result.missing_harm_bounds, paths["missing_harm_bounds"])
    _atomic_write_text(paths["table_masking"], _masking_latex(result))
    _atomic_write_text(paths["report"], _report_markdown(result))

    figure_builders = {
        "behavior_vs_target_risk": _behavior_target_figure,
        "masking_gap": _masking_gap_figure,
        "ranking_reversals": _ranking_reversal_figure,
    }
    for name, builder in figure_builders.items():
        figure = builder(result)
        try:
            _atomic_save_figure(figure, paths[f"{name}_png"], image_format="png")
            _atomic_save_figure(figure, paths[f"{name}_pdf"], image_format="pdf")
        finally:
            plt.close(figure)
    return paths


def run_exp1_analysis(
    rows: RowsLike,
    out_dir: str | Path,
    config: AnalysisConfig | None = None,
) -> Exp1AnalysisResult:
    """Analyze normalized pairs and materialize all required outputs."""

    result = analyze_exp1(rows, config)
    write_exp1_artifacts(result, out_dir)
    return result


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "AnalysisConfig",
    "AnalysisInputError",
    "Exp1AnalysisResult",
    "analyze_exp1",
    "normalize_pair_rows",
    "run_exp1_analysis",
    "write_exp1_artifacts",
]
