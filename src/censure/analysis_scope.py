"""Explicit, outcome-independent actor scopes for partial Experiment 1 analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from censure.config import ConfigurationError, load_yaml
from censure.serialization import canonical_sha256


class FeasibilityExclusion(BaseModel):
    """One prespecified actor excluded without inspecting its harm outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_alias: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_]*$")]
    disposition: Literal["feasibility_deferred"]
    decision_basis: Literal["run_status_and_infrastructure_only"]
    observed_paired_sessions: Annotated[int, Field(ge=1)]
    behavior_invalid_count: Annotated[int, Field(ge=0)]
    oracle_invalid_count: Annotated[int, Field(ge=0)]
    invalid_pair_rate_lower_bound: Annotated[float, Field(ge=0.0, le=1.0)]
    continuation_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    threshold_status: Literal["post_hoc_application_of_pilot_threshold"]
    completed_shards: tuple[Annotated[int, Field(ge=0)], ...]
    planned_shards: Annotated[int, Field(ge=1)]
    outcome_values_inspected: Literal[False]
    rationale: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_evidence(self) -> FeasibilityExclusion:
        if (
            max(self.behavior_invalid_count, self.oracle_invalid_count)
            > self.observed_paired_sessions
        ):
            raise ValueError("invalid trajectory count exceeds observed paired sessions")
        observed_lower_bound = max(self.behavior_invalid_count, self.oracle_invalid_count) / float(
            self.observed_paired_sessions
        )
        if abs(observed_lower_bound - self.invalid_pair_rate_lower_bound) > 1e-12:
            raise ValueError("invalid_pair_rate_lower_bound does not match the status counts")
        if self.invalid_pair_rate_lower_bound <= self.continuation_threshold:
            raise ValueError("feasibility exclusion requires a lower bound above its threshold")
        if len(set(self.completed_shards)) != len(self.completed_shards):
            raise ValueError("completed_shards contains duplicates")
        if any(index >= self.planned_shards for index in self.completed_shards):
            raise ValueError("completed_shards contains an index outside planned_shards")
        return self


class AnalysisScopeConfig(BaseModel):
    """Frozen declaration of a deliberately partial actor analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["censure.analysis-scope.v1"]
    scope_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
    source_experiment_id: Annotated[str, Field(min_length=1)]
    inferential_status: Literal["post_hoc_partial_prespecified_actor_analysis"]
    selection_basis: Literal["completed_actors_after_status_only_feasibility_review"]
    decision_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    decision_timezone: Annotated[str, Field(min_length=1)]
    included_actor_aliases: Annotated[tuple[str, ...], Field(min_length=2)]
    excluded_actors: Annotated[tuple[FeasibilityExclusion, ...], Field(min_length=1)]
    limitations: Annotated[tuple[str, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_actor_sets(self) -> AnalysisScopeConfig:
        included = self.included_actor_aliases
        excluded = tuple(item.actor_alias for item in self.excluded_actors)
        if len(set(included)) != len(included):
            raise ValueError("included_actor_aliases contains duplicates")
        if len(set(excluded)) != len(excluded):
            raise ValueError("excluded_actors contains duplicate aliases")
        overlap = set(included) & set(excluded)
        if overlap:
            raise ValueError(f"actor aliases are both included and excluded: {sorted(overlap)}")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedAnalysisScope:
    config: AnalysisScopeConfig
    sha256: str
    included_actor_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_config": self.config.model_dump(mode="json"),
            "scope_config_sha256": self.sha256,
            "included_actor_ids": list(self.included_actor_ids),
        }


def load_analysis_scope(path: str | Path) -> AnalysisScopeConfig:
    try:
        return AnalysisScopeConfig.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid analysis scope {path}: {exc}") from exc


def resolve_analysis_scope(
    scope: AnalysisScopeConfig,
    experiment_config: Mapping[str, Any],
) -> ResolvedAnalysisScope:
    experiment_id = str(experiment_config.get("experiment_id", ""))
    if scope.source_experiment_id != experiment_id:
        raise ConfigurationError(
            f"analysis scope targets {scope.source_experiment_id!r}, not {experiment_id!r}"
        )
    raw_aliases = experiment_config.get("actors")
    resolved_models = experiment_config.get("resolved_models")
    if not isinstance(raw_aliases, (list, tuple)) or not isinstance(resolved_models, Mapping):
        raise ConfigurationError("resolved experiment config has no actor catalog")
    experiment_aliases = tuple(str(value) for value in raw_aliases)
    included = set(scope.included_actor_aliases)
    excluded = {item.actor_alias for item in scope.excluded_actors}
    unknown = (included | excluded) - set(experiment_aliases)
    if unknown:
        raise ConfigurationError(f"analysis scope contains unknown actors: {sorted(unknown)}")
    uncovered = set(experiment_aliases) - included - excluded
    if uncovered:
        raise ConfigurationError(f"analysis scope does not disposition actors: {sorted(uncovered)}")
    actor_ids: list[str] = []
    for alias in scope.included_actor_aliases:
        model = resolved_models.get(alias)
        if not isinstance(model, Mapping) or not isinstance(model.get("model_id"), str):
            raise ConfigurationError(f"analysis scope actor {alias!r} has no resolved model ID")
        actor_ids.append(str(model["model_id"]))
    if len(set(actor_ids)) != len(actor_ids):
        raise ConfigurationError("analysis scope aliases resolve to duplicate actor IDs")
    return ResolvedAnalysisScope(
        config=scope,
        sha256=canonical_sha256(scope),
        included_actor_ids=tuple(actor_ids),
    )


__all__ = [
    "AnalysisScopeConfig",
    "FeasibilityExclusion",
    "ResolvedAnalysisScope",
    "load_analysis_scope",
    "resolve_analysis_scope",
]
