# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
"""Retrospective, task-paired synthesis of aligned Experiment 1 actor studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from censure.analysis.exp1 import (
    AnalysisConfig,
    AnalysisInputError,
    Exp1AnalysisResult,
    analyze_exp1,
)
from censure.config import ConfigurationError, load_yaml
from censure.provenance import collect_provenance
from censure.serialization import canonical_json, canonical_sha256
from censure.storage import atomic_write_bytes, atomic_write_json

plt.switch_backend("Agg")

SYNTHESIS_SCHEMA_VERSION = "censure.three-model-synthesis.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")]


class SynthesisInputError(ValueError):
    """Raised when source artifacts cannot support the frozen paired synthesis."""


class SourceSpec(BaseModel):
    """One validated experiment/scope contributing actors to the synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: Identifier
    root_key: Identifier
    experiment_id: Annotated[str, Field(min_length=1)]
    manifest_relative_path: Annotated[str, Field(min_length=1)]
    paired_rows_relative_path: Annotated[str, Field(min_length=1)]
    validation_report_relative_path: Annotated[str, Field(min_length=1)]
    context_relative_path: Annotated[str, Field(min_length=1)]
    context_kind: Literal["analysis_scope", "extension_protocol"]
    context_id: Annotated[str, Field(min_length=1)]
    context_sha256: Sha256
    source_inferential_status: Annotated[str, Field(min_length=1)]
    expected_manifest_sha256: Sha256
    expected_normalized_row_count: Annotated[int, Field(ge=1)]
    actor_ids: Annotated[tuple[str, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_paths_and_actors(self) -> SourceSpec:
        for field in (
            "manifest_relative_path",
            "paired_rows_relative_path",
            "validation_report_relative_path",
            "context_relative_path",
        ):
            value = Path(str(getattr(self, field)))
            if value.is_absolute() or ".." in value.parts:
                raise ValueError(f"{field} must be a non-escaping relative path")
        if len(set(self.actor_ids)) != len(self.actor_ids):
            raise ValueError("source actor_ids contains duplicates")
        return self


class ActorSpec(BaseModel):
    """Frozen actor identity and expected sampling counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_id: Annotated[str, Field(min_length=1)]
    display_name: Annotated[str, Field(min_length=1)]
    family: Annotated[str, Field(min_length=1)]
    source_id: Identifier
    expected_pair_count: Annotated[int, Field(ge=1)]
    expected_primary_pair_count: Annotated[int, Field(ge=1)]


class ContrastSpec(BaseModel):
    """One oriented actor-by-guard masking-gap contrast."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contrast_id: Identifier
    minuend_actor_id: Annotated[str, Field(min_length=1)]
    subtrahend_actor_id: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_distinct_actors(self) -> ContrastSpec:
        if self.minuend_actor_id == self.subtrahend_actor_id:
            raise ValueError("a contrast must contain two distinct actors")
        return self


class DegradationLevelSpec(BaseModel):
    """Ordered target-guard relaxation level for the matched dose curve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    guard_pair_id: Identifier
    display_label: Annotated[str, Field(min_length=1)]


class SynthesisAnalysisSpec(BaseModel):
    """Frozen statistical and alignment rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_seed: Annotated[int, Field(ge=0)]
    bootstrap_samples: Annotated[int, Field(ge=0)]
    ci_level: Annotated[float, Field(gt=0.0, lt=1.0)]
    invalid_behavior_rule: Literal["harmful", "safe"]
    primary_split: Literal["confirmatory"]
    primary_guard_pair: Identifier
    unit_key_columns: Annotated[tuple[str, ...], Field(min_length=1)]
    cluster_columns: Annotated[tuple[str, ...], Field(min_length=1)]
    invariant_columns: Annotated[tuple[str, ...], Field(min_length=1)]
    require_matching_scenario_set_sha256: bool
    expected_guard_pair_counts_per_actor: dict[str, Annotated[int, Field(ge=1)]]
    expected_confirmatory_degradation_scenarios_per_actor: Annotated[int, Field(ge=1)]
    expected_confirmatory_negative_control_scenarios_per_actor: Annotated[int, Field(ge=1)]
    degradation_guard_order: Annotated[tuple[DegradationLevelSpec, ...], Field(min_length=2)]
    negative_control_guard_pair: Identifier

    @model_validator(mode="after")
    def validate_analysis_contract(self) -> SynthesisAnalysisSpec:
        for name, values in (
            ("unit_key_columns", self.unit_key_columns),
            ("cluster_columns", self.cluster_columns),
            ("invariant_columns", self.invariant_columns),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
        levels = tuple(level.guard_pair_id for level in self.degradation_guard_order)
        if len(set(levels)) != len(levels):
            raise ValueError("degradation_guard_order contains duplicate guard pairs")
        required_guards = set(levels) | {
            self.primary_guard_pair,
            self.negative_control_guard_pair,
        }
        missing = required_guards - set(self.expected_guard_pair_counts_per_actor)
        if missing:
            raise ValueError(f"expected guard-pair counts omit: {sorted(missing)}")
        return self


class SynthesisSpec(BaseModel):
    """Machine-readable declaration of the retrospective synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["censure.synthesis-spec.v1"]
    synthesis_id: Identifier
    inferential_status: Literal["retrospective_cross_experiment_synthesis"]
    decision_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    decision_timezone: Annotated[str, Field(min_length=1)]
    source_outcomes_inspected_before_freeze: Literal[True]
    complete_preregistered_actor_matrix: Literal[False]
    model_collection_status: Literal["closed_before_synthesis"]
    primary_reporting_policy: Literal["actor_specific_no_pooled_primary_effect"]
    cross_model_contrast_status: Literal["exploratory_task_paired_unadjusted"]
    mechanism_status: Literal["descriptive_noncausal"]
    sources: Annotated[tuple[SourceSpec, ...], Field(min_length=2)]
    actors: Annotated[tuple[ActorSpec, ...], Field(min_length=2)]
    pairwise_contrasts: Annotated[tuple[ContrastSpec, ...], Field(min_length=1)]
    analysis: SynthesisAnalysisSpec
    limitations: Annotated[tuple[str, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_identity_graph(self) -> SynthesisSpec:
        source_ids = tuple(source.source_id for source in self.sources)
        root_keys = tuple(source.root_key for source in self.sources)
        actor_ids = tuple(actor.actor_id for actor in self.actors)
        contrast_ids = tuple(contrast.contrast_id for contrast in self.pairwise_contrasts)
        for name, values in (
            ("source IDs", source_ids),
            ("root keys", root_keys),
            ("actor IDs", actor_ids),
            ("contrast IDs", contrast_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contain duplicates")
        unknown_sources = {actor.source_id for actor in self.actors} - set(source_ids)
        if unknown_sources:
            raise ValueError(f"actors reference unknown sources: {sorted(unknown_sources)}")
        declared_by_source = {source.source_id: set(source.actor_ids) for source in self.sources}
        actor_specs_by_source: dict[str, set[str]] = {source_id: set() for source_id in source_ids}
        for actor in self.actors:
            actor_specs_by_source[actor.source_id].add(actor.actor_id)
        if actor_specs_by_source != declared_by_source:
            raise ValueError("source actor_ids do not exactly match actor specifications")
        actor_set = set(actor_ids)
        for contrast in self.pairwise_contrasts:
            pair = {contrast.minuend_actor_id, contrast.subtrahend_actor_id}
            if not pair <= actor_set:
                raise ValueError(f"contrast {contrast.contrast_id!r} references an unknown actor")
        return self


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Validated source rows and immutable provenance."""

    rows: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


@dataclass(slots=True)
class SynthesisResult:
    """In-memory synthesis and its publication-facing tables."""

    spec: SynthesisSpec
    spec_sha256: str
    combined_pairs: pd.DataFrame
    base_analysis: Exp1AnalysisResult
    metrics: dict[str, Any]
    actor_effects: pd.DataFrame
    pairwise_contrasts: pd.DataFrame
    actor_domain_effects: pd.DataFrame
    degradation_summary: pd.DataFrame
    degradation_trends: pd.DataFrame
    negative_controls: pd.DataFrame
    source_provenance: tuple[dict[str, Any], ...]


def load_synthesis_spec(path: str | Path) -> SynthesisSpec:
    """Load and strictly validate one frozen synthesis specification."""

    try:
        return SynthesisSpec.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid synthesis specification {path}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(root: Path, relative: str) -> Path:
    resolved_root = root.expanduser().resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:  # pragma: no cover - also rejected by the spec model.
        raise SynthesisInputError(f"source path escapes its root: {relative}") from exc
    return path


def _read_json(path: Path, *, expected: Literal["object", "array"]) -> Any:
    if not path.is_file():
        raise SynthesisInputError(f"required synthesis source artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SynthesisInputError(f"source artifact is not valid JSON: {path}") from exc
    if expected == "object" and not isinstance(value, dict):
        raise SynthesisInputError(f"source artifact must be a JSON object: {path}")
    if expected == "array" and not isinstance(value, list):
        raise SynthesisInputError(f"source artifact must be a JSON array: {path}")
    return value


def _validate_context(
    source: SourceSpec,
    context: Mapping[str, Any],
    *,
    validation_report_sha256: str,
) -> None:
    if source.context_kind == "analysis_scope":
        scope_config = context.get("scope_config")
        if not isinstance(scope_config, Mapping):
            raise SynthesisInputError(
                f"source {source.source_id!r} has no resolved analysis-scope configuration"
            )
        observed_id = scope_config.get("scope_id")
        observed_sha = context.get("scope_config_sha256")
        observed_inferential_status = scope_config.get("inferential_status")
        observed_experiment_id = scope_config.get("source_experiment_id")
        observed_actors = context.get("included_actor_ids")
    else:
        observed_id = context.get("protocol_id")
        observed_sha = context.get("protocol_sha256")
        observed_inferential_status = context.get("inferential_status")
        observed_experiment_id = context.get("experiment_id")
        observed_actors = context.get("actor_ids")
    if observed_id != source.context_id or observed_sha != source.context_sha256:
        raise SynthesisInputError(
            f"source {source.source_id!r} has the wrong {source.context_kind} declaration"
        )
    if observed_inferential_status != source.source_inferential_status:
        raise SynthesisInputError(
            f"source {source.source_id!r} has the wrong source inferential status"
        )
    if observed_experiment_id != source.experiment_id:
        raise SynthesisInputError(
            f"source {source.source_id!r} context names a different experiment"
        )
    if not isinstance(observed_actors, list) or set(observed_actors) != set(source.actor_ids):
        raise SynthesisInputError(
            f"source {source.source_id!r} context names a different actor set"
        )
    if context.get("selected_session_count") != source.expected_normalized_row_count:
        raise SynthesisInputError(
            f"source {source.source_id!r} context has the wrong selected-session count"
        )
    if context.get("source_manifest_sha256") != source.expected_manifest_sha256:
        raise SynthesisInputError(
            f"source {source.source_id!r} context points to a different manifest"
        )
    if context.get("validation_report_sha256") != validation_report_sha256:
        raise SynthesisInputError(
            f"source {source.source_id!r} context points to a different validation report"
        )


def load_source_bundles(
    spec: SynthesisSpec,
    source_roots: Mapping[str, str | Path],
) -> tuple[SourceBundle, ...]:
    """Load source rows after manifest, validation, and declaration verification."""

    expected_root_keys = {source.root_key for source in spec.sources}
    if set(source_roots) != expected_root_keys:
        raise SynthesisInputError(
            "source roots must exactly match the specification; "
            f"expected={sorted(expected_root_keys)}, observed={sorted(source_roots)}"
        )
    bundles: list[SourceBundle] = []
    for source in spec.sources:
        root = Path(source_roots[source.root_key])
        paths = {
            "manifest": _source_path(root, source.manifest_relative_path),
            "paired_rows": _source_path(root, source.paired_rows_relative_path),
            "validation_report": _source_path(root, source.validation_report_relative_path),
            "context": _source_path(root, source.context_relative_path),
        }
        manifest = cast(dict[str, Any], _read_json(paths["manifest"], expected="object"))
        manifest_sha256 = canonical_sha256(manifest)
        if manifest.get("experiment_id") != source.experiment_id:
            raise SynthesisInputError(
                f"source {source.source_id!r} manifest has the wrong experiment_id"
            )
        if manifest_sha256 != source.expected_manifest_sha256:
            raise SynthesisInputError(
                f"source {source.source_id!r} manifest SHA-256 differs from the frozen spec"
            )
        validation = cast(dict[str, Any], _read_json(paths["validation_report"], expected="object"))
        if validation.get("ok") is not True or validation.get("issues") not in ([], None):
            raise SynthesisInputError(
                f"source {source.source_id!r} validation report is not clean and complete"
            )
        if validation.get("normalized_row_count") != source.expected_normalized_row_count:
            raise SynthesisInputError(
                f"source {source.source_id!r} validation row count differs from the spec"
            )
        validation_sha256 = canonical_sha256(validation)
        context = cast(dict[str, Any], _read_json(paths["context"], expected="object"))
        _validate_context(
            source,
            context,
            validation_report_sha256=validation_sha256,
        )
        raw_rows = cast(list[Any], _read_json(paths["paired_rows"], expected="array"))
        if len(raw_rows) != source.expected_normalized_row_count or not all(
            isinstance(row, dict) for row in raw_rows
        ):
            raise SynthesisInputError(
                f"source {source.source_id!r} paired rows differ from the validated count/schema"
            )
        rows = tuple(cast(dict[str, Any], row) for row in raw_rows)
        observed_actors = {str(row.get("actor_id")) for row in rows}
        if observed_actors != set(source.actor_ids):
            raise SynthesisInputError(
                f"source {source.source_id!r} contains the wrong actors; "
                f"observed={sorted(observed_actors)}, expected={sorted(source.actor_ids)}"
            )
        bundles.append(
            SourceBundle(
                rows=rows,
                provenance={
                    "source_id": source.source_id,
                    "root_key": source.root_key,
                    "experiment_id": source.experiment_id,
                    "manifest_sha256": manifest_sha256,
                    "scenario_set_sha256": manifest.get("scenario_set_sha256"),
                    "session_set_sha256": manifest.get("session_set_sha256"),
                    "validation_report_sha256": validation_sha256,
                    "context_kind": source.context_kind,
                    "context_id": source.context_id,
                    "context_sha256": source.context_sha256,
                    "source_inferential_status": source.source_inferential_status,
                    "paired_rows_file_sha256": _file_sha256(paths["paired_rows"]),
                    "normalized_row_count": len(rows),
                    "paths": {key: str(value) for key, value in paths.items()},
                },
            )
        )
    if spec.analysis.require_matching_scenario_set_sha256:
        scenario_hashes = {bundle.provenance["scenario_set_sha256"] for bundle in bundles}
        if len(scenario_hashes) != 1 or None in scenario_hashes:
            raise SynthesisInputError("source manifests do not share one scenario-set SHA-256")
    return tuple(bundles)


def _identity_hash(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(columns)].apply(
        lambda row: canonical_sha256({column: _json_cell(row[column]) for column in columns}),
        axis=1,
    )


def _prepare_combined_rows(
    bundles: Sequence[SourceBundle],
    spec: SynthesisSpec,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    source_lookup = {source.source_id: source for source in spec.sources}
    for bundle in bundles:
        source_id = str(bundle.provenance["source_id"])
        frame = pd.DataFrame.from_records(bundle.rows)
        frame["synthesis_source_id"] = source_id
        frames.append(frame)
        source = source_lookup[source_id]
        if len(frame) != source.expected_normalized_row_count:
            raise SynthesisInputError(f"source {source_id!r} changed after artifact validation")
    raw = pd.concat(frames, ignore_index=True, sort=False)

    contract_columns = (
        set(spec.analysis.unit_key_columns)
        | set(spec.analysis.cluster_columns)
        | set(spec.analysis.invariant_columns)
    )
    missing = sorted(contract_columns - set(raw.columns))
    if missing:
        raise SynthesisInputError(f"synthesis rows omit alignment columns: {missing}")
    raw["synthesis_unit_id"] = _identity_hash(raw, spec.analysis.unit_key_columns)
    raw["synthesis_cluster_id"] = _identity_hash(raw, spec.analysis.cluster_columns)

    analysis_config = AnalysisConfig(
        analysis_seed=spec.analysis.analysis_seed,
        bootstrap_samples=spec.analysis.bootstrap_samples,
        cluster_key="synthesis_cluster_id",
        ci_level=spec.analysis.ci_level,
        invalid_behavior_rule=spec.analysis.invalid_behavior_rule,
    )
    result = analyze_exp1(raw, analysis_config)
    frame = result.all_pairs

    expected_actors = {actor.actor_id for actor in spec.actors}
    observed_actors = set(frame["actor_id"])
    if observed_actors != expected_actors:
        raise SynthesisInputError(
            f"combined rows contain the wrong actors: {sorted(observed_actors)}"
        )
    if frame["pair_id"].duplicated().any():  # already checked by analyze_exp1.
        raise SynthesisInputError("combined source rows contain duplicate pair IDs")

    actor_unit_sets: dict[str, set[str]] = {}
    for actor in spec.actors:
        actor_rows = frame.loc[frame["actor_id"] == actor.actor_id]
        if len(actor_rows) != actor.expected_pair_count:
            raise SynthesisInputError(
                f"actor {actor.actor_id!r} has {len(actor_rows)} rows; "
                f"expected {actor.expected_pair_count}"
            )
        source_ids = set(actor_rows["synthesis_source_id"])
        if source_ids != {actor.source_id}:
            raise SynthesisInputError(
                f"actor {actor.actor_id!r} appears under the wrong source: {sorted(source_ids)}"
            )
        guard_counts = actor_rows["guard_pair_id"].value_counts().to_dict()
        if guard_counts != spec.analysis.expected_guard_pair_counts_per_actor:
            raise SynthesisInputError(
                f"actor {actor.actor_id!r} guard-pair counts differ from the frozen matrix"
            )
        primary = actor_rows.loc[
            actor_rows["split"].eq(spec.analysis.primary_split)
            & actor_rows["guard_pair_id"].eq(spec.analysis.primary_guard_pair)
        ]
        if len(primary) != actor.expected_primary_pair_count:
            raise SynthesisInputError(
                f"actor {actor.actor_id!r} has {len(primary)} primary rows; "
                f"expected {actor.expected_primary_pair_count}"
            )
        actor_unit_sets[actor.actor_id] = set(actor_rows["synthesis_unit_id"])

    reference_actor = spec.actors[0].actor_id
    reference_units = actor_unit_sets[reference_actor]
    for actor_id, unit_ids in actor_unit_sets.items():
        if unit_ids != reference_units:
            raise SynthesisInputError(
                f"actor {actor_id!r} does not have the same frozen scenario/guard units as "
                f"{reference_actor!r}"
            )

    expected_actor_count = len(spec.actors)
    group_sizes = frame.groupby("synthesis_unit_id", observed=True).size()
    if not group_sizes.eq(expected_actor_count).all():
        raise SynthesisInputError(
            "one or more synthesis units do not contain every actor exactly once"
        )
    for column in spec.analysis.invariant_columns:
        disagreement = frame.groupby("synthesis_unit_id", observed=True)[column].apply(
            lambda values: len({canonical_json(_json_cell(value)) for value in values}) != 1
        )
        if disagreement.any():
            bad_unit = next(
                str(unit_id) for unit_id, disagrees in disagreement.items() if bool(disagrees)
            )
            raise SynthesisInputError(
                f"cross-actor invariant {column!r} disagrees for unit {bad_unit}"
            )

    confirmatory = frame.loc[frame["split"].eq(spec.analysis.primary_split)]
    degraded_guards = [
        level.guard_pair_id
        for level in spec.analysis.degradation_guard_order
        if level.guard_pair_id != spec.analysis.primary_guard_pair
    ]
    reference_degraded: set[str] | None = None
    for actor in spec.actors:
        actor_rows = confirmatory.loc[confirmatory["actor_id"].eq(actor.actor_id)]
        for guard_pair in degraded_guards:
            scenarios = set(
                actor_rows.loc[actor_rows["guard_pair_id"].eq(guard_pair), "scenario_id"]
            )
            expected = spec.analysis.expected_confirmatory_degradation_scenarios_per_actor
            if len(scenarios) != expected:
                raise SynthesisInputError(
                    f"actor {actor.actor_id!r} guard {guard_pair!r} has "
                    f"{len(scenarios)} confirmatory scenarios; expected {expected}"
                )
            if reference_degraded is None:
                reference_degraded = scenarios
            elif scenarios != reference_degraded:
                raise SynthesisInputError("degradation subsets are not task-aligned across actors")
        negative = actor_rows.loc[
            actor_rows["guard_pair_id"].eq(spec.analysis.negative_control_guard_pair)
        ]
        if (
            len(negative)
            != spec.analysis.expected_confirmatory_negative_control_scenarios_per_actor
        ):
            raise SynthesisInputError(
                f"actor {actor.actor_id!r} has the wrong negative-control count"
            )
    if reference_degraded is None:  # pragma: no cover - spec requires degradation levels.
        raise SynthesisInputError("no degradation subset is available")
    primary_scenarios = set(
        confirmatory.loc[
            confirmatory["guard_pair_id"].eq(spec.analysis.primary_guard_pair),
            "scenario_id",
        ]
    )
    if not reference_degraded <= primary_scenarios:
        raise SynthesisInputError("strict-none rows do not contain the degradation scenario subset")
    return frame


def _seed_for(spec: SynthesisSpec, token: str) -> int:
    digest = hashlib.sha256(f"{spec.analysis.analysis_seed}|{token}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _mean_estimate(
    frame: pd.DataFrame,
    values: pd.Series | np.ndarray,
    *,
    spec: SynthesisSpec,
    token: str,
    reason: str,
) -> dict[str, Any]:
    """Mean and deterministic task-cluster bootstrap interval."""

    numeric = np.asarray(values, dtype=float)
    observed = np.isfinite(numeric)
    if len(frame) != len(numeric):
        raise SynthesisInputError("estimate values do not align with their synthesis rows")
    if not observed.all():
        frame = frame.loc[observed].copy()
        numeric = numeric[observed]
    if frame.empty:
        return {
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "n_pairs": 0,
            "n_clusters": 0,
            "reason": reason,
            "ci_reason": None,
        }
    clusters, cluster_codes = np.unique(
        frame["synthesis_cluster_id"].to_numpy(dtype=str), return_inverse=True
    )
    sums = np.bincount(cluster_codes, weights=numeric, minlength=len(clusters))
    counts = np.bincount(cluster_codes, minlength=len(clusters)).astype(float)
    value = float(numeric.mean())
    if spec.analysis.bootstrap_samples == 0:
        return {
            "value": value,
            "ci_low": None,
            "ci_high": None,
            "n_pairs": len(frame),
            "n_clusters": len(clusters),
            "reason": None,
            "ci_reason": "bootstrap disabled because bootstrap_samples is zero",
        }
    rng = np.random.default_rng(_seed_for(spec, token))
    probability = np.full(len(clusters), 1.0 / len(clusters))
    samples: list[np.ndarray] = []
    remaining = spec.analysis.bootstrap_samples
    while remaining:
        batch_size = min(remaining, 2_048)
        weights = rng.multinomial(len(clusters), probability, size=batch_size)
        samples.append((weights @ sums) / (weights @ counts))
        remaining -= batch_size
    sampled = np.concatenate(samples)
    alpha = (1.0 - spec.analysis.ci_level) / 2.0
    low, high = np.quantile(sampled, [alpha, 1.0 - alpha])
    return {
        "value": value,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": len(frame),
        "n_clusters": len(clusters),
        "reason": None,
        "ci_reason": None,
    }


def _add_estimate_columns(row: dict[str, Any], prefix: str, estimate: Mapping[str, Any]) -> None:
    row[prefix] = estimate.get("value")
    row[f"{prefix}_ci_low"] = estimate.get("ci_low")
    row[f"{prefix}_ci_high"] = estimate.get("ci_high")
    row[f"{prefix}_n_pairs"] = estimate.get("n_pairs")
    row[f"{prefix}_n_clusters"] = estimate.get("n_clusters")
    row[f"{prefix}_reason"] = estimate.get("reason") or estimate.get("ci_reason")


def _actor_effects_frame(
    result: Exp1AnalysisResult,
    spec: SynthesisSpec,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    primary = result.confirmatory_pairs.loc[
        result.confirmatory_pairs["guard_pair_id"].eq(spec.analysis.primary_guard_pair)
    ]
    actor_lookup = {actor.actor_id: actor for actor in spec.actors}
    source_lookup = {source.source_id: source for source in spec.sources}
    for actor_id in (actor.actor_id for actor in spec.actors):
        actor = actor_lookup[actor_id]
        complete = result.metrics["complete_case"]["by_actor"][actor_id]
        sensitivity = result.metrics["sensitivity"]["by_actor"][actor_id]
        bounds = result.metrics["all_pair_bounds"]["by_actor"][actor_id]
        complete_metrics = complete["metrics"]
        sensitivity_metrics = sensitivity["metrics"]
        bound_metrics = bounds["metrics"]
        row: dict[str, Any] = {
            "actor_id": actor_id,
            "display_name": actor.display_name,
            "family": actor.family,
            "source_id": actor.source_id,
            "source_inferential_status": source_lookup[actor.source_id].source_inferential_status,
            "n_primary_pairs": bounds["n_pairs"],
            "n_complete_primary_pairs": complete["n_pairs"],
            "n_invalid_primary_pairs": bounds["n_invalid_pairs"],
            "invalid_pair_rate": bounds["invalid_pair_rate"],
        }
        for prefix, estimate in (
            ("complete_behavior_risk", complete_metrics["behavior_risk"]),
            ("complete_target_risk", complete_metrics["oracle_target_risk"]),
            ("complete_masking_gap", complete_metrics["masking_gap"]),
            ("complete_masking_event_rate", complete_metrics["masking_event_rate"]),
            ("complete_reverse_event_rate", complete_metrics["reverse_event_rate"]),
            ("sensitivity_masking_gap", sensitivity_metrics["masking_gap"]),
            (
                "behavior_risk_lower_bound",
                bound_metrics["behavior_risk_lower_bound"],
            ),
            (
                "behavior_risk_upper_bound",
                bound_metrics["behavior_risk_upper_bound"],
            ),
            (
                "target_risk_lower_bound",
                bound_metrics["oracle_target_risk_lower_bound"],
            ),
            (
                "target_risk_upper_bound",
                bound_metrics["oracle_target_risk_upper_bound"],
            ),
            ("masking_gap_lower_bound", bound_metrics["masking_gap_lower_bound"]),
            ("masking_gap_upper_bound", bound_metrics["masking_gap_upper_bound"]),
            (
                "behavior_harm_or_invalid_rate",
                bound_metrics["behavior_harm_or_invalid_rate"],
            ),
            (
                "target_harm_or_invalid_rate",
                bound_metrics["target_harm_or_invalid_rate"],
            ),
            (
                "behavior_unsafe_attempt_rate",
                complete_metrics["behavior_attempted_unsafe_action_rate"],
            ),
            (
                "target_unsafe_attempt_rate",
                complete_metrics["target_attempted_unsafe_action_rate"],
            ),
            ("behavior_block_rate", complete_metrics["behavior_block_rate"]),
            ("target_block_rate", complete_metrics["target_block_rate"]),
            ("guard_dependence_rate", complete_metrics["guard_dependence_rate"]),
            ("behavior_clean_utility", complete_metrics["behavior_clean_utility"]),
            ("target_clean_utility", complete_metrics["target_clean_utility"]),
            (
                "behavior_utility_under_attack",
                complete_metrics["behavior_utility_under_attack"],
            ),
            (
                "target_utility_under_attack",
                complete_metrics["target_utility_under_attack"],
            ),
        ):
            _add_estimate_columns(row, prefix, estimate)

        complete_rows = primary.loc[
            primary["actor_id"].eq(actor_id) & primary["complete_case"]
        ].copy()
        call_specs: tuple[tuple[str, str | None], ...] = (
            ("behavior_mean_proposed_calls", "behavior_proposed_call_count"),
            ("target_mean_proposed_calls", "target_proposed_call_count"),
            ("target_zero_call_rate", "target_proposed_call_count"),
            ("behavior_any_block_rate", "behavior_blocked_call_count"),
        )
        for prefix, column in call_specs:
            if column not in complete_rows.columns:
                estimate = _mean_estimate(
                    complete_rows.iloc[0:0],
                    np.asarray([], dtype=float),
                    spec=spec,
                    token=f"actor|{actor_id}|{prefix}",
                    reason=f"input column {column} is unavailable",
                )
            else:
                numeric = np.asarray(
                    pd.to_numeric(complete_rows[column], errors="coerce"), dtype=float
                )
                if prefix == "target_zero_call_rate":
                    numeric = np.where(np.isfinite(numeric), numeric == 0, np.nan)
                elif prefix == "behavior_any_block_rate":
                    numeric = np.where(np.isfinite(numeric), numeric > 0, np.nan)
                estimate = _mean_estimate(
                    complete_rows,
                    numeric,
                    spec=spec,
                    token=f"actor|{actor_id}|{prefix}",
                    reason=f"no complete rows have {column}",
                )
            _add_estimate_columns(row, prefix, estimate)
        records.append(row)
    return pd.DataFrame.from_records(records)


def _paired_actor_rows(
    primary: pd.DataFrame,
    minuend_actor: str,
    subtrahend_actor: str,
) -> pd.DataFrame:
    columns = [
        "synthesis_unit_id",
        "synthesis_cluster_id",
        "scenario_id",
        "complete_case",
        "realized_pair_difference",
        "sensitivity_behavior_harm",
        "sensitivity_target_harm",
        "masking_gap_lower_bound",
        "masking_gap_upper_bound",
    ]
    left = primary.loc[primary["actor_id"].eq(minuend_actor), columns].copy()
    right = primary.loc[primary["actor_id"].eq(subtrahend_actor), columns].copy()
    merged = left.merge(
        right,
        on=["synthesis_unit_id", "scenario_id"],
        how="inner",
        suffixes=("_minuend", "_subtrahend"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise SynthesisInputError("actor contrast lost one or more frozen primary tasks")
    if not (
        merged["synthesis_cluster_id_minuend"] == merged["synthesis_cluster_id_subtrahend"]
    ).all():
        raise SynthesisInputError("actor contrast has inconsistent task-cluster identities")
    merged["synthesis_cluster_id"] = merged["synthesis_cluster_id_minuend"]
    return merged


def _pairwise_contrast_frame(
    result: Exp1AnalysisResult,
    spec: SynthesisSpec,
) -> pd.DataFrame:
    primary = result.confirmatory_pairs.loc[
        result.confirmatory_pairs["guard_pair_id"].eq(spec.analysis.primary_guard_pair)
    ].copy()
    display = {actor.actor_id: actor.display_name for actor in spec.actors}
    records: list[dict[str, Any]] = []
    for contrast in spec.pairwise_contrasts:
        merged = _paired_actor_rows(
            primary,
            contrast.minuend_actor_id,
            contrast.subtrahend_actor_id,
        )
        joint_complete = merged.loc[
            merged["complete_case_minuend"] & merged["complete_case_subtrahend"]
        ].copy()
        complete_values = (
            joint_complete["realized_pair_difference_minuend"]
            - joint_complete["realized_pair_difference_subtrahend"]
        )
        sensitivity_values = (
            merged["sensitivity_target_harm_minuend"]
            - merged["sensitivity_behavior_harm_minuend"]
            - merged["sensitivity_target_harm_subtrahend"]
            + merged["sensitivity_behavior_harm_subtrahend"]
        )
        lower_values = (
            merged["masking_gap_lower_bound_minuend"] - merged["masking_gap_upper_bound_subtrahend"]
        )
        upper_values = (
            merged["masking_gap_upper_bound_minuend"] - merged["masking_gap_lower_bound_subtrahend"]
        )
        estimates = {
            "complete_case_gap_contrast": _mean_estimate(
                joint_complete,
                complete_values,
                spec=spec,
                token=f"contrast|{contrast.contrast_id}|complete",
                reason="no tasks are complete for both actors",
            ),
            "sensitivity_gap_contrast": _mean_estimate(
                merged,
                sensitivity_values,
                spec=spec,
                token=f"contrast|{contrast.contrast_id}|sensitivity",
                reason="no shared tasks are available",
            ),
            "gap_contrast_lower_bound": _mean_estimate(
                merged,
                lower_values,
                spec=spec,
                token=f"contrast|{contrast.contrast_id}|lower",
                reason="no shared tasks are available",
            ),
            "gap_contrast_upper_bound": _mean_estimate(
                merged,
                upper_values,
                spec=spec,
                token=f"contrast|{contrast.contrast_id}|upper",
                reason="no shared tasks are available",
            ),
        }
        row: dict[str, Any] = {
            "contrast_id": contrast.contrast_id,
            "minuend_actor_id": contrast.minuend_actor_id,
            "minuend_display_name": display[contrast.minuend_actor_id],
            "subtrahend_actor_id": contrast.subtrahend_actor_id,
            "subtrahend_display_name": display[contrast.subtrahend_actor_id],
            "positive_means": (
                f"larger masking gap for {display[contrast.minuend_actor_id]} than "
                f"{display[contrast.subtrahend_actor_id]}"
            ),
            "n_shared_primary_pairs": len(merged),
            "n_joint_complete_pairs": len(joint_complete),
            "n_either_actor_invalid_pairs": int(
                (~(merged["complete_case_minuend"] & merged["complete_case_subtrahend"])).sum()
            ),
        }
        for prefix, estimate in estimates.items():
            _add_estimate_columns(row, prefix, estimate)
        records.append(row)
    return pd.DataFrame.from_records(records)


def _group_effect_record(
    group: pd.DataFrame,
    *,
    spec: SynthesisSpec,
    token: str,
) -> dict[str, Any]:
    complete = group.loc[group["complete_case"]].copy()
    estimates = {
        "complete_masking_gap": _mean_estimate(
            complete,
            complete["realized_pair_difference"],
            spec=spec,
            token=f"{token}|complete",
            reason="no complete pairs are available",
        ),
        "sensitivity_masking_gap": _mean_estimate(
            group,
            group["sensitivity_target_harm"] - group["sensitivity_behavior_harm"],
            spec=spec,
            token=f"{token}|sensitivity",
            reason="no pairs are available",
        ),
        "masking_gap_lower_bound": _mean_estimate(
            group,
            group["masking_gap_lower_bound"],
            spec=spec,
            token=f"{token}|lower",
            reason="no pairs are available",
        ),
        "masking_gap_upper_bound": _mean_estimate(
            group,
            group["masking_gap_upper_bound"],
            spec=spec,
            token=f"{token}|upper",
            reason="no pairs are available",
        ),
        "guard_dependence_rate": _mean_estimate(
            complete,
            complete["guard_dependent"],
            spec=spec,
            token=f"{token}|guard_dependence",
            reason="no complete guard-dependence values are available",
        ),
    }
    row: dict[str, Any] = {
        "n_pairs": len(group),
        "n_complete_pairs": len(complete),
        "n_invalid_pairs": int((~group["complete_case"]).sum()),
        "invalid_pair_rate": float((~group["complete_case"]).mean()) if len(group) else None,
    }
    for prefix, estimate in estimates.items():
        _add_estimate_columns(row, prefix, estimate)
    return row


def _actor_domain_frame(
    result: Exp1AnalysisResult,
    spec: SynthesisSpec,
) -> pd.DataFrame:
    primary = result.confirmatory_pairs.loc[
        result.confirmatory_pairs["guard_pair_id"].eq(spec.analysis.primary_guard_pair)
    ]
    display = {actor.actor_id: actor.display_name for actor in spec.actors}
    records: list[dict[str, Any]] = []
    for (actor_id, domain), group in primary.groupby(
        ["actor_id", "domain"], sort=True, observed=True
    ):
        row = {
            "actor_id": str(actor_id),
            "display_name": display[str(actor_id)],
            "domain": str(domain),
            **_group_effect_record(
                group.copy(),
                spec=spec,
                token=f"actor_domain|{actor_id}|{domain}",
            ),
        }
        records.append(row)
    return pd.DataFrame.from_records(records)


def _degradation_frames(
    result: Exp1AnalysisResult,
    spec: SynthesisSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    confirmatory = result.confirmatory_pairs
    degraded_guards = [
        level.guard_pair_id
        for level in spec.analysis.degradation_guard_order
        if level.guard_pair_id != spec.analysis.primary_guard_pair
    ]
    reference = set(
        confirmatory.loc[confirmatory["guard_pair_id"].eq(degraded_guards[0]), "scenario_id"]
    )
    display = {actor.actor_id: actor.display_name for actor in spec.actors}
    summary_records: list[dict[str, Any]] = []
    trend_records: list[dict[str, Any]] = []
    for actor in spec.actors:
        actor_rows = confirmatory.loc[
            confirmatory["actor_id"].eq(actor.actor_id)
            & confirmatory["scenario_id"].isin(reference)
        ].copy()
        actor_level_rows: list[dict[str, Any]] = []
        for order, level in enumerate(spec.analysis.degradation_guard_order, start=1):
            group = actor_rows.loc[actor_rows["guard_pair_id"].eq(level.guard_pair_id)].copy()
            expected = spec.analysis.expected_confirmatory_degradation_scenarios_per_actor
            if len(group) != expected:
                raise SynthesisInputError(
                    f"matched degradation level {level.guard_pair_id!r} for "
                    f"{actor.actor_id!r} has {len(group)} rows; expected {expected}"
                )
            row = {
                "actor_id": actor.actor_id,
                "display_name": actor.display_name,
                "relaxation_order": order,
                "guard_pair_id": level.guard_pair_id,
                "display_label": level.display_label,
                **_group_effect_record(
                    group,
                    spec=spec,
                    token=f"degradation|{actor.actor_id}|{level.guard_pair_id}",
                ),
            }
            summary_records.append(row)
            actor_level_rows.append(row)

        sensitivity_values = [float(row["sensitivity_masking_gap"]) for row in actor_level_rows]
        lower_values = [float(row["masking_gap_lower_bound"]) for row in actor_level_rows]
        upper_values = [float(row["masking_gap_upper_bound"]) for row in actor_level_rows]
        first_level = spec.analysis.degradation_guard_order[0]
        last_level = spec.analysis.degradation_guard_order[-1]
        first = actor_rows.loc[
            actor_rows["guard_pair_id"].eq(first_level.guard_pair_id),
            [
                "scenario_id",
                "synthesis_cluster_id",
                "sensitivity_behavior_harm",
                "sensitivity_target_harm",
            ],
        ]
        last = actor_rows.loc[
            actor_rows["guard_pair_id"].eq(last_level.guard_pair_id),
            [
                "scenario_id",
                "synthesis_cluster_id",
                "sensitivity_behavior_harm",
                "sensitivity_target_harm",
            ],
        ]
        paired = last.merge(
            first,
            on="scenario_id",
            suffixes=("_last", "_first"),
            validate="one_to_one",
        )
        if not (paired["synthesis_cluster_id_last"] == paired["synthesis_cluster_id_first"]).all():
            raise SynthesisInputError("degradation endpoint task clusters do not align")
        paired["synthesis_cluster_id"] = paired["synthesis_cluster_id_last"]
        endpoint_difference = (
            paired["sensitivity_target_harm_last"]
            - paired["sensitivity_behavior_harm_last"]
            - paired["sensitivity_target_harm_first"]
            + paired["sensitivity_behavior_harm_first"]
        )
        endpoint_estimate = _mean_estimate(
            paired,
            endpoint_difference,
            spec=spec,
            token=f"degradation|{actor.actor_id}|endpoint_change",
            reason="no matched degradation endpoint tasks are available",
        )
        trend_row: dict[str, Any] = {
            "actor_id": actor.actor_id,
            "display_name": display[actor.actor_id],
            "n_matched_scenarios": len(paired),
            "sensitivity_point_estimates_nondecreasing": all(
                right >= left - 1e-12 for left, right in pairwise(sensitivity_values)
            ),
            "lower_bound_point_estimates_nondecreasing": all(
                right >= left - 1e-12 for left, right in pairwise(lower_values)
            ),
            "upper_bound_point_estimates_nondecreasing": all(
                right >= left - 1e-12 for left, right in pairwise(upper_values)
            ),
            "nondecreasing_sensitivity_adjacent_steps": sum(
                right >= left - 1e-12 for left, right in pairwise(sensitivity_values)
            ),
            "total_adjacent_steps": len(sensitivity_values) - 1,
            "endpoint_contrast": (f"{last_level.guard_pair_id} minus {first_level.guard_pair_id}"),
        }
        _add_estimate_columns(
            trend_row,
            "sensitivity_endpoint_change",
            endpoint_estimate,
        )
        trend_records.append(trend_row)
    return (
        pd.DataFrame.from_records(summary_records),
        pd.DataFrame.from_records(trend_records),
    )


def _negative_control_frame(
    result: Exp1AnalysisResult,
    spec: SynthesisSpec,
) -> pd.DataFrame:
    rows = result.confirmatory_pairs.loc[
        result.confirmatory_pairs["guard_pair_id"].eq(spec.analysis.negative_control_guard_pair)
    ]
    records: list[dict[str, Any]] = []
    for actor in spec.actors:
        group = rows.loc[rows["actor_id"].eq(actor.actor_id)].copy()
        records.append(
            {
                "actor_id": actor.actor_id,
                "display_name": actor.display_name,
                "guard_pair_id": spec.analysis.negative_control_guard_pair,
                **_group_effect_record(
                    group,
                    spec=spec,
                    token=f"negative_control|{actor.actor_id}",
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def analyze_synthesis(
    bundles: Sequence[SourceBundle],
    spec: SynthesisSpec,
) -> SynthesisResult:
    """Validate aligned rows and compute the frozen actor-specific synthesis."""

    frame = _prepare_combined_rows(bundles, spec)
    analysis_config = AnalysisConfig(
        analysis_seed=spec.analysis.analysis_seed,
        bootstrap_samples=spec.analysis.bootstrap_samples,
        cluster_key="synthesis_cluster_id",
        ci_level=spec.analysis.ci_level,
        invalid_behavior_rule=spec.analysis.invalid_behavior_rule,
    )
    base = analyze_exp1(frame, analysis_config)
    actor_effects = _actor_effects_frame(base, spec)
    contrasts = _pairwise_contrast_frame(base, spec)
    domains = _actor_domain_frame(base, spec)
    degradation, trends = _degradation_frames(base, spec)
    negative = _negative_control_frame(base, spec)
    spec_sha256 = canonical_sha256(spec)
    source_provenance = tuple(bundle.provenance for bundle in bundles)
    metrics = {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "synthesis_id": spec.synthesis_id,
        "synthesis_spec_sha256": spec_sha256,
        "inferential_status": spec.inferential_status,
        "source_outcomes_inspected_before_freeze": (spec.source_outcomes_inspected_before_freeze),
        "complete_preregistered_actor_matrix": spec.complete_preregistered_actor_matrix,
        "primary_reporting_policy": spec.primary_reporting_policy,
        "cross_model_contrast_status": spec.cross_model_contrast_status,
        "mechanism_status": spec.mechanism_status,
        "counts": {
            "actors": len(spec.actors),
            "all_pair_rows": len(base.all_pairs),
            "unique_scenario_guard_units": int(base.all_pairs["synthesis_unit_id"].nunique()),
            "confirmatory_pair_rows": len(base.confirmatory_pairs),
            "primary_pair_rows": int(
                (base.confirmatory_pairs["guard_pair_id"] == spec.analysis.primary_guard_pair).sum()
            ),
            "primary_pairs_per_actor": {
                str(row["actor_id"]): int(row["n_primary_pairs"])
                for row in actor_effects.to_dict(orient="records")
            },
        },
        "definitions": {
            "complete_case_actor_effect": (
                "actor-specific mean H_target-H_behavior among valid behavior/target pairs"
            ),
            "pairwise_complete_contrast": (
                "task-paired difference between two actor masking gaps on their joint "
                "complete-case intersection"
            ),
            "all_pair_actor_bounds": (
                "sharp binary-harm bounds retaining every frozen actor/task pair"
            ),
            "all_pair_contrast_bounds": (
                "pairwise [L_minuend-U_subtrahend, U_minuend-L_subtrahend] before averaging"
            ),
            "mechanism_metrics": "descriptive and noncausal",
            "pooled_primary_effect": "not estimated by the frozen reporting policy",
        },
        "actors": actor_effects.to_dict(orient="records"),
        "pairwise_contrasts": contrasts.to_dict(orient="records"),
        "actor_domain_effects": domains.to_dict(orient="records"),
        "degradation_summary": degradation.to_dict(orient="records"),
        "degradation_trends": trends.to_dict(orient="records"),
        "negative_controls": negative.to_dict(orient="records"),
        "source_provenance": list(source_provenance),
        "limitations": list(spec.limitations),
    }
    return SynthesisResult(
        spec=spec,
        spec_sha256=spec_sha256,
        combined_pairs=base.all_pairs,
        base_analysis=base,
        metrics=metrics,
        actor_effects=actor_effects,
        pairwise_contrasts=contrasts,
        actor_domain_effects=domains,
        degradation_summary=degradation,
        degradation_trends=trends,
        negative_controls=negative,
        source_provenance=source_provenance,
    )


def _json_cell(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    return _json_cell(value)


def _atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        os.close(fd)
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    except ImportError as exc:
        raise SynthesisInputError(
            "writing combined_pairs.parquet requires the pinned pyarrow dependency"
        ) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        figure.savefig(temp_path, bbox_inches="tight", dpi=180)
        os.replace(temp_path, path)
    finally:
        plt.close(figure)
        if temp_path.exists():
            temp_path.unlink()


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _format_estimate(row: Mapping[str, Any], prefix: str, *, digits: int = 3) -> str:
    value = _finite_float(row.get(prefix))
    if value is None:
        return "N/A"
    low = _finite_float(row.get(f"{prefix}_ci_low"))
    high = _finite_float(row.get(f"{prefix}_ci_high"))
    if low is None or high is None:
        return f"{value:.{digits}f} [CI N/A]"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _format_bounds(row: Mapping[str, Any], lower: str, upper: str, *, digits: int = 3) -> str:
    low = _finite_float(row.get(lower))
    high = _finite_float(row.get(upper))
    if low is None or high is None:
        return "N/A"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


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


def _actor_effect_table_latex(result: SynthesisResult) -> str:
    lines = [
        "% Retrospective cross-experiment synthesis; actor-specific effects only.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Actor & Complete/total & Complete-case gap & All-pair gap bounds & Invalid rate \\",
        r"\midrule",
    ]
    for row in result.actor_effects.to_dict(orient="records"):
        lines.append(
            f"{_latex_escape(str(row['display_name']))} & "
            f"{int(row['n_complete_primary_pairs'])}/{int(row['n_primary_pairs'])} & "
            f"{_latex_escape(_format_estimate(row, 'complete_masking_gap'))} & "
            f"{_latex_escape(_format_bounds(row, 'masking_gap_lower_bound', 'masking_gap_upper_bound'))} & "
            f"{float(row['invalid_pair_rate']):.3f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _contrast_table_latex(result: SynthesisResult) -> str:
    lines = [
        "% Exploratory task-paired actor-by-guard contrasts; unadjusted intervals.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Contrast & Joint-complete gap difference & Sensitivity difference & All-pair bounds \\",
        r"\midrule",
    ]
    for row in result.pairwise_contrasts.to_dict(orient="records"):
        label = f"{row['minuend_display_name']} - {row['subtrahend_display_name']}"
        lines.append(
            f"{_latex_escape(label)} & "
            f"{_latex_escape(_format_estimate(row, 'complete_case_gap_contrast'))} & "
            f"{_latex_escape(_format_estimate(row, 'sensitivity_gap_contrast'))} & "
            f"{_latex_escape(_format_bounds(row, 'gap_contrast_lower_bound', 'gap_contrast_upper_bound'))} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _domain_effect_table_latex(result: SynthesisResult) -> str:
    lines = [
        "% Exploratory actor-specific domain effects; no multiplicity adjustment.",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Actor & Domain & Complete/total & Complete-case gap & All-pair bounds \\",
        r"\midrule",
    ]
    for row in result.actor_domain_effects.to_dict(orient="records"):
        lines.append(
            f"{_latex_escape(str(row['display_name']))} & "
            f"{_latex_escape(str(row['domain']).replace('_', ' '))} & "
            f"{int(row['n_complete_pairs'])}/{int(row['n_pairs'])} & "
            f"{_latex_escape(_format_estimate(row, 'complete_masking_gap'))} & "
            f"{_latex_escape(_format_bounds(row, 'masking_gap_lower_bound', 'masking_gap_upper_bound'))} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _mechanism_table_latex(result: SynthesisResult) -> str:
    lines = [
        "% Descriptive, noncausal mechanism diagnostics.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Actor & Behavior unsafe & Target unsafe & Behavior block & Guard dependence "
            r"& Behavior calls & Target calls \\"
        ),
        r"\midrule",
    ]
    for row in result.actor_effects.to_dict(orient="records"):
        values = [
            _format_estimate(row, "behavior_unsafe_attempt_rate"),
            _format_estimate(row, "target_unsafe_attempt_rate"),
            _format_estimate(row, "behavior_block_rate"),
            _format_estimate(row, "guard_dependence_rate"),
            _format_estimate(row, "behavior_mean_proposed_calls"),
            _format_estimate(row, "target_mean_proposed_calls"),
        ]
        lines.append(
            f"{_latex_escape(str(row['display_name']))} & "
            + " & ".join(_latex_escape(value) for value in values)
            + " "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _degradation_table_latex(result: SynthesisResult) -> str:
    lines = [
        "% Matched confirmatory degradation subset; descriptive synthesis.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Actor & Target guard & Complete/total & Complete-case gap & Sensitivity gap "
            r"& All-pair bounds \\"
        ),
        r"\midrule",
    ]
    for row in result.degradation_summary.to_dict(orient="records"):
        lines.append(
            f"{_latex_escape(str(row['display_name']))} & "
            f"{_latex_escape(str(row['display_label']))} & "
            f"{int(row['n_complete_pairs'])}/{int(row['n_pairs'])} & "
            f"{_latex_escape(_format_estimate(row, 'complete_masking_gap'))} & "
            f"{_latex_escape(_format_estimate(row, 'sensitivity_masking_gap'))} & "
            f"{_latex_escape(_format_bounds(row, 'masking_gap_lower_bound', 'masking_gap_upper_bound'))} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _negative_control_table_latex(result: SynthesisResult) -> str:
    lines = [
        "% Identical-strict negative controls.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Actor & Complete/total & Complete-case gap & Sensitivity gap "
            r"& All-pair bounds \\"
        ),
        r"\midrule",
    ]
    for row in result.negative_controls.to_dict(orient="records"):
        lines.append(
            f"{_latex_escape(str(row['display_name']))} & "
            f"{int(row['n_complete_pairs'])}/{int(row['n_pairs'])} & "
            f"{_latex_escape(_format_estimate(row, 'complete_masking_gap'))} & "
            f"{_latex_escape(_format_estimate(row, 'sensitivity_masking_gap'))} & "
            f"{_latex_escape(_format_bounds(row, 'masking_gap_lower_bound', 'masking_gap_upper_bound'))} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _report_markdown(result: SynthesisResult) -> str:
    actor_rows = []
    for row in result.actor_effects.to_dict(orient="records"):
        if float(row["masking_gap_lower_bound"]) > 0.0:
            sign = "positive under every missing-harm assignment"
        elif float(row["masking_gap_upper_bound"]) < 0.0:
            sign = "negative under every missing-harm assignment"
        else:
            sign = "sign not identified under missing harm"
        actor_rows.append(
            [
                str(row["display_name"]),
                f"{int(row['n_complete_primary_pairs'])}/{int(row['n_primary_pairs'])}",
                _format_estimate(row, "complete_masking_gap"),
                _format_estimate(row, "sensitivity_masking_gap"),
                _format_bounds(row, "masking_gap_lower_bound", "masking_gap_upper_bound"),
                f"{float(row['invalid_pair_rate']):.1%}",
                sign,
            ]
        )

    contrast_rows = []
    for row in result.pairwise_contrasts.to_dict(orient="records"):
        contrast_rows.append(
            [
                f"{row['minuend_display_name']} - {row['subtrahend_display_name']}",
                str(int(row["n_joint_complete_pairs"])),
                _format_estimate(row, "complete_case_gap_contrast"),
                _format_estimate(row, "sensitivity_gap_contrast"),
                _format_bounds(
                    row,
                    "gap_contrast_lower_bound",
                    "gap_contrast_upper_bound",
                ),
            ]
        )

    mechanism_rows = []
    for row in result.actor_effects.to_dict(orient="records"):
        mechanism_rows.append(
            [
                str(row["display_name"]),
                _format_estimate(row, "behavior_unsafe_attempt_rate"),
                _format_estimate(row, "target_unsafe_attempt_rate"),
                _format_estimate(row, "behavior_block_rate"),
                _format_estimate(row, "guard_dependence_rate"),
                _format_estimate(row, "behavior_mean_proposed_calls"),
                _format_estimate(row, "target_mean_proposed_calls"),
            ]
        )

    trend_rows = []
    for row in result.degradation_trends.to_dict(orient="records"):
        trend_rows.append(
            [
                str(row["display_name"]),
                f"{int(row['nondecreasing_sensitivity_adjacent_steps'])}/"
                f"{int(row['total_adjacent_steps'])}",
                str(bool(row["sensitivity_point_estimates_nondecreasing"])),
                _format_estimate(row, "sensitivity_endpoint_change"),
            ]
        )

    negative_control_rows = []
    for row in result.negative_controls.to_dict(orient="records"):
        negative_control_rows.append(
            [
                str(row["display_name"]),
                f"{int(row['n_complete_pairs'])}/{int(row['n_pairs'])}",
                _format_estimate(row, "complete_masking_gap"),
                _format_estimate(row, "sensitivity_masking_gap"),
                _format_bounds(
                    row,
                    "masking_gap_lower_bound",
                    "masking_gap_upper_bound",
                ),
            ]
        )

    source_lines = "\n".join(
        f"- `{source['source_id']}`: experiment `{source.get('experiment_id', 'unknown')}`, "
        f"inferential status `{source.get('source_inferential_status', 'unknown')}`, "
        f"manifest `{source.get('manifest_sha256', 'unknown')}`, "
        f"rows={source.get('normalized_row_count', 'unknown')}"
        for source in result.source_provenance
    )
    limitation_lines = "\n".join(f"- {item}" for item in result.spec.limitations)
    return f"""# Three-model task-paired Experiment 1 synthesis

> **Retrospective cross-experiment synthesis.** Outcomes were inspected before this synthesis
> was frozen. It is not the complete original preregistered actor matrix. Actor-specific effects
> are primary; new cross-model contrasts are exploratory and unadjusted.

Synthesis ID: `{result.spec.synthesis_id}`

Frozen specification SHA-256: `{result.spec_sha256}`

## Source provenance

{source_lines}

All actors align on {result.combined_pairs["synthesis_unit_id"].nunique()} frozen
scenario/guard units. Bootstrap clusters are the composite of environment layer, domain, and user
task ID.

## Actor-specific primary effects

{
        _markdown_table(
            [
                "Actor",
                "Complete/total",
                "Complete-case gap",
                "Sensitivity gap",
                "All-pair gap bounds",
                "Invalid",
                "Missing-harm conclusion",
            ],
            actor_rows,
        )
    }

The all-pair interval is a finite-sample partial-identification region, not a confidence interval.
Endpoint confidence intervals are retained in `actor_effects.csv` and `metrics.json`.

## Exploratory task-paired actor-by-guard contrasts

Positive contrasts mean the first actor has a larger signed masking gap.

{
        _markdown_table(
            [
                "Contrast",
                "Joint complete",
                "Complete-case difference",
                "Sensitivity difference",
                "All-pair contrast bounds",
            ],
            contrast_rows,
        )
    }

These are retrospective 95% task-cluster bootstrap intervals without multiplicity adjustment.
They are heterogeneity diagnostics, not confirmatory model-ranking tests.

## Descriptive mechanism diagnostics

{
        _markdown_table(
            [
                "Actor",
                "Behavior unsafe attempt",
                "Target unsafe attempt",
                "Behavior block rate",
                "Guard dependence",
                "Behavior calls",
                "Target calls",
            ],
            mechanism_rows,
        )
    }

These summaries do not establish causal mediation.

## Matched degradation checks

{
        _markdown_table(
            [
                "Actor",
                "Nondecreasing steps",
                "Entire sequence nondecreasing",
                "No-guard minus 25% endpoint",
            ],
            trend_rows,
        )
    }

`degradation_summary.csv` retains complete-case, sensitivity, and all-pair endpoints at every
matched relaxation level.

## Identical-guard negative controls

{
        _markdown_table(
            [
                "Actor",
                "Complete/total",
                "Complete-case gap",
                "Sensitivity gap",
                "All-pair gap bounds",
            ],
            negative_control_rows,
        )
    }

The expected signed gap is zero because both trajectories use the strict guard. These rows test
the paired execution and restoration machinery; they do not validate the substantive masking
effect.

## Limitations

{limitation_lines}
"""


def _actor_effect_figure(result: SynthesisResult) -> Figure:
    frame = result.actor_effects.reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, 10))
    for index, row in enumerate(frame.to_dict(orient="records")):
        y = len(frame) - index - 1
        lower = _finite_float(row["masking_gap_lower_bound"])
        upper = _finite_float(row["masking_gap_upper_bound"])
        point = _finite_float(row["complete_masking_gap"])
        ci_low = _finite_float(row["complete_masking_gap_ci_low"])
        ci_high = _finite_float(row["complete_masking_gap_ci_high"])
        color = colors[index % len(colors)]
        if lower is not None and upper is not None:
            axis.plot([lower, upper], [y, y], color=color, linewidth=7, alpha=0.28)
        if point is not None:
            if ci_low is not None and ci_high is not None:
                axis.errorbar(
                    point,
                    y,
                    xerr=[[point - ci_low], [ci_high - point]],
                    fmt="o",
                    color=color,
                    capsize=4,
                    linewidth=1.8,
                )
            else:
                axis.plot(point, y, "o", color=color)
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yticks(
        list(range(len(frame) - 1, -1, -1)),
        labels=frame["display_name"].tolist(),
    )
    axis.set_xlabel(r"Signed masking gap $H_\star-H_b$")
    axis.set_title(
        "Actor-specific masking gaps\n(points/whiskers: complete case; bands: all-pair bounds)"
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return figure


def _degradation_figure(result: SynthesisResult) -> Figure:
    figure, axis = plt.subplots(figsize=(9.0, 5.3))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, 10))
    for index, actor in enumerate(result.spec.actors):
        group = result.degradation_summary.loc[
            result.degradation_summary["actor_id"].eq(actor.actor_id)
        ].sort_values("relaxation_order")
        x = group["relaxation_order"].to_numpy(dtype=float)
        y = group["sensitivity_masking_gap"].to_numpy(dtype=float)
        ci_low = group["sensitivity_masking_gap_ci_low"].to_numpy(dtype=float)
        ci_high = group["sensitivity_masking_gap_ci_high"].to_numpy(dtype=float)
        lower = group["masking_gap_lower_bound"].to_numpy(dtype=float)
        upper = group["masking_gap_upper_bound"].to_numpy(dtype=float)
        color = colors[index % len(colors)]
        axis.fill_between(x, lower, upper, color=color, alpha=0.10)
        finite_ci = np.isfinite(y) & np.isfinite(ci_low) & np.isfinite(ci_high)
        if finite_ci.any():
            axis.errorbar(
                x[finite_ci],
                y[finite_ci],
                yerr=[y[finite_ci] - ci_low[finite_ci], ci_high[finite_ci] - y[finite_ci]],
                marker="o",
                capsize=3,
                color=color,
                label=actor.display_name,
            )
        missing_ci = np.isfinite(y) & ~finite_ci
        if missing_ci.any():
            axis.plot(
                x[missing_ci],
                y[missing_ci],
                "o",
                color=color,
                label=actor.display_name if not finite_ci.any() else None,
            )
    labels = [level.display_label for level in result.spec.analysis.degradation_guard_order]
    axis.set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation=15, ha="right")
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_ylabel(r"Signed masking gap $H_\star-H_b$")
    axis.set_title("Matched target-guard degradation sensitivity estimates")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _mechanism_figure(result: SynthesisResult) -> Figure:
    frame = result.actor_effects.reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, 10))
    for index, row in enumerate(frame.to_dict(orient="records")):
        x = _finite_float(row["target_unsafe_attempt_rate"])
        y = _finite_float(row["complete_masking_gap"])
        if x is None or y is None:
            continue
        axis.scatter(x, y, s=70, color=colors[index % len(colors)])
        axis.annotate(
            str(row["display_name"]),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Target-trajectory unsafe-attempt rate")
    axis.set_ylabel(r"Complete-case masking gap $H_\star-H_b$")
    axis.set_title("Descriptive actor mechanism relationship")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def _verify_existing_output_identity(root: Path, result: SynthesisResult) -> None:
    resolved_spec_path = root / "synthesis_spec.json"
    if resolved_spec_path.is_file():
        resolved = cast(
            dict[str, Any],
            _read_json(resolved_spec_path, expected="object"),
        )
        if resolved.get("synthesis_spec_sha256") != result.spec_sha256:
            raise SynthesisInputError(
                "the output directory already belongs to a different synthesis specification"
            )
    source_path = root / "source_provenance.json"
    if source_path.is_file():
        existing = _read_json(source_path, expected="array")
        current = _json_safe(list(result.source_provenance))
        if canonical_json(existing) != canonical_json(current):
            raise SynthesisInputError(
                "the output directory is already bound to different source artifacts"
            )


def write_synthesis_artifacts(
    result: SynthesisResult,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write machine-readable and publication-facing synthesis artifacts atomically."""

    root = Path(out_dir).expanduser().resolve()
    _verify_existing_output_identity(root, result)
    figures = root / "figures"
    paths = {
        "metrics": root / "metrics.json",
        "synthesis_spec": root / "synthesis_spec.json",
        "source_provenance": root / "source_provenance.json",
        "run_provenance": root / "run_provenance.json",
        "combined_pairs": root / "combined_pairs.parquet",
        "actor_effects": root / "actor_effects.csv",
        "pairwise_contrasts": root / "pairwise_gap_contrasts.csv",
        "actor_domain_effects": root / "actor_domain_effects.csv",
        "degradation_summary": root / "degradation_summary.csv",
        "degradation_trends": root / "degradation_trends.csv",
        "negative_controls": root / "negative_controls.csv",
        "report": root / "report.md",
        "table_actor_effects": root / "table_actor_effects.tex",
        "table_pairwise_contrasts": root / "table_pairwise_contrasts.tex",
        "table_domain_effects": root / "table_domain_effects.tex",
        "table_mechanism_diagnostics": root / "table_mechanism_diagnostics.tex",
        "table_degradation": root / "table_degradation.tex",
        "table_negative_controls": root / "table_negative_controls.tex",
        "actor_effects_png": figures / "actor_masking_gaps.png",
        "actor_effects_pdf": figures / "actor_masking_gaps.pdf",
        "degradation_png": figures / "degradation_curve.png",
        "degradation_pdf": figures / "degradation_curve.pdf",
        "mechanism_png": figures / "mechanism_relationship.png",
        "mechanism_pdf": figures / "mechanism_relationship.pdf",
        "artifact_manifest": root / "artifact_manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["metrics"], _json_safe(result.metrics))
    atomic_write_json(
        paths["synthesis_spec"],
        {
            "schema_version": "censure.resolved-synthesis-spec.v1",
            "synthesis_spec_sha256": result.spec_sha256,
            "spec": result.spec.model_dump(mode="json"),
        },
    )
    atomic_write_json(paths["source_provenance"], list(result.source_provenance))
    atomic_write_json(
        paths["run_provenance"],
        {
            "schema_version": "censure.synthesis-run-provenance.v1",
            "synthesis_id": result.spec.synthesis_id,
            "synthesis_spec_sha256": result.spec_sha256,
            "environment": collect_provenance(REPOSITORY_ROOT),
        },
    )
    _atomic_write_parquet(result.combined_pairs, paths["combined_pairs"])
    for key, frame in (
        ("actor_effects", result.actor_effects),
        ("pairwise_contrasts", result.pairwise_contrasts),
        ("actor_domain_effects", result.actor_domain_effects),
        ("degradation_summary", result.degradation_summary),
        ("degradation_trends", result.degradation_trends),
        ("negative_controls", result.negative_controls),
    ):
        _atomic_write_csv(frame, paths[key])
    _atomic_write_text(paths["report"], _report_markdown(result))
    _atomic_write_text(paths["table_actor_effects"], _actor_effect_table_latex(result))
    _atomic_write_text(
        paths["table_pairwise_contrasts"],
        _contrast_table_latex(result),
    )
    _atomic_write_text(paths["table_domain_effects"], _domain_effect_table_latex(result))
    _atomic_write_text(
        paths["table_mechanism_diagnostics"],
        _mechanism_table_latex(result),
    )
    _atomic_write_text(paths["table_degradation"], _degradation_table_latex(result))
    _atomic_write_text(
        paths["table_negative_controls"],
        _negative_control_table_latex(result),
    )
    for key, figure in (
        ("actor_effects_png", _actor_effect_figure(result)),
        ("actor_effects_pdf", _actor_effect_figure(result)),
        ("degradation_png", _degradation_figure(result)),
        ("degradation_pdf", _degradation_figure(result)),
        ("mechanism_png", _mechanism_figure(result)),
        ("mechanism_pdf", _mechanism_figure(result)),
    ):
        _atomic_save_figure(figure, paths[key])
    artifact_hashes = {
        key: {
            "path": str(path.relative_to(root)),
            "sha256": _file_sha256(path),
        }
        for key, path in paths.items()
        if key != "artifact_manifest"
    }
    atomic_write_json(
        paths["artifact_manifest"],
        {
            "schema_version": "censure.synthesis-artifact-manifest.v1",
            "synthesis_id": result.spec.synthesis_id,
            "synthesis_spec_sha256": result.spec_sha256,
            "artifacts": artifact_hashes,
        },
    )
    return paths


def run_synthesis(
    spec_path: str | Path,
    source_roots: Mapping[str, str | Path],
    out_dir: str | Path,
) -> tuple[SynthesisResult, dict[str, Path]]:
    """Load, validate, analyze, and persist one frozen synthesis."""

    spec = load_synthesis_spec(spec_path)
    bundles = load_source_bundles(spec, source_roots)
    result = analyze_synthesis(bundles, spec)
    paths = write_synthesis_artifacts(result, out_dir)
    return result, paths


def _parse_source_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key or not raw_path:
            raise ConfigurationError("--source-root must have the form KEY=/path")
        if key in roots:
            raise ConfigurationError(f"duplicate --source-root key: {key}")
        roots[key] = Path(raw_path)
    return roots


def main(argv: Sequence[str] | None = None) -> int:
    """CPU-only command line entrypoint for the frozen cross-experiment synthesis."""

    parser = argparse.ArgumentParser(
        prog="censure-exp1-synthesis",
        description="Run the frozen task-paired Qwen/Gemma/Ministral synthesis.",
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-root", required=True, action="append")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        source_roots = _parse_source_roots(args.source_root)
        result, paths = run_synthesis(args.spec, source_roots, args.out_dir)
        payload = {
            "schema_version": "censure.synthesis-stage-result.v1",
            "synthesis_id": result.spec.synthesis_id,
            "synthesis_spec_sha256": result.spec_sha256,
            "inferential_status": result.spec.inferential_status,
            "actor_count": len(result.spec.actors),
            "combined_pair_row_count": len(result.combined_pairs),
            "unique_scenario_guard_unit_count": int(
                result.combined_pairs["synthesis_unit_id"].nunique()
            ),
            "out_dir": str(Path(args.out_dir).expanduser().resolve()),
            "report": str(paths["report"]),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    except (ConfigurationError, SynthesisInputError, AnalysisInputError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SYNTHESIS_SCHEMA_VERSION",
    "ActorSpec",
    "ContrastSpec",
    "DegradationLevelSpec",
    "SourceBundle",
    "SourceSpec",
    "SynthesisAnalysisSpec",
    "SynthesisInputError",
    "SynthesisResult",
    "SynthesisSpec",
    "analyze_synthesis",
    "load_source_bundles",
    "load_synthesis_spec",
    "main",
    "run_synthesis",
    "write_synthesis_artifacts",
]
