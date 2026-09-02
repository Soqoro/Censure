"""Command-line workflow for CPU-only Phase 2 calibration experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from censure.actors.transformers_backend import TransformersActor
from censure.config import load_yaml, resolved_experiment_config
from censure.estimation.agent_analysis import (
    agent_audit_seal_payload,
    summarize_agent_audit_study,
    validate_complete_agent_ledgers,
)
from censure.estimation.agent_cohort import (
    AGENT_BUDGET_FRACTIONS,
    AgentAuditCohortCollection,
    AgentCohortStore,
    agent_allocation_seed,
    agent_budget_rounds,
    extract_agent_audit_cohorts,
)
from censure.estimation.agent_live import LiveAgentSuffixOracle, SelectedSuffixRunStore
from censure.estimation.auditor import CensureAuditor, InMemoryEvaluationOracle
from censure.estimation.calibration import (
    run_calibration_repetition,
    summarize_calibration_results,
)
from censure.estimation.calibration_storage import (
    CalibrationRunStore,
    calibration_chunk_count,
    calibration_chunk_repetition_indices,
    calibration_chunk_shard,
)
from censure.estimation.protocol import (
    CalibrationCatalogEntry,
    FrozenCalibrationCatalog,
    FrozenRobustnessCatalog,
    FrozenSharedSupportCatalog,
    load_frozen_calibration_catalog,
    load_frozen_robustness_catalog,
    load_frozen_shared_support_catalog,
)
from censure.estimation.robustness import (
    run_robustness_repetition,
    summarize_robustness_results,
)
from censure.estimation.robustness_storage import RobustnessRunStore
from censure.estimation.schemas import AllocationPolicyName
from censure.estimation.shared_support import (
    run_shared_support_repetition,
    summarize_shared_support_results,
)
from censure.estimation.shared_support_storage import SharedSupportRunStore
from censure.estimation.storage import AuditorRunStore
from censure.manifest import ExperimentManifest
from censure.schemas import FrozenScenario
from censure.storage import RunStore, atomic_write_json

DEFAULT_BASE_CONFIG = "configs/experiments/phase2_estimator_v1.yaml"
DEFAULT_AMENDMENT_1 = "configs/experiments/phase2_estimator_v1_amendment_1.yaml"
DEFAULT_AMENDMENT_2 = "configs/experiments/phase2_estimator_v1_amendment_2.yaml"
DEFAULT_AMENDMENT_3 = "configs/experiments/phase2_estimator_v1_amendment_3.yaml"
DEFAULT_AMENDMENT_4 = "configs/experiments/phase2_estimator_v1_amendment_4.yaml"
DEFAULT_AGENT_CONFIG = "configs/experiments/phase2_held_out_agents_v1.yaml"
DEFAULT_AGENT_FREEZE = "configs/experiments/phase2_held_out_agents_v1.freeze.yaml"


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _catalog_from_args(args: argparse.Namespace) -> FrozenCalibrationCatalog:
    return load_frozen_calibration_catalog(
        base_config_path=args.base_config,
        amendment_1_path=args.amendment_1,
        amendment_2_path=args.amendment_2,
    )


def _entries_for_purpose(
    catalog: FrozenCalibrationCatalog, purpose: str
) -> tuple[CalibrationCatalogEntry, ...]:
    if purpose == "all":
        return catalog.entries
    return tuple(entry for entry in catalog.entries if purpose in entry.purposes)


def _robustness_catalog_from_args(args: argparse.Namespace) -> FrozenRobustnessCatalog:
    return load_frozen_robustness_catalog(
        base_config_path=args.base_config,
        amendment_3_path=args.amendment_3,
    )


def _shared_support_catalog_from_args(
    args: argparse.Namespace,
) -> FrozenSharedSupportCatalog:
    return load_frozen_shared_support_catalog(
        base_config_path=args.base_config,
        amendment_4_path=args.amendment_4,
    )


def _catalog_summary(catalog: FrozenCalibrationCatalog) -> dict[str, Any]:
    validity = sum("validity" in entry.purposes for entry in catalog.entries)
    efficiency = sum("efficiency" in entry.purposes for entry in catalog.entries)
    return {
        "schema_version": "censure.calibration-catalog-summary.v1",
        "protocol_id": catalog.protocol_id,
        "catalog_sha256": catalog.catalog_sha256,
        "unique_cell_count": len(catalog.entries),
        "validity_cell_count": validity,
        "efficiency_cell_count": efficiency,
        "shared_cell_count": sum(len(entry.purposes) == 2 for entry in catalog.entries),
        "repetition_count": sum(entry.spec.repetitions for entry in catalog.entries),
        "repetitions_per_chunk": catalog.repetitions_per_chunk,
        "work_item_count": sum(
            calibration_chunk_count(entry.spec.repetitions, catalog.repetitions_per_chunk)
            for entry in catalog.entries
        ),
    }


def _run_catalog(args: argparse.Namespace) -> int:
    _json_print(_catalog_summary(_catalog_from_args(args)))
    return 0


def _run_robustness_catalog(args: argparse.Namespace) -> int:
    catalog = _robustness_catalog_from_args(args)
    _json_print(
        {
            "schema_version": "censure.robustness-catalog-summary.v1",
            "protocol_id": catalog.protocol_id,
            "catalog_sha256": catalog.catalog_sha256,
            "cell_count": len(catalog.specs),
            "repetition_count": sum(spec.repetitions for spec in catalog.specs),
            "repetitions_per_chunk": catalog.repetitions_per_chunk,
            "work_item_count": sum(
                calibration_chunk_count(spec.repetitions, catalog.repetitions_per_chunk)
                for spec in catalog.specs
            ),
        }
    )
    return 0


def _run_shared_support_catalog(args: argparse.Namespace) -> int:
    catalog = _shared_support_catalog_from_args(args)
    _json_print(
        {
            "schema_version": "censure.shared-support-catalog-summary.v1",
            "protocol_id": catalog.protocol_id,
            "catalog_sha256": catalog.catalog_sha256,
            "cell_count": len(catalog.specs),
            "repetition_count": sum(spec.repetitions for spec in catalog.specs),
            "repetitions_per_chunk": catalog.repetitions_per_chunk,
            "work_item_count": sum(
                calibration_chunk_count(spec.repetitions, catalog.repetitions_per_chunk)
                for spec in catalog.specs
            ),
        }
    )
    return 0


def _validate_shard(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must lie in [0, num_shards)")
    if args.max_work_items is not None and args.max_work_items < 1:
        raise ValueError("--max-work-items must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")


def _prepare_store(
    args: argparse.Namespace, catalog: FrozenCalibrationCatalog
) -> CalibrationRunStore:
    store = CalibrationRunStore(args.out_root, args.experiment_id)
    store.write_resolved_catalog(catalog)
    store.write_catalog(catalog.specs)
    return store


def _run_calibration(args: argparse.Namespace) -> int:
    _validate_shard(args)
    catalog = _catalog_from_args(args)
    entries = _entries_for_purpose(catalog, args.purpose)
    store = _prepare_store(args, catalog)
    selected = 0
    written = 0
    skipped_complete = 0
    for entry in entries:
        spec = entry.spec
        store.write_cell_spec(spec)
        for chunk_index in range(
            calibration_chunk_count(spec.repetitions, catalog.repetitions_per_chunk)
        ):
            if (
                calibration_chunk_shard(
                    cell_id=spec.cell_id,
                    chunk_index=chunk_index,
                    num_shards=args.num_shards,
                )
                != args.shard_index
            ):
                continue
            selected += 1
            if args.resume and store.is_chunk_complete(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
            ):
                skipped_complete += 1
                continue
            rows = tuple(
                row
                for repetition_index in calibration_chunk_repetition_indices(
                    repetitions=spec.repetitions,
                    repetitions_per_chunk=catalog.repetitions_per_chunk,
                    chunk_index=chunk_index,
                )
                for row in run_calibration_repetition(spec, repetition_index)
            )
            store.write_chunk(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
                rows=rows,
            )
            written += 1
            if written % args.progress_every == 0:
                _json_print(
                    {
                        "selected": selected,
                        "written": written,
                        "skipped_complete": skipped_complete,
                        "cell_id": spec.cell_id,
                        "chunk_index": chunk_index,
                    }
                )
            if args.max_work_items is not None and written >= args.max_work_items:
                _json_print(
                    {
                        "status": "work_limit_reached",
                        "selected": selected,
                        "written": written,
                        "skipped_complete": skipped_complete,
                        "catalog_sha256": catalog.catalog_sha256,
                    }
                )
                return 0
    _json_print(
        {
            "status": "complete",
            "selected": selected,
            "written": written,
            "skipped_complete": skipped_complete,
            "catalog_sha256": catalog.catalog_sha256,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
    )
    return 0


def _run_status(args: argparse.Namespace) -> int:
    _validate_shard(args)
    catalog = _catalog_from_args(args)
    entries = _entries_for_purpose(catalog, args.purpose)
    store = CalibrationRunStore(args.out_root, args.experiment_id)
    selected = 0
    complete = 0
    for entry in entries:
        for chunk_index in range(
            calibration_chunk_count(entry.spec.repetitions, catalog.repetitions_per_chunk)
        ):
            if (
                calibration_chunk_shard(
                    cell_id=entry.spec.cell_id,
                    chunk_index=chunk_index,
                    num_shards=args.num_shards,
                )
                != args.shard_index
            ):
                continue
            selected += 1
            complete += store.is_chunk_complete(
                entry.spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
            )
    _json_print(
        {
            "catalog_sha256": catalog.catalog_sha256,
            "purpose": args.purpose,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "selected": selected,
            "complete": complete,
            "remaining": selected - complete,
        }
    )
    return 0


def _run_summarize(args: argparse.Namespace) -> int:
    catalog = _catalog_from_args(args)
    entries = _entries_for_purpose(catalog, args.purpose)
    store = _prepare_store(args, catalog)
    all_summaries: list[dict[str, Any]] = []
    for entry in entries:
        rows = store.read_completed_cell_chunks(
            entry.spec,
            repetitions_per_chunk=catalog.repetitions_per_chunk,
            require_all=True,
        )
        summaries = summarize_calibration_results(rows)
        store.write_summaries(entry.spec, summaries)
        all_summaries.extend(
            {
                "purposes": list(entry.purposes),
                "cell_spec": entry.spec.model_dump(mode="json"),
                "summary": summary.model_dump(mode="json"),
            }
            for summary in summaries
        )
    payload = {
        "schema_version": "censure.calibration-combined-summary.v1",
        "catalog_sha256": catalog.catalog_sha256,
        "purpose": args.purpose,
        "cell_count": len(entries),
        "rows": all_summaries,
    }
    path = store.root / "results" / f"{args.purpose}_summary.json"
    digest = atomic_write_json(path, payload)
    _json_print(
        {
            "status": "complete",
            "path": str(path),
            "sha256": digest,
            "cell_count": len(entries),
            "summary_row_count": len(all_summaries),
        }
    )
    return 0


def _prepare_robustness_store(
    args: argparse.Namespace, catalog: FrozenRobustnessCatalog
) -> RobustnessRunStore:
    store = RobustnessRunStore(args.out_root, args.experiment_id)
    store.write_catalog(catalog)
    return store


def _run_robustness(args: argparse.Namespace) -> int:
    _validate_shard(args)
    catalog = _robustness_catalog_from_args(args)
    store = _prepare_robustness_store(args, catalog)
    selected = 0
    written = 0
    skipped_complete = 0
    for spec in catalog.specs:
        store.write_cell_spec(spec)
        for chunk_index in range(
            calibration_chunk_count(spec.repetitions, catalog.repetitions_per_chunk)
        ):
            if (
                calibration_chunk_shard(
                    cell_id=spec.cell_id,
                    chunk_index=chunk_index,
                    num_shards=args.num_shards,
                )
                != args.shard_index
            ):
                continue
            selected += 1
            if args.resume and store.is_chunk_complete(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
            ):
                skipped_complete += 1
                continue
            rows = tuple(
                run_robustness_repetition(spec, repetition_index)
                for repetition_index in calibration_chunk_repetition_indices(
                    repetitions=spec.repetitions,
                    repetitions_per_chunk=catalog.repetitions_per_chunk,
                    chunk_index=chunk_index,
                )
            )
            store.write_chunk(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
                rows=rows,
            )
            written += 1
            if written % args.progress_every == 0:
                _json_print(
                    {
                        "selected": selected,
                        "written": written,
                        "skipped_complete": skipped_complete,
                        "cell_id": spec.cell_id,
                        "chunk_index": chunk_index,
                    }
                )
            if args.max_work_items is not None and written >= args.max_work_items:
                _json_print(
                    {
                        "status": "work_limit_reached",
                        "selected": selected,
                        "written": written,
                        "skipped_complete": skipped_complete,
                        "catalog_sha256": catalog.catalog_sha256,
                    }
                )
                return 0
    _json_print(
        {
            "status": "complete",
            "selected": selected,
            "written": written,
            "skipped_complete": skipped_complete,
            "catalog_sha256": catalog.catalog_sha256,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
    )
    return 0


def _run_robustness_status(args: argparse.Namespace) -> int:
    _validate_shard(args)
    catalog = _robustness_catalog_from_args(args)
    store = RobustnessRunStore(args.out_root, args.experiment_id)
    selected = 0
    complete = 0
    for spec in catalog.specs:
        for chunk_index in range(
            calibration_chunk_count(spec.repetitions, catalog.repetitions_per_chunk)
        ):
            if (
                calibration_chunk_shard(
                    cell_id=spec.cell_id,
                    chunk_index=chunk_index,
                    num_shards=args.num_shards,
                )
                != args.shard_index
            ):
                continue
            selected += 1
            complete += store.is_chunk_complete(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
            )
    _json_print(
        {
            "catalog_sha256": catalog.catalog_sha256,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "selected": selected,
            "complete": complete,
            "remaining": selected - complete,
        }
    )
    return 0


def _run_robustness_summarize(args: argparse.Namespace) -> int:
    catalog = _robustness_catalog_from_args(args)
    store = _prepare_robustness_store(args, catalog)
    combined: list[dict[str, Any]] = []
    for spec in catalog.specs:
        rows = store.read_completed_cell(spec, repetitions_per_chunk=catalog.repetitions_per_chunk)
        summary = summarize_robustness_results(rows)
        store.write_summary(spec, summary)
        combined.append(
            {
                "cell_spec": spec.model_dump(mode="json"),
                "summary": summary.model_dump(mode="json"),
            }
        )
    payload = {
        "schema_version": "censure.robustness-combined-summary.v1",
        "catalog_sha256": catalog.catalog_sha256,
        "cell_count": len(catalog.specs),
        "rows": combined,
    }
    path = store.root / "results" / "robustness_summary.json"
    digest = atomic_write_json(path, payload)
    _json_print(
        {
            "status": "complete",
            "path": str(path),
            "sha256": digest,
            "cell_count": len(catalog.specs),
        }
    )
    return 0


def _prepare_shared_support_store(
    args: argparse.Namespace, catalog: FrozenSharedSupportCatalog
) -> SharedSupportRunStore:
    store = SharedSupportRunStore(args.out_root, args.experiment_id)
    store.write_catalog(catalog)
    return store


def _run_shared_support(args: argparse.Namespace) -> int:
    _validate_shard(args)
    catalog = _shared_support_catalog_from_args(args)
    store = _prepare_shared_support_store(args, catalog)
    selected = 0
    written = 0
    skipped_complete = 0
    for spec in catalog.specs:
        store.write_cell_spec(spec)
        for chunk_index in range(
            calibration_chunk_count(spec.repetitions, catalog.repetitions_per_chunk)
        ):
            if (
                calibration_chunk_shard(
                    cell_id=spec.cell_id,
                    chunk_index=chunk_index,
                    num_shards=args.num_shards,
                )
                != args.shard_index
            ):
                continue
            selected += 1
            if args.resume and store.is_chunk_complete(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
            ):
                skipped_complete += 1
                continue
            rows = tuple(
                run_shared_support_repetition(spec, repetition_index)
                for repetition_index in calibration_chunk_repetition_indices(
                    repetitions=spec.repetitions,
                    repetitions_per_chunk=catalog.repetitions_per_chunk,
                    chunk_index=chunk_index,
                )
            )
            store.write_chunk(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
                rows=rows,
            )
            written += 1
            if written % args.progress_every == 0:
                _json_print(
                    {
                        "selected": selected,
                        "written": written,
                        "skipped_complete": skipped_complete,
                        "cell_id": spec.cell_id,
                        "chunk_index": chunk_index,
                    }
                )
            if args.max_work_items is not None and written >= args.max_work_items:
                _json_print(
                    {
                        "status": "work_limit_reached",
                        "selected": selected,
                        "written": written,
                        "skipped_complete": skipped_complete,
                        "catalog_sha256": catalog.catalog_sha256,
                    }
                )
                return 0
    _json_print(
        {
            "status": "complete",
            "selected": selected,
            "written": written,
            "skipped_complete": skipped_complete,
            "catalog_sha256": catalog.catalog_sha256,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
    )
    return 0


def _run_shared_support_status(args: argparse.Namespace) -> int:
    _validate_shard(args)
    catalog = _shared_support_catalog_from_args(args)
    store = SharedSupportRunStore(args.out_root, args.experiment_id)
    selected = 0
    complete = 0
    for spec in catalog.specs:
        for chunk_index in range(
            calibration_chunk_count(spec.repetitions, catalog.repetitions_per_chunk)
        ):
            if (
                calibration_chunk_shard(
                    cell_id=spec.cell_id,
                    chunk_index=chunk_index,
                    num_shards=args.num_shards,
                )
                != args.shard_index
            ):
                continue
            selected += 1
            complete += store.is_chunk_complete(
                spec,
                chunk_index=chunk_index,
                repetitions_per_chunk=catalog.repetitions_per_chunk,
            )
    _json_print(
        {
            "catalog_sha256": catalog.catalog_sha256,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "selected": selected,
            "complete": complete,
            "remaining": selected - complete,
        }
    )
    return 0


def _run_shared_support_summarize(args: argparse.Namespace) -> int:
    catalog = _shared_support_catalog_from_args(args)
    store = _prepare_shared_support_store(args, catalog)
    combined: list[dict[str, Any]] = []
    for spec in catalog.specs:
        rows = store.read_completed_cell(spec, repetitions_per_chunk=catalog.repetitions_per_chunk)
        summary = summarize_shared_support_results(
            rows,
            max_importance_ratio=spec.max_importance_ratio,
            model_condition=spec.model_condition,
        )
        store.write_summary(spec, summary)
        combined.append(
            {
                "cell_spec": spec.model_dump(mode="json"),
                "summary": summary.model_dump(mode="json"),
            }
        )
    payload = {
        "schema_version": "censure.shared-support-combined-summary.v1",
        "catalog_sha256": catalog.catalog_sha256,
        "cell_count": len(catalog.specs),
        "rows": combined,
    }
    path = store.root / "results" / "shared_support_summary.json"
    digest = atomic_write_json(path, payload)
    _json_print(
        {
            "status": "complete",
            "path": str(path),
            "sha256": digest,
            "cell_count": len(catalog.specs),
        }
    )
    return 0


def _load_agent_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], ExperimentManifest, RunStore]:
    config = resolved_experiment_config(
        args.config,
        resolve_remote=False,
        model_root=args.model_root,
    )
    freeze = load_yaml(args.freeze)
    if freeze.get("schema_version") != "censure.phase2-held-out-freeze.v1":
        raise ValueError("unsupported held-out-agent freeze schema")
    if freeze.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("held-out freeze and experiment config IDs differ")
    expected_config_sha256 = str(freeze.get("resolved_config_sha256", ""))
    if config.get("resolved_config_hash") != expected_config_sha256:
        raise ValueError(
            "resolved held-out config differs from the prospective freeze: "
            f"expected {expected_config_sha256}, found {config.get('resolved_config_hash')}"
        )
    store = RunStore(args.out_root, str(config["experiment_id"]))
    # Reuse the Experiment 1 manifest loader because it verifies the resolved
    # configuration hash before returning any stored sampling unit.
    from censure.cli import _load_manifest

    manifest = _load_manifest(config, store)
    expected_manifest_sha256 = str(freeze.get("manifest_sha256", ""))
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "frozen held-out manifest differs from the prospective freeze: "
            f"expected {expected_manifest_sha256}, found {manifest.manifest_sha256}"
        )
    for field, observed in (
        ("scenario_count", manifest.summary.scenario_count),
        ("paired_session_count", manifest.summary.paired_session_count),
        ("trajectory_count", manifest.summary.trajectory_count),
    ):
        if int(freeze.get(field, -1)) != observed:
            raise ValueError(f"held-out freeze {field} differs from the manifest")
    return config, freeze, manifest, store


def _agent_collection_summary(collection: AgentAuditCohortCollection) -> dict[str, Any]:
    return {
        "schema_version": "censure.agent-cohort-summary.v1",
        "protocol_id": collection.protocol_id,
        "source_manifest_sha256": collection.source_manifest_sha256,
        "collection_sha256": collection.collection_sha256,
        "actor_count": len(collection.cohorts),
        "cohorts": [
            {
                "actor_id": cohort.actor_id,
                "cohort_id": cohort.cohort_id,
                "cohort_sha256": cohort.cohort_sha256,
                "cohort_size": cohort.envelope.cohort_size,
                "candidate_count": len(cohort.envelope.candidates),
                "auditable_candidate_count": len(cohort.envelope.auditable_candidates),
                "nonauditable_candidate_count": (
                    len(cohort.envelope.candidates) - len(cohort.envelope.auditable_candidates)
                ),
                "supported_session_count": len(cohort.supported_session_ids),
                "supported_harm_unit_count": cohort.supported_harm_unit_count,
                "unresolved_session_count": len(cohort.unresolved_session_ids),
                "theta_env": cohort.envelope.theta_env,
                "target_frontier_mass": cohort.envelope.target_frontier_mass,
                "auditable_mass": cohort.envelope.auditable_mass,
                "nonauditable_mass": cohort.envelope.nonauditable_mass,
            }
            for cohort in collection.cohorts
        ],
    }


def _run_agent_cohort(args: argparse.Namespace) -> int:
    config, _freeze, manifest, store = _load_agent_context(args)
    cohort_store = AgentCohortStore(args.out_root, str(config["experiment_id"]))
    if cohort_store.collection_path.is_file():
        collection = cohort_store.read_collection()
        if collection.source_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("persisted agent cohort belongs to a different manifest")
        payload = _agent_collection_summary(collection)
        payload["status"] = "already_frozen"
        _json_print(payload)
        return 0

    selected_sessions = [
        session
        for session in manifest.sessions
        if session.guard_pair_id == "strict_none"
        and session.behavior_guard_id == "strict"
        and session.target_guard_id == "none"
    ]
    missing_behavior = [
        session.session_id
        for session in selected_sessions
        if not store.is_complete(session_id=session.session_id, role="behavior")
    ]
    if missing_behavior:
        raise ValueError(
            "behavior stage is incomplete; refusing to freeze a prematurely unresolved cohort "
            f"({len(missing_behavior)} missing)"
        )
    preexisting_targets = [
        session.session_id
        for session in selected_sessions
        if store.is_complete(session_id=session.session_id, role="target")
    ]
    if preexisting_targets:
        raise ValueError(
            "private target trajectories already exist; the agent cohort must be frozen first"
        )

    from censure.cli import _bindings_factory

    restore_cache: dict[str, Any] = {}

    def restore_check(scenario: FrozenScenario, checkpoint: Any) -> bool:
        bindings = restore_cache.get(scenario.scenario_id)
        if bindings is None:
            bindings = _bindings_factory(scenario)()
            restore_cache[scenario.scenario_id] = bindings
        bindings.environment.restore(checkpoint)
        return bindings.environment.snapshot() == checkpoint

    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("held-out execution config must be a mapping")
    collection = extract_agent_audit_cohorts(
        manifest,
        store,
        checkpoint_restore_check=restore_check,
        max_tool_steps=int(execution.get("max_tool_steps", 12)),
    )
    cohort_store.write_collection(collection)
    auditor_store = AuditorRunStore(args.out_root, str(config["experiment_id"]))
    for cohort in collection.cohorts:
        auditor_store.write_envelope(cohort.envelope)
    payload = _agent_collection_summary(collection)
    payload["status"] = "frozen"
    atomic_write_json(
        cohort_store.root / "agent_cohorts" / "cohort_summary.json",
        payload,
    )
    _json_print(payload)
    return 0


def _selected_agent_policies(raw: str) -> tuple[AllocationPolicyName, ...]:
    if raw == "all":
        return tuple(AllocationPolicyName)
    return (AllocationPolicyName(raw),)


def _run_agent_audits(args: argparse.Namespace) -> int:
    config, _freeze, manifest, store = _load_agent_context(args)
    cohort_store = AgentCohortStore(args.out_root, str(config["experiment_id"]))
    collection = cohort_store.read_collection()
    if collection.source_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("agent cohort and manifest hashes differ")
    if (
        cohort_store.audit_seal_path.is_file()
        or cohort_store.audit_seal_path.with_suffix(".sha256").is_file()
    ):
        raise ValueError("agent audits are already sealed and cannot be extended")
    resolved_models = cast(Mapping[str, Mapping[str, Any]], config["resolved_models"])
    if args.model not in resolved_models:
        raise ValueError(
            f"unknown --model {args.model!r}; available: {', '.join(sorted(resolved_models))}"
        )
    model_config = dict(resolved_models[args.model])
    actor_id = str(model_config["model_id"])
    cohort = next(
        (item for item in collection.cohorts if item.actor_id == actor_id),
        None,
    )
    if cohort is None:
        raise ValueError(f"selected model {args.model!r} has no frozen agent cohort")
    preexisting_targets = [
        session_id
        for item in collection.cohorts
        for session_id in item.source_session_ids
        if store.is_complete(session_id=session_id, role="target")
    ]
    if preexisting_targets:
        raise ValueError(
            "full target outcomes already exist; selective audits must be sealed before "
            "the full-oracle stage"
        )
    policies = _selected_agent_policies(args.policy)
    auditor_store = AuditorRunStore(args.out_root, str(config["experiment_id"]))
    rows: list[dict[str, Any]] = []
    budget_rounds = agent_budget_rounds(len(cohort.envelope.candidates))
    max_rounds = max(budget_rounds.values())
    if not cohort.envelope.auditable_candidates:
        max_rounds = 0
    allocation_seed = agent_allocation_seed(cohort.cohort_id)
    completed: dict[AllocationPolicyName, tuple[Any, Any]] = {}
    for policy in policies:
        replay = CensureAuditor(
            envelope=cohort.envelope,
            oracle=InMemoryEvaluationOracle({}),
            policy=policy,
            allocation_seed=allocation_seed,
            alpha=0.05,
            exploration_epsilon=0.10,
        )
        template = replay.initial_ledger()
        if not auditor_store.has_ledger(template):
            continue
        if not args.resume:
            raise FileExistsError(
                f"agent audit ledger already exists for {cohort.actor_id}/{policy.value}; "
                "use --resume"
            )
        ledger = auditor_store.read_ledger(template)
        replay.validate_ledger(ledger)
        if len(ledger.disclosures) == max_rounds:
            completed[policy] = (
                ledger,
                auditor_store.read_certificate_path(ledger),
            )

    actor: Any | None = None
    oracle: LiveAgentSuffixOracle | None = None
    if len(completed) != len(policies):
        if os.getenv("HF_TOKEN"):
            model_config["token"] = os.environ["HF_TOKEN"]
        actor = TransformersActor(model_config)
        if actor.actor_revision != next(
            session.actor_revision
            for session in manifest.sessions
            if session.actor_id == cohort.actor_id
        ):
            raise ValueError("loaded live suffix actor revision differs from the frozen cohort")
        expected_template = next(
            session.chat_template_sha256
            for session in manifest.sessions
            if session.actor_id == cohort.actor_id
        )
        if actor.chat_template_hash != expected_template:
            raise ValueError("loaded live suffix chat template differs from the frozen cohort")
        execution = config.get("execution", {})
        if not isinstance(execution, Mapping):
            raise ValueError("held-out execution config must be a mapping")
        from censure.cli import _bindings_factory

        oracle = LiveAgentSuffixOracle(
            cohort=cohort,
            manifest=manifest,
            behavior_store=store,
            suffix_store=SelectedSuffixRunStore(
                args.out_root,
                str(config["experiment_id"]),
            ),
            actor=actor,
            bindings_factory=lambda scenario: _bindings_factory(scenario)(),
            max_tool_steps=int(execution.get("max_tool_steps", 12)),
            wall_clock_seconds=float(execution.get("wall_clock_seconds", 600)),
            retries=int(execution.get("retries", 0)),
        )

    for policy in policies:
        if policy in completed:
            ledger, points = completed[policy]
        else:
            if oracle is None:  # pragma: no cover - preflight invariant
                raise AssertionError("live suffix oracle was not constructed")
            auditor = CensureAuditor(
                envelope=cohort.envelope,
                oracle=oracle,
                policy=policy,
                allocation_seed=allocation_seed,
                alpha=0.05,
                exploration_epsilon=0.10,
            )
            template = auditor.initial_ledger()
            existing = (
                auditor_store.read_ledger(template) if auditor_store.has_ledger(template) else None
            )
            ledger, points = auditor.run(total_rounds=max_rounds, ledger=existing)
            auditor_store.write_ledger(ledger)
            auditor_store.write_certificate_path(ledger, points)

            selected_ids = {disclosure.candidate_id for disclosure in ledger.disclosures}
            for candidate_id in sorted(selected_ids):
                oracle.evaluate_selected(candidate_id)
            diagnostics = {
                candidate_id: oracle.diagnostics[candidate_id] for candidate_id in selected_ids
            }
            cohort_store.write_private_diagnostics(
                cohort=cohort,
                policy=policy.value,
                allocation_seed=allocation_seed,
                diagnostics=diagnostics,
            )
        for fraction, planned_round in budget_rounds.items():
            round_index = min(planned_round, len(ledger.disclosures))
            point = points[round_index]
            rows.append(
                {
                    "actor_id": cohort.actor_id,
                    "cohort_id": cohort.cohort_id,
                    "cohort_sha256": cohort.cohort_sha256,
                    "policy": policy.value,
                    "allocation_seed": allocation_seed,
                    "budget_fraction": float(fraction),
                    "planned_round": planned_round,
                    "realized_round": round_index,
                    "candidate_count": len(cohort.envelope.candidates),
                    "auditable_candidate_count": len(cohort.envelope.auditable_candidates),
                    "certificate": point.model_dump(mode="json"),
                }
            )

    payload = {
        "schema_version": "censure.agent-audit-run-summary.v1",
        "protocol_id": collection.protocol_id,
        "source_manifest_sha256": collection.source_manifest_sha256,
        "collection_sha256": collection.collection_sha256,
        "actor_alias": args.model,
        "actor_id": cohort.actor_id,
        "policies": [policy.value for policy in policies],
        "budget_fractions": list(AGENT_BUDGET_FRACTIONS),
        "executed_unique_suffix_count": (
            0 if oracle is None else len(set(oracle.executed_candidate_ids))
        ),
        "persisted_cache_hit_count": (
            0 if oracle is None else len(oracle.persisted_cache_candidate_ids)
        ),
        "skipped_complete_policy_count": len(completed),
        "rows": rows,
    }
    result_path = (
        cohort_store.root / "agent_audits" / "actors" / cohort.cohort_id / "run_summary.json"
    )
    digest = atomic_write_json(result_path, payload)
    if actor is not None:
        del actor
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    _json_print(
        {
            "status": "complete",
            "path": str(result_path),
            "sha256": digest,
            "row_count": len(rows),
            "collection_sha256": collection.collection_sha256,
        }
    )
    return 0


def _run_agent_seal(args: argparse.Namespace) -> int:
    config, _freeze, manifest, store = _load_agent_context(args)
    cohort_store = AgentCohortStore(args.out_root, str(config["experiment_id"]))
    collection = cohort_store.read_collection()
    if collection.source_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("agent cohort and manifest hashes differ")
    preexisting_targets = [
        session_id
        for cohort in collection.cohorts
        for session_id in cohort.source_session_ids
        if store.is_complete(session_id=session_id, role="target")
    ]
    if preexisting_targets:
        raise ValueError("cannot seal audits after full target outcomes have been generated")
    auditor_store = AuditorRunStore(args.out_root, str(config["experiment_id"]))
    ledgers, certificates = validate_complete_agent_ledgers(
        collection=collection,
        auditor_store=auditor_store,
    )
    payload = agent_audit_seal_payload(
        collection=collection,
        ledgers=ledgers,
        certificates=certificates,
    )
    digest = cohort_store.write_audit_seal(payload)
    _json_print(
        {
            "status": "sealed",
            "path": str(cohort_store.audit_seal_path),
            "sha256": digest,
            "actor_count": len(collection.cohorts),
            "ledger_count": len(ledgers),
            "full_target_outcomes_present_at_seal": False,
        }
    )
    return 0


def _run_agent_status(args: argparse.Namespace) -> int:
    config, _freeze, manifest, store = _load_agent_context(args)
    cohort_store = AgentCohortStore(args.out_root, str(config["experiment_id"]))
    collection = cohort_store.read_collection()
    if collection.source_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("agent cohort and manifest hashes differ")
    auditor_store = AuditorRunStore(args.out_root, str(config["experiment_id"]))
    suffix_store = SelectedSuffixRunStore(
        args.out_root,
        str(config["experiment_id"]),
    )
    rows: list[dict[str, Any]] = []
    all_complete = True
    for cohort in collection.cohorts:
        expected_rounds = max(agent_budget_rounds(len(cohort.envelope.candidates)).values())
        if not cohort.envelope.auditable_candidates:
            expected_rounds = 0
        for policy in AllocationPolicyName:
            auditor = CensureAuditor(
                envelope=cohort.envelope,
                oracle=InMemoryEvaluationOracle({}),
                policy=policy,
                allocation_seed=agent_allocation_seed(cohort.cohort_id),
                alpha=0.05,
                exploration_epsilon=0.10,
            )
            template = auditor.initial_ledger()
            completed_rounds = 0
            unique_candidates: set[str] = set()
            valid = False
            if auditor_store.has_ledger(template):
                ledger = auditor_store.read_ledger(template)
                auditor.validate_ledger(ledger)
                completed_rounds = len(ledger.disclosures)
                unique_candidates = {disclosure.candidate_id for disclosure in ledger.disclosures}
                valid = completed_rounds == expected_rounds
            all_complete = all_complete and valid
            rows.append(
                {
                    "actor_id": cohort.actor_id,
                    "cohort_id": cohort.cohort_id,
                    "policy": policy.value,
                    "expected_rounds": expected_rounds,
                    "completed_rounds": completed_rounds,
                    "complete": valid,
                    "unique_selected_candidate_count": len(unique_candidates),
                    "persisted_selected_suffix_count": sum(
                        suffix_store.has_run(
                            cohort_id=cohort.cohort_id,
                            candidate_id=candidate_id,
                        )
                        for candidate_id in unique_candidates
                    ),
                }
            )
    full_target_count = sum(
        store.is_complete(session_id=session_id, role="target")
        for cohort in collection.cohorts
        for session_id in cohort.source_session_ids
    )
    seal_present = cohort_store.audit_seal_path.is_file() and (
        cohort_store.audit_seal_path.with_suffix(".sha256").is_file()
    )
    _json_print(
        {
            "schema_version": "censure.agent-audit-status.v1",
            "protocol_id": collection.protocol_id,
            "collection_sha256": collection.collection_sha256,
            "all_maximum_budget_ledgers_complete": all_complete,
            "full_target_trajectory_count": full_target_count,
            "ready_to_seal": all_complete and full_target_count == 0 and not seal_present,
            "audit_seal_present": seal_present,
            "rows": rows,
        }
    )
    return 0


def _run_agent_summarize(args: argparse.Namespace) -> int:
    config, _freeze, manifest, store = _load_agent_context(args)
    cohort_store = AgentCohortStore(args.out_root, str(config["experiment_id"]))
    collection = cohort_store.read_collection()
    auditor_store = AuditorRunStore(args.out_root, str(config["experiment_id"]))
    payload = summarize_agent_audit_study(
        collection=collection,
        manifest=manifest,
        run_store=store,
        auditor_store=auditor_store,
        cohort_store=cohort_store,
    )
    result_path = cohort_store.root / "agent_audits" / "study_summary.json"
    digest = atomic_write_json(result_path, payload)
    _json_print(
        {
            "status": "complete",
            "path": str(result_path),
            "sha256": digest,
            "actor_count": len(payload["actor_rows"]),
            "audit_row_count": len(payload["audit_rows"]),
            "post_audit_full_oracle_revealed": True,
        }
    )
    return 0


def _add_protocol_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-config", type=Path, default=Path(DEFAULT_BASE_CONFIG))
    parser.add_argument("--amendment-1", type=Path, default=Path(DEFAULT_AMENDMENT_1))
    parser.add_argument("--amendment-2", type=Path, default=Path(DEFAULT_AMENDMENT_2))
    parser.add_argument("--amendment-3", type=Path, default=Path(DEFAULT_AMENDMENT_3))
    parser.add_argument("--amendment-4", type=Path, default=Path(DEFAULT_AMENDMENT_4))


def _add_store_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default="phase2_estimator_v1")
    parser.add_argument("--purpose", choices=("all", "validity", "efficiency"), default="all")


def _add_robustness_store_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default="phase2_estimator_v1")


def _add_shard(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)


def _add_agent_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_AGENT_CONFIG))
    parser.add_argument("--freeze", type=Path, default=Path(DEFAULT_AGENT_FREEZE))
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="censure-phase2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="inspect the frozen calibration catalog")
    _add_protocol_paths(catalog)
    catalog.set_defaults(handler=_run_catalog)

    robustness_catalog = subparsers.add_parser(
        "robustness-catalog", help="inspect the frozen robustness catalog"
    )
    _add_protocol_paths(robustness_catalog)
    robustness_catalog.set_defaults(handler=_run_robustness_catalog)

    shared_support_catalog = subparsers.add_parser(
        "shared-support-catalog", help="inspect the frozen shared-support OPE catalog"
    )
    _add_protocol_paths(shared_support_catalog)
    shared_support_catalog.set_defaults(handler=_run_shared_support_catalog)

    run = subparsers.add_parser("run-calibration", help="run a deterministic CPU calibration shard")
    _add_protocol_paths(run)
    _add_store_paths(run)
    _add_shard(run)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-work-items", type=int)
    run.add_argument("--progress-every", type=int, default=25)
    run.set_defaults(handler=_run_calibration)

    status = subparsers.add_parser("calibration-status", help="count checksum-valid work items")
    _add_protocol_paths(status)
    _add_store_paths(status)
    _add_shard(status)
    status.set_defaults(handler=_run_status, max_work_items=None, progress_every=1)

    summarize = subparsers.add_parser(
        "summarize-calibration", help="summarize only after all selected cells are complete"
    )
    _add_protocol_paths(summarize)
    _add_store_paths(summarize)
    summarize.set_defaults(handler=_run_summarize)

    run_robustness = subparsers.add_parser(
        "run-robustness", help="run a deterministic CPU robustness shard"
    )
    _add_protocol_paths(run_robustness)
    _add_robustness_store_paths(run_robustness)
    _add_shard(run_robustness)
    run_robustness.add_argument("--resume", action="store_true")
    run_robustness.add_argument("--max-work-items", type=int)
    run_robustness.add_argument("--progress-every", type=int, default=25)
    run_robustness.set_defaults(handler=_run_robustness)

    robustness_status = subparsers.add_parser(
        "robustness-status", help="count checksum-valid robustness chunks"
    )
    _add_protocol_paths(robustness_status)
    _add_robustness_store_paths(robustness_status)
    _add_shard(robustness_status)
    robustness_status.set_defaults(
        handler=_run_robustness_status, max_work_items=None, progress_every=1
    )

    robustness_summarize = subparsers.add_parser(
        "summarize-robustness", help="summarize only after every robustness cell is complete"
    )
    _add_protocol_paths(robustness_summarize)
    _add_robustness_store_paths(robustness_summarize)
    robustness_summarize.set_defaults(handler=_run_robustness_summarize)

    run_shared_support = subparsers.add_parser(
        "run-shared-support", help="run a deterministic shared-support OPE shard"
    )
    _add_protocol_paths(run_shared_support)
    _add_robustness_store_paths(run_shared_support)
    _add_shard(run_shared_support)
    run_shared_support.add_argument("--resume", action="store_true")
    run_shared_support.add_argument("--max-work-items", type=int)
    run_shared_support.add_argument("--progress-every", type=int, default=25)
    run_shared_support.set_defaults(handler=_run_shared_support)

    shared_support_status = subparsers.add_parser(
        "shared-support-status", help="count checksum-valid shared-support chunks"
    )
    _add_protocol_paths(shared_support_status)
    _add_robustness_store_paths(shared_support_status)
    _add_shard(shared_support_status)
    shared_support_status.set_defaults(
        handler=_run_shared_support_status, max_work_items=None, progress_every=1
    )

    shared_support_summarize = subparsers.add_parser(
        "summarize-shared-support",
        help="summarize only after every shared-support cell is complete",
    )
    _add_protocol_paths(shared_support_summarize)
    _add_robustness_store_paths(shared_support_summarize)
    shared_support_summarize.set_defaults(handler=_run_shared_support_summarize)

    agent_cohort = subparsers.add_parser(
        "freeze-agent-cohort",
        help="freeze behavior-derived held-out frontiers before target execution",
    )
    _add_agent_paths(agent_cohort)
    agent_cohort.set_defaults(handler=_run_agent_cohort)

    agent_audits = subparsers.add_parser(
        "run-agent-audits",
        help="run propensity-recorded audits against private held-out targets",
    )
    _add_agent_paths(agent_audits)
    agent_audits.add_argument(
        "--policy",
        choices=("all", *(policy.value for policy in AllocationPolicyName)),
        default="all",
    )
    agent_audits.add_argument(
        "--model",
        required=True,
        help="one frozen actor alias; load and audit one GPU model at a time",
    )
    agent_audits.add_argument("--resume", action="store_true")
    agent_audits.set_defaults(handler=_run_agent_audits)

    agent_seal = subparsers.add_parser(
        "seal-agent-audits",
        help="commit every maximum-budget ledger before releasing full target outcomes",
    )
    _add_agent_paths(agent_seal)
    agent_seal.set_defaults(handler=_run_agent_seal)

    agent_status = subparsers.add_parser(
        "agent-audit-status",
        help="inspect actor/policy ledger completion without loading a model",
    )
    _add_agent_paths(agent_status)
    agent_status.set_defaults(handler=_run_agent_status)

    agent_summarize = subparsers.add_parser(
        "summarize-agent-audits",
        help="reveal full targets only after every maximum-budget ledger is frozen",
    )
    _add_agent_paths(agent_summarize)
    agent_summarize.set_defaults(handler=_run_agent_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
