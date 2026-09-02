"""Command-line workflow for CPU-only Phase 2 calibration experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    load_frozen_calibration_catalog,
    load_frozen_robustness_catalog,
)
from censure.estimation.robustness import (
    run_robustness_repetition,
    summarize_robustness_results,
)
from censure.estimation.robustness_storage import RobustnessRunStore
from censure.storage import atomic_write_json

DEFAULT_BASE_CONFIG = "configs/experiments/phase2_estimator_v1.yaml"
DEFAULT_AMENDMENT_1 = "configs/experiments/phase2_estimator_v1_amendment_1.yaml"
DEFAULT_AMENDMENT_2 = "configs/experiments/phase2_estimator_v1_amendment_2.yaml"
DEFAULT_AMENDMENT_3 = "configs/experiments/phase2_estimator_v1_amendment_3.yaml"


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
            calibration_chunk_count(
                entry.spec.repetitions, catalog.repetitions_per_chunk
            )
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
            calibration_chunk_count(
                entry.spec.repetitions, catalog.repetitions_per_chunk
            )
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
        rows = store.read_completed_cell(
            spec, repetitions_per_chunk=catalog.repetitions_per_chunk
        )
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


def _add_protocol_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-config", type=Path, default=Path(DEFAULT_BASE_CONFIG))
    parser.add_argument("--amendment-1", type=Path, default=Path(DEFAULT_AMENDMENT_1))
    parser.add_argument("--amendment-2", type=Path, default=Path(DEFAULT_AMENDMENT_2))
    parser.add_argument("--amendment-3", type=Path, default=Path(DEFAULT_AMENDMENT_3))


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
    robustness_status.set_defaults(handler=_run_robustness_status, max_work_items=None, progress_every=1)

    robustness_summarize = subparsers.add_parser(
        "summarize-robustness", help="summarize only after every robustness cell is complete"
    )
    _add_protocol_paths(robustness_summarize)
    _add_robustness_store_paths(robustness_summarize)
    robustness_summarize.set_defaults(handler=_run_robustness_summarize)
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
