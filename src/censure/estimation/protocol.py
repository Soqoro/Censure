"""Resolve the frozen Phase 2 protocol and calibration work catalog."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from censure.config import ConfigurationError, load_yaml
from censure.estimation.calibration import CalibrationCellSpec
from censure.estimation.enumerable import SupportRegime
from censure.estimation.schemas import AllocationPolicyName
from censure.schemas import FrozenModel, Sha256Hex
from censure.serialization import canonical_sha256

CalibrationPurpose = Literal["validity", "efficiency"]


class CalibrationCatalogEntry(FrozenModel):
    schema_version: Literal["censure.calibration-catalog-entry.v1"] = (
        "censure.calibration-catalog-entry.v1"
    )
    purposes: tuple[CalibrationPurpose, ...]
    spec: CalibrationCellSpec

    @model_validator(mode="after")
    def validate_purposes(self) -> CalibrationCatalogEntry:
        if not self.purposes:
            raise ValueError("calibration catalog entry requires a purpose")
        if tuple(sorted(set(self.purposes))) != self.purposes:
            raise ValueError("calibration purposes must be unique and sorted")
        return self


class FrozenCalibrationCatalog(FrozenModel):
    schema_version: Literal["censure.frozen-calibration-catalog.v1"] = (
        "censure.frozen-calibration-catalog.v1"
    )
    protocol_id: str
    base_config_sha256: Sha256Hex
    amendment_1_sha256: Sha256Hex
    amendment_2_sha256: Sha256Hex
    repetitions_per_chunk: Annotated[int, Field(ge=1)]
    entries: Annotated[tuple[CalibrationCatalogEntry, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_cell_ids(self) -> FrozenCalibrationCatalog:
        cell_ids = [entry.spec.cell_id for entry in self.entries]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("frozen calibration catalog contains duplicate cell IDs")
        return self

    @property
    def catalog_sha256(self) -> str:
        return canonical_sha256(self)

    @property
    def specs(self) -> tuple[CalibrationCellSpec, ...]:
        return tuple(entry.spec for entry in self.entries)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"Phase 2 protocol field {field!r} must be a mapping")
    return value


def _tuple_float(value: Any, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"Phase 2 protocol field {field!r} must be a list")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Phase 2 protocol field {field!r} is not numeric") from exc


def _tuple_int(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ConfigurationError(f"Phase 2 protocol field {field!r} must be an integer list")
    return tuple(value)


def load_frozen_calibration_catalog(
    *,
    base_config_path: str | Path,
    amendment_1_path: str | Path,
    amendment_2_path: str | Path,
) -> FrozenCalibrationCatalog:
    base = load_yaml(base_config_path)
    amendment_1 = load_yaml(amendment_1_path)
    amendment_2 = load_yaml(amendment_2_path)
    protocol_id = str(base.get("protocol_id", ""))
    if protocol_id != "censure-phase2-estimator-v1":
        raise ConfigurationError("unexpected Phase 2 protocol ID")
    if amendment_1.get("parent_protocol_id") != protocol_id:
        raise ConfigurationError("Phase 2 amendment 1 does not target the base protocol")
    if amendment_2.get("parent_amendment_id") != amendment_1.get("amendment_id"):
        raise ConfigurationError("Phase 2 amendment 2 does not target amendment 1")

    auditing = _mapping(base.get("auditing"), field="auditing")
    confidence = _mapping(base.get("confidence"), field="confidence")
    calibration = _mapping(base.get("calibration"), field="calibration")
    dgp = _mapping(amendment_1.get("enumerable_dgp"), field="enumerable_dgp")
    validity = _mapping(amendment_2.get("validity_grid"), field="validity_grid")
    efficiency = _mapping(amendment_2.get("efficiency_grid"), field="efficiency_grid")
    execution = _mapping(amendment_2.get("execution"), field="execution")

    base_budgets = _tuple_float(auditing.get("budgets_fraction"), field="auditing.budgets_fraction")
    validity_budgets = _tuple_float(
        validity.get("budgets_fraction"), field="validity_grid.budgets_fraction"
    )
    efficiency_budgets = _tuple_float(
        efficiency.get("budgets_fraction"), field="efficiency_grid.budgets_fraction"
    )
    if validity_budgets != base_budgets or efficiency_budgets != base_budgets:
        raise ConfigurationError("amendment budgets differ from the base protocol")

    common = {
        "protocol_id": protocol_id,
        "seed_namespace": str(calibration["seed_namespace"]),
        "base_seed": int(execution["base_seed"]),
        "mixed_auditable_probability": float(dgp["mixed_auditable_probability"]),
        "delayed_harm_probability": float(dgp["delayed_harm_probability"]),
        "budget_fractions": base_budgets,
        "alpha": float(confidence["cohort_alpha_audit"]),
        "exploration_epsilon": float(auditing["exploration_epsilon"]),
        "release_threshold_eta": float(calibration["release_threshold_eta"]),
    }
    entries_by_id: dict[str, CalibrationCatalogEntry] = {}

    validity_regime_masses = {
        SupportRegime.FULL_OVERLAP: _tuple_float(
            validity.get("full_overlap_zero_support_mass"),
            field="validity_grid.full_overlap_zero_support_mass",
        ),
        SupportRegime.DETERMINISTIC_CLONEABLE_NONOVERLAP: _tuple_float(
            validity.get("nonoverlap_zero_support_mass"),
            field="validity_grid.nonoverlap_zero_support_mass",
        ),
        SupportRegime.MIXED_AUDITABILITY: _tuple_float(
            validity.get("nonoverlap_zero_support_mass"),
            field="validity_grid.nonoverlap_zero_support_mass",
        ),
    }
    validity_policy = AllocationPolicyName(str(validity["policy"]))
    validity_sizes = _tuple_int(validity.get("cohort_sizes"), field="validity_grid.cohort_sizes")
    validity_harm = _tuple_float(
        validity.get("target_harm_prevalence"),
        field="validity_grid.target_harm_prevalence",
    )
    for regime, cohort_size, harm_prevalence in itertools.product(
        SupportRegime, validity_sizes, validity_harm
    ):
        for zero_support_mass in validity_regime_masses[regime]:
            spec = CalibrationCellSpec(
                **common,
                support_regime=regime,
                cohort_size=cohort_size,
                target_harm_prevalence=harm_prevalence,
                zero_support_mass=zero_support_mass,
                policy=validity_policy,
                repetitions=int(validity["repetitions"]),
            )
            entries_by_id[spec.cell_id] = CalibrationCatalogEntry(
                purposes=("validity",), spec=spec
            )

    efficiency_regimes = tuple(
        SupportRegime(str(value)) for value in efficiency["support_regimes"]
    )
    efficiency_policies = tuple(
        AllocationPolicyName(str(value)) for value in efficiency["policies"]
    )
    efficiency_sizes = _tuple_int(
        efficiency.get("cohort_sizes"), field="efficiency_grid.cohort_sizes"
    )
    efficiency_harm = _tuple_float(
        efficiency.get("target_harm_prevalence"),
        field="efficiency_grid.target_harm_prevalence",
    )
    efficiency_mass = _tuple_float(
        efficiency.get("zero_support_mass"), field="efficiency_grid.zero_support_mass"
    )
    for regime, cohort_size, harm_prevalence, zero_support_mass, policy in itertools.product(
        efficiency_regimes,
        efficiency_sizes,
        efficiency_harm,
        efficiency_mass,
        efficiency_policies,
    ):
        spec = CalibrationCellSpec(
            **common,
            support_regime=regime,
            cohort_size=cohort_size,
            target_harm_prevalence=harm_prevalence,
            zero_support_mass=zero_support_mass,
            policy=policy,
            repetitions=int(efficiency["repetitions"]),
        )
        existing = entries_by_id.get(spec.cell_id)
        purposes: tuple[CalibrationPurpose, ...] = (
            ("efficiency",) if existing is None else ("efficiency", "validity")
        )
        entries_by_id[spec.cell_id] = CalibrationCatalogEntry(
            purposes=purposes, spec=spec
        )

    entries = tuple(
        sorted(
            entries_by_id.values(),
            key=lambda entry: (
                entry.spec.support_regime.value,
                entry.spec.cohort_size,
                entry.spec.target_harm_prevalence,
                entry.spec.zero_support_mass,
                entry.spec.policy.value,
            ),
        )
    )
    return FrozenCalibrationCatalog(
        protocol_id=protocol_id,
        base_config_sha256=canonical_sha256(base),
        amendment_1_sha256=canonical_sha256(amendment_1),
        amendment_2_sha256=canonical_sha256(amendment_2),
        repetitions_per_chunk=int(execution["repetitions_per_chunk"]),
        entries=entries,
    )


__all__ = [
    "CalibrationCatalogEntry",
    "FrozenCalibrationCatalog",
    "load_frozen_calibration_catalog",
]
