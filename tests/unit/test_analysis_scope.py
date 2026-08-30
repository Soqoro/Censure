from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from censure.analysis_scope import (
    AnalysisScopeConfig,
    load_analysis_scope,
    resolve_analysis_scope,
)
from censure.config import ConfigurationError, resolved_experiment_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCOPE_PATH = REPOSITORY_ROOT / "configs" / "analysis" / "exp1_qwen_gemma_v1.yaml"
EXPERIMENT_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "exp1_full_v2.yaml"


def test_qwen_gemma_scope_is_complete_outcome_free_and_resolvable() -> None:
    scope = load_analysis_scope(SCOPE_PATH)
    experiment = resolved_experiment_config(EXPERIMENT_PATH, resolve_remote=False)
    resolved = resolve_analysis_scope(scope, experiment)

    assert scope.included_actor_aliases == ("qwen3_8b", "gemma3_12b")
    assert [item.actor_alias for item in scope.excluded_actors] == ["llama31_8b"]
    assert scope.excluded_actors[0].outcome_values_inspected is False
    assert scope.excluded_actors[0].invalid_pair_rate_lower_bound == pytest.approx(81 / 172)
    assert resolved.included_actor_ids == (
        "Qwen/Qwen3-8B",
        "google/gemma-3-12b-it",
    )
    assert len(resolved.sha256) == 64


def test_scope_rejects_inconsistent_status_evidence() -> None:
    raw = load_analysis_scope(SCOPE_PATH).model_dump(mode="python")
    raw["excluded_actors"][0]["invalid_pair_rate_lower_bound"] = 0.2

    with pytest.raises(ValidationError, match="does not match the status counts"):
        AnalysisScopeConfig.model_validate(raw)


def test_scope_rejects_outcome_inspection_and_actor_overlap() -> None:
    raw = load_analysis_scope(SCOPE_PATH).model_dump(mode="python")
    raw["excluded_actors"][0]["outcome_values_inspected"] = True
    with pytest.raises(ValidationError, match="False"):
        AnalysisScopeConfig.model_validate(raw)

    raw = load_analysis_scope(SCOPE_PATH).model_dump(mode="python")
    raw["excluded_actors"][0]["actor_alias"] = "qwen3_8b"
    with pytest.raises(ValidationError, match="both included and excluded"):
        AnalysisScopeConfig.model_validate(raw)


def test_scope_must_disposition_every_frozen_actor() -> None:
    scope = load_analysis_scope(SCOPE_PATH)
    experiment = resolved_experiment_config(EXPERIMENT_PATH, resolve_remote=False)
    experiment["actors"] = [*experiment["actors"], "future_actor"]
    experiment["resolved_models"]["future_actor"] = {"model_id": "example/future"}

    with pytest.raises(ConfigurationError, match="does not disposition"):
        resolve_analysis_scope(scope, experiment)
