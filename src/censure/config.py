"""Configuration loading and immutable model-revision resolution."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from censure.serialization import canonical_sha256


class ConfigurationError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {config_path}")
    return data


def model_config_path(actor_id: str, root: str | Path = "configs/models") -> Path:
    path = Path(root) / f"{actor_id}.yaml"
    if not path.is_file():
        known = ", ".join(item.stem for item in sorted(Path(root).glob("*.yaml")))
        raise ConfigurationError(f"unknown model alias {actor_id!r}; available: {known}")
    return path


def resolve_model_revision(config: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    """Resolve a model branch to a 40-character immutable Hub commit.

    This performs metadata access only; it never downloads tokenizer/model files.
    """

    resolved = copy.deepcopy(config)
    model_id = str(resolved["model_id"])
    revision = resolved.get("model_revision")
    tokenizer_revision = resolved.get("tokenizer_revision")
    if revision != "resolve_at_doctor" and tokenizer_revision != "resolve_at_doctor":
        for label, value in (
            ("model_revision", revision),
            ("tokenizer_revision", tokenizer_revision),
        ):
            if not isinstance(value, str) or len(value) != 40:
                raise ConfigurationError(f"{label} must be a frozen 40-character commit SHA")
        return resolved
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token or os.getenv("HF_TOKEN")).model_info(model_id, revision="main")
    except Exception as exc:
        raise ConfigurationError(
            f"cannot resolve gated model {model_id!r}; accept its license and provide HF_TOKEN: {exc}"
        ) from exc
    if not isinstance(info.sha, str) or len(info.sha) != 40:
        raise ConfigurationError(f"Hub returned no immutable SHA for {model_id}")
    resolved["model_revision"] = info.sha
    resolved["tokenizer_revision"] = info.sha
    return resolved


def resolved_experiment_config(
    experiment_path: str | Path,
    *,
    selected_models: list[str] | None = None,
    resolve_remote: bool = False,
    model_root: str | Path | None = None,
) -> dict[str, Any]:
    experiment = Path(experiment_path).resolve()
    config = load_yaml(experiment)
    _resolve_agentdojo_exclusions(config, experiment)
    actor_ids = selected_models or list(config.get("actors", []))
    if not actor_ids:
        raise ConfigurationError("experiment config must select at least one actor")
    models: dict[str, Any] = {}
    resolved_model_root = (
        Path(model_root) if model_root is not None else experiment.parent.parent / "models"
    )
    for actor_id in actor_ids:
        model = load_yaml(model_config_path(actor_id, resolved_model_root))
        models[actor_id] = resolve_model_revision(model) if resolve_remote else model
    config["resolved_models"] = models
    config["resolved_config_hash"] = canonical_sha256(config)
    return config


def _resolve_agentdojo_exclusions(config: dict[str, Any], experiment_path: Path) -> None:
    """Load a checksummed, outcome-free task-pair exclusion declaration."""

    raw_agent = config.get("agentdojo")
    if not isinstance(raw_agent, dict):
        return
    raw_path = raw_agent.get("exclusion_file")
    if raw_path is None:
        return
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ConfigurationError("agentdojo exclusion_file must be a non-empty path")
    expected_sha256 = raw_agent.get("exclusion_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ConfigurationError("agentdojo exclusion_sha256 must be a SHA-256 digest")

    exclusion_path = Path(raw_path)
    if not exclusion_path.is_absolute():
        exclusion_path = experiment_path.parent / exclusion_path
    document = load_yaml(exclusion_path)
    actual_sha256 = canonical_sha256(document)
    if actual_sha256 != expected_sha256:
        raise ConfigurationError(
            "AgentDojo exclusion declaration hash differs from exclusion_sha256: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    if document.get("schema_version") != "censure.agentdojo-exclusions.v1":
        raise ConfigurationError("unsupported AgentDojo exclusion declaration schema")
    raw_by_suite = document.get("pairs_by_suite")
    if not isinstance(raw_by_suite, dict):
        raise ConfigurationError("AgentDojo exclusion declaration lacks pairs_by_suite")

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_suite, raw_pairs in sorted(raw_by_suite.items()):
        suite = str(raw_suite)
        if not isinstance(raw_pairs, list):
            raise ConfigurationError(f"AgentDojo exclusions for {suite} must be a list")
        for raw_pair in raw_pairs:
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or not all(isinstance(value, str) and value.strip() for value in raw_pair)
            ):
                raise ConfigurationError(
                    f"AgentDojo exclusion for {suite} must be [user_task_id, injection_task_id]"
                )
            identity = (suite, raw_pair[0], raw_pair[1])
            if identity in seen:
                raise ConfigurationError(f"duplicate AgentDojo exclusion: {identity}")
            seen.add(identity)
            normalized.append(
                {
                    "suite": suite,
                    "user_task_id": raw_pair[0],
                    "injection_task_id": raw_pair[1],
                }
            )

    raw_agent["excluded_task_injection_pairs"] = normalized
    raw_agent["resolved_exclusion"] = {
        "schema_version": document["schema_version"],
        "source_experiment_id": str(document.get("source_experiment_id", "unknown")),
        "source_manifest_sha256": str(document.get("source_manifest_sha256", "unknown")),
        "selection": str(document.get("selection", "unknown")),
        "sha256": actual_sha256,
        "pair_count": len(normalized),
    }
