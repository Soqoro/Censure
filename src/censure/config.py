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
