"""Portable run provenance collection with no secret values."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance(repository: str | Path = ".") -> dict[str, Any]:
    repo = Path(repository).resolve()
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            continue
        if name:
            versions[name.lower()] = distribution.version
    versions = dict(sorted(versions.items()))
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
        }
    except ImportError:
        pass
    return {
        "schema_version": "censure.provenance.v1",
        "repository_git_sha": _git(["rev-parse", "HEAD"], repo),
        "repository_dirty": bool(_git(["status", "--porcelain"], repo)),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": versions,
        "agentdojo_benchmark_version": "v1.2.2",
        "cuda": cuda,
    }
