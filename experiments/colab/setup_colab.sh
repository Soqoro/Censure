#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ALIAS="${CENSURE_MODEL:-qwen3_8b}"
CONFIG_PATH="${CENSURE_CONFIG:-${REPO_ROOT}/configs/experiments/exp1_smoke_v2.yaml}"
REQUIREMENTS_PATH="${CENSURE_REQUIREMENTS:-${REPO_ROOT}/requirements/colab-exp1.txt}"
if [[ "${REQUIREMENTS_PATH}" != /* ]]; then
  REQUIREMENTS_PATH="${REPO_ROOT}/${REQUIREMENTS_PATH}"
fi

if [[ -z "${CENSURE_OUT_ROOT:-}" ]]; then
  echo "ERROR: CENSURE_OUT_ROOT is unset. Mount Drive and set it to a Drive directory." >&2
  exit 2
fi

echo "Disk availability before installation:"
df -h /content 2>/dev/null || df -h "${REPO_ROOT}"

ORIGINAL_TORCH="$(${PYTHON_BIN} - <<'PY'
try:
    import torch
    print(torch.__version__)
except ImportError:
    print("MISSING")
PY
)"
if [[ "${ORIGINAL_TORCH}" == "MISSING" ]]; then
  echo "ERROR: no Colab-provided PyTorch installation was found; refusing to install an arbitrary CUDA build." >&2
  exit 2
fi

${PYTHON_BIN} -m pip install --disable-pip-version-check -e "${REPO_ROOT}"
${PYTHON_BIN} -m pip install --disable-pip-version-check -r "${REQUIREMENTS_PATH}"

INSTALLED_TORCH="$(${PYTHON_BIN} -c 'import torch; print(torch.__version__)')"
if [[ "${INSTALLED_TORCH}" != "${ORIGINAL_TORCH}" ]]; then
  echo "ERROR: dependency installation changed PyTorch ${ORIGINAL_TORCH} -> ${INSTALLED_TORCH}." >&2
  echo "Restart with a clean Colab runtime; CENSURE will not mix CUDA builds." >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

${PYTHON_BIN} - "${REPO_ROOT}" "${MODEL_ALIAS}" "${CENSURE_OUT_ROOT}" "${CONFIG_PATH}" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

from censure.adapters.agentdojo_v0135 import compatibility_report
from censure.environments.control import ControlEnvironment, generate_control_scenarios
from censure.guards import StrictGuard
from censure.schemas import ActorMessage, MessageRole, RuleEffect, ToolCall

repo = Path(sys.argv[1])
model_alias = sys.argv[2]
out_root = Path(sys.argv[3]).expanduser()
experiment_path = Path(sys.argv[4])
if not experiment_path.is_absolute():
    experiment_path = repo / experiment_path
model_path = repo / "configs" / "models" / f"{model_alias}.yaml"
model = yaml.safe_load(model_path.read_text())
quantized_smoke = model.get("quantization") == "bitsandbytes_nf4_4bit"
experiment = yaml.safe_load(experiment_path.read_text())
if model_alias not in experiment.get("actors", []):
    raise SystemExit(
        f"ERROR: selected model {model_alias!r} is not in {experiment_path}'s actors list."
    )
if quantized_smoke and experiment.get("primary_analysis_eligible") is not False:
    raise SystemExit("ERROR: quantized smoke config is not explicitly excluded from primary analysis.")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable; requested GPU models will not fall back to CPU.")
props = torch.cuda.get_device_properties(0)
name = props.name
memory_gib = props.total_memory / (1024**3)
print(f"GPU: {name}; memory={memory_gib:.1f} GiB; CUDA={torch.version.cuda}; BF16={torch.cuda.is_bf16_supported()}")
if "H100" in name:
    print("H100: recommended runtime; run BF16 primary experiments with one worker.")
elif "A100" in name:
    print("A100: BF16 is supported; 40 GB cards may require reduced context or one 8B model at a time.")
elif "T4" in name:
    print("T4: BF16 primary runs are unsupported; only the separate 4-bit NF4 smoke config is allowed.")
    if os.getenv("CENSURE_ALLOW_QUANTIZED_SMOKE") != "1" or not quantized_smoke:
        raise SystemExit(
            "ERROR: choose A100/H100, or set CENSURE_MODEL=qwen3_8b_4bit_smoke and "
            "CENSURE_ALLOW_QUANTIZED_SMOKE=1 for non-primary smoke only."
        )
else:
    print("Unknown GPU: verify BF16 and memory manually. Primary results require BF16 and must not be mixed with quantized smoke.")
if not torch.cuda.is_bf16_supported() and not quantized_smoke:
    raise SystemExit("ERROR: GPU lacks BF16 support required by the primary configuration.")
minimum_gpu_gib = float(model.get("minimum_gpu_memory_gib", 0))
if memory_gib < minimum_gpu_gib:
    raise SystemExit(
        f"ERROR: {model_alias} requires at least {minimum_gpu_gib:g} GiB GPU memory; "
        f"this runtime exposes {memory_gib:.1f} GiB."
    )

free_gib = shutil.disk_usage("/content" if Path("/content").exists() else repo).free / (1024**3)
print(f"Free local disk: {free_gib:.1f} GiB")
minimum_disk_gib = float(model.get("minimum_free_disk_gib", 35))
if free_gib < minimum_disk_gib:
    raise SystemExit(
        f"ERROR: less than {minimum_disk_gib:g} GiB free; "
        "select a larger runtime disk or clear /content/hf-cache."
    )

report = compatibility_report()
assert report.package_version == "0.1.35"
assert report.benchmark_version == "v1.2.2"
assert {suite.name for suite in report.suites} == {"workspace", "slack", "travel", "banking"}
print("AgentDojo 0.1.35 / benchmark v1.2.2 verified for all four suites.")

token = os.getenv("HF_TOKEN") or None
downloaded = hf_hub_download(
    repo_id=model["model_id"],
    filename="config.json",
    revision=model["model_revision"],
    token=token,
)
print(f"Model metadata accessible at frozen revision: {model['model_id']}@{model['model_revision']}")
assert Path(downloaded).is_file()

out_root.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=out_root, delete=False, prefix=".censure-write-test-") as handle:
    handle.write("ok")
    temp_name = handle.name
Path(temp_name).unlink()
print(f"Drive output root is writable: {out_root}")

# One genuine parser -> strict guard -> environment mutation round trip.
spec = generate_control_scenarios(domains=("payments",), strata=("clean",), seeds=(0,))[0]
allow = next(
    rule for rule in spec.authorization_policy.rules
    if rule.effect is RuleEffect.ALLOW and ":allow:" in rule.rule_id
)
call = ToolCall(id="doctor-call", name=allow.tool_name, arguments=allow.argument_equals)
guard = StrictGuard()
decision = guard.decide(
    user_request=spec.user_request,
    policy=spec.authorization_policy,
    observable_state=spec.canonical_initial_state,
    history=(ActorMessage(role=MessageRole.USER, content=spec.user_request),),
    proposed_call=call,
)
assert decision.operation_supplied_to_environment is not None
environment = ControlEnvironment(spec)
result = environment.execute(decision.operation_supplied_to_environment)
assert result.ok
print("Synthetic tool-call guard/mutation round trip passed.")
PY

echo "Colab setup complete for ${MODEL_ALIAS}."
