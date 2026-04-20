#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REQUIRED_CPUS="${REQUIRED_CPUS:-16}"
REQUIRED_MEM_GB="${REQUIRED_MEM_GB:-64}"
GPU_PREFERENCE="${GPU_PREFERENCE:-h100,h200,a100,v100,p100}"
SBATCH_OUTPUT_MODE="${SBATCH_OUTPUT_MODE:-submit}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-45}"

select_node() {
python3 - <<'PY'
import os
import re
import subprocess
import sys

required_cpus = int(os.environ.get("REQUIRED_CPUS", "16"))
required_mem_gb = float(os.environ.get("REQUIRED_MEM_GB", "64"))
gpu_preference = [part.strip().lower() for part in os.environ.get("GPU_PREFERENCE", "h100,h200,a100,v100,p100").split(",") if part.strip()]
gpu_rank = {gpu: idx for idx, gpu in enumerate(gpu_preference)}

def parse_tres(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values

def mem_to_mb(raw: str) -> float:
    match = re.match(r"(\d+)([KMGTP]?)", raw.strip())
    if not match:
        return 0.0
    number = float(match.group(1))
    unit = match.group(2)
    scale = {
        "": 1.0 / (1024.0 * 1024.0),
        "K": 1.0 / 1024.0,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024.0,
        "P": 1024.0 * 1024.0 * 1024.0,
    }
    return number * scale[unit]

node_names = [
    line.strip()
    for line in subprocess.check_output(["sinfo", "-N", "-h", "-p", "GPUQ", "-o", "%N"], text=True).splitlines()
    if line.strip()
]
eligible: list[tuple[int, int, int, str, str, float, int]] = []

for node_name in node_names:
    block = subprocess.check_output(["scontrol", "show", "node", node_name], text=True)
    if "Partitions=GPUQ" not in block:
        continue

    node_name_match = re.search(r"NodeName=(\S+)", block)
    gres_match = re.search(r"Gres=(\S+)", block)
    state_match = re.search(r"State=(\S+)", block)
    cfg_match = re.search(r"CfgTRES=([^\n]+)", block)
    alloc_match = re.search(r"AllocTRES=([^\n]+)", block)
    if not (node_name_match and gres_match and state_match and cfg_match):
        continue

    node_name = node_name_match.group(1)
    gres = gres_match.group(1)
    state = state_match.group(1)
    if any(flag in state for flag in ("DOWN", "DRAIN", "FAIL")):
        continue

    gpu_type_match = re.search(r"gpu:([a-zA-Z0-9]+):(\d+)", gres)
    if not gpu_type_match:
        continue
    gpu_type = gpu_type_match.group(1).lower()

    cfg_tres = parse_tres(cfg_match.group(1))
    alloc_tres = parse_tres(alloc_match.group(1)) if alloc_match else {}
    cfg_cpu = int(cfg_tres.get("cpu", "0"))
    alloc_cpu = int(alloc_tres.get("cpu", "0"))
    cfg_gpu = int(cfg_tres.get("gres/gpu", "0"))
    alloc_gpu = int(alloc_tres.get("gres/gpu", "0"))
    cfg_mem_mb = mem_to_mb(cfg_tres.get("mem", "0M"))
    alloc_mem_mb = mem_to_mb(alloc_tres.get("mem", "0M")) if "mem" in alloc_tres else 0.0

    free_cpu = cfg_cpu - alloc_cpu
    free_gpu = cfg_gpu - alloc_gpu
    free_mem_gb = (cfg_mem_mb - alloc_mem_mb) / 1024.0
    if free_cpu < required_cpus or free_gpu < 1 or free_mem_gb < required_mem_gb:
        continue

    pref_rank = gpu_rank.get(gpu_type, len(gpu_rank))
    state_penalty = 0 if "IDLE" in state else 1
    eligible.append((pref_rank, state_penalty, -free_gpu, node_name, gpu_type, free_mem_gb, free_cpu))

if not eligible:
    sys.exit(1)

eligible.sort()
_, _, _, node_name, gpu_type, free_mem_gb, free_cpu = eligible[0]
print(f"{node_name}|gpu:{gpu_type}:1|free_cpu={free_cpu}|free_mem_gb={free_mem_gb:.1f}")
PY
}

waited_seconds=0

wait_for_node() {
  while true; do
    local selection
    selection="$(select_node)" || true
    if [[ -n "${selection}" ]]; then
      echo "${selection}"
      return 0
    fi

    if [[ "${WAIT_FOR_GPU}" != "1" ]]; then
      break
    fi

    if (( MAX_WAIT_SECONDS > 0 && waited_seconds >= MAX_WAIT_SECONDS )); then
      break
    fi

    echo "No eligible GPUQ node yet for ${REQUIRED_CPUS} CPUs / ${REQUIRED_MEM_GB} GB / 1 GPU. Retrying in ${POLL_SECONDS}s..." >&2
    sleep "${POLL_SECONDS}"
    waited_seconds=$((waited_seconds + POLL_SECONDS))
  done

  if [[ "${WAIT_FOR_GPU}" == "1" && "${MAX_WAIT_SECONDS}" != "0" ]]; then
    echo "No eligible GPUQ node appeared within ${MAX_WAIT_SECONDS}s for >= ${REQUIRED_CPUS} CPUs, >= ${REQUIRED_MEM_GB} GB RAM, and >= 1 free GPU." >&2
  else
    echo "No eligible GPUQ node currently has >= ${REQUIRED_CPUS} CPUs, >= ${REQUIRED_MEM_GB} GB RAM, and >= 1 free GPU." >&2
  fi
  return 1
}

wait_for_job_start() {
  local job_id="$1"
  local waited=0
  while (( waited < START_TIMEOUT_SECONDS )); do
    local status
    status="$(squeue -j "${job_id}" -h -o '%T|%R')" || true
    if [[ -z "${status}" ]]; then
      return 0
    fi

    local state="${status%%|*}"
    local reason="${status#*|}"
    if [[ "${state}" == "RUNNING" || "${state}" == "CONFIGURING" || "${state}" == "COMPLETING" ]]; then
      echo "Job ${job_id} entered state ${state} on ${reason}."
      return 0
    fi

    sleep 5
    waited=$((waited + 5))
  done

  return 1
}

while true; do
  selection="$(wait_for_node)" || exit 1

  IFS='|' read -r SELECTED_NODE SELECTED_GRES SELECTED_META <<<"${selection}"
  echo "Selected GPU node: ${SELECTED_NODE}"
  echo "Selected GRES: ${SELECTED_GRES}"
  echo "Capacity: ${SELECTED_META}"

  if [[ "${SBATCH_OUTPUT_MODE}" == "show" ]]; then
    echo "sbatch --partition=GPUQ --nodelist=${SELECTED_NODE} --gres=${SELECTED_GRES} --cpus-per-task=${REQUIRED_CPUS} --mem=${REQUIRED_MEM_GB}G $* scripts/idun_feature_augmentation_gpu.sbatch"
    exit 0
  fi

  job_id="$(
    sbatch --parsable \
      --partition=GPUQ \
      --nodelist="${SELECTED_NODE}" \
      --gres="${SELECTED_GRES}" \
      --cpus-per-task="${REQUIRED_CPUS}" \
      --mem="${REQUIRED_MEM_GB}G" \
      "$@" \
      scripts/idun_feature_augmentation_gpu.sbatch
  )"
  echo "Submitted job ${job_id}. Waiting up to ${START_TIMEOUT_SECONDS}s for it to start..."

  if wait_for_job_start "${job_id}"; then
    echo "Job ${job_id} accepted without falling back to the normal queue."
    exit 0
  fi

  status="$(squeue -j "${job_id}" -h -o '%T|%R')" || true
  echo "Job ${job_id} did not start within ${START_TIMEOUT_SECONDS}s (${status:-unknown}). Canceling and retrying..." >&2
  scancel "${job_id}" || true

  if [[ "${WAIT_FOR_GPU}" != "1" ]]; then
    exit 1
  fi
done
