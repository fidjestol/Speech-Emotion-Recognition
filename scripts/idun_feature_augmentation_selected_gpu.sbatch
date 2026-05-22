#!/usr/bin/env bash
#SBATCH --job-name=feat-aug-sel-gpu
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/slurm-feature-augmentation-selected-gpu-%j.out
#SBATCH --error=logs/slurm-feature-augmentation-selected-gpu-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

source .venv-idun/bin/activate

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export ENABLE_GPU_MODELS="${ENABLE_GPU_MODELS:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RUN_NAME="${RUN_NAME:-feature_aug_selected_$(date +%Y%m%d_%H%M%S)}"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-${RUN_NAME}_sources}"
SELECTED_METHODS="${SELECTED_METHODS:-anova pca ga}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ser-feature-augmentation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RUN_BOTH_VARIANTS="${RUN_BOTH_VARIANTS:-0}"
INCLUDE_XXX="${INCLUDE_XXX:-0}"
REQUIRE_AGREEMENT="${REQUIRE_AGREEMENT:-1}"
TARGET_PER_CLASS="${TARGET_PER_CLASS:-10000}"
GROUP_SIZE="${GROUP_SIZE:-5}"
USE_NROWS="${USE_NROWS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SAVE_AUGMENTED_MATRICES="${SAVE_AUGMENTED_MATRICES:-0}"
SAVE_MODELS="${SAVE_MODELS:-1}"
TRAIN_AUGMENTED_FINAL_STACKING="${TRAIN_AUGMENTED_FINAL_STACKING:-0}"
EXCLUDED_EMOTIONS="${EXCLUDED_EMOTIONS:-sur fea oth dis}"
MODELS="${MODELS:-logreg linear_svc_cal polynomial_svc_cal xgboost catboost soft_voting stacking}"
DATASET_KEYS_OVERRIDE="${DATASET_KEYS_OVERRIDE:-}"
SKIP_PREP="${SKIP_PREP:-0}"

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Run name: ${RUN_NAME}"
echo "Source bundle: ${SOURCE_BUNDLE}"
echo "Selected methods: ${SELECTED_METHODS}"
echo "Report to: ${REPORT_TO}"
echo "Enable GPU models: ${ENABLE_GPU_MODELS}"
echo "Models: ${MODELS}"
echo "Train augmented_final stacking: ${TRAIN_AUGMENTED_FINAL_STACKING}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"

python - <<'PY'
import json
import socket
import torch
print(json.dumps({
    "host": socket.gethostname(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}, indent=2))
PY

prep_cmd=(
  python feature_augmentation/mean_std_oversampling/prepare_selected_feature_sources.py
  --bundle-name "${SOURCE_BUNDLE}"
  --methods
)

for method in ${SELECTED_METHODS}; do
  prep_cmd+=("${method}")
done

prep_cmd+=(--excluded-emotions)
for emotion in ${EXCLUDED_EMOTIONS}; do
  prep_cmd+=("${emotion}")
done

if [[ "${INCLUDE_XXX}" == "1" ]]; then
  prep_cmd+=(--include-xxx)
else
  prep_cmd+=(--no-include-xxx)
fi
if [[ "${REQUIRE_AGREEMENT}" == "1" ]]; then
  prep_cmd+=(--require-agreement)
else
  prep_cmd+=(--no-require-agreement)
fi

if [[ "${SKIP_PREP}" == "1" ]]; then
  echo "Skipping source preparation and reusing existing bundle."
else
  printf 'Prepare command:\n  %q' "${prep_cmd[@]}"
  printf '\n'
  "${prep_cmd[@]}"
fi

MANIFEST_PATH="${REPO_ROOT}/feature_augmentation/selected_feature_sources/${SOURCE_BUNDLE}/manifest.json"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi
export SER_FEATURE_SOURCE_MANIFEST="${MANIFEST_PATH}"

if [[ -n "${DATASET_KEYS_OVERRIDE}" ]]; then
  DATASET_KEYS="${DATASET_KEYS_OVERRIDE}"
else
  DATASET_KEYS="$(
    python - <<'PY'
import json
import os
from pathlib import Path
manifest = json.loads(Path(os.environ["SER_FEATURE_SOURCE_MANIFEST"]).read_text())
print(" ".join(manifest["datasets"].keys()))
PY
  )"
fi

echo "Dataset keys: ${DATASET_KEYS}"
echo "Manifest path: ${SER_FEATURE_SOURCE_MANIFEST}"

cmd=(
  python feature_augmentation/mean_std_oversampling/run_all_augmentation.py
  --run-name "${RUN_NAME}"
  --use-variant-dirs
  --report-to "${REPORT_TO}"
  --wandb-project "${WANDB_PROJECT}"
  --target-per-class "${TARGET_PER_CLASS}"
  --group-size "${GROUP_SIZE}"
  --dataset-keys
)

for key in ${DATASET_KEYS}; do
  cmd+=("${key}")
done

cmd+=(--excluded-emotions)
for emotion in ${EXCLUDED_EMOTIONS}; do
  cmd+=("${emotion}")
done

cmd+=(--models)
for model in ${MODELS}; do
  cmd+=("${model}")
done

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${RUN_BOTH_VARIANTS}" == "1" ]]; then
  cmd+=(--run-both-variants)
fi
if [[ "${INCLUDE_XXX}" == "1" ]]; then
  cmd+=(--include-xxx)
fi
if [[ "${REQUIRE_AGREEMENT}" == "1" ]]; then
  cmd+=(--require-agreement)
else
  cmd+=(--no-require-agreement)
fi
if [[ -n "${USE_NROWS}" ]]; then
  cmd+=(--use-nrows "${USE_NROWS}")
fi
if [[ "${ENABLE_GPU_MODELS}" == "1" ]]; then
  cmd+=(--enable-gpu-models)
else
  cmd+=(--no-enable-gpu-models)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
else
  cmd+=(--no-skip-existing)
fi
if [[ "${SAVE_AUGMENTED_MATRICES}" == "1" ]]; then
  cmd+=(--save-augmented-matrices)
else
  cmd+=(--no-save-augmented-matrices)
fi
if [[ "${SAVE_MODELS}" == "1" ]]; then
  cmd+=(--save-models)
else
  cmd+=(--no-save-models)
fi
if [[ "${TRAIN_AUGMENTED_FINAL_STACKING}" == "1" ]]; then
  cmd+=(--train-augmented-final-stacking)
else
  cmd+=(--no-train-augmented-final-stacking)
fi

printf 'Run command:\n  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "Latest run marker: feature_augmentation/mean_std_oversampling/artifacts/latest_run.txt"
echo "End: $(date)"
