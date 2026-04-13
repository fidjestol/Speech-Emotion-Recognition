#!/usr/bin/env bash
#SBATCH --job-name=feat-aug-gpu
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-feature-augmentation-gpu-%j.out
#SBATCH --error=logs/slurm-feature-augmentation-gpu-%j.err

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

RUN_NAME="${RUN_NAME:-feature_augmentation_gpu_$(date +%Y%m%d_%H%M%S)}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ser-feature-augmentation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DATASET_KEYS="${DATASET_KEYS:-combined mfcc_raw prosody_energy prosody_pitch representations rhythm_pauses ssl_hubert ssl_wav2vec tonality bert}"
RUN_BOTH_VARIANTS="${RUN_BOTH_VARIANTS:-1}"
REQUIRE_AGREEMENT="${REQUIRE_AGREEMENT:-1}"
TARGET_PER_CLASS="${TARGET_PER_CLASS:-10000}"
GROUP_SIZE="${GROUP_SIZE:-5}"
USE_NROWS="${USE_NROWS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SAVE_AUGMENTED_MATRICES="${SAVE_AUGMENTED_MATRICES:-0}"
SAVE_MODELS="${SAVE_MODELS:-1}"
EXCLUDED_EMOTIONS="${EXCLUDED_EMOTIONS:-sur fea oth dis}"
MODELS="${MODELS:-logreg linear_svc_cal polynomial_svc_cal xgboost catboost soft_voting stacking}"

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Run name: ${RUN_NAME}"
echo "Report to: ${REPORT_TO}"
echo "Dataset keys: ${DATASET_KEYS}"
echo "Run both variants: ${RUN_BOTH_VARIANTS}"
echo "Enable GPU models: ${ENABLE_GPU_MODELS}"
echo "Models: ${MODELS}"
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

printf 'Command:\n  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "Latest run marker: feature_augmentation/mean_std_oversampling/artifacts/latest_run.txt"
echo "End: $(date)"
