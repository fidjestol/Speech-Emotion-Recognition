#!/usr/bin/env bash
# End-to-end Idun Slurm launcher for:
# 1. extracting frame-level emotion2vec_base embeddings
# 2. preparing 6-way IEMOCAP splits
# 3. training the downstream non-sequential SER classifier
#
# Submit:
#   sbatch scripts/idun_emotion2vec_downstream_full.sbatch

#SBATCH --job-name=e2v-downstream
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/slurm-e2v-downstream-%j.out
#SBATCH --error=logs/slurm-e2v-downstream-%j.err
#SBATCH --account=share-ie-idi

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
module load FFmpeg/6.0-GCCcore-13.2.0

if [[ ! -f .venv-idun/bin/activate ]]; then
  echo "Missing .venv-idun. Create it before submitting this job."
  exit 1
fi

source .venv-idun/bin/activate

echo "=== Job metadata ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Requested GRES: ${SLURM_JOB_GRES:-${SBATCH_GRES:-unknown}}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

if ! python - <<'PY' >/dev/null 2>&1
import funasr
import huggingface_hub
import modelscope
PY
then
  echo "Missing emotion2vec extraction dependencies in .venv-idun; installing them now."
  bash scripts/setup_emotion2vec_idun_env.sh
  source .venv-idun/bin/activate
fi

if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  export WANDB_API_KEY="$(
    python - <<'PY'
from netrc import netrc
from pathlib import Path
info = netrc(str(Path.home() / ".netrc")).authenticators("api.wandb.ai")
print("" if info is None or not info[2] else info[2])
PY
  )"
fi

WANDB_PROJECT="${WANDB_PROJECT:-emotion2vec-downstream-ser}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-emotion2vec-base-full}"

RUN_NAME="${RUN_NAME:-emotion2vec_base_iemocap_full_$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MODEL_ID:-iic/emotion2vec_base}"
PRIMARY_HUB="${PRIMARY_HUB:-hf}"
FALLBACK_HUB="${FALLBACK_HUB:-ms}"
DATASET_CSV="${DATASET_CSV:-datasets/IEMOCAP/iemocap_full_dataset.csv}"
AUDIO_ROOT="${AUDIO_ROOT:-datasets/IEMOCAP}"
EMBED_ROOT="${EMBED_ROOT:-extracted_features/emotion2vec_base/frame}"
MANIFEST_PATH="${MANIFEST_PATH:-extracted_features/emotion2vec_base/frame_manifest.csv}"
SUMMARY_PATH="${SUMMARY_PATH:-extracted_features/emotion2vec_base/frame_summary.json}"
SPLIT_OUTPUT_DIR="${SPLIT_OUTPUT_DIR:-datasets/IEMOCAP/splits}"
SPLIT_PREFIX="${SPLIT_PREFIX:-iemocap_emotion2vec_base_6way}"
MIN_AGREEMENT="${MIN_AGREEMENT:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
LR_SCHEDULER="${LR_SCHEDULER:-reduce_on_plateau}"
LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-2}"
LR_SCHEDULER_FACTOR="${LR_SCHEDULER_FACTOR:-0.5}"
LR_SCHEDULER_MIN_LR="${LR_SCHEDULER_MIN_LR:-1e-6}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"
WANDB_WATCH_MODEL="${WANDB_WATCH_MODEL:-0}"
RESUME_EXTRACTION="${RESUME_EXTRACTION:-1}"
TRAIN_RESUME_CHECKPOINT="${TRAIN_RESUME_CHECKPOINT:-}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
INCLUDE_EMOTIONS="${INCLUDE_EMOTIONS:-sur ang hap sad neu fru}"

echo "=== W&B ==="
echo "Project: ${WANDB_PROJECT}"
echo "Entity: ${WANDB_ENTITY:-<default>}"
echo "Group: ${WANDB_GROUP}"
echo "Watch model hooks: $([[ "${WANDB_WATCH_MODEL}" == "1" ]] && echo enabled || echo disabled)"
echo "Resume checkpoint: ${TRAIN_RESUME_CHECKPOINT:-<disabled>}"
echo "Early stopping patience: ${EARLY_STOPPING_PATIENCE:-<disabled>}"
echo "Early stopping min delta: ${EARLY_STOPPING_MIN_DELTA}"
echo "LR scheduler: ${LR_SCHEDULER}"
echo "LR scheduler patience: ${LR_SCHEDULER_PATIENCE}"
echo "LR scheduler factor: ${LR_SCHEDULER_FACTOR}"
echo "LR scheduler min lr: ${LR_SCHEDULER_MIN_LR}"
echo "API key present: $([[ -n "${WANDB_API_KEY:-}" ]] && echo yes || echo no)"

EXTRACT_CMD=(
  python -u Emotion2Vec/extraction/extract_hf_embeddings.py
  --dataset-csv "${DATASET_CSV}"
  --audio-root "${AUDIO_ROOT}"
  --output-root "${EMBED_ROOT}"
  --manifest-path "${MANIFEST_PATH}"
  --summary-path "${SUMMARY_PATH}"
  --model-id "${MODEL_ID}"
  --hub "${PRIMARY_HUB}"
  --granularity frame
  --label-col emotion
  --agreement-col agreement
  --path-col path
  --min-agreement "${MIN_AGREEMENT}"
  --include-emotions
)

for emotion in ${INCLUDE_EMOTIONS}; do
  EXTRACT_CMD+=("${emotion}")
done

if [[ "${RESUME_EXTRACTION}" == "1" ]]; then
  EXTRACT_CMD+=(--resume)
else
  EXTRACT_CMD+=(--no-resume)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRACT_CMD+=(--max-samples "${MAX_SAMPLES}")
fi

echo "=== Extraction command ==="
printf '%q ' "${EXTRACT_CMD[@]}"
printf '\n'

if ! "${EXTRACT_CMD[@]}"; then
  if [[ "${FALLBACK_HUB}" != "${PRIMARY_HUB}" ]]; then
    echo "Primary extraction hub ${PRIMARY_HUB} failed; retrying with ${FALLBACK_HUB}."
    EXTRACT_CMD_RETRY=("${EXTRACT_CMD[@]}")
    for idx in "${!EXTRACT_CMD_RETRY[@]}"; do
      if [[ "${EXTRACT_CMD_RETRY[$idx]}" == "${PRIMARY_HUB}" ]]; then
        EXTRACT_CMD_RETRY[$idx]="${FALLBACK_HUB}"
        break
      fi
    done
    printf '%q ' "${EXTRACT_CMD_RETRY[@]}"
    printf '\n'
    "${EXTRACT_CMD_RETRY[@]}"
  else
    exit 1
  fi
fi

SPLIT_CMD=(
  python -u Emotion2Vec/data/prepare_iemocap_splits.py
  --metadata-csv "${DATASET_CSV}"
  --embeddings-root "${EMBED_ROOT}"
  --output-dir "${SPLIT_OUTPUT_DIR}"
  --prefix "${SPLIT_PREFIX}"
  --min-agreement "${MIN_AGREEMENT}"
  --labels
)

for emotion in ${INCLUDE_EMOTIONS}; do
  SPLIT_CMD+=("${emotion}")
done

SPLIT_CMD+=(--require-embeddings)

echo "=== Split command ==="
printf '%q ' "${SPLIT_CMD[@]}"
printf '\n'
"${SPLIT_CMD[@]}"

TRAIN_CMD=(
  python -u Emotion2Vec/training/train_downstream.py
  --config configs/downstream_config.yaml
  --run-name "${RUN_NAME}"
  --train-csv "${SPLIT_OUTPUT_DIR}/${SPLIT_PREFIX}_train.csv"
  --val-csv "${SPLIT_OUTPUT_DIR}/${SPLIT_PREFIX}_val.csv"
  --test-csv "${SPLIT_OUTPUT_DIR}/${SPLIT_PREFIX}_test.csv"
  --embeddings-root "${EMBED_ROOT}"
  --label-col label
  --embedding-path-col embedding_path
  --sample-id-col sample_id
  --device cuda
  --epochs "${EPOCHS}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --lr-scheduler "${LR_SCHEDULER}"
  --lr-scheduler-patience "${LR_SCHEDULER_PATIENCE}"
  --lr-scheduler-factor "${LR_SCHEDULER_FACTOR}"
  --lr-scheduler-min-lr "${LR_SCHEDULER_MIN_LR}"
  --num-workers "${NUM_WORKERS}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --report-to wandb
  --wandb-mode online
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-tags emotion2vec,downstream,iemocap,base,full
  --save-figures
  --save-predictions
)

if [[ "${WANDB_WATCH_MODEL}" == "1" ]]; then
  TRAIN_CMD+=(--wandb-watch-model)
else
  TRAIN_CMD+=(--no-wandb-watch-model)
fi

if [[ -n "${TRAIN_RESUME_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--resume-from-checkpoint "${TRAIN_RESUME_CHECKPOINT}")
fi

if [[ -n "${EARLY_STOPPING_PATIENCE}" ]]; then
  TRAIN_CMD+=(--early-stopping-patience "${EARLY_STOPPING_PATIENCE}")
fi

if [[ -n "${EARLY_STOPPING_MIN_DELTA}" ]]; then
  TRAIN_CMD+=(--early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "=== Train command ==="
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

echo "End: $(date)"
