#!/usr/bin/env bash
# Idun GPU job for extracting emotion2vec_base embeddings from IEMOCAP.
#
# Submit with:
#   sbatch scripts/idun_emotion2vec_extract_base.sbatch

#SBATCH --job-name=e2v-base-ext
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --account=share-ie-idi

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
module load FFmpeg/6.0-GCCcore-13.2.0

source .venv-idun/bin/activate

DATASET_CSV="${DATASET_CSV:-datasets/IEMOCAP/iemocap_full_dataset.csv}"
AUDIO_ROOT="${AUDIO_ROOT:-datasets/IEMOCAP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-extracted_features/emotion2vec_base/frame}"
MANIFEST_PATH="${MANIFEST_PATH:-extracted_features/emotion2vec_base/frame_manifest.csv}"
SUMMARY_PATH="${SUMMARY_PATH:-extracted_features/emotion2vec_base/frame_summary.json}"
MODEL_ID="${MODEL_ID:-iic/emotion2vec_base}"
HUB="${HUB:-hf}"
GRANULARITY="${GRANULARITY:-frame}"
MIN_AGREEMENT="${MIN_AGREEMENT:-1}"
EXCLUDE_EMOTIONS="${EXCLUDE_EMOTIONS:-xxx}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-1}"

echo "=== Job metadata ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model: ${MODEL_ID}"
echo "Hub: ${HUB}"
echo "Granularity: ${GRANULARITY}"
echo "Output root: ${OUTPUT_ROOT}"

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
  python Emotion2Vec/extraction/extract_hf_embeddings.py
  --dataset-csv "${DATASET_CSV}"
  --audio-root "${AUDIO_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --manifest-path "${MANIFEST_PATH}"
  --summary-path "${SUMMARY_PATH}"
  --model-id "${MODEL_ID}"
  --hub "${HUB}"
  --granularity "${GRANULARITY}"
  --min-agreement "${MIN_AGREEMENT}"
  --exclude-emotions
)

for emotion in ${EXCLUDE_EMOTIONS}; do
  cmd+=("${emotion}")
done

if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
else
  cmd+=(--no-resume)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
fi

printf 'Command:\n  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "End: $(date)"
