#!/bin/bash
# Idun Slurm launcher for Whisper SER finetuning.
#
# Submit:
#   sbatch scripts/idun_whisper_finetune.sbatch
#
# Monitor after disconnect:
#   squeue -u "$USER"
#   sacct -j <jobid> --format=JobID,State,Start,End,Elapsed,ExitCode
#   tail -f logs/slurm-whisper-ft-<jobid>.out
#   seff <jobid>
#
# Optional TensorBoard from a compute/login session:
#   source .venv-idun/bin/activate
#   tensorboard --logdir whisper/finetuning/artifacts

#SBATCH --job-name=whisper-ft
#SBATCH --partition=GPUQ
# Request a GPU type instead of generic VRAM. On Idun, Slurm exposes types
# such as p100, v100, a100, h100, h200 rather than a direct VRAM selector.
# For Whisper large-v3 finetuning, use H100 for the safest memory headroom.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=logs/slurm-whisper-ft-%j.out
#SBATCH --error=logs/slurm-whisper-ft-%j.err
#SBATCH --account=ie-idi

set -euo pipefail

echo "=== Job metadata ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Requested GRES: gpu:h100:1"

cd /cluster/home/bekadb/Speech-Emotion-Recognition
mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

if [[ ! -f .venv-idun/bin/activate ]]; then
  echo "Missing .venv-idun. Create it before submitting this job."
  exit 1
fi

source .venv-idun/bin/activate

python - <<'PY'
import torch
import transformers
import tensorboard
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("tensorboard:", tensorboard.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-}"
PRECISION="${PRECISION:-auto}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_BOTH_VARIANTS="${RUN_BOTH_VARIANTS:-true}"
RESUME_MODE="${RESUME_MODE:-none}"

CMD=(
  python -u whisper/finetuning/finetune_whisper_ser.py
  --device cuda
  --precision "${PRECISION}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --num-train-epochs "${EPOCHS}"
  --schedule-mode steps
  --logging-steps "${LOGGING_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --report-to tensorboard
  --dataloader-num-workers "${NUM_WORKERS}"
)

if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
fi

if [[ "${RUN_BOTH_VARIANTS}" == "true" ]]; then
  CMD+=(--run-both-xxx-variants)
else
  CMD+=(--no-run-both-xxx-variants)
fi

if [[ "${RESUME_MODE}" != "none" ]]; then
  CMD+=(--resume-from-checkpoint "${RESUME_MODE}")
fi

echo "=== Launch command ==="
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "End: $(date)"
