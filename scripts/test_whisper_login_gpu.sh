#!/bin/bash
# Tiny login-node GPU test for Whisper SER finetuning.
# Uses its own isolated env because the login-node P100 needs a different
# PyTorch/CUDA wheel line than the main Slurm environment.

set -euo pipefail

cd /cluster/home/bekadb/Speech-Emotion-Recognition

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

if [[ ! -f .venv-login-gpu/bin/activate ]]; then
  echo "Missing .venv-login-gpu. Run scripts/setup_whisper_login_gpu_env.sh first."
  exit 1
fi

source .venv-login-gpu/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u whisper/finetuning/finetune_whisper_ser.py \
  --model-id openai/whisper-small \
  --device cuda \
  --precision fp32 \
  --max-samples-per-label 10 \
  --num-train-epochs 0.2 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-duration 30 \
  --freeze-encoder \
  --logging-steps 1 \
  --eval-steps 5 \
  --save-steps 5 \
  --dataloader-num-workers 0 \
  --report-to tensorboard \
  --run-name login_gpu_test_10_per_label_whisper_small_both_variants
