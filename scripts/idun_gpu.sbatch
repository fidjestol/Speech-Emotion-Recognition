#!/bin/bash
# Idun GPU batch template for this repository.
#
# Usage:
#   1) Edit account/modules/venv/command below.
#   2) Submit with: sbatch scripts/idun_gpu.sbatch
#   3) Monitor with: squeue -u "$USER"

#SBATCH --job-name=ser-gpu
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --account=ie-idi

set -euo pipefail

echo "=== Job metadata ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

cd /cluster/home/bekadb/Speech-Emotion-Recognition
mkdir -p logs

# Start clean module environment and load toolchain.
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

# Activate your project environment.
# Option A: venv
# source .venv-idun/bin/activate
#
# Option B: conda
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate ser

# Optional sanity check for GPU availability in PyTorch.
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
PY

# Replace this with your actual entrypoint.
# Examples:
# python main.py
# python -m feature_extraction.pipelines.some_pipeline
python main.py

echo "End: $(date)"
