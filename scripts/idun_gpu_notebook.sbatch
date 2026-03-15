#!/bin/bash
# Idun GPU batch template for executing a Jupyter notebook.
#
# Usage:
#   1) Edit account/modules/env below.
#   2) Set NOTEBOOK_PATH to the notebook you want to run.
#   3) Submit with: sbatch scripts/idun_gpu_notebook.sbatch

#SBATCH --job-name=ser-nb-gpu
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
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

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

# Activate your project environment.
# Prefer the Idun-specific venv when present.
if [[ -f .venv-idun/bin/activate ]]; then
  source .venv-idun/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
# source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate ser

# Notebook to execute.
NOTEBOOK_PATH="feature_test/DNN/mfcc_dnn_test.ipynb"

# GPU sanity check.
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
PY

NOTEBOOK_BASENAME="$(basename "${NOTEBOOK_PATH}" .ipynb)"
OUTPUT_NOTEBOOK="logs/${NOTEBOOK_BASENAME}.executed.${SLURM_JOB_ID}.ipynb"

python -m nbconvert \
  --to notebook \
  --execute "${NOTEBOOK_PATH}" \
  --output "${OUTPUT_NOTEBOOK}"

echo "Executed notebook written to: ${OUTPUT_NOTEBOOK}"
echo "End: $(date)"
