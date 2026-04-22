#!/bin/bash
# Idun GPU job for the Emotion2Vec sanity/evaluation script.
#
# Submit with:
#   sbatch scripts/idun_emotion2vec.sbatch

#SBATCH --job-name=emotion2vec
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --account=share-ie-idi

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
module load FFmpeg/6.0-GCCcore-13.2.0

source .venv-idun/bin/activate

echo "Python: $(which python)"
echo "ffmpeg: $(which ffmpeg)"
ffmpeg -version | head -n 1

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
PY

python Emotion2Vec/Iemocap_evaluation/e2vec_test.py --device cuda:0

echo "End: $(date)"
