#!/bin/bash
# Isolated login-node GPU test environment for Tesla P100 (sm_60).
# This is separate from .venv-idun on purpose.

set -euo pipefail

cd /cluster/home/bekadb/Speech-Emotion-Recognition

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

ENV_DIR=".venv-login-gpu"

python -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

python -m pip install -U pip wheel "setuptools<81"

# Keep this env isolated from the main project env. Pin a CUDA build line that
# still supports P100 / sm_60 for login-node validation.
python -m pip install \
  torch==2.6.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install \
  accelerate \
  huggingface-hub \
  librosa \
  matplotlib \
  pandas \
  scikit-learn \
  seaborn \
  soundfile \
  tensorboard \
  transformers

echo "Created ${ENV_DIR}"
echo "Activate with:"
echo "  module purge"
echo "  module load Python/3.11.5-GCCcore-13.2.0"
echo "  module load CUDA/12.4.0"
echo "  source ${ENV_DIR}/bin/activate"
