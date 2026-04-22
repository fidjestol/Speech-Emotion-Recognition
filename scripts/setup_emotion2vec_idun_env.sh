#!/bin/bash
# Install Emotion2Vec/FunASR dependencies into the Idun project venv.
# No sudo is needed; packages are installed under .venv-idun.

set -euo pipefail

cd /cluster/home/bekadb/Speech-Emotion-Recognition

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
module load FFmpeg/6.0-GCCcore-13.2.0

source .venv-idun/bin/activate

python -m pip install -U pip wheel "setuptools<81"
python -m pip install -U funasr modelscope

python - <<'PY'
from shutil import which

import funasr

print("funasr:", getattr(funasr, "__version__", "unknown"))
print("ffmpeg:", which("ffmpeg"))
PY
