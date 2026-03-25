# Whisper SER Finetuning

This folder contains the Whisper finetuning workflow for adapting Whisper to IEMOCAP with two supported variants:

- `fru`
- `neu`
- `ang`
- `sad`
- `exc`
- `hap`

The script defaults to running both:

- `with_xxx`
- `without_xxx`

The shared retained labels are:

- `fru`
- `neu`
- `ang`
- `sad`
- `exc`
- `hap`

Optional label:

- `xxx`

Still excluded:

- `oth`
- `dis`

## Recommended workflow

Use the script for real training on Idun and keep the notebook for inspection or prototyping. The script is better for:

- surviving SSH disconnects through Slurm
- saving checkpoints and resumable state
- writing clean logs for `tail -f`
- saving TensorBoard event files
- exporting final test-set artifacts such as confusion matrices, prediction tables, and metric summaries

## Notebook

- `whisper/finetuning/finetune_whisper_ser.ipynb`

The notebook mirrors the same general flow as the script and is still useful for:

- checking the environment
- exploring data filters and splits
- inspecting intermediate tables interactively

## Script

- `whisper/finetuning/finetune_whisper_ser.py`

The script adds:

- step-based logging, evaluation, and checkpoint saving
- TensorBoard logging
- optional Weights & Biases logging with `--report-to wandb` or `--report-to tensorboard,wandb`
- resumable checkpoints with `--resume-from-checkpoint latest`
- saved `train/val/test` split summaries
- saved trainer history as CSV/JSON
- saved test predictions table
- saved classification report CSV/JSON
- saved confusion matrix CSV/PNG
- saved per-run comparison table across variants

## Idun launcher

- `scripts/idun_whisper_finetune.sbatch`

Default resource request in the Slurm launcher:

- `1 x H100 GPU`
- `8` CPU cores
- `48G` system RAM
- `30:00:00` walltime

That is a reasonable starting point for Whisper finetuning. Start with one GPU first; only consider multi-GPU after measuring memory pressure and throughput.

On Idun, you typically request GPU type rather than a raw VRAM number. The cluster exposes types like `p100`, `v100`, `a100`, `h100`, and `h200`. For this Whisper workflow, the launcher currently targets `h100` to maximize memory headroom and reduce the chance of a late OOM.

The login-node GPU test is only for a tiny sanity check. Use the Slurm launcher for any real training run.
The Slurm launcher also forces `--device cuda` so a misconfigured job fails early instead of silently falling back to CPU.

## Run

```bash
python whisper/finetuning/finetune_whisper_ser.py
```

Smoke run example:

```bash
python whisper/finetuning/finetune_whisper_ser.py --max-train-samples 64 --max-eval-samples 32 --eval-steps 20 --save-steps 20
```

Balanced per-label GPU sanity test:

```bash
python whisper/finetuning/finetune_whisper_ser.py --model-id openai/whisper-small --device cuda --max-samples-per-label 10 --num-train-epochs 0.2 --train-batch-size 1 --eval-batch-size 1 --max-duration 30 --freeze-encoder --eval-steps 5 --save-steps 5 --run-name login_gpu_test_10_per_label_whisper_small_both_variants
```

Use Weights & Biases as well:

```bash
WANDB_API_KEY=... WANDB_PROJECT=ser-whisper python whisper/finetuning/finetune_whisper_ser.py --report-to tensorboard,wandb
```

Single variant example:

```bash
python whisper/finetuning/finetune_whisper_ser.py --no-run-both-xxx-variants --include-xxx
```

Resume a previous run:

```bash
python whisper/finetuning/finetune_whisper_ser.py --run-name 20260315_whisper_retry --resume-from-checkpoint latest
```

Submit on Idun:

```bash
sbatch scripts/idun_whisper_finetune.sbatch
```

Login-node GPU test helper:

```bash
./scripts/setup_whisper_login_gpu_env.sh
./scripts/test_whisper_login_gpu.sh
```

The login-node GPU test uses its own isolated environment, `.venv-login-gpu`, because the login-node Tesla P100 requires a different PyTorch wheel line than the main Slurm environment.
It also uses `openai/whisper-small` instead of `openai/whisper-large-v3`, because the login-node P100 does not have enough memory for a realistic `large-v3` finetuning pass.

## Monitoring on Idun

After SSH disconnects, you can still track the job with:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,State,Start,End,Elapsed,ExitCode
tail -f logs/slurm-whisper-ft-<jobid>.out
seff <jobid>
```

TensorBoard logs are saved inside each variant directory under `runs/`. You can inspect them later with:

```bash
source .venv-idun/bin/activate
tensorboard --logdir whisper/finetuning/artifacts
```

## Output layout

Each run creates a timestamped directory under `whisper/finetuning/artifacts/` and writes:

- `run_config.json`
- `environment_summary.json`
- `variant_comparison.csv`
- `with_xxx/` and/or `without_xxx/`

Each variant directory contains:

- checkpoints and final model files
- `split_summary.csv`
- `label_distribution.csv`
- `run_metadata.json`
- `reports/test_predictions.csv`
- `reports/test_metrics.csv`
- `reports/classification_report.csv`
- `reports/confusion_matrix_counts.csv`
- `reports/confusion_matrix_normalized.csv`
- `figures/train_loss_curve.png`
- `figures/eval_loss_curve.png`
- `figures/eval_f1_macro_curve.png`
- `figures/confusion_matrix_counts.png`
- `figures/confusion_matrix_normalized.png`

The most recent run path is also recorded in:

```bash
whisper/finetuning/artifacts/latest_run.txt
```

## Idun environment

Use the Idun-specific virtual environment for this workflow:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-idun/bin/activate
```

If it does not exist yet:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
python -m venv .venv-idun
source .venv-idun/bin/activate
python -m pip install -U pip wheel "setuptools<81"
python -m pip install -e .
```

The script defaults to `--precision auto`, which now prefers `bf16` on GPUs that support it and falls back to `fp16` otherwise.

## Login-node GPU test environment

For the special P100 login-node sanity check, use the isolated login test env:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-login-gpu/bin/activate
```

Create it with:

```bash
./scripts/setup_whisper_login_gpu_env.sh
```
