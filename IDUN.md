# IDUN.md

This document is the Idun runbook for this repository. It focuses on the workflows we actually use in this codebase, not generic Slurm advice.

## Scope

This repo currently has two serious Idun workflows:

- Whisper finetuning on GPU
- Feature augmentation experiments on CPU or GPU

There are also two convenience templates:

- `scripts/idun_gpu.sbatch`
- `scripts/idun_gpu_notebook.sbatch`

Use the workflow-specific scripts when possible.

## Repository Paths That Matter

- Whisper finetuning code: [whisper/finetuning](/cluster/home/bekadb/Speech-Emotion-Recognition/whisper/finetuning)
- Feature augmentation code: [feature_augmentation/mean_std_oversampling](/cluster/home/bekadb/Speech-Emotion-Recognition/feature_augmentation/mean_std_oversampling)
- Shared feature-selection filtering logic: [feature_selection/common.py](/cluster/home/bekadb/Speech-Emotion-Recognition/feature_selection/common.py)
- Slurm scripts: [scripts](/cluster/home/bekadb/Speech-Emotion-Recognition/scripts)
- Logs: `logs/`

## Idun Environment

For this repository, the stable cluster environment is:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
```

Add CUDA when the job needs GPU access:

```bash
module load CUDA/12.4.0
```

### Main virtual environment

Use `.venv-idun` for real Idun jobs:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

python -m venv .venv-idun
source .venv-idun/bin/activate
python -m pip install -U pip wheel "setuptools<81"
python -m pip install -e .
```

Why `setuptools<81` matters:
- TensorBoard in this setup still expects `pkg_resources`
- newer `setuptools` may break `tensorboard`

If TensorBoard fails with `ModuleNotFoundError: No module named 'pkg_resources'`, fix it with:

```bash
python -m pip install --force-reinstall "setuptools<81"
```

### Login-node Whisper sanity env

The special login-node Whisper GPU sanity path uses a separate environment:

- `.venv-login-gpu`

That exists because the login node has Tesla P100 GPUs, which needed a different PyTorch wheel line than the real H100 Slurm path.

## Accounts and Queueing

In this repo, `share-ie-idi` has been the reliable account for actual job queueing.

We saw `ie-idi` get blocked by:
- `AssocGrpCPUMinutesLimit`

So the current Slurm scripts for real work default to:

- `#SBATCH --account=share-ie-idi`

If a job is pending with:
- `Reason=Priority`

that is normal.

If a job is pending with:
- `Reason=AssocGrpCPUMinutesLimit`

that is a quota/policy block, not a code problem.

Useful checks:

```bash
squeue -u "$USER"
squeue --start -j <jobid>
scontrol show job <jobid> | egrep "JobState|Reason|StartTime|Account|QOS"
```

Finished jobs are not visible in `squeue`; use:

```bash
sacct -j <jobid> --format=JobID,State,Start,End,Elapsed,ExitCode
seff <jobid>
```

## Storage and Quota

Cluster compute allocation stops when the job ends. Storage usage does not.

Check quota and usage with:

```bash
lfs quota -u "$USER" /cluster
du -sh /cluster/home/$USER
du -sh /cluster/work/$USER 2>/dev/null
df -h /cluster/home/$USER
df -h /cluster/work/$USER 2>/dev/null
```

What continues to count against storage quota after jobs finish:
- logs
- checkpoints
- artifacts
- TensorBoard event files
- saved models

What stops immediately when jobs finish:
- CPU allocation
- GPU allocation
- RAM allocation
- wall-clock compute usage

## Monitoring After SSH Disconnects

For all jobs:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,State,Start,End,Elapsed,ExitCode
seff <jobid>
```

Follow logs live once the job is running:

```bash
tail -f logs/<slurm-output-file>.out
```

If the output file does not exist yet, the job likely has not started.

## Whisper Finetuning on Idun

Main script:
- [scripts/idun_whisper_finetune.sbatch](/cluster/home/bekadb/Speech-Emotion-Recognition/scripts/idun_whisper_finetune.sbatch)

Current intent:
- GPU workflow
- `GPUQ`
- `1 x H100`
- `8` CPU cores
- `48G` RAM
- `30:00:00` walltime

Environment:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-idun/bin/activate
```

Submit:

```bash
sbatch scripts/idun_whisper_finetune.sbatch
```

The script forces CUDA training so a bad job fails early instead of silently running on CPU.

### Whisper outputs

Main artifact root:
- `whisper/finetuning/artifacts/`

Typical outputs:
- `run_config.json`
- `environment_summary.json`
- `variant_comparison.csv`
- `variant_comparison.json`
- `with_xxx/`
- `without_xxx/`

Per variant:
- checkpoints
- final model files
- split summaries
- label distributions
- trainer history
- test predictions
- classification reports
- confusion matrices
- figures
- TensorBoard logs under `runs/`

Latest run marker:

```bash
cat whisper/finetuning/artifacts/latest_run.txt
```

### Whisper monitoring backends

TensorBoard:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-idun/bin/activate

tensorboard --logdir whisper/finetuning/artifacts/<run_dir> --port 6006
```

Weights & Biases can be enabled with:

```bash
REPORT_TO=tensorboard,wandb sbatch scripts/idun_whisper_finetune.sbatch
```

### Important Whisper-specific lessons from this repo

- Do not rely on login-node P100 GPUs for real Whisper finetuning.
- The login-node sanity check uses `whisper-small`; real training uses `whisper-large-v3`.
- TensorBoard event files are written under `runs/`, not a custom `tensorboard/` path.
- The login-node test is a code-path sanity check, not a faithful performance test.

## Feature Augmentation on Idun

Main scripts:
- CPU path: [scripts/idun_feature_augmentation.sbatch](/cluster/home/bekadb/Speech-Emotion-Recognition/scripts/idun_feature_augmentation.sbatch)
- GPU path: [scripts/idun_feature_augmentation_gpu.sbatch](/cluster/home/bekadb/Speech-Emotion-Recognition/scripts/idun_feature_augmentation_gpu.sbatch)

Main runner:
- [run_all_augmentation.py](/cluster/home/bekadb/Speech-Emotion-Recognition/feature_augmentation/mean_std_oversampling/run_all_augmentation.py)

Per-dataset experiment:
- [run_augmentation_experiment.py](/cluster/home/bekadb/Speech-Emotion-Recognition/feature_augmentation/mean_std_oversampling/run_augmentation_experiment.py)

### Current experiment design

The current feature-augmentation pipeline now supports:
- leave-one-session-out cross-validation using the 5 IEMOCAP sessions
- baseline and augmented training
- logistic regression
- calibrated linear SVM
- random forest
- XGBoost
- soft-voting ensemble
- stacking ensemble
- saved models
- W&B logging
- figures and CSV summaries

### CPU vs GPU guidance

Use CPU first unless you explicitly want GPU-backed XGBoost.

Why:
- most models here are classic tabular models
- CPU queueing is often easier/faster
- GPU only really helps the XGBoost path in the current implementation

CPU submit:

```bash
sbatch scripts/idun_feature_augmentation.sbatch
```

GPU submit:

```bash
sbatch scripts/idun_feature_augmentation_gpu.sbatch
```

### Recommended smoke run

Use one dataset first, not the full matrix of datasets.

Recommended smoke run:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv-idun/bin/activate

RUN_NAME=feature_aug_cpu_smoke_mfcc_loso_5session \
WANDB_PROJECT=ser-feature-augmentation \
REPORT_TO=wandb \
DATASET_KEYS="mfcc_raw" \
RUN_BOTH_VARIANTS=1 \
REQUIRE_AGREEMENT=1 \
USE_NROWS=8000 \
TARGET_PER_CLASS=300 \
GROUP_SIZE=3 \
SAVE_MODELS=1 \
SAVE_AUGMENTED_MATRICES=0 \
ENABLE_GPU_MODELS=0 \
sbatch scripts/idun_feature_augmentation.sbatch
```

Why `8000` rows for the smoke run:
- LOSO cross-validation needs multiple sessions
- small row counts can accidentally include only one session
- `mfcc_raw` needed a larger prefix to cover all 5 sessions

### Feature augmentation outputs

Artifact root:
- `feature_augmentation/mean_std_oversampling/artifacts/`

The runner writes:
- `run_manifest.csv`
- `run_manifest.json`
- `results_summary.csv`
- `failed_runs.csv`
- `failed_runs.json`
- `environment_summary.json`

Per dataset/variant:
- environment summary
- fold metrics
- aggregate metrics
- classification reports
- per-class accuracy tables
- confusion matrices
- normalized confusion matrices
- fold predictions
- saved-model manifests
- final-model manifests
- class counts
- augmentation summaries
- figures
- saved `.joblib` model files when enabled

### Agreement filtering

`REQUIRE_AGREEMENT=1` means:
- keep only samples with annotator agreement
- cleaner labels
- fewer examples

`REQUIRE_AGREEMENT=0` means:
- keep more data
- accept more label noise

For this repo, agreement-filtered smoke runs are the better default.

### Failure behavior

The all-run feature augmentation orchestrator is designed so one failed dataset or variant does not stop the others.

That is why it always writes:
- `run_manifest.*`
- `failed_runs.*`

This makes post-run review much easier.

## W&B

W&B is used in both the Whisper and feature-augmentation flows.

If a job should report to W&B:
- ensure `wandb` is installed in `.venv-idun`
- ensure you are logged in or the right auth environment is present

Useful env vars:

```bash
export WANDB_PROJECT=...
export WANDB_ENTITY=...
```

In this repo:
- Whisper supports `tensorboard`, `wandb`, or both
- Feature augmentation uses W&B for runner-level and per-dataset reporting

## TensorBoard

TensorBoard is mainly relevant for Whisper runs.

If `tensorboard` fails with `pkg_resources` import problems:

```bash
python -m pip install --force-reinstall "setuptools<81"
```

## Generic GPU Template Scripts

These are convenience templates, not the primary research entrypoints:

- [scripts/idun_gpu.sbatch](/cluster/home/bekadb/Speech-Emotion-Recognition/scripts/idun_gpu.sbatch)
- [scripts/idun_gpu_notebook.sbatch](/cluster/home/bekadb/Speech-Emotion-Recognition/scripts/idun_gpu_notebook.sbatch)

Use them when you want a quick custom run outside the hardened workflows.

## Common Problems We Already Hit

### Wrong Python module loaded

Symptom:
- `libpython3.11.so.1.0: cannot open shared object file`

Cause:
- activating the venv without first loading the matching Idun Python module

Fix:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv-idun/bin/activate
```

### Job pending forever under association limit

Symptom:
- `Reason=AssocGrpCPUMinutesLimit`

Fix:
- use the right account for this repo, usually `share-ie-idi`
- cancel stale jobs you do not need

### TensorBoard missing dashboards

Cause:
- pointing TensorBoard at the wrong directory

Fix:
- point it at the actual run directory or the variant `runs/` folder

### No log file yet

Cause:
- the job has not started, so Slurm has not created the output file yet

Fix:

```bash
squeue -j <jobid>
squeue --start -j <jobid>
```

### Git lock file errors

Symptom:
- `.git/index.lock` exists

Fix:
- make sure no Git command is still running
- then remove the stale lock file

## Minimal Daily Cheatsheet

Activate Idun env:

```bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-idun/bin/activate
```

Queue Whisper:

```bash
sbatch scripts/idun_whisper_finetune.sbatch
```

Queue feature augmentation on CPU:

```bash
sbatch scripts/idun_feature_augmentation.sbatch
```

Queue feature augmentation on GPU:

```bash
sbatch scripts/idun_feature_augmentation_gpu.sbatch
```

Monitor jobs:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,State,Start,End,Elapsed,ExitCode
seff <jobid>
```

Follow logs:

```bash
tail -f logs/<slurm-output>.out
```

Check quota:

```bash
lfs quota -u "$USER" /cluster
```
