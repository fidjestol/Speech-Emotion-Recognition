# Speech Emotion Recognition Project Spec (Code Overview)

## 1) Project Summary

- Project: `speech-emotion-recognition`
- Purpose: Research codebase for Speech Emotion Recognition (SER) using:
  - classical feature engineering + tabular ML
  - feature selection and augmentation workflows
  - Whisper-based end-to-end audio classification finetuning/evaluation
- Primary dataset contract in code: `datasets/IEMOCAP/iemocap_full_dataset.csv` + referenced audio files under `datasets/IEMOCAP/`

## 2) Tech Stack

- Python: `>=3.10` (`pyproject.toml`)
- Packaging/build: `setuptools`, `uv`
- Core libs:
  - Audio/features: `librosa`, `opensmile`, `praat-parselmouth`, `soundfile`
  - ML (classical): `scikit-learn`, `xgboost`, `catboost`
  - DL/SER: `torch`, `torchaudio`, `transformers`, `datasets`, `accelerate`
  - Analysis/plotting: `pandas`, `numpy`, `matplotlib`, `seaborn`
  - Tracking: `tensorboard`, `wandb`

## 3) High-Level Architecture

The repo is split into mostly independent research workflows that share common dataset and label filtering patterns.

1. Feature extraction layer
- Location: `feature_extraction/`
- Produces per-utterance feature CSVs into `extracted_features/...`
- Families include MFCC, prosody, spectral, tonality, rhythm, SSL embeddings, openSMILE, text representations.

2. Feature selection layer
- Location: `feature_selection/`
- Notebook pipelines for PCA and ANOVA `SelectKBest`.
- Produces selected/reduced feature artifacts (CSV + JSON metadata).

3. Feature augmentation layer
- Location: `feature_augmentation/mean_std_oversampling/`
- Oversamples training folds with class-conditional mean/std prototypes.
- Runs LOSO-style model benchmarks and persists detailed artifacts.

4. Classical model testing layer
- Location: `feature_test/`
- Contains baselines/ensemble experiments over extracted and selected features.

5. Whisper deep-learning layer
- Location: `whisper/finetuning/`, `whisper/evaluation/`
- Finetunes Whisper audio classifier on IEMOCAP variants.
- Evaluates published Whisper SER checkpoint on a clean 6-way label subset.

6. Utilities + operations
- Shared helper modules in `utils/`, `feature_extraction/common.py`, `feature_selection/common.py`, `feature_augmentation/common.py`
- Idun/Slurm runbooks and job scripts in `IDUN.md` and `scripts/*.sbatch`.

## 4) Repository Map (Code-Centric)

- `feature_extraction/`
  - `common.py`: machine detection, worker/thread resolution, torch/cuda settings.
  - `feature_families/*.py`: callable extraction APIs (`extract`, `extract_batch`) for each family.
  - `pipelines/`: notebook orchestrations for all-family and ANOVA-selected feature generation.

- `feature_selection/`
  - `common.py`: dataset-key paths, variant handling (`with_xxx` / `without_xxx`), filtering helpers.
  - `anova/`: elbow + select-k notebook workflows and artifacts.
  - `pca/`: PCA notebook workflows and artifacts.

- `feature_augmentation/`
  - `common.py`: mean/std oversampling primitives and reporting frame builders.
  - `mean_std_oversampling/run_augmentation_experiment.py`: single dataset/variant runner.
  - `mean_std_oversampling/run_all_augmentation.py`: orchestrator across dataset keys and variants.
  - `gan_workflow/`: scaffold/planning assets for future GAN augmentation.

- `whisper/`
  - `finetuning/finetune_whisper_ser.py`: training entrypoint with artifactized run outputs.
  - `evaluation/evaluate_whisper_ser.py`: standalone evaluation entrypoint.

- `utils/`
  - `threading.py`: safe threaded map abstraction + worker sizing.
  - `stats.py`: NaN-safe statistic summarization for feature curves.

- `scripts/`
  - Slurm job launchers for Whisper and augmentation workflows (`idun_*.sbatch`).

- `analysis/`, `inspections/`
  - EDA notebooks/scripts and feature inspection scripts used for sanity checks.

## 5) Data and Label Contract

### Metadata columns used repeatedly

Commonly expected fields:
- `path`, `session`, `method`, `gender`, `emotion`, `agreement` (plus optional columns like `text`, `split`, etc.)

### Label policy patterns in code

- Excluded often: `sur`, `fea`, `oth`, `dis` (classical feature-selection/augmentation contexts)
- Optional class: `xxx`
- Two major variant modes used by scripts:
  - `with_xxx`
  - `without_xxx`

Whisper workflows have separate explicit label policies defined in each script.

## 6) Core Module Specs

### `feature_extraction/feature_families`

Each module is designed as reusable extraction code (not only notebooks), typically exposing:
- `extract(audio, sr, ...) -> dict`
- `extract_batch(items, ...) -> list[dict]`

Implemented families (Python modules):
- `prosody_energy.py`: RMS energy contour stats with voiced masking.
- `prosody_pitch.py`: F0 extraction (pYIN with YIN fallback) + voicing/silence gating.
- `representations.py`: log-mel representation summaries.
- `rhythm_pauses.py`: silence ratio, pause durations, onset rate/count.
- `spectral_shape.py`: centroid/bandwidth/rolloff/flatness/flux/contrast/ZCR pooled stats.
- `tonality.py`: chroma + tonnetz summary stats.
- `voice_quality.py`: HNR via parselmouth, HPSS proxy fallback.
- `ssl_embeddings.py`: HuBERT/wav2vec embeddings, pooled mean/std vectors.
- `smilesets.py`: openSMILE functional feature extraction (eGeMAPS/GeMAPS/ComParE).

### `feature_extraction/pipelines/pipeline_common.py`

Provides orchestration helpers for notebook pipelines:
- loading base metadata with configurable filtering
- loading/merging feature family CSVs
- family-source path mapping and alias support
- optional auto-execution of missing family generator notebooks
- selection helpers for prefixed feature columns

### `feature_selection/common.py`

Shared behavior for PCA/ANOVA workflows:
- machine and env-driven variant config
- artifact directory resolution (`*_with_xxx`, `*_without_xxx`)
- feature frame filtering rules
- dataset-key to CSV path contract

### `feature_augmentation/common.py`

Defines augmentation core:
- mean/std class-conditional oversampling
- class count, confusion matrix, classification report helpers
- artifact path resolution

### `whisper/finetuning/finetune_whisper_ser.py`

Main training pipeline features:
- configurable train/val/test split from IEMOCAP metadata
- label-space variants (`with_xxx` / `without_xxx`)
- Hugging Face `Trainer`-based finetuning
- step/epoch scheduling options
- precision/device controls (`fp32`, `fp16`, `bf16`, auto)
- optional checkpoint resume
- TensorBoard and optional W&B reporting
- comprehensive artifact export (metrics, predictions, confusion matrices, plots, metadata)

### `whisper/evaluation/evaluate_whisper_ser.py`

Evaluation pipeline features:
- evaluates HF checkpoint `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`
- fixed clean 6-way label subset (`ang, fea, hap, neu, sad, sur`)
- stratified 80/20 split
- exports metrics, predictions, classification report, confusion matrix, run metadata

## 7) Execution Entry Points

Primary script entrypoints:

- Whisper finetuning:
  - `python whisper/finetuning/finetune_whisper_ser.py`

- Whisper evaluation:
  - `python whisper/evaluation/evaluate_whisper_ser.py`

- Full augmentation sweep:
  - `python feature_augmentation/mean_std_oversampling/run_all_augmentation.py --run-name <name> ...`

- Single augmentation experiment:
  - `python feature_augmentation/mean_std_oversampling/run_augmentation_experiment.py --dataset-key <key> --run-name <name> ...`

- Classical ensemble baseline:
  - `python feature_test/classical_feature_system/classical_feature_ensemble.py`

Operational launchers:
- Slurm templates and workflow launchers in `scripts/`.

## 8) Artifact Contracts

### Feature extraction outputs

Expected under `extracted_features/<family>/...csv`, typically keyed by `path`.

### Feature selection outputs

- ANOVA artifacts in `feature_selection/anova/select_k_best_anova/artifacts*`
- PCA artifacts in `feature_selection/pca/select_pca/artifacts*`
- Include selected feature JSONs, transformed train/test matrices, rankings/variance files, metadata.

### Augmentation outputs

Under `feature_augmentation/mean_std_oversampling/artifacts/<run_name>/...`:
- run manifests
- per-dataset per-variant reports/figures/models/metadata
- failure manifests for partial-run resilience

### Whisper outputs

Under `whisper/finetuning/artifacts/<run>/...`:
- run + environment config
- per-variant checkpoints, reports, figures, tensorboard logs
- cross-variant comparison tables
- `latest_run.txt` pointer

Under `whisper/evaluation/artifacts/<run>/...`:
- results CSV
- prediction CSV
- classification report CSV
- confusion matrix PNG
- run metadata JSON

## 9) Environment and Runtime Configuration

Common env/config controls used in code:
- `SER_MACHINE` for machine profile selection
- `SER_INCLUDE_XXX`, `SER_RUN_BOTH_XXX_VARIANTS`, `SER_USE_VARIANT_ARTIFACT_DIRS`
- Torch device/precision are script flags in Whisper workflows
- W&B controlled by CLI flags and credentials (`WANDB_*` environment)

## 10) Observed Project Characteristics

- Notebook-heavy research workflow with script hardening in key paths (Whisper + augmentation).
- Strong artifactization: most mature scripts save run metadata + tables + figures.
- Multiple parallel SER tracks coexist:
  - handcrafted/tabular classical pipelines
  - end-to-end Whisper fine-tuning/evaluation
- Cluster-first operations documented for Idun usage (`IDUN.md`, Slurm scripts).

## 11) Current Gaps (from repository state)

- Root `README.md` is minimal and does not describe full workflows.
- No single unified pipeline CLI that chains extraction -> selection -> augmentation -> training.
- Tests are mostly notebooks/experiments; there is no dedicated automated unit/integration test suite in the repo root.
- Some paths appear notebook-legacy and may need consistency checks when standardizing pipelines.

---

This spec is based on current source structure and script behavior in this repository.
