# Sub-Family Genetic Algorithm Feature Selection

This folder contains a second-stage genetic algorithm for selecting feature
sub-families after broad feature-family selection.

The family-level GA answers:

```text
Which feature families should be used?
```

This GA answers:

```text
Which feature groups inside those families should be used?
```

## Layout

```text
feature_selection/ga/
  common.py
  subfamily_groups.py
  evaluate_subfamily_subset.py
  ga_subfamily_selection.py
  artifacts/
```

## Grouping Strategy

The GA does not search individual columns directly. It searches grouped feature
blocks:

- engineered acoustic features use name-pattern groups, such as `pitch_f0`,
  `energy_rms`, `rhythm_pauses`, `tonality_chroma`
- embedding and sparse text features use contiguous blocks, such as
  `bert_emb_block_0000_0127` or `tfidf_block_0128_0255`

Each group maps to explicit column names, written to `subfamily_groups.csv`.

## Basic Run

Run on manually chosen families:

```bash
python feature_selection/ga/ga_subfamily_selection.py \
  --run-name ga_subfamily_without_xxx_v1 \
  --families mfcc_raw prosody_pitch prosody_energy rhythm_pauses \
  --no-include-xxx \
  --validation-mode loso \
  --report-to wandb
```

Run from the family-level GA output:

```bash
python feature_selection/ga/ga_subfamily_selection.py \
  --run-name ga_subfamily_from_family_ga_v1 \
  --family-selection-json feature_family_selection/genetic_algorithm/artifacts/ga_feature_family_selection_without_xxx_v1/without_xxx/selected_families.json \
  --no-include-xxx \
  --validation-mode loso \
  --report-to wandb
```

## Dual W&B Reporting

The runner uses `utils.wandb_multi.init_multi_wandb_run`, so it logs to the
normal W&B account and mirrors to a second account when these environment
variables are set:

```bash
export WANDB_SECONDARY_API_KEY=...
export WANDB_SECONDARY_PROJECT=ser-feature-subfamily-selection
export WANDB_SECONDARY_ENTITY=...
```

If `WANDB_SECONDARY_API_KEY` is not set, only the primary W&B run is created.

## Outputs

Artifacts are written under:

```text
feature_selection/ga/artifacts/<run_name>/<with_xxx|without_xxx>/
```

Important files:

```text
ga_checkpoint.pkl
ga_checkpoint.json
generation_history.csv
subfamily_groups.csv
selected_subfamilies.json
selected_features.json
best_solution.json
run_config.json
```

## Fitness

The default fitness favors macro F1 and penalizes large selections:

```text
fitness = alpha * macro_f1
          - beta * selected_group_fraction
          - gamma * selected_feature_fraction
```

Defaults:

```text
alpha = 1.0
beta  = 0.03
gamma = 0.02
```

## Validation

Real runs should use session-independent validation:

```text
--validation-mode loso
```

`loso` means leave-one-session-out. Each fold trains on four IEMOCAP sessions
and tests on the remaining held-out session, then fitness uses mean macro F1
across folds.

For tiny smoke tests only, use:

```text
--validation-mode stratified
```

That uses a fast stratified random train/test split and is not
speaker/session independent.
