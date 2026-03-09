# GAN Workflow (Feature Augmentation)

This folder is a scaffold for adding GAN-based augmentation on top of extracted
or ANOVA-selected tabular features.

## Default Location

Current default location is:

- `feature_augmentation/gan_workflow`

If you want this colocated with ANOVA artifacts instead, we can move it under:

- `feature_selection/anova/gan_workflow`

## Intended Workflow

1. Choose the feature input table (for example combined ANOVA train features).
2. Decide balancing target per class.
3. Train a class-conditional GAN on training features only.
4. Generate synthetic rows for minority classes.
5. Export augmented training table + run metadata.
6. Re-run model evaluation with identical test split.

## Included Files

- `decisions.md`: major decisions that need your confirmation.
- `configs/gan_experiment_template.json`: config template for experiments.
- `scripts/plan_gan_augmentation.py`: class-balance planning helper.

