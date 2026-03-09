# GAN Decisions To Confirm

Use this file to lock major choices before implementation.

## 1) Folder Placement

- Current default: `feature_augmentation/gan_workflow`

## 2) Input Feature Table

- Candidate A: `feature_selection/anova/select_k_best_anova/artifacts/combined_X_train_selected.csv`
- Candidate B: family-specific train tables (mfcc/tfidf/bert/etc.)
- Candidate C: merged table built in `feature_test/classical_feature_system`

## 3) Label Source

- Confirm where training labels come from for each row.
- If labels are not in the feature CSV, define join key (usually `path`) and label file.

## 4) GAN Variant

- Recommended start: class-conditional MLP GAN for tabular numeric features.
- Alternative: CTGAN-style training for mixed data.

## 5) Balancing Policy

- `max`: raise all classes to the majority class count.
- `median`: raise only classes below median count.
- `fixed`: set explicit per-class target counts.

## 6) Leakage Controls

- Train GAN on train split only.
- No synthetic rows in validation/test.
- Keep a synthetic provenance flag in augmented outputs.

## 7) Success Criteria

- Primary metric (for example macro-F1 on held-out test).
- Required minimum lift to keep GAN augmentation in pipeline.

