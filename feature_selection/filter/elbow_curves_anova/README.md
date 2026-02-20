# ANOVA Elbow Curve Notebooks

This folder contains per-family notebooks that sweep top-`k` ANOVA-ranked features and plot elbow curves for:
- `accuracy`
- `f1_weighted`

## Notebooks
- `elbow_curve_bert.ipynb`
- `elbow_curve_mfcc_raw.ipynb`
- `elbow_curve_mfcc_normalized.ipynb`
- `elbow_curve_prosody_energy.ipynb`
- `elbow_curve_prosody_pitch.ipynb`
- `elbow_curve_tfidf.ipynb`

Each notebook saves outputs to `feature_selection/filter/elbow_curves_anova/artifacts/`:
- `<family>_elbow_curve_scores.csv`
- `<family>_elbow_summary.csv`

If a source CSV is missing for a family (for example `text` or `prosody_pitch`), the notebook will raise a clear `FileNotFoundError` describing what to generate first.
