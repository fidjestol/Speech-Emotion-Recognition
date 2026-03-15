# Whisper SER Evaluation

This folder evaluates the Hugging Face checkpoint [`firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) on the local IEMOCAP metadata in a way that matches the repo's newer `feature_selection` notebooks:

- stratified `train_test_split(test_size=0.20, random_state=42)`
- `accuracy`, weighted `precision/recall/f1`, and `f1_macro`
- saved predictions, classification report, confusion matrix, and run metadata

## Label policy

The upstream Whisper checkpoint predicts:

- `angry`
- `fearful`
- `happy`
- `neutral`
- `sad`
- `surprised`

This evaluation now keeps only the clean label overlaps between IEMOCAP and the Whisper checkpoint, and explicitly excludes `disgust`.

The notebook/script therefore evaluates this 6-way comparison set:

- `ang -> ang`
- `fea -> fea`
- `hap -> hap`
- `neu -> neu`
- `sad -> sad`
- `sur -> sur`

It excludes:

- `exc`
- `fru`
- `oth`
- `xxx`
- `dis`

## Notebook

The main evaluation workflow is now in:

- `whisper/evaluation/evaluate_whisper_ser.ipynb`

It is organized like the PCA notebooks, with each section printing or displaying something useful:

- environment and config checks
- filtered label counts
- train/test split summaries
- checkpoint label mapping
- metrics table
- prediction preview
- classification report
- confusion matrix
- artifact export summary

## Script

The original script version is still available as:

- `whisper/evaluation/evaluate_whisper_ser.py`

## Run

```bash
uv run python whisper/evaluation/evaluate_whisper_ser.py
```

Smoke test on a few rows:

```bash
uv run python whisper/evaluation/evaluate_whisper_ser.py --limit 16 --batch-size 2
```

Outputs are written under `whisper/evaluation/artifacts/whisper_large_v3_iemocap_clean_6way/`.
