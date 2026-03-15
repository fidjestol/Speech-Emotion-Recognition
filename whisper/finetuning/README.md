# Whisper SER Finetuning

This folder now contains a finetuning scaffold for adapting Whisper to IEMOCAP with two supported variants:

- `with_xxx`
- `without_xxx`

By default, the training workflow runs both so you can compare them directly.

The shared retained labels are:

- `ang`
- `fea`
- `hap`
- `neu`
- `sad`
- `sur`

Optional label:

- `xxx`

Still excluded:

- `exc`
- `fru`
- `oth`
- `dis`

## Notebook

- `whisper/finetuning/finetune_whisper_ser.ipynb`

The notebook is organized in the same sectioned style as the rest of the repo and shows:

- environment and config checks
- filtered class counts
- split summaries
- model label setup
- dataset sample preview
- training argument summary
- per-variant train/validation/test metrics
- variant comparison summary
- saved artifact paths

## Script

- `whisper/finetuning/finetune_whisper_ser.py`

The script version runs the same finetuning pipeline directly from the terminal and saves separate outputs for `with_xxx` and `without_xxx`.

## Run

```bash
uv run python whisper/finetuning/finetune_whisper_ser.py
```

Smoke run example:

```bash
uv run python whisper/finetuning/finetune_whisper_ser.py --max-train-samples 64 --max-eval-samples 32
```

Single variant example:

```bash
uv run python whisper/finetuning/finetune_whisper_ser.py --no-run-both-xxx-variants --include-xxx
```
