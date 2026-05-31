# Speech-Emotion-Recognition
Masters project on Speech Emotion Recognition in call automation.

## Repository Contents
This repository tracks source code, notebooks, configuration files, and documentation for
the thesis experiments. Experiment artifacts and dataset material needed for final
delivery are kept in Git. Local runtime files, logs, environments, and generated thesis
report exports are intentionally left out.

To reproduce experiments, use the tracked project layout and rerun the relevant feature
extraction, feature selection, augmentation, or evaluation scripts as needed.

## Idun GPU Job Template
Use `scripts/idun_gpu.sbatch` as a starting point for GPU jobs on Idun.

```bash
sbatch scripts/idun_gpu.sbatch
```

For notebook batch execution on GPU, use `scripts/idun_gpu_notebook.sbatch`.
