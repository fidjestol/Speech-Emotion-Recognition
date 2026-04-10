# GA Implementation Plan For Feature Family Selection

## Goal
Use a genetic algorithm to choose the best subset of feature families for speech emotion recognition.

## Agreed Design Choices
- Search space: feature families rather than individual features.
- Family list:
  - `mfcc_raw`
  - `prosody_energy`
  - `prosody_pitch`
  - `ssl_hubert`
  - `ssl_wav2vec`
  - `bert`
  - `tfidf`
  - `tonality`
  - `rhythm_pauses`
  - `representations`
- Chromosome: binary vector, one bit per feature family.
- `include_xxx`: toggleable boolean switch.
  - Default: `False`, which means exclude `xxx`.
- Filtering logic: reuse the existing project logic from [feature_selection/common.py](/cluster/home/bekadb/Speech-Emotion-Recognition/feature_selection/common.py).
- Fitness metric: `macro F1`.
- Fitness formula:
  - `fitness = alpha * macro_f1 - beta * (num_selected_families / d)`
- Starting values:
  - `alpha = 1.0`
  - `beta = 0.05`
- Evaluator model:
  - configurable
  - default implementation: `random_forest`
  - optional support: `logreg`, `linear_svc`
- Data split for GA search:
  - `train_test_split(test_size=0.20, random_state=42, stratify=y)`
- Selection: tournament selection.
- Crossover: uniform crossover.
- Mutation: bit-flip mutation with default rate `1 / d`.
- Elitism: keep top `3` individuals.
- Population size: `40`.
- Number of generations: `30`.
- Minimum selected families: `1`.

## Checkpointing And Resume
- The GA should checkpoint after every completed generation.
- Resume should continue from the next generation in a later run.
- Checkpoint should save:
  - current population
  - evaluated fitness cache
  - history so far
  - best solution so far
  - Python random state
  - NumPy random state
- This is generation-level resume, not inside-model resume.

## W&B Reporting
- Use W&B for live reporting during GA runs.
- Helpful live metrics:
  - current generation
  - best fitness
  - average fitness
  - best macro F1
  - average macro F1
  - average number of selected families
  - number of unique chromosomes in the population
- Helpful final outputs:
  - history table
  - best chromosome
  - selected family names
  - best macro F1 and accuracy summary

## Saved Outputs
Under the GA artifacts directory, save:
- run config
- generation history CSV
- checkpoint files
- best solution JSON
- selected families JSON
- optional W&B run files

## Validation Strategy
- Use the GA to find promising feature-family subsets efficiently.
- Validate the best family subsets later with the stronger project evaluation pipeline if needed.
