# Genetic Algorithm for Feature Family Selection

This folder is for genetic-algorithm-based selection of **feature families** for speech emotion recognition (SER).

The idea follows a standard genetic algorithm setup and adapts it to this repository's feature-family setting.

## 1. Encoding

Represent a solution as a binary string of length `d`, where `d` is the number of feature families.

- `1` means the feature family is selected
- `0` means the feature family is not selected

Example for 8 feature families:

```text
[1, 0, 1, 1, 0, 0, 1, 0]
```

This selects families `1`, `3`, `4`, and `7`.

In this repository, a chromosome can represent inclusion/exclusion of families such as:

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

## 2. Initial Population

Create a population of random chromosomes, for example:

- `30` to `100` candidate subsets

It is also useful to seed some individuals with:

1. all feature families
2. top-k families from a filter method or earlier experiments
3. sparse random subsets

## 3. Fitness Function

Evaluate each chromosome by training and validating a model using only the selected feature families.

A common fitness function is:

```text
Fitness = alpha * Performance - beta * (#selected_features / d)
```

where:

- `Performance` can be macro F1, accuracy, AUC, RMSE, etc.
- the second term penalizes large subsets
- `alpha` and `beta` control the tradeoff between prediction quality and subset size

For SER in this repository, a good default is:

- `Performance = macro F1`

because class balance matters and macro F1 is more informative than plain accuracy.

## 4. Selection

Selection decides which chromosomes are more likely to become parents for the next generation.

The goal is to:

- give strong solutions a better chance to reproduce
- still keep enough diversity so the search does not collapse too early

Common choices:

1. tournament selection
2. roulette wheel selection
3. rank selection

### Tournament Selection

Tournament selection works by:

1. randomly sampling a small group of chromosomes from the population
2. comparing their fitness values
3. choosing the best one from that small group as a parent

Example:

- randomly pick 3 chromosomes
- compare their fitness
- keep the best one

Then repeat to pick the second parent.

Why it is useful:

- simple to implement
- does not require probabilities to be perfectly scaled
- robust when fitness values are close together or noisy
- easy to control selection pressure by changing tournament size

Tradeoff:

- if the tournament size is too large, the algorithm can become too greedy and lose diversity

Practical default:

- tournament size `3` or `4`

### Roulette Wheel Selection

Roulette wheel selection chooses parents with probability proportional to fitness.

Idea:

- each chromosome gets a slice of a roulette wheel
- better fitness means a larger slice
- spinning the wheel selects a parent

Why it is useful:

- directly reflects relative fitness
- weak but promising individuals can still be selected sometimes

Tradeoff:

- sensitive to fitness scaling
- if one chromosome is much stronger than the others, it can dominate too early
- can be unstable when fitness values are negative or tightly clustered unless normalized carefully

For this project, it is usable, but usually less convenient than tournament selection.

### Rank Selection

Rank selection sorts chromosomes by fitness and assigns selection probability based on rank rather than raw score.

Idea:

- best chromosome gets highest rank
- worst gets lowest rank
- selection depends on position in the ranking, not on the absolute fitness gap

Why it is useful:

- more stable than roulette wheel when raw fitness values have large or irregular gaps
- helps avoid one very strong individual dominating too quickly

Tradeoff:

- adds an extra sorting step
- ignores the exact size of fitness differences, which can sometimes slow convergence

### Which One Fits This Repository Best?

For this codebase, a practical default is:

- tournament selection

Reason:

- the fitness values may come from session-based SER evaluation and can be noisy
- tournament selection is simple, stable, and works well for discrete family-selection search spaces
- it is easy to tune without overcomplicating the pipeline

A good starting rule is:

- use tournament selection with tournament size `3`

## 5. Crossover

Combine two parent chromosomes to create offspring.

Common methods:

1. one-point crossover
2. two-point crossover
3. uniform crossover

Example:

```text
Parent 1: [1,0,1,1,0,0]
Parent 2: [0,1,0,1,1,0]
Offspring: [1,0,1,1,1,0]
```

A practical default for feature family selection is:

- uniform crossover

## 6. Mutation

Flip some bits with a small probability to maintain diversity.

Example:

```text
[1,0,1,1,1,0] -> [1,0,0,1,1,0]
```

A practical default is:

- mutation rate around `1/d` or a small constant such as `0.01`

## 7. Replacement

Form the next generation from offspring.

A common strategy is to keep the best current solution using **elitism**.

A practical default is:

- keep top `2` to `5` individuals unchanged

## 8. Stopping Criterion

Stop when one of these is met:

1. fixed number of generations
2. no improvement for several generations
3. desired performance reached

Practical defaults:

- `30` to `50` generations
- patience of `10`

## Basic Algorithm

1. Generate initial population
2. Evaluate fitness of each chromosome
3. Repeat until stopping condition:
   - select parents
   - apply crossover
   - apply mutation
   - evaluate offspring
   - form next generation
4. Return the best chromosome as the selected feature-family subset

## Adaptation to This Repository

In this project, the chromosome should usually represent **feature families**, not individual features, at least in the first phase.

That makes the search:

- smaller
- easier to interpret
- cheaper to run
- more useful scientifically

After identifying strong families, a second-stage search can do finer-grained feature subset selection inside the best families.

## Suggested First Defaults

Recommended starting choices for this codebase:

- encoding: binary family-inclusion vector
- population size: `40` to `60`
- selection: tournament
- crossover: uniform
- mutation: `1/d`
- elitism: `2` to `5`
- metric: macro F1
- fitness:

```text
fitness = alpha * macro_f1 - beta * (num_selected / d)
```

Suggested initial values:

- `alpha = 1.0`
- `beta = 0.05` to `0.2`

## Notes

Genetic algorithms are not limited to feature selection. They can also be used for:

- feature weighting
- hyperparameter tuning
- ensemble weighting
- model or pipeline search

For this repository, the most natural first use is:

- feature family selection

before moving to more fine-grained feature subset search.
