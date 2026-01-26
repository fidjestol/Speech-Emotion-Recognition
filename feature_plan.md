# MELD Audio Feature Sets Plan

## Goal
Create a structured, repeatable way to define, extract, cache, and evaluate multiple audio feature sets for the MELD dataset, grounded in the feature families listed in the notes (MFCCs, log-mel/spectrograms, prosodic/LLDs, and SSL embeddings like wav2vec2/HuBERT).

## Core principles
- Consistency: one shared preprocessing pipeline so feature sets are comparable.
- Traceability: every feature set has a config, version, and dimensionality recorded.
- Reproducibility: train-only statistics for normalization, deterministic parameters.
- Scalability: cache features on disk, avoid recomputation, allow partial reruns.

## Plan (phased)
### Phase 1: Standardize audio preprocessing
- Sample rate: 16 kHz (or 22.05 kHz if matching legacy specs); pick one and stick to it.
- Channel: mono conversion.
- Loudness: optional RMS normalization or peak normalization. (dont normalize for now)
- Trim: keep leading/trailing silence unless specifically testing silence trimming.
- Windowing defaults: n_fft=1024, hop=256 (adjust if matching prior work).
- Save a single canonical preprocessing config file.

### Phase 2: Define a feature set registry
Create a single source of truth (YAML or JSON) that defines each feature set.
Recommended schema:
- name
- family (mfcc, logmel, spectrogram, prosody, ssl, hybrid)
- inputs (audio waveform, spectrogram, etc.)
- params (n_mfcc, n_mels, n_fft, hop, window, fmin/fmax, etc.)
- postproc (deltas, deltas2, normalization, statistics)
- output_dim (explicit or derived)
- notes (link to paper or rationale)

### Phase 3: Implement feature extractors
Implement a modular extractor per family:
- MFCC family: MFCC (13), MFCC+delta+delta2, 20/40 MFCC variants, log-energy.
- Log-mel / Spectrogram: 40/80/120 mel bands, linear spectrogram, spectrogram images.
- Prosodic + spectral LLDs: pitch, energy, ZCR, centroid, bandwidth, rolloff, chroma,
  contrast, tonnetz.
- OpenSMILE sets: eGeMAPS, ComParE (if available via python wrappers).
  https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf, might be worth re-reading this.
- SSL embeddings: wav2vec2.0, HuBERT (base/large), pooled per-utterance.

Each extractor should:
- Accept the canonical preprocessed waveform + sr.
- Emit a fixed-length vector or a time series with a known shape.
- Record shape and stats in a sidecar metadata file.

### Phase 4: Caching and versioning
- Cache features in `outputs/features/<feature_set>/<split>/`.
- Use a hash of preprocessing + feature params in the cache key.
- Save a small JSON metadata file per utterance with:
  - duration, sr, params, shape
  - normalization stats used

### Phase 5: Baselines and evaluation protocol
- Baselines:
  - MFCC-13 + delta + delta2
  - 40-mel log-mel spectrogram (frame-level)
  - Prosody + spectral LLDs (summary stats)
  - wav2vec2 base pooled embeddings
- Metrics: F1 Score, recall, accuracy, precision, recall
- Always report mean and std across runs or folds.
- Maintain an ablation table: feature families on/off. 

### Phase 6: Iterative expansion
Add complex sets only after baselines are stable:
- Spectrogram CNN encoders (2D CNN on log-mel)
- Hybrid features: concatenate MFCC stats + SSL pooled embeddings
- Feature selection: SFS/SBS or dimensionality reduction (PCA)

## Suggested feature set catalog (initial)
1. mfcc_13_d_dd
2. mfcc_40
3. logmel_40
4. logmel_80
5. spectrogram_linear
6. llds_basic (pitch, energy, zcr, centroid, rolloff, etc.)
7. chroma_contrast_tonnetz
8. eGeMAPS (if openSMILE available)
9. wav2vec2_base_meanpool
10. hubert_base_meanpool
11. hybrid_mfcc_stats_plus_w2v2

## Things to look out for (ML emotion recognition)
- Data leakage: never compute normalization stats on dev/test.
- Speaker leakage: MELD splits are predefined; keep them fixed.
- Class imbalance: report macro metrics and consider class weights.
- Label noise: MELD labels can be subjective; expect confusion between
  similar emotions (e.g., neutral vs. sadness).
- Temporal context: utterance-only may miss dialog context; keep this in mind
  when comparing to works using context.
- Feature scaling: per-utterance vs. global normalization changes performance.
- Overfitting risk: high-dimensional features (ComParE, spectrogram images)
  require stronger regularization and larger models.
- Pitch extraction: unvoiced regions and noisy audio can bias pitch features.
- Silent frames: decide whether to trim or keep; it affects energy-based features.
- Reproducibility: fix random seeds and log all configs.

## Next implementation steps (repo-oriented)
- Add a feature registry file: `configs/feature_sets.json`.
- Add a runner script that loads registry entries and extracts to cache.
- Add a simple report generator that prints feature shapes, summary stats,
  and sample plots.

