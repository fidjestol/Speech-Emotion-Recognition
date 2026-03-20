Feature family notebooks

- cepstral_mfcc.ipynb: MFCCs with deltas plus compact summary export for inspection.
- prosody_energy.ipynb: RMS energy contour with voiced masking and summary export.
- prosody_pitch.ipynb: F0 tracking with voicing logic and summary export.
- representations.ipynb: Log-mel spectrogram representation summaries.
- rhythm_pauses.ipynb: Pause and onset-rate statistics from nonsilent intervals.
- smilesets.ipynb: openSMILE standard feature sets (eGeMAPS/GeMAPS/ComParE).
- spectral_shape.ipynb: Spectral-shape contours (centroid, rolloff, contrast, etc.) with summaries.
- ssl_embeddings.ipynb: HuBERT/wav2vec2 embeddings with pooled mean/std and caching.
- text_lda_topics.ipynb: LDA topic-distribution features from utterance transcripts, plus dialogue-context topic features.
- tonality.ipynb: Chroma and tonnetz summaries over time.
- voice_quality.ipynb: HNR via parselmouth with HPSS fallback proxy.

Batch extraction (Python modules)

- Every `*.py` family module now includes `extract_batch(...)`.
- `extract_batch(...)` runs on multiple threads only when available (and enabled).
- Use `max_workers` to cap thread count for your host.
