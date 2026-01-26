Feature family modules

- cepstral_mfcc.py: MFCCs with deltas plus compact summary export for inspection.
- prosody_energy.py: RMS energy contour with voiced masking and summary export.
- prosody_pitch.py: F0 tracking with voicing logic and summary export.
- representations.py: Log-mel spectrogram representation summaries.
- rhythm_pauses.py: Pause and onset-rate statistics from nonsilent intervals.
- smilesets.py: openSMILE standard feature sets (eGeMAPS/GeMAPS/ComParE).
- spectral_shape.py: Spectral-shape contours (centroid, rolloff, contrast, etc.) with summaries.
- ssl_embeddings.py: HuBERT/wav2vec2 embeddings with pooled mean/std and caching.
- tonality.py: Chroma and tonnetz summaries over time.
- voice_quality.py: HNR via parselmouth with HPSS fallback proxy.
