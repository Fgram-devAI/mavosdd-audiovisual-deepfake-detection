# Audio Channel Normalization Audit

Date: 2026-07-02

## Verdict

Partial: the normalization branch is implemented correctly and produced complete artifacts, but it did not achieve the stronger goal of removing acoustic/channel shortcuts from the audio spoof task.

## Scope

This audit reviewed the audio channel normalization pipeline, the normalized manifest and feature artifacts, and the downstream validation results. The branch normalizes the codec-matched WAV inputs into `data/derived/audio_normalized/`, rewrites the audio spoof manifest to point at those normalized WAVs, re-extracts frozen SSL embeddings, and retrains the audio MLP baselines.

The normalization pipeline is:

```text
decode -> mono -> 16 kHz resample -> silence trim -> 7 kHz lowpass
       -> EBU R128 loudness normalization at -23 LUFS
       -> peak safety cap -> PCM-16 WAV
```

## Implementation Findings

No blocking implementation bugs were found.

- `src/data/audio_normalize.py` is deterministic. The transform stack does not use randomness or mutable global state.
- The lowpass transform uses a zero-phase Butterworth SOS filter and the corrected SciPy-style pad length formula, then raises a typed error when the clip is too short for filtering.
- Loudness normalization uses a silence guard and peak safety scaling, which prevents silent or near-silent rows from being amplified into junk.
- `src/data/normalize_audio_channel.py` writes WAVs atomically through a temporary `.wav` path and `os.replace`, so interrupted writes should not leave accepted partial files.
- Path-token validation rejects traversal, separators, whitespace, colon, NUL, and `.`/`..`, which is the right level of defensive checking for `provider/sample_id` paths.
- The `_run.json` provenance sidecar records row counts, parameters, failure summaries, fallback counts, and git commit.

Test status:

```text
.venv/bin/python -m pytest tests/data -q
187 passed
```

The full suite had unrelated MediaPipe/OpenGL failures in visual prediction tests:

```text
.venv/bin/python -m pytest -q
354 passed, 4 failed
```

Those failures are not evidence against the audio normalization branch.

## Artifact Findings

The normalized audio artifact set is complete.

- `data/derived/audio_normalized/_run.json` reports `n_rows_in=6367`, `n_rows_valid=6367`, `n_rows_written=6367`, `n_rows_failed=0`, and `n_fallbacks=5`.
- `data/derived/audio_spoof_manifest_normalized.csv` preserves all 6367 rows and adds `original_audio_path`.
- The voice-disjoint split is preserved: train/val/test voice-id intersections are empty.
- The normalized manifest points `audio_path` to normalized WAVs. The only expected `_codec` references are in `original_audio_path`, which is provenance, not the training audio path.
- The input to normalization was already codec-matched WAV, not mixed raw MP3/WAV. The source manifest `audio_spoof_manifest_voice_split.csv` points every row at `data/audio_wav_codec_matched/*.wav`.
- Spot checks show normalized WAVs are 16 kHz, mono, PCM-16, peak-safe, and strongly attenuated above the 7 kHz lowpass.

Embedding artifacts are also complete:

- `data/features/audio_wav2vec2_normalized/`: 6367 `.npy`
- `data/features/audio_wavlm_normalized/`: 6367 `.npy`
- `data/features/audio_hubert_normalized/`: 6367 `.npy`

All checked arrays are shape `(199, 768)` and dtype `float16`, with one file per manifest `sample_id`.

The normalized checkpoints point at normalized feature directories, so the near-perfect metrics are not explained by accidentally training on codec-matched embeddings.

## Evaluation Findings

The key result is that normalization reduced shallow acoustic leakage but did not remove it.

Before normalization, the hand-crafted acoustic probe was nearly saturated:

```text
LR ROC-AUC: 0.9914
RF ROC-AUC: 0.9971
```

After normalization, the same probe still remains very high:

```text
LR ROC-AUC: 0.9713
RF ROC-AUC: 0.9889
```

The strongest residual single-feature shortcuts after normalization are still spectral and channel-like:

```text
spectral_bandwidth_std: 0.9222
spectral_rolloff_std: 0.8979
spectral_centroid_std: 0.8468
noise_floor_db: 0.8336
zcr_std: 0.7831
rms: 0.7350
trailing_silence_s: 0.7278
```

Provider leave-one-engine-out checks are mixed:

```text
held elevenlabs: 0.7971
held google_tts: 0.9699
held openai_tts: 0.9774
```

That means some engine-specific leakage was weakened, especially for ElevenLabs, but Google/OpenAI TTS remain highly separable from original audio using shallow features.

Direct logistic-regression probes on mean-pooled SSL embeddings show the same story:

```text
wav2vec2 codec:      0.9658
wav2vec2 normalized: 0.9659
wavlm codec:         0.9999
wavlm normalized:    0.9992
hubert codec:        0.9998
hubert normalized:   0.9992
```

The normalized deep baselines are therefore not just an MLP-capacity artifact. The frozen pretrained encoders still expose highly discriminative cues.

Normalized mel-CNN shows the same pattern. The previous notebook-based codec mel baseline is recorded in `report/val_eval/all_checkpoints_val_metrics.json` as ROC-AUC `1.0`, EER `0.0`, and F1 `1.0`. The normalized mel notebook run on `data/features/audio_mel_normalized` produced:

```text
mel-CNN normalized ROC-AUC: 0.9999381647291615
mel-CNN normalized EER:     0.0028897683238519206
mel-CNN normalized F1:      0.9961587708066582
```

Per-provider validation recall for the normalized mel-CNN was:

```text
elevenlabs: 1.0000
google_tts: 1.0000
openai_tts: 0.9942
original:   0.9968
```

This confirms that normalized mel-spectrograms remain almost perfectly separable too. The result strengthens the interpretation that remaining separability is visible in relatively direct time-frequency structure, not only in abstract SSL embeddings.

## Root-Cause Ranking

1. Residual acoustic/channel shortcuts remain. This is the most directly supported explanation because the normalized acoustic probe still reaches LR ROC-AUC 0.9713 and RF ROC-AUC 0.9889.

2. SSL encoders preserve discriminative synthesis cues that the normalization does not remove. WavLM and HuBERT remain near-perfect even with mean-pooled linear probes, which means separability exists before the trainable MLP.

3. The dataset construction still allows real-vs-TTS differences beyond simple channel effects. Prosody, phonetic timing, speaker/timbre distributions, vocoder texture, and generator-specific artifacts are not neutralized by trim, lowpass, and loudness normalization.

4. Loader/path leakage is unlikely. The manifest and checkpoints point at normalized audio/features. `original_audio_path` retains codec-matched provenance but is not used by the feature-store dataset loader for audio tensors.

5. Evaluation split leakage is unlikely based on the checked voice-disjoint intersections. The split deserves continued protection, but it is not the current best explanation.

## Recommendations

Merge the branch as a valuable preprocessing and audit improvement, but do not claim that channel confounds are solved.

For the report, phrase the result as:

```text
Channel normalization reduced obvious nuisance cues but did not eliminate
dataset-level separability. The remaining near-perfect WavLM/HuBERT scores
should be interpreted as performance on this constructed benchmark, not proof
of robust real-world audio deepfake detection.
```

Do not keep changing audio results indefinitely. The current branch is enough to close the audio-normalization story:

1. Keep the normalized acoustic-probe numbers as the honest limitation.
2. Keep the normalized mel-CNN result as additional evidence that the audio-only task remains nearly saturated after normalization.
3. Move to lip-sync/visual consistency and late fusion.

The next substantial work should be visual/lip-sync completion, not another audio-only cleanup pass.

## Normalized Mel Extraction

The mel extractor already follows the manifest `audio_path`, so normalized mels are created by pointing it at the normalized manifest:

```bash
.venv/bin/python -m src.features.extract_mel \
  --manifest data/derived/audio_spoof_manifest_normalized.csv \
  --out-dir data/features/audio_mel_normalized
```

Optional smoke run:

```bash
.venv/bin/python -m src.features.extract_mel \
  --manifest data/derived/audio_spoof_manifest_normalized.csv \
  --out-dir data/features/audio_mel_normalized_smoke \
  --limit 32 \
  --overwrite
```

Verify count and shape:

```bash
find data/features/audio_mel_normalized -name '*.npy' | wc -l

.venv/bin/python - <<'PY'
import glob
import numpy as np

p = sorted(glob.glob("data/features/audio_mel_normalized/*.npy"))[0]
a = np.load(p)
print(p, a.shape, a.dtype, float(a.min()), float(a.max()))
PY
```

The expected count is 6367. The arrays are log-power mel-spectrograms from the normalized WAV paths. Training on these currently belongs in the mel notebook path, because the main `src/train.py` audio baseline is built around fixed `(199, 768)` SSL embeddings.

Notebook hygiene note: the normalized notebook constants should keep `CKPT = Path("models/checkpoints/best_mel_cnn_normalized.pt")` for the full run. If a later cell resets `CKPT` to `best_mel_cnn.pt`, it will overwrite the local ignored codec checkpoint. The previous codec metric is still preserved in `report/val_eval/all_checkpoints_val_metrics.json`.

Current mel gap:

- `src/features/extract_mel.py` can produce normalized mel features.
- `src/data/feature_store.py --validate` does not yet validate mel stores.
- `src/train.py` and `src/evaluate.py` do not yet expose a mel-CNN backend.
- The existing mel-CNN baseline lives in `notebooks/01_mel_cnn_baseline.ipynb`.

So the normalized mel workflow is currently hybrid: CLI for extraction, notebook for training/evaluation. That is acceptable for closing this branch if the goal is to compare against the previous notebook-based mel baseline. Making mel a first-class CLI backend should be a separate cleanup branch, for example `feat/mel-cnn-cli`.

## Close-Out Plan

After this branch lands, the recommended order is:

1. `feat/lip-sync-consistency`: finish repository-level lip/visual feature setup and evaluate audio-visual consistency. This should not rely only on colleague notebook results.
2. `feat/generated-video-batch-eval`: ingest the new fully AI-generated videos as a separate evaluation batch with explicit labels for audio fake, visual fake, and pair consistency.
3. Fusion pass: compare audio-only, visual-only, lip-sync/consistency, and late-fusion models under the same split/evaluation policy.

For the new generated-video batch, avoid collapsing every generated source into one vague "total fake" label too early. Keep source/provider metadata and evaluate at least:

- real vs any fake;
- original MAVOS generated families vs new fully generated videos;
- audio-fake, visual-fake, and audio-visual consistency as separate axes.

That structure will make the final fusion result easier to explain and much less brittle.
