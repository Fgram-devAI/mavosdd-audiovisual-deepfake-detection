# Audiovisual Deepfake Detection

Late-fusion detector for MAVOS-DD videos, combining frozen audio embeddings
(Wav2Vec2 / WavLM / HuBERT) with MediaPipe lip-landmark motion features.

Phase 1–5 baselines (PRs #7–#9) were trained on the original **1,000-video
cap** (500 real, 250 EchoMimic, 250 MEMO). The roadmap is now revised to
**~4,149 videos** across five MAVOS-DD source folders (real, EchoMimic,
MEMO, LivePortrait, Sonic) for Phase 6+ work — see
[`docs/roadmap-audio-visual-speech-detection.md`](docs/roadmap-audio-visual-speech-detection.md)
Revision 1.

The pipeline is intentionally feature-first:

1. Stream a capped MAVOS-DD subset.
2. Extract frozen audio embeddings and MediaPipe lip-landmark sequences.
3. Train a compact (< 2M trainable params) late-fusion classifier on
   serialized `.npy`/`.npz` features only — raw video never enters the
   training loop.
4. Evaluate once on the locked test split and package a reproducible
   predictor.

## Quick Start

Python 3.10 virtual environment.

macOS / Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Smoke-test the model module:

```bash
python -m src.models.late_fusion
```

## Pipeline (Phase 1–4 — Data Preparation)

Each step writes gitignored artifacts under `data/`. Pass `--help` to any
module for full flags. The fetch is deterministic for a fixed MAVOS-DD
repository state; cap changes are picked up from `src.common.CAPS` and the
downloader is idempotent on re-runs.

1. **Fetch the MAVOS-DD subset.**

   ```bash
   python -m src.data.download_subset
   python -m src.data.download_subset --validate    # expect VALIDATION OK
   ```

2. **Freeze splits and extract features** (70/15/15 stratified on
   `source_folder`, seed 42; one `.npy` per video for audio, one `.npz`
   for lips).

   ```bash
   python -m src.data.make_splits
   python -m src.features.extract_audio
   python -m src.features.extract_lips
   ```

3. **Transcribe bonafide WAVs** (optional, prerequisite for TTS spoof
   generation). Requires `GOOGLE_APPLICATION_CREDENTIALS` and
   `GOOGLE_CLOUD_PROJECT` in `.env`.

   ```bash
   python scripts/export_wav.py
   python scripts/transcribe_google_stt_v2.py
   ```

4. **Generate spoof audio** from those transcripts. Each script supports
   `--estimate-only` (character count, no spend) and `--limit N` (smoke
   run). Outputs land under `data/tts_audio/`.

   | Engine             | Script                                                | Notes                                  |
   |--------------------|-------------------------------------------------------|----------------------------------------|
   | ElevenLabs TTS     | `scripts/synthesize_tts_from_transcripts.py`           | Text → speech, paid API                |
   | ElevenLabs STS     | `scripts/convert_real_speech_elevenlabs.py`            | Speech → speech (preserves prosody)    |
   | Google Neural2 TTS | `scripts/synthesize_google_tts_from_transcripts.py`    | Paid API; default voice rotation       |
   | Coqui XTTS-v2      | `scripts/synthesize_coqui_xtts_from_transcripts.py`    | Local, free; Phase 6+ multi-engine     |
   | OpenAI TTS         | `scripts/synthesize_openai_tts_from_transcripts.py`    | Paid API; Phase 6+ multi-engine        |

5. **Build derived manifests** (`audio_spoof`, `visual_speech`,
   `fusion_speech`).

   ```bash
   python -m src.data.build_speech_manifests
   ```

## Anti-Leakage (Phase 4 Hardening)

Two confounders surfaced during the mel-CNN baseline (PR #7) and were
neutralized in place. Training and evaluation default to the neutralized
inputs; the original (leaky) embeddings/splits remain on disk but are
unused by the merged baselines.

**Codec footprint match.** Bonafide rows are clean 16 kHz PCM WAV; every
TTS spoof row is lossy MP3 (ElevenLabs 44.1 kHz / 128 kbps, Google TTS
24 kHz / 64 kbps). That 100 % format/label correlation lets any mel-input
model shortcut to a WAV-vs-MP3 detector. Fix: round-trip every bonafide
through MP3 (codec spec sampled per row from the spoof distribution) and
decode all rows back to 16 kHz mono WAV, so codec history becomes
label-independent. Requires `ffmpeg` + `libmp3lame` on PATH.

```bash
python -m src.data.codec_match_audio                    # writes data/audio_wav_codec_matched/ + manifest

# Re-extract from the codec-matched WAVs (the legacy stores are now stale):
for B in wav2vec2 wavlm hubert; do
  python -m src.features.extract_audio_embeddings --backend $B \
    --manifest data/derived/audio_spoof_manifest_codec_matched.csv --overwrite
done
python -m src.features.extract_mel \
  --manifest data/derived/audio_spoof_manifest_codec_matched.csv --overwrite
```

**Voice-disjoint split.** Even with codec neutralized, the same TTS voice
appeared in train, val, and test simultaneously. Fix: confine each
`(provider, voice_id_or_name)` to exactly one split.

```bash
python -m src.data.make_voice_disjoint_manifest        # data/derived/audio_spoof_manifest_voice_split.csv
python -m src.data.apply_voice_split --target data/derived/visual_speech_manifest.csv \
  --out data/derived/visual_speech_manifest_voice_split.csv
python -m src.data.apply_voice_split --target data/derived/fusion_speech_manifest.csv \
  --out data/derived/fusion_speech_manifest_voice_split.csv
```

The audio voice-split manifest's `audio_path` already points at the
codec-matched WAVs, so this single file neutralizes **both** confounders.
The `apply_voice_split` helper rewrites only the `split` column of the
visual/fusion manifests, preserving `pair_label_binary` byte-identically.

## Phase 5 Baselines — Results

Validation-only model selection; the test split is locked for Phase 6's
single consolidated pass. Honest in-distribution val ROC-AUC on the
**codec-matched + voice-disjoint** inputs:

| Modality               | wav2vec2           | wavlm              | hubert             |
|------------------------|--------------------|--------------------|--------------------|
| audio                  | 0.9508 (EER 0.106) | 1.0000 (EER 0.006) | 1.0000 (EER 0.000) |
| fusion (audio ⊕ lips)  | 0.9509 (EER 0.107) | 1.0000 (EER 0.000) | 1.0000 (EER 0.000) |
| visual (lips only)     | 0.5688 (EER 0.433) | —                  | —                  |

Train:

```bash
# Audio anti-spoof (per backend)
python -m src.train --backend {wav2vec2,wavlm,hubert} --run-name audio_<backend>_codec

# Visual + fusion
python -m src.train --modality visual --run-name visual_bigru
python -m src.train --modality fusion --backend wav2vec2 --run-name fusion_wav2vec2_codec
```

Evaluate any checkpoint on val (test refused unless `--allow-test`):

```bash
python -m src.evaluate --checkpoint models/checkpoints/best_<name>.pt --split val
```

Full per-checkpoint metric battery (roc_auc, eer, eer_threshold, f1,
precision, recall, confusion, per-provider recall) is committed at
[`reports/val_eval/all_checkpoints_val_metrics.json`](reports/val_eval/all_checkpoints_val_metrics.json).

### Known Limitations

**Per-TTS-engine spectral fingerprinting.** WavLM and HuBERT saturate at
val ROC-AUC = 1.0 even after both confounders are neutralized; Wav2Vec2
reaches 0.95. The remaining shortcut is generator fingerprinting — every
TTS engine leaves vocoder/encoder artifacts (high-frequency residuals,
silence padding, bandwidth ceilings) invariant to voice, codec, and text.
The trained head is therefore closer to a *two-class TTS-engine detector*
(ElevenLabs OR Google TTS vs MAVOS-DD bonafide) than a generalized
deepfake detector — a new engine the model has not seen will likely evade
detection. The honest evaluation protocol is **engine-disjoint** /
**leave-one-engine-out** across ≥ 3 generators, scoped as the future
`feat/multi-engine-spoof` branch (Coqui XTTS-v2 and OpenAI TTS entry
points already shipped in `scripts/`).

**Visual-only collapses to a constant predictor.** Val confusion
`tn=0 / fp=150 / fn=0 / tp=217` — the BiGRU labels every row as spoof.
ROC-AUC 0.57 confirms the underlying scores don't separate the classes at
any threshold. Structural cause: every video contributes a paired bonafide
row and one or more matched spoof rows, all sharing the same
`source_video_id` and therefore the **same `.npz` lip features**. A model
asked to discriminate two rows whose input is byte-identical cannot do
better than class-frequency bias.

**Late fusion by concat ≈ audio-only.** Fusion wav2vec2 0.9509 vs audio
wav2vec2 0.9508 — statistically indistinguishable. Concat fusion can't
carry cross-modal interaction when one stream is at chance; it just
inherits the dominant stream. The signal that *would* help is
**audio-visual consistency** (does the mouth motion match the audio?) —
a SyncNet-style lip-sync head. That requires a roadmap revision and is
explicitly deferred.
