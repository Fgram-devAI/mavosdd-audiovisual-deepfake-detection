# Audio Curation Pipeline — TTS Output → Training-Ready Features

What to run, in order, **after every TTS engine has finished writing files
under `data/tts_audio/<engine>/<source_folder>/*`** and lip features are
extracted, so that everything lines up before you launch training.

This doc is operational. It documents the steps and the reason each one
exists; the deeper "what failed in Phase 1–5 that justified this design"
discussion lives in `README.md` (Anti-Leakage) and `CLAUDE.md`.

## TL;DR — the run order

1. Sanity-check TTS outputs vs transcripts
2. Build derived manifests
3. Codec-match bonafide audio (round-trip through MP3)
4. Re-extract audio embeddings on codec-matched WAVs
5. Generate voice-disjoint splits
6. Apply voice-split assignment to visual + fusion manifests
7. Validate feature store
8. Train

Each step writes new artifacts only — none destroy upstream inputs except
`make_splits.py`, which is the only truly destructive script in the
repo (see "Backups" below).

---

## 1. Sanity-check TTS outputs

Before building anything, confirm each engine produced what you expect.

```bash
# Counts per engine
for engine in elevenlabs google_tts coqui_xtts openai_tts; do
  count=$(find "data/tts_audio/$engine" -type f \( -name '*.mp3' -o -name '*.wav' \) 2>/dev/null | wc -l)
  echo "$engine: $count files"
done

# Any zero-byte files?
find data/tts_audio -type f -size 0
```

If a count is suspiciously low, re-run the corresponding synthesize script
— it skips existing files, so re-running only catches the gaps.

## 2. Build derived manifests

The build script reads `data/manifest.csv` + every TTS engine's manifest
JSONL and emits the three manifests used downstream:

```bash
python -m src.data.build_speech_manifests
```

Outputs under `data/derived/`:

- `audio_spoof_manifest.csv` — every bonafide and TTS-spoof row paired by `source_video_id`. Audio-only training input.
- `visual_speech_manifest.csv` — one row per (real video, fake-source) pair. Visual-only training input.
- `fusion_speech_manifest.csv` — joined audio+lip rows. Fusion training input.

All three follow the 17-column schema documented in
`src/data/build_speech_manifests.py`. Run the script's own validator to
confirm:

```bash
python -m src.data.build_speech_manifests --validate
```

## 3. Codec-match bonafide audio

**Why:** bonafide WAVs are clean 16 kHz PCM; TTS spoofs come back as
lossy MP3 (44.1 kHz / 128 kbps for ElevenLabs, 24 kHz / 64 kbps for
Google, etc.). A 100% format↔label correlation lets any mel-input model
learn "WAV ≡ real, MP3 ≡ fake" instead of an actual deepfake feature.

The script round-trips each bonafide WAV through a random MP3 codec spec
sampled from the spoof distribution, then decodes back to 16 kHz mono
WAV. Codec history becomes label-independent.

```bash
# Requires ffmpeg + libmp3lame on PATH
python -m src.data.codec_match_audio
```

Outputs:
- `data/audio_wav_codec_matched/` — new bonafide WAVs (lossy-history but PCM container)
- `data/derived/audio_spoof_manifest_codec_matched.csv` — manifest pointing at the new paths

## 4. Re-extract audio embeddings (codec-matched)

The original embedding stores under `data/features/audio_<backend>/` were
extracted from the clean bonafide WAVs — they're now stale relative to
the codec-matched manifest. Re-extract into the dedicated codec stores:

```bash
for B in wav2vec2 wavlm hubert; do
  python -m src.features.extract_audio_embeddings \
    --backend $B \
    --manifest data/derived/audio_spoof_manifest_codec_matched.csv \
    --overwrite
done
```

Outputs land at `data/features/audio_<backend>_codec/<sample_id>.npy`.
The extractor reads each row's `audio_path` directly, so spoof rows pull
from `data/tts_audio/...` and bonafide rows pull from
`data/audio_wav_codec_matched/`.

The extractor is per-file idempotent: drop `--overwrite` after the first
full run if you only want to backfill new rows. Use `--overwrite` after
any change to the codec pipeline or the manifest.

## 5. Voice-disjoint splits

**Why:** even with codecs neutralized, the same TTS voice can appear in
train, val and test simultaneously — the model can shortcut to "I've
heard this voice in train, label spoof." Voice-disjoint splitting
confines each `(provider, voice_id_or_name)` to exactly one split.

```bash
python -m src.data.make_voice_disjoint_manifest
```

Output: `data/derived/audio_spoof_manifest_voice_split.csv`. Same rows as
the codec-matched manifest, with the `split` column rewritten so no
voice straddles splits. This is the canonical audio training input.

## 6. Apply voice-split to visual + fusion manifests

The voice-disjoint split is computed on the audio manifest. The visual
and fusion manifests must carry the **same** split assignment so a row's
"split = val" decision is consistent across views. `apply_voice_split`
rewrites only the `split` column; every other field (especially
`pair_label_binary` and the feature paths) is byte-identical.

```bash
python -m src.data.apply_voice_split \
  --target data/derived/visual_speech_manifest.csv \
  --out    data/derived/visual_speech_manifest_voice_split.csv

python -m src.data.apply_voice_split \
  --target data/derived/fusion_speech_manifest.csv \
  --out    data/derived/fusion_speech_manifest_voice_split.csv
```

`apply_voice_split` raises ValueError if any target `sample_id` is
missing from the source — that's how it guarantees the assignment is
total, not partial.

## 7. Validate the feature store

Before launching any training run, validate that every manifest row has
a present, well-shaped feature file on disk:

```bash
# Audio (per backend)
for B in wav2vec2 wavlm hubert; do
  python -m src.data.feature_store --validate --view audio --backend $B
done

# Visual
python -m src.data.feature_store --validate --view visual

# Fusion (per backend)
for B in wav2vec2 wavlm hubert; do
  python -m src.data.feature_store --validate --view fusion --backend $B
done
```

Exit code is non-zero iff there are `missing` or `bad_shape` errors.
`path_mismatches` are warnings only — they indicate a manifest column
disagrees with the canonical reconstructed path, which is informational
unless you've been editing manifests by hand.

## 8. Train

At this point every dataset view (audio, visual, fusion) is honest:
codec-matched, voice-disjoint, validated. Train with:

```bash
# Audio anti-spoof, per backend
python -m src.train --modality audio --backend wav2vec2 --run-name audio_wav2vec2_codec
python -m src.train --modality audio --backend wavlm     --run-name audio_wavlm_codec
python -m src.train --modality audio --backend hubert    --run-name audio_hubert_codec

# Visual (use --drop-no-face to filter zero-mask rows from train; val/test stay honest)
python -m src.train --modality visual --drop-no-face --run-name visual_bigru

# Fusion (per backend)
python -m src.train --modality fusion --backend wav2vec2 --drop-no-face --run-name fusion_wav2vec2_codec
python -m src.train --modality fusion --backend wavlm    --drop-no-face --run-name fusion_wavlm_codec
python -m src.train --modality fusion --backend hubert   --drop-no-face --run-name fusion_hubert_codec
```

Train evaluates on val every epoch and saves the best-val-ROC-AUC
checkpoint. Test split is gated — `python -m src.evaluate
--checkpoint ... --split test` refuses unless you pass `--allow-test`,
and you should only do that once per checkpoint, after model selection
on val.

---

## Backups before any destructive step

`make_splits.py` is the **only** destructive script in the pipeline. It
unconditionally overwrites `data/splits/*.csv`. Splits are
seed-deterministic but seed-42 reproducibility breaks across cap
changes because the input DataFrame size and stratum proportions differ
— the same `video_id` can land in a different split between regenerations.

Before re-running `make_splits` after a dataset cap change, take a
one-time backup:

```bash
cp -r data/splits   data/splits.phase1_5_backup
cp    data/manifest.csv data/manifest.phase1_5_backup.csv
cp -r data/derived  data/derived.phase1_5_backup
```

(These three paths are gitignored.)

Every other curation script (`build_speech_manifests`,
`codec_match_audio`, `extract_audio_embeddings`,
`make_voice_disjoint_manifest`, `apply_voice_split`) writes new files
without touching the originals. Idempotency: each per-file extractor
skips existing outputs unless `--overwrite` is passed.

## Skip-if-exists matrix

| Script | Re-run safe? | Skip mechanism | `--overwrite` |
|---|---|---|---|
| `make_splits.py` | NO — destructive | None | n/a |
| `build_speech_manifests.py` | Yes — deterministic | Always writes fresh | n/a |
| `codec_match_audio.py` | Yes | Per-file existence check | Yes |
| `extract_audio_embeddings.py` | Yes | Per-file existence check | Yes |
| `extract_lips.py` | Yes | Per-file existence check | n/a (delete to re-extract) |
| `make_voice_disjoint_manifest.py` | Yes — deterministic | Always writes fresh | n/a |
| `apply_voice_split.py` | Yes — deterministic | Always writes fresh | n/a |
| `synthesize_*_from_transcripts.py` | Yes | Per-file existence check | Yes |
| `transcribe_google_stt_v2.py` | Yes | Per-file existence check | Yes |

Anything in this table that's "Yes" with a `--overwrite` flag is the
controlled way to force a refresh after the underlying inputs change
(new codec spec, new TTS voice pool, new backend, etc.).
