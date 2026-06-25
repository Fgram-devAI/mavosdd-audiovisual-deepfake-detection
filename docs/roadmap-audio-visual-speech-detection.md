# Roadmap: Audio-Visual AI-Generated Speech Detection

This roadmap aligns the implementation with the assignment:

> Detect AI-generated speech using both audio features and visual cues. Use pretrained audio classifiers or embeddings, such as Wav2Vec2/ASVspoof-style features, and optionally visual lip movement features. Dataset sources include real audiovisual speech plus synthetic speech from commercial generators.

The key framing is per-modality:

- Original MAVOS-DD audio is treated as `audio_label=bonafide`, even when the video source is `echomimic` or `memo`.
- ElevenLabs and Google TTS outputs are treated as `audio_label=spoof`.
- MAVOS-DD `real` videos are `video_label=real`.
- MAVOS-DD `echomimic` and `memo` videos are `video_label=fake`.
- Audio-only generated assets inherit the split of their source video to prevent transcript/content leakage.

## Goal

Build and evaluate a compact deep-learning system that detects AI-generated speech with:

1. An audio anti-spoof branch.
2. A lightweight visual lip-motion cue branch.
3. A late-fusion classifier that tests whether visual cues improve audio-spoof detection.

The project should avoid a full SyncNet-style research detour. The visual branch is a practical cue for mouth-motion compatibility, not a full audio-lip synchronization model.

## Data Views

Create derived manifests rather than mutating the raw `data/manifest.csv`.

Recommended outputs:

```text
data/derived/audio_spoof_manifest.csv
data/derived/visual_speech_manifest.csv
data/derived/fusion_speech_manifest.csv
```

All rows should include:

```text
sample_id
source_video_id
split
media_type
source_folder
audio_path
audio_feature_path
lip_feature_path
provider
voice_id_or_name
audio_label
video_label
pair_label
```

Use `seed=42` and keep existing video train/val/test splits frozen.

## Label Rules

### Audio Label

```text
bonafide:
  - original real audio
  - original echomimic audio
  - original memo audio

spoof:
  - ElevenLabs TTS audio
  - Google TTS audio
  - optional ElevenLabs speech-to-speech audio
```

Never label original EchoMimic/MEMO audio as spoof.

### Video Label

```text
real:
  - MAVOS-DD real videos

fake:
  - MAVOS-DD echomimic videos
  - MAVOS-DD memo videos

na:
  - audio-only generated assets without a native video
```

### Pair Label

Use this only for the visual/fusion speech task:

```text
matched_bonafide:
  - source video lips + original source audio

generated_same_transcript:
  - source video lips + generated TTS audio from that source transcript

mismatched_negative:
  - source video lips + different source audio from the same split
```

For the first implementation, `matched_bonafide` is the positive class and generated/mismatched pairs are the suspicious class.

## Phase 1: Manifests

Build `src/data/build_speech_manifests.py`.

Inputs:

```text
data/manifest.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
data/transcripts/google_stt_v2/
data/tts_audio/elevenlabs/
data/tts_audio/google_tts/
```

Tasks:

1. Load the frozen native video splits.
2. Add bonafide audio rows for all original MAVOS-DD videos.
3. Add spoof audio rows for generated TTS files.
4. Map every generated TTS file back to its `source_video_id`.
5. Assign every generated row to the same split as its source video.
6. Write provider and voice metadata.
7. Validate that no `source_video_id` appears in multiple splits.

Acceptance checks:

```text
all generated rows have source_video_id
all generated rows inherit source split
all original echomimic/memo audio rows are audio_label=bonafide
all generated provider rows are audio_label=spoof
no generated data is committed to git
```

## Phase 2: Generated Audio Features

Extend or add feature extraction for generated audio files.

Recommended script:

```text
src/features/extract_generated_audio.py
```

Requirements:

- Use the same frozen Wav2Vec2 model and 4-second, 16 kHz processing window as `src/features/extract_audio.py`.
- Write `.npy` features only.
- Skip existing features.
- Log failures explicitly.

Recommended output:

```text
data/features/audio_generated/{sample_id}.npy
```

Keep original MAVOS-DD audio features in:

```text
data/features/audio/{video_id}.npy
```

## Phase 3: Training Views

Implement dataset views rather than new architectures.

### Task A: Audio Anti-Spoof

Question:

```text
Is this speech generated?
```

Input:

```text
audio_feature_path
```

Label:

```text
audio_label: bonafide=0, spoof=1
```

Training pool:

- Bonafide: original real + original echomimic + original memo audio.
- Spoof: ElevenLabs and Google TTS audio.

Model:

```text
LateFusionClassifier(modality="audio")
```

### Task B: Visual Lip Cue

Question:

```text
Do lip-motion features provide evidence that the speech pairing is suspicious?
```

Input:

```text
lip_feature_path
```

Label:

```text
pair_label: matched_bonafide=0, generated_same_transcript/mismatched_negative=1
```

Model:

```text
LateFusionClassifier(modality="visual")
```

Important caveat:

The visual-only branch cannot hear the generated audio. It learns whether lip-motion patterns and pairing setup provide a weak cue. The main value is in fusion.

### Task C: Audio-Visual Fusion

Question:

```text
Does combining generated-speech audio features with lip-motion cues improve spoof detection?
```

Input:

```text
audio_feature_path + lip_feature_path
```

Label:

```text
audio_label or pair_label, depending on the experiment
```

First fusion experiment:

```text
source lips + original audio -> 0
source lips + generated TTS audio from same transcript -> 1
```

Optional hard negatives:

```text
source lips + different original audio from same split -> 1
```

Model:

```text
LateFusionClassifier(modality="fusion")
```

## Phase 4: Metrics

Evaluate once on the locked test split.

Report:

```text
AUC
F1
EER
accuracy
confusion matrix
```

Required breakdowns:

```text
bonafide original real
bonafide original echomimic
bonafide original memo
spoof elevenlabs
spoof google_tts
```

Money plots:

1. Audio spoof score by provider/source.
2. Fusion score for original pairs vs generated-speech pairs.
3. Visual-only score distribution, with an explicit caveat if weak.

## Expected Findings

Expected main result:

```text
Audio branch separates original speech from commercial TTS.
Visual cues alone are weaker.
Fusion may improve robustness when generated speech is paired with real/fake video lips.
```

Expected important nuance:

```text
EchoMimic/MEMO are video-generation attacks with bonafide audio in this setup.
The audio anti-spoof head should not call their original audio spoof.
```

This is a feature of the label design, not a bug.

## Out Of Scope

Do not build:

- Full SyncNet-style contrastive synchronization training.
- Dense phoneme-to-viseme alignment.
- Frame-level lag estimation as a primary deliverable.
- End-to-end video/audio training on raw media.

Allowed lightweight visual cue:

- Pair original lips with generated or mismatched audio features.
- Use existing lip landmark features.
- Train the existing visual/fusion heads.

## Implementation Order

1. Add `src/data/build_speech_manifests.py`.
2. Add tests for split inheritance and label rules.
3. Add generated-audio feature extraction.
4. Add dataset views for `audio_spoof`, `visual_speech`, and `fusion_speech`.
5. Implement `src/train.py --task audio_spoof`.
6. Add evaluation for AUC/F1/EER.
7. Train audio anti-spoof baseline.
8. Train visual cue baseline.
9. Train fusion baseline.
10. Write final report tables and plots.

## Hard Constraints

- Keep Wav2Vec2 frozen.
- Train only from cached `.npy`/`.npz` features.
- Keep seed 42.
- Existing video splits frozen **per cap regime** (see Revision 1 below).
- Generated audio inherits source-video split.
- Keep trainable parameters under 2M.
- Do not commit generated audio, transcripts, features, checkpoints, or raw data.

## Revision 1 — Dataset Expansion (Phase 6+ scope)

Phases 1–5 (audio anti-spoof + visual + fusion baselines, PRs #7/#8/#9) were
trained on the original 1,000-video cap: `real=500, echomimic=250, memo=250`.
Those splits are frozen for that cap and remain valid for retrospective
comparison.

Starting from Phase 6, the cap is expanded to **~4,149 videos** across
**five** MAVOS-DD source folders:

```
real:         2500
echomimic:     600
memo:          400
liveportrait:  314   # new fake source
sonic:         335   # new fake source
```

Rationale: the Phase 5 visual baseline saturates at chance (val ROC-AUC
≈ 0.57) because every video contributes a paired bonafide/spoof row that
shares the same lip sequence — the visual head sees identical input under
two labels and cannot discriminate. Adding `liveportrait` and `sonic`
diversifies the fake-video sources and gives the visual/fusion heads a
broader generator distribution to learn from. The larger bonafide pool
(2,500 vs 500) also stress-tests recording-condition shortcuts.

Implications:

- `src/common.CAPS` and `LABEL_MAP` are updated accordingly. `liveportrait`
  and `sonic` map to `1` (fake) under the existing real-vs-fake binary
  scheme.
- `src/data/download_subset.py` is idempotent: re-running it after a CAPS
  change rebuilds per-class counts from the manifest and only fetches the
  missing delta.
- Splits will be **re-frozen at the new cap** before Phase 6 training.
  Phase 1–5 checkpoints stay tied to the old cap's splits; the Phase 6
  consolidated test pass uses the new cap's locked test split.
- The 4.0 s window, 5 fps lip sampling, codec-matched audio re-extraction,
  and voice-disjoint split remain unchanged.

This revision was made explicit per the project rule that the cap may only
be raised through a documented roadmap change.
