# CLAUDE.md — Project Memory

> Claude's living memory for this repo. Read this first each session. This is a **public, tracked** doc — distinct from the local, git-ignored assistant/operator context described below. Keep it concise and current.

## Project overview

**Audiovisual (multimodal) deepfake detection** on a capped MAVOS-DD video subset. The approach is **feature-first late fusion**: two modalities are extracted once into serialized `.npy` feature files, then a lightweight classifier learns over them. The two modalities are (1) **frozen Wav2Vec2 audio embeddings** (`facebook/wav2vec2-base-960h`) and (2) **MediaPipe Face Mesh lip-landmark visual motion** features. A late-fusion neural net (audio MLP head + visual BiGRU head → concat → MLP logit) produces the real/fake logit. **Serialized `.npy` feature files are the only training input — raw video never enters the training loop.** This is not speech-only / audio-only detection.

## Repo map

```
CLAUDE.md                      # this file (public project memory)
README.md
requirements.txt
config/default.yaml            # seed, caps, paths, feature + training specs
docs/
  workflow.md                  # phase-based implementation guide (public)
  prompts/                     # [gitignored] LOCAL: claude.md tracker, codex.md playbook
  superpowers/                 # [gitignored] LOCAL material (whole folder)
    CLAUDE.md                  # [gitignored] LOCAL brainstorming sidecar
    spec-fetching-data.md      # [gitignored] LOCAL spec (data ingestion slice)
src/
  common.py
  data/{download_subset.py, make_splits.py, dataset.py (legacy), build_speech_manifests.py, feature_store.py}
  features/{extract_audio.py, extract_lips.py, audio_io.py, audio_backends.py, extract_audio_embeddings.py}
  models/late_fusion.py
  train.py
  evaluate.py
  predict.py
reports/{data_audit.md, final_report.md}
data/                          # [gitignored] raw/, quarantine/, features/audio|lips/, manifest.csv, splits/
models/checkpoints/            # [gitignored]
runs/                          # [gitignored]
release/                       # [gitignored]
```

Source roles: `data/download_subset.py` (streaming MAVOS-DD filter + hard cap); `data/make_splits.py` (stratified frozen split); `data/build_speech_manifests.py` (writes the three derived manifests with 17-column SCHEMA + validate CLI); `data/feature_store.py` (**canonical** split-safe PyTorch datasets, `validate_feature_store` CLI, train-only normalization, strict collate); `data/dataset.py` (legacy `MultimodalFeatureDataset`, kept for historical reasons — new training code should use `feature_store.py`); `features/audio_io.py` (deterministic 4 s / 16 kHz windowing); `features/audio_backends.py` (Wav2Vec2 / WavLM / HuBERT registry, all 768-dim, frozen); `features/extract_audio_embeddings.py` (multi-backend `.npy` extractor CLI); `features/extract_audio.py` (legacy Wav2Vec2-only extractor); `features/extract_lips.py` (MediaPipe lip landmarks → `.npz` with `feats` + `mask`); `models/late_fusion.py`, `train.py`, `evaluate.py`, `predict.py`.

## Docs & context model

**Public, tracked (source of truth):** `README.md`, `docs/workflow.md`, `reports/`, `src/`, `config/default.yaml`, and this `CLAUDE.md`.

**Local, git-ignored (private scratch/sidecar — never cite as public truth):** `.claude/`, `docs/prompts/` (older `claude.md` state tracker + `codex.md` playbook), and the entire `docs/superpowers/` folder — both the `CLAUDE.md` brainstorming sidecar and the implementation specs (`spec-*.md`, e.g. `spec-fetching-data.md`) are local working material, not public deliverables. `.gitignore` ignores `.claude/`, `docs/prompts/`, `docs/superpowers/`, and root-level lowercase `/claude.md`, `/codex.md`, `/workflow.md`; the root `CLAUDE.md` is explicitly re-included so this memory file stays tracked even on case-insensitive filesystems.

## Hard constraints

- **Dataset cap** (Phase 6+): ~4,149 videos — 2500 real / 600 EchoMimic / 400 MEMO / 314 LivePortrait / 335 Sonic. Phase 1–5 baselines stay tied to the original 1,000-video cap (500 real / 250 EchoMimic / 250 MEMO); see `docs/roadmap-audio-visual-speech-detection.md` Revision 1.
- **English-only** subset.
- **Raw video never enters the training loop** — `.npy` features only.
- **Wav2Vec2 frozen**: extracted once, never fine-tuned.
- **Splits frozen once created**: 70/15/15 stratified on `source_folder`, seed 42.
- **Test split evaluated exactly once**, after model selection.
- **Normalization statistics computed on the train split only.**
- **Under 2M trainable params** (Wav2Vec2 excluded).
- **Seed 42 everywhere.**
- **Binary label map**: real→0, echomimic→1, memo→1.
- **Deep-learning-only classifier** — no classical hand-engineered final model.
- **Log MediaPipe face-detection misses via masks; never silently drop.**

## Current status

Phases 1–5 complete. MAVOS-DD subset ingested (1000 native videos at the original cap), splits frozen (seed 42, 70/15/15 stratified on `source_folder`), derived manifests built under `data/derived/`, Wav2Vec2/WavLM/HuBERT audio features extracted (2171 `.npy` per backend, codec-matched), MediaPipe lip features extracted (1000 `.npz`). Feature-store dataset loader (`src/data/feature_store.py`) green; reusable modality-aware training harness (`src/train.py`) + evaluator (`src/evaluate.py`) shipped; audio anti-spoof baselines (PR #8) and visual + fusion baselines on voice-disjoint splits (PR #9) trained. **Entering Phase 6** (consolidated test-split pass + final report) — and starting the roadmap-Revision-1 dataset expansion (~4,149 videos, +liveportrait/+sonic source folders) for Phase 6+ work.

- **PHASE 5 VAL ROC-AUC (codec-matched + voice-disjoint, honest in-distribution):**

  | Modality | wav2vec2 | wavlm | hubert |
  |---|---|---|---|
  | audio | 0.9508 (EER 0.106) | 1.0000 (EER 0.006) | 1.0000 (EER 0.000) |
  | fusion (audio ⊕ lips) | 0.9509 (EER 0.107) | 1.0000 (EER 0.000) | 1.0000 (EER 0.000) |
  | visual (lips only) | — | — | — |

  Visual-only: val ROC-AUC **0.5688**, EER **0.4333**, confusion `tn=0/fp=150/fn=0/tp=217` — i.e. the model collapses to the constant predictor "always spoof" (val has 217 positives vs 150 negatives, so "always spoof" wins on accuracy).

- **TEST AUC:** n/a (locked; Phase 6 will run the consolidated test-split pass once per checkpoint).

- **Honest reads (Phase 5):**
  - **WavLM/HuBERT saturate at val ROC-AUC = 1.0** for both audio-only and fusion. Wav2Vec2 stays at ~0.95 for both. The remaining shortcut is per-TTS-engine spectral fingerprinting (ElevenLabs vs Google TTS).
  - **Visual-only is structurally at chance** because the matched bonafide row and the matched spoof row(s) for any given video share the **same `source_video_id` and therefore the same `.npz` lip features**. A BiGRU asked to discriminate between two rows whose input is byte-identical cannot do better than class-frequency bias.
  - **Fusion = audio-only.** Concat fusion can't carry cross-modal interaction when one stream is at chance; fusion just inherits the audio component's score. The signal that would help is **audio-visual lip-sync** (deferred — would require a roadmap revision for the consistency-head architecture).
  - Full per-checkpoint val metric battery (roc_auc, eer, eer_threshold, f1, precision, recall, confusion, per_provider_recall) is in `reports/val_eval/all_checkpoints_val_metrics.json`.

## Workflow (phase-based — see `docs/workflow.md`)

Work is organized into **phases**, not calendar days. Detail lives in `docs/workflow.md`.

- **Phase 1** — Environment, dependency smoke tests, and MAVOS-DD subset ingestion
- **Phase 2** — Dataset audit, manifest validation, and frozen train/val/test splits
- **Phase 3** — Audio and visual feature extraction
- **Phase 4** — Feature-store validation and dataset loader
- **Phase 5** — Late-fusion model, training harness, and unimodal baselines
- **Phase 6** — Evaluation, error analysis, prediction CLI, and report packaging

## Decision log (append-only)

- **Audio window**: 4.0s fixed @ 16kHz (Wav2Vec2 native SR).
- **Lip sequence**: 5 fps × 4s = 20-frame lip-landmark sequences.
- **Visual head**: BiGRU over LSTM (fewer params at equal hidden size).
- **Fusion**: late fusion via concat over cross-attention (simplicity within param budget; attention is future work).
- **Multi-backend audio**: Wav2Vec2 is the canonical baseline; WavLM and HuBERT are additive ablations. All three are frozen, all output 768-dim. Stores live at `data/features/audio_{backend}/{sample_id}.npy`.
- **Loader contract**: `feature_store.py` is the **only** path raw features enter training. Validate before training: `python -m src.data.feature_store --validate --view {audio|visual|fusion} --backend {…}`. `path_mismatches` is a warning channel; exit code is driven by `missing` + `bad_shape` only.
- **Manifest sample_id convention**: bonafide rows in every derived manifest use the bare `{source_video_id}` (no `pos__` prefix). This keeps a single `.npy` per `(video_id, backend)` pair instead of duplicating bonafide features under a prefixed key.
- **Lip-sync consistency (feat/lip-sync-consistency)**: SyncNet-style consistency head (WavLM SSL + BiGRU lip encoder + concat similarity MLP, 460,801 trainable params) trained on deterministic pair manifest (`data/derived/lipsync_pairs_manifest.csv`, negatives-per-positive=2, seed 42). Positive class = `async_inconsistent_pair` (label=1). Val metrics captured at `report/val_eval/lipsync_wavlm_val.txt`: ROC-AUC 0.8409, EER 0.2527, threshold 0.6726, F1 0.826, positive-sync accuracy 0.748. Per-negative-type recall reveals the honest reading: `generated_same_transcript=1.00`, `mismatched_generated=1.00`, `mismatched_original=0.48` — the model perfectly detects any TTS audio (all three providers ≥ 0.998 recall) but is near-chance on mismatched-original audio. It is therefore an audio-spoof detector with lip data along for the ride, not a true fine-grained audio-visual sync head. Test split is intentionally excluded from the pair manifest in this branch; a future final-eval branch regenerates the manifest with `--splits train val test` and adds an `--allow-test` gate.

## Key config facts (`config/default.yaml`)

Dataset `unibuc-cs/MAVOS-DD`, split `train`, language english; caps real:500 / echomimic:250 / memo:250. Audio: sr 16000, 4.0s, `facebook/wav2vec2-base-960h`. Lips: 5 fps, 20 frames. Training: batch 32, max_epochs 50, lr 1e-4, weight_decay 1e-2, early_stop_patience 7, dropout 0.3, max_trainable_params 2,000,000. Seed 42.

## TODO / next actions

**Phase 6 — close out the Phase-1–5 milestone first (small):**
1. Single consolidated test-split pass per checkpoint: `python -m src.evaluate --checkpoint <ckpt> --split test --allow-test` for mel-CNN (PR #7), audio × 3 (PR #8), visual + fusion × {1 or 3} (PR #9).
2. Assemble the test-split table (per-modality + per-provider recall + confusion).
3. Write `reports/final_report.md` documenting honest reads: WavLM/HuBERT engine fingerprinting, visual-only structural collapse, fusion = audio.

**Then (optional, after Phase 6):**
- `feat/dataset-expansion` — re-ingest at the Revision-1 cap (~4,149 videos), re-freeze splits, re-extract features/embeddings. Phase 1–5 baselines remain valid for the old cap.
- `feat/multi-engine-spoof` — wire up `scripts/synthesize_coqui_xtts_from_transcripts.py` and `scripts/synthesize_openai_tts_from_transcripts.py`; run leave-one-engine-out (LOEO) evaluation across ≥ 3 TTS engines for the engine-fingerprint generalization story.
- `feat/av-consistency-head` — audio-visual lip-sync as a real multimodal signal. Three flavors (lite in-domain contrastive head → frozen pretrained SyncNet → full SyncNet-style training); each requires its own roadmap revision and spec.

## Handoff notes

- **Done:** ingestion, frozen splits, derived manifests, three codec-matched audio embedding stores, MediaPipe lip features, feature-store dataset loader + validate CLI (PR #6); mel-CNN baseline (PR #7); audio anti-spoof baselines wav2vec2/wavlm/hubert (PR #8); visual + fusion baselines on voice-disjoint splits (PR #9). 223/223 tests green on `feat/visual-and-fusion-baseline`.
- **Next:** Phase 6 — single consolidated test-split pass per checkpoint, then `reports/final_report.md`. After that, optional `feat/dataset-expansion`, `feat/multi-engine-spoof`, and/or `feat/av-consistency-head` (see TODO).
- **To resume:** read this file, then `docs/workflow.md` (Phase 6) and `docs/roadmap-audio-visual-speech-detection.md` Revision 1. Always pre-flight training with `python -m src.data.feature_store --validate --view <view> [--backend <backend>]`. Honor the hard constraints — especially the per-cap-regime frozen splits, seed 42, single test evaluation, and feature-only training input. Use `feature_store.py` (not legacy `dataset.py`) for any new dataset code. Val metric snapshots live at `reports/val_eval/all_checkpoints_val_metrics.json`.
