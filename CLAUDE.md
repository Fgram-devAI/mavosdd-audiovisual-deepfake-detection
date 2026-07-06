# CLAUDE.md — Project Memory

> Claude's living memory for this repo. Read this first each session. This is a **public, tracked** doc — distinct from the local, git-ignored assistant/operator context described below. Keep it concise and current.

## Project overview

**Audiovisual (multimodal) deepfake detection** on a capped MAVOS-DD video subset. Keep three labels separate: (1) **audio anti-spoof** asks whether speech audio is TTS/generated, (2) **visual fake-video** asks whether the video source is generated (`echomimic`/`memo`/`liveportrait`/`sonic` vs `real`), and (3) **audio-visual sync** asks whether mouth motion matches the audio. The mature pipeline is feature-first: audio SSL embeddings, MediaPipe lip-landmark features, mel-spectrograms, and/or sampled frame features are extracted once, then lightweight heads learn over cached artifacts. **Raw video should only enter extraction/notebook baselines, not the core feature-store training loop.**

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
report/{audio_channel_normalization_audit.md, val_eval/, visual_frame_baseline/}
data/                          # [gitignored] raw/, quarantine/, features/audio|lips/, manifest.csv, splits/
models/checkpoints/            # [gitignored]
runs/                          # [gitignored]
release/                       # [gitignored]
```

Source roles: `data/download_subset.py` (streaming MAVOS-DD filter + hard cap); `data/make_splits.py` (stratified frozen split); `data/build_speech_manifests.py` (writes the three derived manifests with 17-column SCHEMA + validate CLI); `data/feature_store.py` (**canonical** split-safe PyTorch datasets, `validate_feature_store` CLI, train-only normalization, strict collate); `data/dataset.py` (legacy `MultimodalFeatureDataset`, kept for historical reasons — new training code should use `feature_store.py`); `features/audio_io.py` (deterministic 4 s / 16 kHz windowing); `features/audio_backends.py` (Wav2Vec2 / WavLM / HuBERT registry, all 768-dim, frozen); `features/extract_audio_embeddings.py` (multi-backend `.npy` extractor CLI); `features/extract_audio.py` (legacy Wav2Vec2-only extractor); `features/extract_lips.py` (MediaPipe lip landmarks → `.npz` with `feats` + `mask`); `models/late_fusion.py`, `train.py`, `evaluate.py`, `predict.py`.

## Docs & context model

**Public, tracked (source of truth):** `README.md`, `docs/workflow.md`, `report/`, `src/`, `config/default.yaml`, and this `CLAUDE.md`.

**Local, git-ignored (private scratch/sidecar — never cite as public truth):** `.claude/`, `docs/prompts/` (older `claude.md` state tracker + `codex.md` playbook), and the entire `docs/superpowers/` folder — both the `CLAUDE.md` brainstorming sidecar and the implementation specs (`spec-*.md`, e.g. `spec-fetching-data.md`) are local working material, not public deliverables. `.gitignore` ignores `.claude/`, `docs/prompts/`, `docs/superpowers/`, and root-level lowercase `/claude.md`, `/codex.md`, `/workflow.md`; the root `CLAUDE.md` is explicitly re-included so this memory file stays tracked even on case-insensitive filesystems.

## Hard constraints

- **Dataset cap** (Phase 6+): ~4,149 videos — 2500 real / 600 EchoMimic / 400 MEMO / 314 LivePortrait / 335 Sonic. Phase 1–5 baselines stay tied to the original 1,000-video cap (500 real / 250 EchoMimic / 250 MEMO); see `docs/roadmap-audio-visual-speech-detection.md` Revision 1.
- **English-only** subset.
- **Core feature-store training uses cached features only** — raw video may enter extraction/notebook baselines, but not `src/train.py` feature-store models.
- **Wav2Vec2 frozen**: extracted once, never fine-tuned.
- **Splits frozen once created**: 70/15/15 stratified on `source_folder`, seed 42.
- **Test split evaluated exactly once**, after model selection.
- **Normalization statistics computed on the train split only.**
- **Under 2M trainable params** (Wav2Vec2 excluded).
- **Seed 42 everywhere.**
- **Binary video label map**: real→0, echomimic/memo/liveportrait/sonic→1.
- **Deep-learning-only final classifier** — classical probes are diagnostics only.
- **Log MediaPipe face-detection misses via masks; never silently drop.**

## Current status

Phases 1–5 complete. MAVOS-DD subset ingested, splits frozen (seed 42, 70/15/15 stratified on `source_folder`), derived manifests built under `data/derived/`, Wav2Vec2/WavLM/HuBERT audio features extracted, MediaPipe lip features extracted, and feature-store dataset loader (`src/data/feature_store.py`) green. The project has moved beyond the original 1,000-video cap into the expanded five-source setup (~4,149 matched-bonafide videos across real/echomimic/memo/liveportrait/sonic), plus TTS/STS/generated-audio rows for the audio-spoof task.

- **`feat/final-fusion-generated-video` shipped** (2026-07-06): logistic-fusion val ROC-AUC 0.9307 (EER 0.1338, F1 0.8376), MLP val ROC-AUC 0.9175, EER-selected threshold 0.3702. Score cache and provenance under `data/derived/final_fusion_scores_{train,val}.csv` + `final_fusion_score_provenance.json` (2904 train / 622 val rows, gitignored). Comparison table at `report/val_eval/final_fusion_comparison.md`. Higgsfield external batch at `report/val_eval/generated_video_batch_scores.csv` — 71/71 scored, 76.06% hit rate at the imported MAVOS-DD val threshold (positive-only stress test — detection rate is a hit rate, not accuracy). Final label semantics: `real → 0`, `echomimic/memo/liveportrait/sonic → 1`; native-audio generated videos stay `1` even when sync-consistency says the audio and lips match. Sync-only ROC-AUC 0.2417 (inverse-informative) validates that sync consistency ≠ deepfake detection.

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
  - Full per-checkpoint val metric battery (roc_auc, eer, eer_threshold, f1, precision, recall, confusion, per_provider_recall) is in `report/val_eval/all_checkpoints_val_metrics.json`.

- **Audio channel normalization shipped**: codec-matched audio was normalized with silence trim, 7 kHz lowpass, EBU R128 loudness normalization, and peak safety. It reduced but did not remove shortcuts: acoustic LR ROC-AUC **0.9914 → 0.9713**, RF ROC-AUC **0.9970 → 0.9889**, normalized mel-CNN ROC-AUC **0.99994**. Full report: `report/audio_channel_normalization_audit.md`.

- **`feat/lip-sync-consistency` shipped** (2026-07-02): SyncNet-style consistency head (WavLM + BiGRU) trained on a deterministic pair manifest (train/val only; test locked out of the manifest itself in this branch). Val ROC-AUC **0.8409**, EER **0.2527**, F1 **0.826**, positive-sync accuracy **0.748**. Per-negative-type recall reveals the model is mostly an audio-spoof detector: `generated_same_transcript=1.00`, `mismatched_generated=1.00`, `mismatched_original=0.48`. Full metrics: `report/val_eval/lipsync_wavlm_val.txt`.

- **Visual frame baseline notebook landed**: EfficientNetB0 over sampled video frames for real vs generated-video (`real=0`, `echomimic/memo/liveportrait/sonic=1`) reaches val ROC-AUC **0.9853**, EER **0.0671**, F1 **0.9167**. Per-source: real specificity **0.9307**, echomimic/memo/sonic recall **1.0**, liveportrait recall **0.6596**. Treat as channel-confounded because sampled source folders have disjoint resolution/FPS/codec signatures. Summary: `report/visual_frame_baseline/visual_frame_baseline_efficientnet_b0_val.json`.

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
- **Final fusion feature set**: `audio_fake_score`, `video_av_fake_score`, `sync_inconsistent_score`. `visual_fake_score` and `source_folder` are excluded from the trainable model — the first because it lives only in a notebook, the second because it would reward dataset-source recognition instead of general deepfake detection.

## Key config facts (`config/default.yaml`)

Dataset `unibuc-cs/MAVOS-DD`, split `train`, language english; caps real:500 / echomimic:250 / memo:250. Audio: sr 16000, 4.0s, `facebook/wav2vec2-base-960h`. Lips: 5 fps, 20 frames. Training: batch 32, max_epochs 50, lr 1e-4, weight_decay 1e-2, early_stop_patience 7, dropout 0.3, max_trainable_params 2,000,000. Seed 42.

## TODO / next actions

- All planned final-fusion work completed. Next: `feat/final-report` for the write-up branch.

## Handoff notes

- **Done:** ingestion, frozen splits, derived manifests, three codec-matched audio embedding stores, MediaPipe lip features, feature-store dataset loader + validate CLI (PR #6); mel-CNN baseline (PR #7); audio anti-spoof baselines wav2vec2/wavlm/hubert (PR #8); visual + fusion baselines on voice-disjoint splits (PR #9). Lip-sync consistency branch: pair-manifest builder, dataset, model, training + val-only evaluator, WavLM baseline metrics (`feat/lip-sync-consistency`, 379/379 tests green).
- **Next:** `feat/pretrained-syncnet`, then `feat/generated-video-batch-fusion`.
- **To resume:** read this file and `README.md`. Always pre-flight feature-store training with `python -m src.data.feature_store --validate --view <view> [--backend <backend>]`. Honor the hard constraints — especially frozen splits, seed 42, single test evaluation, and feature-only core training input. Use `feature_store.py` (not legacy `dataset.py`) for dataset code. Val metric snapshots live at `report/val_eval/all_checkpoints_val_metrics.json`.
- Final fusion: run `python -m src.data.build_final_fusion_scores` → `python -m src.train_final_fusion` → `python -m src.evaluate_final_fusion` in that order. External generated videos: `python -m scripts.score_generated_video_batch --input-dir <folder> --batch-name <label>`. Fusion checkpoints live at `models/checkpoints/best_final_fusion_{logreg,mlp}.pt`; both stay gitignored.
