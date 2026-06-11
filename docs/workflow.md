# workflow.md — Phase-Based Implementation Guide
**Project:** Audiovisual Deepfake Detection (Wav2Vec2 audio ⊕ MediaPipe lip-motion late fusion)
**Dataset:** MAVOS-DD subset (1,000 videos) · **Paradigm:** pre-extracted features → late-fusion DL classifier

This document describes *what each phase must accomplish and the order of dependencies*, not a
calendar. Phases are gated by **exit criteria**, not by elapsed days — advance only when the prior
phase's exit criteria are met. Sub-bullets are ordered by dependency, so a phase can be executed in
one sitting or spread out; sequence matters, dates do not.

---

## Global Invariants (apply to every phase)

| Constraint | Value |
|---|---|
| Total videos | **1,000** — 500 `english/real`, 250 `english/echomimic`, 250 `english/memo` |
| Language | **English only** |
| Training input | **Serialized `.npy` features only** — raw video never enters the training loop |
| Audio encoder | `facebook/wav2vec2-base-960h`, **frozen**, embeddings extracted **once** |
| Visual features | MediaPipe Face Mesh lip-region landmarks (sequence arrays) |
| Loss | Binary Cross-Entropy (`BCEWithLogitsLoss`) |
| Label map | `real → 0`, `echomimic → 1`, `memo → 1` (binary: Real vs Fake) |
| Split | 70/15/15 stratified train/val/test, seed `42`, **frozen once created** |
| Test protocol | Held-out test split is **evaluated exactly once**, after model selection |
| Normalization | Statistics computed on the **train split only** |
| Parameter budget | **< 2M trainable params** (Wav2Vec2 excluded — it is offline) |
| Reproducibility | Seed `42` everywhere; splits + configs committed; deterministic flags on |

---

## Phase 1 — Environment, Smoke Tests & MAVOS-DD Subset Ingestion

**Goal:** a reproducible environment and a capped, on-disk MAVOS-DD subset.

- Create an isolated Python 3.10 environment; `pip install -r requirements.txt` (pinned versions).
- Verify the `ffmpeg` binary is on `PATH` (audio demux dependency).
- Dependency smoke tests, run before any ingestion:
  - Wav2Vec2: load processor + model, push 1 s of random noise through it, assert output shape `(1, T, 768)`.
  - MediaPipe `FaceMesh`: run on a sample image, assert 478 landmarks returned.
- Ingest the capped subset via `src/data/download_subset.py`:
  - Stream `unibuc-cs/MAVOS-DD` with `datasets.load_dataset(..., streaming=True)`.
  - Filter predicate: `language == "english"` AND `generation_method ∈ {real, echomimic, memo}`,
    with a path-prefix fallback (`english/real/*`, `english/echomimic/*`, `english/memo/*`) when
    metadata fields differ.
  - Enforce hard caps `{"real": 500, "echomimic": 250, "memo": 250}` via a counter dict; **break the
    stream** the moment all counters saturate.
  - Persist each video to `data/raw/{source_folder}/{video_id}.mp4`.
  - Append manifest rows as files land (resumable via manifest diff).

> The detailed, implementation-ready contract for this phase lives in
> `docs/superpowers/spec-fetching-data.md` — read it before writing ingestion code.

**Exit criteria:** environment installed, `ffmpeg` confirmed, both extractor smoke tests pass, and a
capped subset (up to 500/250/250) is on disk with a manifest written.

---

## Phase 2 — Dataset Audit, Manifest Validation & Frozen Splits

**Goal:** a verified manifest of exactly 1,000 usable videos and an immutable split.

- Integrity audit: probe every file with `cv2.VideoCapture` + an audio-stream check; quarantine
  corrupt or audio-less files to `data/quarantine/` and top up class counts from the stream if any
  class falls short of its cap.
- Manifest validation: assert 1,000 rows, correct per-class counts (500/250/250), unique
  `video_id`s, valid `binary_label` per the label map, and non-degenerate `duration_s`/`fps`.
- Distribution sanity report → `reports/data_audit.md`: duration histogram, fps modes, resolution
  modes, per-class counts.
- Generate stratified `data/splits/{train,val,test}.csv` via `src/data/make_splits.py`
  (70/15/15, seed `42`, stratify on `source_folder` so both fake generators appear in every split).
- **Freeze the split:** commit the split files; they are immutable from here on.

**Exit criteria:** 1,000 verified videos, validated manifest, `reports/data_audit.md` written, and
frozen `train/val/test` split CSVs committed.

---

## Phase 3 — Audio & Visual Feature Extraction

**Goal:** convert all 1,000 videos into stationary, serialized NumPy artifacts. After this phase,
raw video is never touched again by the training path.

- Audio branch — `src/features/extract_audio.py`:
  - Demux to mono 16 kHz (`librosa.load(path, sr=16000, mono=True)`).
  - Clip/pad to a fixed **4.0 s** window (64,000 samples; center-crop long, zero-pad short).
  - Forward pass through **frozen** Wav2Vec2 (`model.eval()`, `torch.no_grad()`), take
    `last_hidden_state` → `(T≈199, 768)`.
  - Save `float16` → `data/features/audio/{video_id}.npy`.
  - Resumable via a done-list checkpoint; optional mean+std pooling to `(1536,)` if disk pressure
    demands it (decide once, apply uniformly).
- Visual branch — `src/features/extract_lips.py`:
  - Sample frames at **5 fps** over the same 4 s window (20 frames).
  - MediaPipe Face Mesh per frame → select the ~40 lip-contour landmark indices.
  - Normalize: translate to lip centroid, scale by lip-box diagonal; also store per-frame normalized
    lip bbox `(x, y, w, h)`.
  - Stack → landmarks + bbox per frame; missing-face frames → zero vector + a per-frame validity
    `mask (20,)`. Never silently drop a frame.
  - Save → `data/features/lips/{video_id}.npy`.
  - Log face-detection failure rate per class → `reports/extraction_log.csv` (heavy fakes can break
    detectors; the rate is itself a signal worth reporting).

**Exit criteria:** every retained manifest row has both an audio and a lip `.npy` artifact;
per-class detection-failure rates logged.

---

## Phase 4 — Feature-Store Validation & Dataset Loader

**Goal:** a trustworthy feature store and a fast `.npy`-only loader.

- Coverage: assert 1:1 dual-modality coverage — every manifest row has both `.npy` artifacts; drop
  (and document) irrecoverable rows **symmetrically** from both modalities.
- Quality: shape / NaN / Inf audit across the full store.
- Normalization: compute dataset-level statistics on the **train split only** (visual branch);
  persist to `data/features/stats.json`.
- Loader: `src/data/dataset.py` — `MultimodalFeatureDataset(torch.utils.data.Dataset)` returning
  `(audio_feat, lip_feat, mask, label)` straight from `.npy` (memory-mapped reads). It must contain
  **no video-decode imports** — features only.

**Exit criteria:** validated dual-modality store, `stats.json` (train-only) written, and a working
`Dataset`/`DataLoader` round-trip benchmark (< 5 ms/sample).

---

## Phase 5 — Late-Fusion Model, Training Harness & Unimodal Baselines

**Goal:** a trained fusion model that beats both unimodal baselines on validation, selected without
ever touching the test split.

- Architecture — `src/models/late_fusion.py`:
  - **Audio head:** temporal mean-pool Wav2Vec2 frames `(T,768) → (768)` → MLP `768→256→128`
    (ReLU, Dropout, BatchNorm).
  - **Visual head:** per-frame landmark+bbox features → **BiGRU** (hidden 128) → masked pooling →
    `128`.
  - **Fusion:** concat `(128+128=256)` → MLP `256→64→1` logit.
  - A `--modality audio|visual|fusion` switch in the same class enables unimodal ablation.
  - **Parameter budget check:** assert `sum(p.numel()) < 2,000,000` trainable params.
- Training harness — `src/train.py`: AdamW (lr `1e-4`, wd `1e-2`), `BCEWithLogitsLoss`, batch 32,
  max 50 epochs, early stopping on **val ROC-AUC** (patience 7), `ReduceLROnPlateau`.
  - Deterministic seeding; per-epoch CSV logging (`runs/{exp_name}/metrics.csv`); best-checkpoint
    saving to `models/checkpoints/`.
  - Overfit-one-batch sanity test (loss → ~0) before any full run.
- Baselines & selection: train audio-only, visual-only, and late-fusion. **All model selection uses
  validation metrics only.** A bounded tuning sweep (keep it small — fast loops are the point of the
  1k cap) over a few hyperparameters; freeze the winning config and tag
  `models/checkpoints/best_fusion.pt`.

**Exit criteria:** fusion model beats both unimodal baselines on val ROC-AUC; best checkpoint +
config committed. **The test split has not been touched.**

---

## Phase 6 — Evaluation, Error Analysis, Prediction CLI & Report Packaging

**Goal:** a single, honest test-set readout, error analysis, a reproducible predictor, and a final
report.

- Test-set evaluation — `src/evaluate.py`, run **once** on the held-out test split with
  `best_fusion.pt`:
  - Metrics: ROC-AUC, accuracy, precision, recall, F1, Equal Error Rate (EER).
  - Artifacts: ROC curve, confusion matrix (threshold from val-set Youden's J), and a per-generator
    breakdown (EchoMimic vs MEMO recall).
- Error analysis: dump top false negatives / false positives by id for manual inspection;
  calibration check (reliability diagram + Brier score); consolidate the audio-vs-visual-vs-fusion
  ablation table to quantify the fusion gain.
- Prediction CLI — `src/predict.py`: single-video path that runs the *same* extraction functions
  end-to-end (video → features → logit → probability), proving the offline pipeline reproduces
  online.
- Packaging: export `best_fusion.pt` + `stats.json` + config as a self-contained `release/` bundle;
  finalize `reports/final_report.md` (dataset card, architecture, metrics/figures, limitations: 1k
  subset, two generators, English-only; future work: cross-lingual eval, attention fusion).

**Exit criteria:** reproducible single-command eval, validated `predict.py`, `release/` bundle, and
`reports/final_report.md` committed.

---

## Risk Register

| Risk | Phase | Mitigation |
|---|---|---|
| MAVOS-DD schema differs from assumed fields | 1 | Inspect the first streamed record before writing the filter; path-prefix fallback |
| Streaming bandwidth slow | 1 | Counters terminate early; ingestion is resumable via manifest diff |
| Class falls short after quarantine | 2 | Top up from the stream before freezing the split |
| MediaPipe misses faces on heavy fakes | 3 | Validity masks + zero-fill; log rate per class; never drop silently |
| Disk pressure from frame-level audio tensors | 3 | `float16` + optional mean/std pooling fallback |
| Overfitting on 1k samples | 5 | Dropout, early stopping, < 2M params, weight decay, frozen encoder |
| Test-set leakage via repeated evaluation | 6 | Single-shot test protocol; all tuning on val only |
