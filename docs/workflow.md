# workflow.md — Master Development Roadmap
**Project:** Audiovisual Deepfake Detection (Audio + Lip-Movement Late Fusion)
**Timeline:** 14 days | **Dataset:** MAVOS-DD subset (1,000 videos) | **Paradigm:** Pre-extracted features → Late Fusion DL classifier

---

## Global Invariants (apply to all phases)

| Constraint | Value |
|---|---|
| Total videos | **1,000** (500 `english/real`, 250 `english/echomimic`, 250 `english/memo`) |
| Training input | **Serialized `.npy` features only** — raw video never enters the training loop |
| Audio encoder | `facebook/wav2vec2-base-960h`, **frozen**, embeddings extracted **once** |
| Visual features | MediaPipe Face Mesh lip-region landmarks (sequence arrays) |
| Loss | Binary Cross-Entropy (`BCEWithLogitsLoss`) |
| Split | 70/15/15 stratified train/val/test, fixed seed `42`, split frozen on Day 3 |
| Label map | `real → 0`, `echomimic → 1`, `memo → 1` (binary: Real vs Fake) |

---

## Phase 1 — Environment Setup & Filtered Data Ingestion (Days 1–3)

### Day 1 — Environment & Repository Bootstrap
- [ ] Create repo skeleton (see `claude.md` architecture blueprint).
- [ ] Create isolated environment (`python==3.10`):
  ```
  torch torchaudio transformers datasets librosa opencv-python mediapipe
  numpy pandas scikit-learn matplotlib tqdm soundfile ffmpeg-python
  ```
- [ ] Verify `ffmpeg` binary on PATH (audio demux dependency).
- [ ] Smoke test: load Wav2Vec2 processor + model, run 1s of random noise through it, assert output shape `(1, T, 768)`.
- [ ] Smoke test: MediaPipe `FaceMesh` on a webcam frame / sample image, assert 478 landmarks returned.
- [ ] `git init`, commit `requirements.txt` + skeleton. Pin all versions.

### Day 2 — Streaming Ingestion of the 1k Subset
- [ ] Implement `src/data/download_subset.py` (see `codex.md` §1) using `datasets.load_dataset(..., streaming=True)`.
- [ ] Filter predicate: `language == "english"` AND `generation_method ∈ {real, echomimic, memo}` (fall back to path-prefix filtering `english/real/*`, `english/echomimic/*`, `english/memo/*` if metadata fields differ).
- [ ] Hard caps enforced by counter dict: `{"real": 500, "echomimic": 250, "memo": 250}`; **break the stream** the moment all counters saturate.
- [ ] Persist each video to `data/raw/{label_folder}/{video_id}.mp4`.
- [ ] Write `data/manifest.csv`: `video_id, relative_path, source_folder, binary_label, duration_s, fps, n_frames`.

### Day 3 — Integrity Audit & Frozen Split
- [ ] Probe every file with `cv2.VideoCapture` + audio stream check; quarantine corrupt/audio-less files to `data/quarantine/`, top up counts from the stream if any class falls short.
- [ ] Distribution sanity report: duration histogram, fps modes, resolution modes → `reports/data_audit.md`.
- [ ] Generate stratified `splits/{train,val,test}.csv` (70/15/15, stratify on `source_folder` so both fake generators appear in every split). Commit splits — **immutable from here on**.

**Phase 1 exit criteria:** 1,000 verified videos on disk, manifest + frozen splits committed, both extractors smoke-tested.

---

## Phase 2 — Feature Extraction Pipeline (Days 4–6)

> Goal: convert all 1,000 videos into stationary, serialized NumPy artifacts. After Day 6, raw video is never touched again.

### Day 4 — Audio Branch: Wav2Vec2 Embedding Extraction
- [ ] `src/features/extract_audio.py`:
  1. Demux audio via ffmpeg → mono 16 kHz waveform (`librosa.load(path, sr=16000, mono=True)`).
  2. Clip/pad to fixed window (recommend **4.0 s** = 64,000 samples; center-crop long, zero-pad short).
  3. Forward pass through **frozen** Wav2Vec2 (`torch.no_grad()`, `model.eval()`), take `last_hidden_state` → `(T≈199, 768)`.
  4. Save `float16` array → `data/features/audio/{video_id}.npy`.
- [ ] Batch on GPU if available; checkpoint progress via a done-list so the job is resumable.
- [ ] Optional storage lever: mean+std pooling to `(1536,)` per clip if disk pressure demands it (decide once, apply uniformly).

### Day 5 — Visual Branch: Lip-Landmark Sequence Extraction
- [ ] `src/features/extract_lips.py`:
  1. Sample frames at fixed rate (**5 fps** over the same 4 s window → 20 frames).
  2. MediaPipe Face Mesh per frame → select the ~40 lip-contour indices (inner + outer ring).
  3. Normalize: translate to lip centroid, scale by inter-landmark bounding-box diagonal (removes head-position/scale variance); also store per-frame lip bounding box `(x, y, w, h)` normalized by frame size.
  4. Stack → `(20, 40, 2)` landmarks + `(20, 4)` bbox; missing-face frames → zero vector + validity mask `(20,)`.
  5. Save dict-style `.npz` or flat `.npy` → `data/features/lips/{video_id}.npy`.
- [ ] Log face-detection failure rate per class → `reports/extraction_log.csv` (deepfake artifacts sometimes break detectors; this is itself a signal worth noting in the report).

### Day 6 — Feature Store Validation
- [ ] Assert 1:1 coverage: every manifest row has both `.npy` artifacts; drop (and document) irrecoverable rows symmetrically from both modalities.
- [ ] Shape/NaN/Inf audit across the full store.
- [ ] Compute dataset-level normalization statistics on **train split only** (visual branch); persist to `data/features/stats.json`.
- [ ] Write `src/data/dataset.py`: `MultimodalFeatureDataset(torch.utils.data.Dataset)` returning `(audio_feat, lip_feat, mask, label)` straight from `.npy` (memory-mapped reads).

**Phase 2 exit criteria:** complete dual-modality `.npy` feature store, validated, with a working PyTorch `Dataset`/`DataLoader` round-trip benchmark (< 5 ms/sample).

---

## Phase 3 — Late Fusion Network Implementation (Days 7–10)

### Day 7 — Architecture Definition
- [ ] `src/models/late_fusion.py` (skeleton in `codex.md` §3):
  - **Audio head:** temporal mean-pool over Wav2Vec2 frames `(T,768) → (768)` → MLP `768→256→128` (ReLU, Dropout 0.3, BatchNorm).
  - **Visual head:** flatten landmarks per frame `(20, 80+4)` → **BiGRU** (hidden 128, 2 layers) → masked last-state/attention pooling → `(256) → 128`.
  - **Fusion:** concat `(128+128=256)` → MLP `256→64→1` logit.
- [ ] Parameter budget check: target **< 2M trainable params** (Wav2Vec2 excluded — it is offline). Print `sum(p.numel())` in CI.
- [ ] Unimodal baseline switches (`--modality audio|visual|fusion`) built into the same class for ablation.

### Day 8 — Training Harness
- [ ] `src/train.py`: AdamW (lr `1e-4`, wd `1e-2`), `BCEWithLogitsLoss`, batch 32, max 50 epochs, early stopping on val ROC-AUC (patience 7), `ReduceLROnPlateau`.
- [ ] Deterministic seeding, per-epoch CSV logging (`runs/{exp_name}/metrics.csv`), best-checkpoint saving to `models/checkpoints/`.
- [ ] Overfit-one-batch sanity test (loss → ~0) before any full run.

### Day 9 — Baselines & Ablations
- [ ] Train: audio-only, visual-only, late-fusion. Three runs each minimum config.
- [ ] Record val ROC-AUC / accuracy / F1 per run into `claude.md` status table.

### Day 10 — Tuning Sweep (bounded)
- [ ] Grid only over: lr `{1e-3, 1e-4}`, dropout `{0.3, 0.5}`, GRU hidden `{64, 128}`, fusion width `{64, 128}` — **max 12 runs**, fast loops are the whole point of the 1k cap.
- [ ] Freeze the winning config; tag checkpoint `models/checkpoints/best_fusion.pt`.

**Phase 3 exit criteria:** trained fusion model beating both unimodal baselines on val ROC-AUC, checkpoint + config committed.

---

## Phase 4 — Evaluation, Validation Tracking & Reporting (Days 11–14)

### Day 11 — Test-Set Evaluation (single shot)
- [ ] `src/evaluate.py`: run **once** on the held-out test split with `best_fusion.pt`.
- [ ] Metrics: ROC-AUC, accuracy, precision, recall, F1, Equal Error Rate (EER — standard in anti-spoofing literature).
- [ ] Artifacts: ROC curve PNG, confusion matrix PNG (threshold from val-set Youden's J), per-generator breakdown (EchoMimic vs MEMO recall — exposes generator-specific blind spots).

### Day 12 — Error Analysis
- [ ] Dump top-20 false negatives / false positives with ids → manual inspection of corresponding raw videos.
- [ ] Calibration check: reliability diagram + Brier score.
- [ ] Ablation table consolidation (audio vs visual vs fusion) → quantify the fusion gain.

### Day 13 — Inference Path & Packaging
- [ ] `src/predict.py`: single-video CLI — runs the *same* extraction functions end-to-end (video → features → logit → probability). Validates that the offline pipeline is reproducible online.
- [ ] Export `best_fusion.pt` + `stats.json` + config YAML as a self-contained `release/` bundle; optional TorchScript trace.

### Day 14 — Final Report & Repo Freeze
- [ ] `reports/final_report.md`: dataset card, pipeline diagram, architecture spec, all metrics/figures, limitations (1k subset, 2 generators, English-only), future work (cross-lingual eval on remaining MAVOS-DD splits, attention-based fusion).
- [ ] README finalization, `claude.md` closed out with final status, tag `v1.0`.

**Phase 4 exit criteria:** reproducible single-command eval, final report committed, repo tagged.

---

## Risk Register

| Risk | Phase | Mitigation |
|---|---|---|
| MAVOS-DD schema differs from assumed fields | 1 | Inspect first streamed record before writing the filter; path-prefix fallback |
| Streaming bandwidth slow | 1 | Counters terminate early; ingestion is resumable via manifest diff |
| MediaPipe misses faces on heavy fakes | 2 | Validity masks + zero-fill; log rate per class; never drop silently |
| Disk pressure from frame-level audio tensors | 2 | `float16` + optional mean/std pooling fallback |
| Overfitting on 1k samples | 3 | Dropout, early stopping, <2M params, weight decay, frozen encoder |
| Test-set leakage via repeated evaluation | 4 | Single-shot test protocol; all tuning on val only |
