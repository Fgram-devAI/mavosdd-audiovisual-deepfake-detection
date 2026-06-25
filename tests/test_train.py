"""Tests for src/train.py harness and LateFusionClassifier audio-only path."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.models.late_fusion import LateFusionClassifier


# ---------- shared synthetic manifest + .npy fixture helpers ----------

_AUDIO_CSV_FIELDS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path",
    "audio_label", "audio_label_binary",
    "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _row(sample_id: str, *, split: str, provider: str, label: int) -> dict:
    blank = {k: "" for k in _AUDIO_CSV_FIELDS}
    blank.update({
        "sample_id": sample_id,
        "source_video_id": sample_id,
        "split": split,
        "media_type": "audio",
        "source_folder": "real" if label == 0 else "tts",
        "provider": provider,
        "audio_path": f"data/audio_wav/{sample_id}.wav",
        "video_path": f"data/raw/{sample_id}.mp4",
        "audio_label": "bonafide" if label == 0 else "spoof",
        "audio_label_binary": str(label),
        "video_label": "real",
        "video_label_binary": "0",
        "pair_label": "matched_bonafide",
        "pair_label_binary": str(label),
    })
    return blank


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AUDIO_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_npy(dir_path: Path, sample_id: str, t: int = 16, dim: int = 768,
               *, signal: float = 0.0) -> None:
    """Write a .npy whose mean across the time dim equals (signal)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(abs(hash(sample_id)) % (2**32))
    arr = rng.randn(t, dim).astype(np.float32) * 0.1 + signal
    np.save(dir_path / f"{sample_id}.npy", arr)


def _build_train_fixture(
    tmp_path: Path, *, n_train: int = 8, n_val: int = 4, separable: bool = False
) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.csv"
    audio_dir = tmp_path / "feat_codec"
    rows: list[dict] = []
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            label = i % 2
            sid = f"{split}_{i}"
            provider = "elevenlabs" if label == 1 else "real"
            rows.append(_row(sid, split=split, provider=provider, label=label))
            sig = (label * 0.5) if separable else 0.0
            _write_npy(audio_dir, sid, signal=sig)
    _write_manifest(manifest, rows)
    return manifest, audio_dir


def _write_npz(dir_path: Path, source_video_id: str, *, t: int = 20, dim: int = 84,
               signal: float = 0.0) -> None:
    """Write a lip .npz with feats (T,84) (centered at `signal`) and mask (T,) of 1s."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(abs(hash(source_video_id)) % (2**32))
    feats = rng.randn(t, dim).astype(np.float32) * 0.1 + signal
    mask = np.ones(t, dtype=np.float32)
    np.savez(dir_path / f"{source_video_id}.npz", feats=feats, mask=mask)


def _build_visual_fixture(
    tmp_path: Path, *, n_train: int = 8, n_val: int = 4, separable: bool = False
) -> tuple[Path, Path]:
    manifest = tmp_path / "visual.csv"
    lips_dir = tmp_path / "lips"
    rows: list[dict] = []
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            label = i % 2
            sid = f"{split}_{i}"
            rows.append(_row(sid, split=split, provider="original", label=label))
            sig = (label * 0.5) if separable else 0.0
            _write_npz(lips_dir, sid, signal=sig)
    _write_manifest(manifest, rows)
    return manifest, lips_dir


def _build_fusion_fixture(
    tmp_path: Path, *, n_train: int = 8, n_val: int = 4, separable: bool = False
) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "fusion.csv"
    audio_dir = tmp_path / "feat_codec"
    lips_dir = tmp_path / "lips"
    rows: list[dict] = []
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            label = i % 2
            sid = f"{split}_{i}"
            provider = "elevenlabs" if label == 1 else "real"
            rows.append(_row(sid, split=split, provider=provider, label=label))
            sig = (label * 0.5) if separable else 0.0
            _write_npy(audio_dir, sid, signal=sig)
            _write_npz(lips_dir, sid, signal=sig)
    _write_manifest(manifest, rows)
    return manifest, audio_dir, lips_dir


class TestLateFusionAudioOnlyForward:
    def test_audio_forward_accepts_audio_only(self):
        model = LateFusionClassifier("audio")
        logits = model(torch.randn(4, 199, 768))
        assert logits.shape == (4,)

    def test_fusion_forward_still_three_args(self):
        model = LateFusionClassifier("fusion")
        logits = model(
            torch.randn(4, 199, 768),
            torch.randn(4, 20, 84),
            torch.ones(4, 20),
        )
        assert logits.shape == (4,)

    def test_visual_forward_requires_lips_and_mask(self):
        model = LateFusionClassifier("visual")
        with pytest.raises(ValueError, match="visual"):
            model(torch.randn(4, 199, 768))

    def test_fusion_forward_requires_lips_and_mask(self):
        model = LateFusionClassifier("fusion")
        with pytest.raises(ValueError, match="fusion"):
            model(torch.randn(4, 199, 768))

    def test_visual_forward_accepts_none_audio(self):
        model = LateFusionClassifier("visual")
        logits = model(None, torch.randn(4, 20, 84), torch.ones(4, 20))
        assert logits.shape == (4,)

    def test_audio_forward_rejects_none_audio(self):
        model = LateFusionClassifier("audio")
        with pytest.raises(ValueError, match="audio"):
            model(None)

    def test_fusion_forward_rejects_none_audio(self):
        model = LateFusionClassifier("fusion")
        with pytest.raises(ValueError, match="audio"):
            model(None, torch.randn(4, 20, 84), torch.ones(4, 20))


# ---------- Task 5: src/train.py harness ----------

from src import train as train_mod


class TestTrainHarness:
    def test_codec_dirs_constant_points_at_codec_features(self):
        from src import common
        assert train_mod.CODEC_DIRS["wav2vec2"] == common.FEAT_AUDIO_WAV2VEC2_CODEC_DIR
        assert train_mod.CODEC_DIRS["wavlm"] == common.FEAT_AUDIO_WAVLM_CODEC_DIR
        assert train_mod.CODEC_DIRS["hubert"] == common.FEAT_AUDIO_HUBERT_CODEC_DIR

    def test_build_resolves_loader_with_audio_dir_override(self, tmp_path):
        manifest, audio_dir = _build_train_fixture(tmp_path)
        cfg = train_mod.RunConfig(
            modality="audio", backend="wav2vec2",
            manifest=manifest, audio_dir=audio_dir,
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, val_ds, stats = train_mod.build_datasets(cfg)
        assert train_ds.split == "train"
        assert val_ds.split == "val"
        assert train_ds._audio_dir == audio_dir
        assert val_ds._audio_dir == audio_dir
        assert stats.audio_mean is not None and stats.audio_std is not None

    def test_param_budget_asserted(self, tmp_path):
        cfg = train_mod.RunConfig(
            modality="audio", backend="wav2vec2",
            manifest=tmp_path / "m.csv", audio_dir=tmp_path / "a",
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        model = train_mod.build_model(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params < 2_000_000

    def test_one_step_decreases_loss(self, tmp_path):
        manifest, audio_dir = _build_train_fixture(tmp_path, separable=True)
        cfg = train_mod.RunConfig(
            modality="audio", backend="wav2vec2",
            manifest=manifest, audio_dir=audio_dir,
            batch_size=4, epochs=1, lr=1e-2, weight_decay=0.0,
            dropout=0.0, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, val_ds, stats = train_mod.build_datasets(cfg)
        model = train_mod.build_model(cfg)
        loader = train_mod.make_dataloader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        batch = next(iter(loader))
        logits = model(batch["audio"])
        loss0 = float(criterion(logits, batch["label"].float()))
        for _ in range(20):
            opt.zero_grad()
            logits = model(batch["audio"])
            loss = criterion(logits, batch["label"].float())
            loss.backward()
            opt.step()
        loss1 = float(criterion(model(batch["audio"]), batch["label"].float()))
        assert loss1 < loss0

    def test_checkpoint_roundtrip(self, tmp_path):
        manifest, audio_dir = _build_train_fixture(tmp_path)
        cfg = train_mod.RunConfig(
            modality="audio", backend="wav2vec2",
            manifest=manifest, audio_dir=audio_dir,
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        rc = train_mod.main([
            "--backend", "wav2vec2",
            "--manifest", str(manifest),
            "--audio-dir", str(audio_dir),
            "--epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--run-name", "rt",
            "--runs-dir", str(tmp_path / "runs"),
            "--checkpoint-path", str(tmp_path / "ckpt.pt"),
        ])
        assert rc == 0
        ckpt = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
        for key in ("state_dict", "modality", "backend", "audio_dir",
                    "model_hparams", "norm_stats", "val_metrics", "seed", "manifest"):
            assert key in ckpt, f"missing {key} in checkpoint"
        assert ckpt["modality"] == "audio"
        assert ckpt["backend"] == "wav2vec2"
        assert ckpt["audio_dir"] == str(audio_dir)
        assert ckpt["seed"] == 42
        rebuild = LateFusionClassifier(
            modality=ckpt["model_hparams"]["modality"],
            emb=ckpt["model_hparams"]["emb"],
            p=ckpt["model_hparams"]["dropout"],
        )
        rebuild.load_state_dict(ckpt["state_dict"])

    def test_early_stop_triggers_after_patience(self, tmp_path):
        cfg = train_mod.RunConfig(
            modality="audio", backend="wav2vec2",
            manifest=tmp_path / "m.csv", audio_dir=tmp_path / "a",
            batch_size=2, epochs=20, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=3, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        stopped_at, best = train_mod._simulate_early_stop(
            val_aucs=[0.6, 0.7, 0.65, 0.66, 0.67, 0.68, 0.5, 0.5, 0.5],
            patience=cfg.patience,
        )
        assert stopped_at == 6
        assert best == 0.7

    def test_train_never_builds_test_split(self, tmp_path):
        manifest, audio_dir = _build_train_fixture(tmp_path)
        seen_splits: list[str] = []
        real_cls = train_mod.AudioFeatureDataset

        def spy(*args, **kwargs):
            seen_splits.append(kwargs.get("split", args[1] if len(args) > 1 else None))
            return real_cls(*args, **kwargs)

        with patch.object(train_mod, "AudioFeatureDataset", side_effect=spy):
            rc = train_mod.main([
                "--backend", "wav2vec2",
                "--manifest", str(manifest),
                "--audio-dir", str(audio_dir),
                "--epochs", "1",
                "--batch-size", "2",
                "--device", "cpu",
                "--run-name", "t",
                "--runs-dir", str(tmp_path / "runs"),
                "--checkpoint-path", str(tmp_path / "ckpt.pt"),
            ])
        assert rc == 0
        assert "test" not in seen_splits, f"train.py built test split: {seen_splits}"

    def test_build_datasets_visual_wires_visual_dataset(self, tmp_path):
        from src.data.feature_store import VisualFeatureDataset
        manifest, lips_dir = _build_visual_fixture(tmp_path)
        cfg = train_mod.RunConfig(
            modality="visual", backend="wav2vec2",
            manifest=manifest, audio_dir=tmp_path / "unused",
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, val_ds, stats = train_mod.build_datasets(cfg, lips_dir=lips_dir)
        assert isinstance(train_ds, VisualFeatureDataset)
        assert isinstance(val_ds, VisualFeatureDataset)
        assert stats.lips_mean is not None and stats.lips_std is not None
        assert stats.audio_mean is None and stats.audio_std is None

    def test_build_datasets_fusion_wires_fusion_dataset(self, tmp_path):
        from src.data.feature_store import FusionFeatureDataset
        manifest, audio_dir, lips_dir = _build_fusion_fixture(tmp_path)
        cfg = train_mod.RunConfig(
            modality="fusion", backend="wav2vec2",
            manifest=manifest, audio_dir=audio_dir,
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, val_ds, stats = train_mod.build_datasets(cfg, lips_dir=lips_dir)
        assert isinstance(train_ds, FusionFeatureDataset)
        assert isinstance(val_ds, FusionFeatureDataset)
        assert stats.audio_mean is not None and stats.audio_std is not None
        assert stats.lips_mean is not None and stats.lips_std is not None

    def test_build_datasets_visual_drop_no_face_filters_train_but_not_val(self, tmp_path):
        manifest, lips_dir = _build_visual_fixture(tmp_path, n_train=4, n_val=2)
        # Zero out one train mask and one val mask, then re-save.
        bad_train_sid = "train_0"
        bad_val_sid = "val_0"
        for sid in (bad_train_sid, bad_val_sid):
            path = lips_dir / f"{sid}.npz"
            feats = np.load(path)["feats"]
            np.savez(path, feats=feats, mask=np.zeros(20, dtype=np.float32))

        cfg = train_mod.RunConfig(
            modality="visual", backend="wav2vec2",
            manifest=manifest, audio_dir=tmp_path / "unused",
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, val_ds, _ = train_mod.build_datasets(
            cfg, lips_dir=lips_dir, drop_no_face_train=True,
        )
        # Train: 4 originally, 1 dropped -> 3 kept.
        assert len(train_ds) == 3
        assert train_ds.dropped_no_face_ids == [bad_train_sid]
        # Val: 2 originally, none dropped (filter is train-only).
        assert len(val_ds) == 2
        assert val_ds.dropped_no_face_ids == []

    def test_build_datasets_fusion_drop_no_face_filters_train_but_not_val(self, tmp_path):
        manifest, audio_dir, lips_dir = _build_fusion_fixture(tmp_path, n_train=4, n_val=2)
        bad_train_sid = "train_0"
        bad_val_sid = "val_0"
        for sid in (bad_train_sid, bad_val_sid):
            path = lips_dir / f"{sid}.npz"
            feats = np.load(path)["feats"]
            np.savez(path, feats=feats, mask=np.zeros(20, dtype=np.float32))

        cfg = train_mod.RunConfig(
            modality="fusion", backend="wav2vec2",
            manifest=manifest, audio_dir=audio_dir,
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, val_ds, _ = train_mod.build_datasets(
            cfg, lips_dir=lips_dir, drop_no_face_train=True,
        )
        assert len(train_ds) == 3
        assert train_ds.dropped_no_face_ids == [bad_train_sid]
        assert len(val_ds) == 2
        assert val_ds.dropped_no_face_ids == []

    def test_build_datasets_drop_no_face_default_is_off(self, tmp_path):
        """Default behavior preserves NO-FACE rows for backward compatibility."""
        manifest, lips_dir = _build_visual_fixture(tmp_path, n_train=2, n_val=2)
        path = lips_dir / "train_0.npz"
        feats = np.load(path)["feats"]
        np.savez(path, feats=feats, mask=np.zeros(20, dtype=np.float32))

        cfg = train_mod.RunConfig(
            modality="visual", backend="wav2vec2",
            manifest=manifest, audio_dir=tmp_path / "unused",
            batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
            dropout=0.3, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, val_ds, _ = train_mod.build_datasets(cfg, lips_dir=lips_dir)
        assert len(train_ds) == 2  # NO-FACE row still present
        assert train_ds.dropped_no_face_ids == []

    def test_param_budget_asserted_visual_and_fusion(self, tmp_path):
        for modality in ("visual", "fusion"):
            cfg = train_mod.RunConfig(
                modality=modality, backend="wav2vec2",
                manifest=tmp_path / "m.csv", audio_dir=tmp_path / "a",
                batch_size=2, epochs=1, lr=1e-4, weight_decay=1e-2,
                dropout=0.3, patience=2, device="cpu", seed=42,
                run_name="t", runs_dir=tmp_path / "runs",
                checkpoint_path=tmp_path / "ckpt.pt",
            )
            model = train_mod.build_model(cfg)
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert n < 2_000_000, f"{modality}: {n:,}"

    def test_one_step_decreases_loss_visual(self, tmp_path):
        manifest, lips_dir = _build_visual_fixture(tmp_path, separable=True)
        cfg = train_mod.RunConfig(
            modality="visual", backend="wav2vec2",
            manifest=manifest, audio_dir=tmp_path / "unused",
            batch_size=4, epochs=1, lr=1e-2, weight_decay=0.0,
            dropout=0.0, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, _, _ = train_mod.build_datasets(cfg, lips_dir=lips_dir)
        model = train_mod.build_model(cfg)
        loader = train_mod.make_dataloader(train_ds, batch_size=cfg.batch_size,
                                           shuffle=True, drop_last=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        batch = next(iter(loader))
        loss0 = float(criterion(
            train_mod._forward_batch(model, batch, "visual", torch.device("cpu")),
            batch["label"].float(),
        ))
        for _ in range(20):
            opt.zero_grad()
            logits = train_mod._forward_batch(model, batch, "visual", torch.device("cpu"))
            loss = criterion(logits, batch["label"].float())
            loss.backward()
            opt.step()
        loss1 = float(criterion(
            train_mod._forward_batch(model, batch, "visual", torch.device("cpu")),
            batch["label"].float(),
        ))
        assert loss1 < loss0

    def test_one_step_decreases_loss_fusion(self, tmp_path):
        manifest, audio_dir, lips_dir = _build_fusion_fixture(tmp_path, separable=True)
        cfg = train_mod.RunConfig(
            modality="fusion", backend="wav2vec2",
            manifest=manifest, audio_dir=audio_dir,
            batch_size=4, epochs=1, lr=1e-2, weight_decay=0.0,
            dropout=0.0, patience=2, device="cpu", seed=42,
            run_name="t", runs_dir=tmp_path / "runs",
            checkpoint_path=tmp_path / "ckpt.pt",
        )
        train_ds, _, _ = train_mod.build_datasets(cfg, lips_dir=lips_dir)
        model = train_mod.build_model(cfg)
        loader = train_mod.make_dataloader(train_ds, batch_size=cfg.batch_size,
                                           shuffle=True, drop_last=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        batch = next(iter(loader))
        loss0 = float(criterion(
            train_mod._forward_batch(model, batch, "fusion", torch.device("cpu")),
            batch["label"].float(),
        ))
        for _ in range(20):
            opt.zero_grad()
            logits = train_mod._forward_batch(model, batch, "fusion", torch.device("cpu"))
            loss = criterion(logits, batch["label"].float())
            loss.backward()
            opt.step()
        loss1 = float(criterion(
            train_mod._forward_batch(model, batch, "fusion", torch.device("cpu")),
            batch["label"].float(),
        ))
        assert loss1 < loss0

    def test_checkpoint_roundtrip_visual(self, tmp_path):
        manifest, lips_dir = _build_visual_fixture(tmp_path)
        ckpt_path = tmp_path / "ckpt_visual.pt"
        rc = train_mod.main([
            "--modality", "visual",
            "--manifest", str(manifest),
            "--lips-dir", str(lips_dir),
            "--epochs", "1", "--batch-size", "2", "--device", "cpu",
            "--run-name", "rt-visual",
            "--runs-dir", str(tmp_path / "runs"),
            "--checkpoint-path", str(ckpt_path),
        ])
        assert rc == 0
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["modality"] == "visual"
        assert ckpt["model_hparams"]["modality"] == "visual"
        assert "lips_mean" in ckpt["norm_stats"] and "lips_std" in ckpt["norm_stats"]
        # visual ckpt has no audio stats
        assert "audio_mean" not in ckpt["norm_stats"]
        rebuild = LateFusionClassifier(
            modality=ckpt["model_hparams"]["modality"],
            emb=ckpt["model_hparams"]["emb"],
            p=ckpt["model_hparams"]["dropout"],
        )
        rebuild.load_state_dict(ckpt["state_dict"])

    def test_checkpoint_roundtrip_fusion(self, tmp_path):
        manifest, audio_dir, lips_dir = _build_fusion_fixture(tmp_path)
        ckpt_path = tmp_path / "ckpt_fusion.pt"
        rc = train_mod.main([
            "--modality", "fusion", "--backend", "wav2vec2",
            "--manifest", str(manifest),
            "--audio-dir", str(audio_dir),
            "--lips-dir", str(lips_dir),
            "--epochs", "1", "--batch-size", "2", "--device", "cpu",
            "--run-name", "rt-fusion",
            "--runs-dir", str(tmp_path / "runs"),
            "--checkpoint-path", str(ckpt_path),
        ])
        assert rc == 0
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["modality"] == "fusion"
        assert ckpt["backend"] == "wav2vec2"
        assert ckpt["audio_dir"] == str(audio_dir)
        for key in ("audio_mean", "audio_std", "lips_mean", "lips_std"):
            assert key in ckpt["norm_stats"], f"missing norm_stats[{key!r}]"

    def test_train_never_builds_test_split_visual(self, tmp_path):
        from src.data.feature_store import VisualFeatureDataset
        manifest, lips_dir = _build_visual_fixture(tmp_path)
        seen_splits: list[str] = []
        real_cls = VisualFeatureDataset

        def spy(*args, **kwargs):
            seen_splits.append(kwargs.get("split"))
            return real_cls(*args, **kwargs)

        with patch.object(train_mod, "VisualFeatureDataset", side_effect=spy):
            rc = train_mod.main([
                "--modality", "visual",
                "--manifest", str(manifest),
                "--lips-dir", str(lips_dir),
                "--epochs", "1", "--batch-size", "2", "--device", "cpu",
                "--run-name", "nots-visual",
                "--runs-dir", str(tmp_path / "runs"),
                "--checkpoint-path", str(tmp_path / "ckpt.pt"),
            ])
        assert rc == 0
        assert "test" not in seen_splits, f"train.py built test split: {seen_splits}"

    def test_train_never_builds_test_split_fusion(self, tmp_path):
        from src.data.feature_store import FusionFeatureDataset
        manifest, audio_dir, lips_dir = _build_fusion_fixture(tmp_path)
        seen_splits: list[str] = []
        real_cls = FusionFeatureDataset

        def spy(*args, **kwargs):
            seen_splits.append(kwargs.get("split"))
            return real_cls(*args, **kwargs)

        with patch.object(train_mod, "FusionFeatureDataset", side_effect=spy):
            rc = train_mod.main([
                "--modality", "fusion", "--backend", "wav2vec2",
                "--manifest", str(manifest),
                "--audio-dir", str(audio_dir),
                "--lips-dir", str(lips_dir),
                "--epochs", "1", "--batch-size", "2", "--device", "cpu",
                "--run-name", "nots-fusion",
                "--runs-dir", str(tmp_path / "runs"),
                "--checkpoint-path", str(tmp_path / "ckpt.pt"),
            ])
        assert rc == 0
        assert "test" not in seen_splits, f"train.py built test split: {seen_splits}"
