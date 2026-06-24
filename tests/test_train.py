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
