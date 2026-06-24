"""Tests for src/features/extract_audio_embeddings.py."""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


CSV_FIELDS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path",
    "audio_label", "audio_label_binary",
    "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _row(sample_id: str, *, split: str = "train", provider: str = "original",
         audio_path: str = "/dev/null") -> dict:
    blank = {k: "" for k in CSV_FIELDS}
    blank.update({
        "sample_id": sample_id,
        "source_video_id": sample_id,
        "split": split,
        "media_type": "audio",
        "source_folder": "tts_audio",
        "provider": provider,
        "audio_path": audio_path,
        "audio_label": "spoof",
        "audio_label_binary": "1",
        "video_label": "na",
        "video_label_binary": "",
        "pair_label": "na",
        "pair_label_binary": "",
    })
    return blank


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------- default_out_dir ----------

def test_default_out_dir_uses_backend_specific_constant():
    from src.features.extract_audio_embeddings import default_out_dir
    from src import common

    assert default_out_dir("wav2vec2") == common.FEAT_AUDIO_WAV2VEC2_DIR
    assert default_out_dir("wavlm") == common.FEAT_AUDIO_WAVLM_DIR
    assert default_out_dir("hubert") == common.FEAT_AUDIO_HUBERT_DIR


def test_default_out_dir_raises_on_unknown_backend():
    from src.features.extract_audio_embeddings import default_out_dir

    with pytest.raises(ValueError, match=r"unknown.*backend"):
        default_out_dir("not-a-backend")


# ---------- iter_manifest_rows ----------

def test_iter_manifest_rows_yields_all_rows_unfiltered(tmp_path):
    from src.features.extract_audio_embeddings import iter_manifest_rows

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("a"), _row("b"), _row("c")])

    out = list(iter_manifest_rows(manifest, split=None, source_providers=None, limit=None))

    assert [r["sample_id"] for r in out] == ["a", "b", "c"]


def test_iter_manifest_rows_split_filter(tmp_path):
    from src.features.extract_audio_embeddings import iter_manifest_rows

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("a", split="train"),
        _row("b", split="val"),
        _row("c", split="train"),
    ])

    out = list(iter_manifest_rows(manifest, split="train", source_providers=None, limit=None))

    assert [r["sample_id"] for r in out] == ["a", "c"]


def test_iter_manifest_rows_provider_filter(tmp_path):
    from src.features.extract_audio_embeddings import iter_manifest_rows

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [
        _row("a", provider="elevenlabs"),
        _row("b", provider="google_tts"),
        _row("c", provider="elevenlabs"),
    ])

    out = list(iter_manifest_rows(
        manifest, split=None, source_providers=("elevenlabs",), limit=None
    ))

    assert [r["sample_id"] for r in out] == ["a", "c"]


def test_iter_manifest_rows_limit(tmp_path):
    from src.features.extract_audio_embeddings import iter_manifest_rows

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row(str(i)) for i in range(10)])

    out = list(iter_manifest_rows(manifest, split=None, source_providers=None, limit=3))

    assert [r["sample_id"] for r in out] == ["0", "1", "2"]


# ---------- extract loop ----------

def _make_fake_backend(out_shape=(199, 768)):
    backend = MagicMock()
    backend.sample_rate = 16000
    backend.encode.return_value = np.random.randn(*out_shape).astype(np.float32)
    return backend


def test_extract_writes_one_npy_per_row_using_sample_id(tmp_path):
    from src.features.extract_audio_embeddings import extract

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha"), _row("beta")])
    out_dir = tmp_path / "out"
    backend = _make_fake_backend()

    with patch("src.features.extract_audio_embeddings.load_audio_window",
               return_value=np.zeros(64000, dtype=np.float32)):
        counts = extract(
            manifest, backend, out_dir,
            split=None, source_providers=None, limit=None,
            overwrite=False, dtype="float16",
        )

    assert (out_dir / "alpha.npy").exists()
    assert (out_dir / "beta.npy").exists()
    assert counts == {"written": 2, "skipped": 0, "failed": 0}


def test_extract_skips_existing_unless_overwrite(tmp_path):
    from src.features.extract_audio_embeddings import extract

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha")])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    np.save(out_dir / "alpha.npy", np.zeros((1, 1), dtype=np.float32))

    backend = _make_fake_backend()
    with patch("src.features.extract_audio_embeddings.load_audio_window",
               return_value=np.zeros(64000, dtype=np.float32)):
        counts_skip = extract(
            manifest, backend, out_dir,
            split=None, source_providers=None, limit=None,
            overwrite=False, dtype="float16",
        )
        counts_overwrite = extract(
            manifest, backend, out_dir,
            split=None, source_providers=None, limit=None,
            overwrite=True, dtype="float16",
        )

    assert counts_skip == {"written": 0, "skipped": 1, "failed": 0}
    assert counts_overwrite == {"written": 1, "skipped": 0, "failed": 0}


def test_extract_writes_float16_by_default_and_float32_when_requested(tmp_path):
    from src.features.extract_audio_embeddings import extract

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha")])
    out_dir_16 = tmp_path / "out16"
    out_dir_32 = tmp_path / "out32"
    backend = _make_fake_backend()

    with patch("src.features.extract_audio_embeddings.load_audio_window",
               return_value=np.zeros(64000, dtype=np.float32)):
        extract(manifest, backend, out_dir_16,
                split=None, source_providers=None, limit=None,
                overwrite=False, dtype="float16")
        extract(manifest, backend, out_dir_32,
                split=None, source_providers=None, limit=None,
                overwrite=False, dtype="float32")

    assert np.load(out_dir_16 / "alpha.npy").dtype == np.float16
    assert np.load(out_dir_32 / "alpha.npy").dtype == np.float32


def test_extract_does_not_mutate_manifest_csv(tmp_path):
    from src.features.extract_audio_embeddings import extract

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha"), _row("beta")])
    before = manifest.read_bytes()
    out_dir = tmp_path / "out"
    backend = _make_fake_backend()

    with patch("src.features.extract_audio_embeddings.load_audio_window",
               return_value=np.zeros(64000, dtype=np.float32)):
        extract(manifest, backend, out_dir,
                split=None, source_providers=None, limit=None,
                overwrite=False, dtype="float16")

    assert manifest.read_bytes() == before


def test_extract_counts_failures_and_continues(tmp_path):
    from src.features.extract_audio_embeddings import extract

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("a"), _row("b"), _row("c")])
    out_dir = tmp_path / "out"

    backend = _make_fake_backend()

    # First call succeeds, second raises, third succeeds.
    def side_effect(path, sr, seconds):
        if "b" in str(path):
            raise RuntimeError("decode error")
        return np.zeros(64000, dtype=np.float32)

    with patch("src.features.extract_audio_embeddings.load_audio_window",
               side_effect=side_effect):
        # audio_path needs to differ per row so the side_effect can branch.
        _write_manifest(manifest, [
            _row("a", audio_path="/tmp/a.mp3"),
            _row("b", audio_path="/tmp/b.mp3"),
            _row("c", audio_path="/tmp/c.mp3"),
        ])
        counts = extract(
            manifest, backend, out_dir,
            split=None, source_providers=None, limit=None,
            overwrite=False, dtype="float16",
        )

    assert counts == {"written": 2, "skipped": 0, "failed": 1}
    assert (out_dir / "a.npy").exists()
    assert (out_dir / "c.npy").exists()
    assert not (out_dir / "b.npy").exists()


# ---------- CLI ----------

def test_main_runs_with_backend_and_manifest(tmp_path, monkeypatch):
    from src.features import extract_audio_embeddings as cli

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha")])
    out_dir = tmp_path / "out"

    fake_backend = _make_fake_backend()
    monkeypatch.setattr(cli, "load_backend",
                        lambda name, device: fake_backend)
    monkeypatch.setattr(cli, "load_audio_window",
                        lambda path, sr, seconds: np.zeros(64000, dtype=np.float32))

    rc = cli.main([
        "--backend", "wav2vec2",
        "--manifest", str(manifest),
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    assert (out_dir / "alpha.npy").exists()


def test_main_limit_flag(tmp_path, monkeypatch):
    from src.features import extract_audio_embeddings as cli

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row(str(i)) for i in range(5)])
    out_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "load_backend",
                        lambda name, device: _make_fake_backend())
    monkeypatch.setattr(cli, "load_audio_window",
                        lambda path, sr, seconds: np.zeros(64000, dtype=np.float32))

    cli.main([
        "--backend", "wav2vec2",
        "--manifest", str(manifest),
        "--out-dir", str(out_dir),
        "--limit", "2",
    ])

    assert sorted(p.name for p in out_dir.glob("*.npy")) == ["0.npy", "1.npy"]


def test_main_uses_backend_default_out_dir_when_omitted(tmp_path, monkeypatch):
    from src.features import extract_audio_embeddings as cli

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha")])

    default_dir = tmp_path / "default"
    monkeypatch.setattr(cli, "default_out_dir", lambda name: default_dir)
    monkeypatch.setattr(cli, "load_backend",
                        lambda name, device: _make_fake_backend())
    monkeypatch.setattr(cli, "load_audio_window",
                        lambda path, sr, seconds: np.zeros(64000, dtype=np.float32))

    cli.main(["--backend", "wav2vec2", "--manifest", str(manifest)])

    assert (default_dir / "alpha.npy").exists()


def test_main_rejects_unknown_backend(tmp_path, capsys):
    from src.features import extract_audio_embeddings as cli

    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [_row("alpha")])

    with pytest.raises(SystemExit):
        cli.main([
            "--backend", "totally-fake",
            "--manifest", str(manifest),
        ])


def test_extract_counts_failure_when_sample_id_column_missing(tmp_path):
    from src.features.extract_audio_embeddings import extract

    manifest = tmp_path / "m.csv"
    # Row missing the sample_id column entirely.
    with manifest.open("w", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["split", "provider", "audio_path"])
        w.writeheader()
        w.writerow({"split": "train", "provider": "x", "audio_path": "/tmp/a.mp3"})

    out_dir = tmp_path / "out"
    backend = _make_fake_backend()

    counts = extract(
        manifest, backend, out_dir,
        split=None, source_providers=None, limit=None,
        overwrite=False, dtype="float16",
    )

    assert counts == {"written": 0, "skipped": 0, "failed": 1}
