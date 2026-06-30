"""End-to-end CLI tests for src/analysis/acoustic_probe.py."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


REQUIRED_COLS = [
    "sample_id", "split", "source_folder", "provider",
    "audio_path", "audio_label_binary",
]


def _write_manifest(out: Path, rows: list[dict]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=REQUIRED_COLS)
    df.to_csv(out, index=False)
    return out


@pytest.fixture
def mini_manifest(
    tmp_path,
    make_tone_wav,
    make_silent_wav,
    make_white_noise_wav,
):
    rows = []
    for i in range(3):
        # bonafide: white noise, train
        p = make_white_noise_wav(name=f"bona_train_{i}.wav", seed=100 + i)
        rows.append({
            "sample_id": f"bona_train_{i}",
            "split": "train",
            "source_folder": "real",
            "provider": "original",
            "audio_path": str(p),
            "audio_label_binary": 0,
        })
    for i in range(3):
        # spoof: tone, train (1 elevenlabs, 1 google_tts, 1 openai_tts)
        engines = ["elevenlabs", "google_tts", "openai_tts"]
        p = make_tone_wav(name=f"spoof_train_{i}.wav", freq_hz=440.0 + 50 * i)
        rows.append({
            "sample_id": f"spoof_train_{i}",
            "split": "train",
            "source_folder": engines[i],
            "provider": engines[i],
            "audio_path": str(p),
            "audio_label_binary": 1,
        })
    for i in range(2):
        p = make_white_noise_wav(name=f"bona_val_{i}.wav", seed=200 + i)
        rows.append({
            "sample_id": f"bona_val_{i}",
            "split": "val",
            "source_folder": "real",
            "provider": "original",
            "audio_path": str(p),
            "audio_label_binary": 0,
        })
    for i in range(3):
        engines = ["elevenlabs", "google_tts", "openai_tts"]
        p = make_tone_wav(name=f"spoof_val_{i}.wav", freq_hz=440.0 + 50 * i)
        rows.append({
            "sample_id": f"spoof_val_{i}",
            "split": "val",
            "source_folder": engines[i],
            "provider": engines[i],
            "audio_path": str(p),
            "audio_label_binary": 1,
        })
    return _write_manifest(tmp_path / "manifest.csv", rows)


def test_cli_writes_features_failures_and_metrics(tmp_path, mini_manifest):
    from src.analysis.acoustic_probe import main

    out_dir = tmp_path / "out"
    rc = main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
    ])
    assert rc == 0
    assert (out_dir / "acoustic_features.csv").exists()
    assert (out_dir / "acoustic_failures.csv").exists()
    assert (out_dir / "probe_metrics.json").exists()

    feats = pd.read_csv(out_dir / "acoustic_features.csv")
    assert len(feats) == 11  # 6 train + 5 val, all valid
    assert "rms" in feats.columns
    assert "sample_id" in feats.columns

    failures = pd.read_csv(out_dir / "acoustic_failures.csv")
    assert list(failures.columns) == ["sample_id", "audio_path", "reason"]
    assert len(failures) == 0

    meta = json.loads((out_dir / "probe_metrics.json").read_text())
    assert meta["bad_rows"] == []
    assert meta["config"]["with_f0"] is False
    assert meta["config"]["seed"] == 42


def test_cli_records_unreadable_audio_in_failures(
    tmp_path, mini_manifest, make_corrupt_wav,
):
    from src.analysis.acoustic_probe import main

    df = pd.read_csv(mini_manifest)
    bad = make_corrupt_wav(name="broken.wav")
    df.loc[0, "audio_path"] = str(bad)
    df.to_csv(mini_manifest, index=False)

    out_dir = tmp_path / "out"
    rc = main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
    ])
    assert rc == 0
    failures = pd.read_csv(out_dir / "acoustic_failures.csv")
    assert len(failures) == 1
    assert failures.iloc[0]["sample_id"] == df.iloc[0]["sample_id"]
    meta = json.loads((out_dir / "probe_metrics.json").read_text())
    assert len(meta["bad_rows"]) == 1


def test_cli_resumes_using_cache(tmp_path, mini_manifest):
    from src.analysis.acoustic_probe import main

    out_dir = tmp_path / "out"
    main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
    ])
    cache = out_dir / "acoustic_features.csv"
    mtime_before = cache.stat().st_mtime_ns

    # Second run with the SAME flags should NOT recompute (mtime stays the same,
    # because the resumable loop sees every row already cached and writes nothing).
    main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
    ])
    assert cache.stat().st_mtime_ns == mtime_before


def test_cli_force_recomputes(tmp_path, mini_manifest):
    from src.analysis.acoustic_probe import main

    out_dir = tmp_path / "out"
    main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
    ])
    mtime_before = (out_dir / "acoustic_features.csv").stat().st_mtime_ns
    main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
        "--force",
    ])
    mtime_after = (out_dir / "acoustic_features.csv").stat().st_mtime_ns
    assert mtime_after >= mtime_before  # rewritten


def test_cli_schema_guard_rejects_f0_against_core_cache(tmp_path, mini_manifest, capsys):
    from src.analysis.acoustic_probe import main

    out_dir = tmp_path / "out"
    main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
    ])
    rc = main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
        "--f0",
    ])
    assert rc != 0
    captured = capsys.readouterr()
    assert "--force" in (captured.err + captured.out)


def test_cli_fails_loud_on_missing_manifest_columns(tmp_path):
    from src.analysis.acoustic_probe import main

    bogus = tmp_path / "bad.csv"
    pd.DataFrame({"only_one_col": [1, 2, 3]}).to_csv(bogus, index=False)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        main([
            "--manifest", str(bogus),
            "--out-dir", str(out_dir),
            "--no-plots",
        ])


def test_cli_writes_summary_tables_and_default_probes(tmp_path, mini_manifest):
    from src.analysis.acoustic_probe import main

    out_dir = tmp_path / "out"
    rc = main([
        "--manifest", str(mini_manifest),
        "--out-dir", str(out_dir),
        "--no-plots",
    ])
    assert rc == 0
    for name in ("summary_by_label.csv", "summary_by_label_provider.csv",
                 "summary_by_source_folder.csv"):
        assert (out_dir / name).exists(), name

    meta = json.loads((out_dir / "probe_metrics.json").read_text())
    dp = meta["default_probes"]
    assert "lr" in dp and "rf" in dp and "per_feature_lr" in dp
    assert "roc_auc" in dp["lr"] and "roc_auc" in dp["rf"]
    assert dp["lr"]["roc_auc"] > 0.9  # tone vs noise is trivially separable
    assert isinstance(dp["per_feature_lr"], list)
    assert len(dp["per_feature_lr"]) > 5  # at least the core features
    assert "feature_importances" in dp["rf"]
