"""Tests for src/data/build_speech_manifests.py."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest


def _write_split_csv(path: Path, video_ids: list[str], source_folder: str = "real") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "relative_path", "source_folder", "binary_label",
                    "duration_s", "fps", "n_frames"])
        for vid in video_ids:
            w.writerow([vid, f"data/raw/{source_folder}/{vid}.mp4", source_folder,
                        0 if source_folder == "real" else 1, "5.0", "24.0", "120"])


def test_load_split_map_maps_each_video_to_its_split(tmp_path):
    from src.data.build_speech_manifests import load_split_map

    splits_dir = tmp_path / "splits"
    _write_split_csv(splits_dir / "train.csv", ["a", "b"])
    _write_split_csv(splits_dir / "val.csv", ["c"])
    _write_split_csv(splits_dir / "test.csv", ["d"])

    m = load_split_map(splits_dir)

    assert m == {"a": "train", "b": "train", "c": "val", "d": "test"}


def test_load_split_map_raises_on_duplicate_across_splits(tmp_path):
    from src.data.build_speech_manifests import load_split_map

    splits_dir = tmp_path / "splits"
    _write_split_csv(splits_dir / "train.csv", ["dup", "x"])
    _write_split_csv(splits_dir / "val.csv", ["dup"])
    _write_split_csv(splits_dir / "test.csv", ["y"])

    with pytest.raises(ValueError, match=r"^split leakage:.*dup"):
        load_split_map(splits_dir)


def _write_manifest_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    """rows = [(video_id, source_folder), ...]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "relative_path", "source_folder", "binary_label",
                    "duration_s", "fps", "n_frames"])
        for vid, src in rows:
            label = 0 if src == "real" else 1
            w.writerow([vid, f"data/raw/{src}/{vid}.mp4", src, label, "5.0", "24.0", "120"])


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _touch_mp3(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


def test_iter_native_rows_real_is_bonafide_audio_and_real_video(tmp_path):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("vid_real", "real")])
    split_map = {"vid_real": "train"}

    rows = list(iter_native_rows(manifest, split_map))

    assert len(rows) == 1
    r = rows[0]
    assert r["source_video_id"] == "vid_real"
    assert r["sample_id"] == "vid_real"
    assert r["split"] == "train"
    assert r["media_type"] == "video"
    assert r["source_folder"] == "real"
    assert r["provider"] == "original"
    assert r["voice_id_or_name"] == ""
    assert r["audio_label"] == "bonafide"
    assert r["audio_label_binary"] == 0
    assert r["video_label"] == "real"
    assert r["video_label_binary"] == 0
    assert r["pair_label"] == "na"
    assert r["pair_label_binary"] == ""
    assert r["video_path"] == "data/raw/real/vid_real.mp4"
    assert r["audio_feature_path"] == "data/features/audio/vid_real.npy"


def test_iter_native_rows_echomimic_is_bonafide_audio_and_fake_video(tmp_path):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("vid_em", "echomimic")])
    rows = list(iter_native_rows(manifest, {"vid_em": "val"}))

    r = rows[0]
    assert r["source_folder"] == "echomimic"
    assert r["audio_label"] == "bonafide"
    assert r["audio_label_binary"] == 0
    assert r["video_label"] == "fake"
    assert r["video_label_binary"] == 1


def test_iter_native_rows_memo_is_bonafide_audio_and_fake_video(tmp_path):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("vid_memo", "memo")])
    rows = list(iter_native_rows(manifest, {"vid_memo": "test"}))

    r = rows[0]
    assert r["source_folder"] == "memo"
    assert r["audio_label"] == "bonafide"
    assert r["video_label"] == "fake"
    assert r["video_label_binary"] == 1


def test_iter_native_rows_skips_videos_missing_from_split_map(tmp_path, caplog):
    from src.data.build_speech_manifests import iter_native_rows

    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("known", "real"), ("orphan", "real")])

    with caplog.at_level(logging.WARNING):
        rows = list(iter_native_rows(manifest, {"known": "train"}))

    assert [r["source_video_id"] for r in rows] == ["known"]
    assert any("orphan" in rec.message for rec in caplog.records)


def test_parse_tts_filename_extracts_id_and_voice():
    from src.data.build_speech_manifests import parse_tts_filename

    sv, voice = parse_tts_filename("abc123__voice-XYZ.mp3")
    assert sv == "abc123"
    assert voice == "XYZ"


def test_parse_tts_filename_returns_none_on_bad_name():
    from src.data.build_speech_manifests import parse_tts_filename

    assert parse_tts_filename("no_voice.mp3") is None
    assert parse_tts_filename("abc__voice-X.wav") is None


def test_iter_tts_records_prefers_jsonl_when_present(tmp_path):
    from src.data.build_speech_manifests import iter_tts_records

    tts_dir = tmp_path / "tts_audio"
    _write_jsonl(tts_dir / "manifest.jsonl", [{
        "provider": "elevenlabs",
        "video_id": "src1",
        "voice_id": "V1",
        "source_folder": "real",
        "synthetic_audio_path": "data/tts_audio/elevenlabs/real/src1__voice-V1.mp3",
    }])
    _write_jsonl(tts_dir / "google_tts_manifest.jsonl", [{
        "provider": "google_tts",
        "video_id": "src2",
        "voice_name": "en-US-Neural2-A",
        "source_folder": "echomimic",
        "synthetic_audio_path": "data/tts_audio/google_tts/echomimic/src2__voice-en-US-Neural2-A.mp3",
    }])

    out = iter_tts_records(tts_dir, providers=["elevenlabs", "google_tts"])

    by_provider = {r["provider"]: r for r in out}
    assert by_provider["elevenlabs"]["source_video_id"] == "src1"
    assert by_provider["elevenlabs"]["voice"] == "V1"
    assert by_provider["elevenlabs"]["source_folder"] == "real"
    assert by_provider["google_tts"]["source_video_id"] == "src2"
    assert by_provider["google_tts"]["voice"] == "en-US-Neural2-A"


def test_iter_tts_records_falls_back_to_filesystem_when_jsonl_missing(tmp_path):
    from src.data.build_speech_manifests import iter_tts_records

    tts_dir = tmp_path / "tts_audio"
    _touch_mp3(tts_dir / "elevenlabs" / "real" / "src3__voice-V3.mp3")

    out = iter_tts_records(tts_dir, providers=["elevenlabs"])

    assert len(out) == 1
    r = out[0]
    assert r["provider"] == "elevenlabs"
    assert r["source_video_id"] == "src3"
    assert r["voice"] == "V3"
    assert r["source_folder"] == "real"
    assert r["synthetic_audio_path"].endswith("real/src3__voice-V3.mp3")


def test_iter_tts_records_skips_provider_not_requested(tmp_path):
    from src.data.build_speech_manifests import iter_tts_records

    tts_dir = tmp_path / "tts_audio"
    _write_jsonl(tts_dir / "sts_manifest.jsonl", [{
        "provider": "elevenlabs",
        "video_id": "sts1",
        "voice_id": "V",
        "source_folder": "real",
        "synthetic_audio_path": "data/tts_audio/elevenlabs_sts/real/sts1__voice-V.mp3",
    }])

    out = iter_tts_records(tts_dir, providers=["elevenlabs", "google_tts"])
    assert out == []
