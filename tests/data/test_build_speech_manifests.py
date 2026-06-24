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


def test_iter_generated_rows_inherits_split_from_source(tmp_path):
    from src.data.build_speech_manifests import iter_generated_rows

    tts = [{
        "provider": "elevenlabs",
        "source_video_id": "src_a",
        "voice": "V1",
        "synthetic_audio_path": "data/tts_audio/elevenlabs/real/src_a__voice-V1.mp3",
        "source_folder": "real",
    }]
    split_map = {"src_a": "train"}

    rows, excluded = iter_generated_rows(tts, split_map)

    assert excluded == []
    assert len(rows) == 1
    r = rows[0]
    assert r["split"] == "train"
    assert r["source_video_id"] == "src_a"
    assert r["provider"] == "elevenlabs"
    assert r["voice_id_or_name"] == "V1"
    assert r["audio_label"] == "spoof"
    assert r["audio_label_binary"] == 1
    assert r["video_label"] == "na"
    assert r["video_label_binary"] == ""
    assert r["media_type"] == "audio"
    assert r["source_folder"] == "real"
    assert r["sample_id"] == "elevenlabs__src_a__voice-V1"
    assert r["audio_feature_path"] == f"data/features/audio_generated/{r['sample_id']}.npy"
    assert r["lip_feature_path"] == ""


def test_iter_generated_rows_excludes_unknown_source_and_logs(tmp_path, caplog):
    from src.data.build_speech_manifests import iter_generated_rows

    tts = [{
        "provider": "elevenlabs",
        "source_video_id": "ghost",
        "voice": "V",
        "synthetic_audio_path": "data/tts_audio/elevenlabs/real/ghost__voice-V.mp3",
        "source_folder": "real",
    }]
    split_map = {"known": "train"}

    with caplog.at_level(logging.WARNING):
        rows, excluded = iter_generated_rows(tts, split_map)

    assert rows == []
    assert excluded == ["ghost"]
    assert any("ghost" in rec.message for rec in caplog.records)


def test_iter_generated_rows_fills_source_folder_from_native_map_when_missing(tmp_path):
    from src.data.build_speech_manifests import iter_generated_rows

    tts = [{
        "provider": "google_tts",
        "source_video_id": "src_b",
        "voice": "en-US-Neural2-A",
        "synthetic_audio_path": "data/tts_audio/google_tts/src_b__voice-en-US-Neural2-A.mp3",
        "source_folder": "",
    }]
    rows, _ = iter_generated_rows(
        tts,
        split_map={"src_b": "val"},
        source_folder_map={"src_b": "memo"},
    )
    assert rows[0]["source_folder"] == "memo"


def test_iter_generated_rows_distinct_sample_ids_per_provider_voice(tmp_path):
    from src.data.build_speech_manifests import iter_generated_rows

    tts = [
        {"provider": "elevenlabs", "source_video_id": "x", "voice": "A",
         "synthetic_audio_path": "p1.mp3", "source_folder": "real"},
        {"provider": "elevenlabs", "source_video_id": "x", "voice": "B",
         "synthetic_audio_path": "p2.mp3", "source_folder": "real"},
        {"provider": "google_tts", "source_video_id": "x", "voice": "A",
         "synthetic_audio_path": "p3.mp3", "source_folder": "real"},
    ]
    rows, _ = iter_generated_rows(tts, {"x": "train"})
    sample_ids = [r["sample_id"] for r in rows]
    assert sample_ids == [
        "elevenlabs__x__voice-A",
        "elevenlabs__x__voice-B",
        "google_tts__x__voice-A",
    ]
    assert len(set(sample_ids)) == 3


def test_write_manifest_emits_schema_columns_in_order(tmp_path):
    from src.data.build_speech_manifests import SCHEMA, write_manifest

    out = tmp_path / "derived" / "x.csv"
    write_manifest(
        [{"sample_id": "s1", "source_video_id": "v1", "split": "train",
          "media_type": "video", "source_folder": "real", "provider": "original",
          "voice_id_or_name": "",
          "audio_path": "", "video_path": "",
          "audio_feature_path": "", "lip_feature_path": "",
          "audio_label": "bonafide", "audio_label_binary": 0,
          "video_label": "real", "video_label_binary": 0,
          "pair_label": "na", "pair_label_binary": ""}],
        out,
    )
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(SCHEMA)
        rows = list(reader)
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["pair_label"] == "na"
    assert rows[0]["voice_id_or_name"] == ""


def test_write_manifest_rejects_unknown_columns(tmp_path):
    from src.data.build_speech_manifests import write_manifest

    out = tmp_path / "x.csv"
    with pytest.raises(ValueError, match=r"unknown column"):
        write_manifest([{"sample_id": "s", "evil_extra": "boom"}], out)


def _fixture_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a tiny manifest + splits + tts_audio tree under tmp_path."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("src_a", "real"), ("src_b", "echomimic")])

    splits = tmp_path / "splits"
    _write_split_csv(splits / "train.csv", ["src_a"])
    _write_split_csv(splits / "val.csv", ["src_b"], source_folder="echomimic")
    _write_split_csv(splits / "test.csv", [])

    tts = tmp_path / "tts_audio"
    _write_jsonl(tts / "manifest.jsonl", [{
        "provider": "elevenlabs", "video_id": "src_a", "voice_id": "V1",
        "source_folder": "real",
        "synthetic_audio_path": "data/tts_audio/elevenlabs/real/src_a__voice-V1.mp3",
    }])
    _write_jsonl(tts / "google_tts_manifest.jsonl", [{
        "provider": "google_tts", "video_id": "src_b", "voice_name": "en-US-Neural2-A",
        "source_folder": "echomimic",
        "synthetic_audio_path": "data/tts_audio/google_tts/echomimic/src_b__voice-en-US-Neural2-A.mp3",
    }])
    return {"manifest": manifest, "splits": splits, "tts": tts}


def test_build_audio_spoof_manifest_writes_native_and_generated_rows(tmp_path):
    from src.data.build_speech_manifests import build_audio_spoof_manifest

    p = _fixture_tree(tmp_path)
    out = tmp_path / "derived" / "audio_spoof_manifest.csv"

    stats = build_audio_spoof_manifest(
        manifest_path=p["manifest"],
        splits_dir=p["splits"],
        tts_dir=p["tts"],
        out_path=out,
        providers=["elevenlabs", "google_tts"],
    )

    assert stats["native_rows"] == 2
    assert stats["generated_rows"] == 2
    assert stats["excluded"] == []

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    by_label = {}
    for r in rows:
        by_label.setdefault(r["audio_label"], 0)
        by_label[r["audio_label"]] += 1
    assert by_label == {"bonafide": 2, "spoof": 2}

    # Spec §2: echomimic original audio is bonafide, video is fake.
    em = [r for r in rows if r["source_video_id"] == "src_b" and r["provider"] == "original"][0]
    assert em["audio_label"] == "bonafide"
    assert em["video_label"] == "fake"

    # Generated rows have video_label=na and blank video_label_binary.
    gens = [r for r in rows if r["audio_label"] == "spoof"]
    for r in gens:
        assert r["video_label"] == "na"
        assert r["video_label_binary"] == ""


def test_build_audio_spoof_manifest_inherits_split_for_generated(tmp_path):
    from src.data.build_speech_manifests import build_audio_spoof_manifest

    p = _fixture_tree(tmp_path)
    out = tmp_path / "derived" / "audio_spoof_manifest.csv"
    build_audio_spoof_manifest(
        manifest_path=p["manifest"],
        splits_dir=p["splits"],
        tts_dir=p["tts"],
        out_path=out,
        providers=["elevenlabs", "google_tts"],
    )

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    gen_by_src = {r["source_video_id"]: r["split"]
                  for r in rows if r["audio_label"] == "spoof"}
    assert gen_by_src["src_a"] == "train"
    assert gen_by_src["src_b"] == "val"


def test_build_visual_speech_manifest_emits_matched_and_generated_pairs(tmp_path):
    from src.data.build_speech_manifests import build_visual_speech_manifest

    p = _fixture_tree(tmp_path)
    out = tmp_path / "derived" / "visual_speech_manifest.csv"

    stats = build_visual_speech_manifest(
        manifest_path=p["manifest"],
        splits_dir=p["splits"],
        tts_dir=p["tts"],
        out_path=out,
        providers=["elevenlabs", "google_tts"],
    )
    assert stats["matched_rows"] == 2
    assert stats["generated_rows"] == 2

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    by_pair_label = {}
    for r in rows:
        by_pair_label.setdefault(r["pair_label"], 0)
        by_pair_label[r["pair_label"]] += 1
    assert by_pair_label == {"matched_bonafide": 2, "generated_same_transcript": 2}

    # Both row kinds carry a lip_feature_path pointing to the source video's lips.
    for r in rows:
        assert r["lip_feature_path"].endswith(f"data/features/lips/{r['source_video_id']}.npz")
        assert r["media_type"] == "pair"

    # Pair binaries match string labels.
    for r in rows:
        if r["pair_label"] == "matched_bonafide":
            assert r["pair_label_binary"] == "0"
        elif r["pair_label"] == "generated_same_transcript":
            assert r["pair_label_binary"] == "1"


def test_build_visual_speech_manifest_excludes_generated_without_native_source(tmp_path):
    from src.data.build_speech_manifests import build_visual_speech_manifest

    p = _fixture_tree(tmp_path)
    # Add a TTS record whose source_video_id is not in data/manifest.csv.
    _write_jsonl(p["tts"] / "manifest.jsonl", [
        {"provider": "elevenlabs", "video_id": "src_a", "voice_id": "V1",
         "source_folder": "real",
         "synthetic_audio_path": "data/tts_audio/elevenlabs/real/src_a__voice-V1.mp3"},
        {"provider": "elevenlabs", "video_id": "stranger", "voice_id": "Z",
         "source_folder": "real",
         "synthetic_audio_path": "data/tts_audio/elevenlabs/real/stranger__voice-Z.mp3"},
    ])
    # Add stranger to the split file so it survives split inheritance but
    # still has no native lip source.
    _write_split_csv(p["splits"] / "train.csv", ["src_a", "stranger"])

    out = tmp_path / "derived" / "visual_speech_manifest.csv"
    stats = build_visual_speech_manifest(
        manifest_path=p["manifest"],
        splits_dir=p["splits"],
        tts_dir=p["tts"],
        out_path=out,
        providers=["elevenlabs", "google_tts"],
    )

    assert "stranger" in stats["excluded_no_native_source"]
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(r["source_video_id"] != "stranger" for r in rows)


def test_build_fusion_speech_manifest_has_bonafide_and_spoof_rows(tmp_path):
    from src.data.build_speech_manifests import build_fusion_speech_manifest

    p = _fixture_tree(tmp_path)
    out = tmp_path / "derived" / "fusion_speech_manifest.csv"

    stats = build_fusion_speech_manifest(
        manifest_path=p["manifest"],
        splits_dir=p["splits"],
        tts_dir=p["tts"],
        out_path=out,
        providers=["elevenlabs", "google_tts"],
    )
    assert stats["bonafide_rows"] == 2
    assert stats["spoof_rows"] == 2

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    counts_audio = {}
    counts_pair = {}
    for r in rows:
        counts_audio.setdefault(r["audio_label"], 0)
        counts_audio[r["audio_label"]] += 1
        counts_pair.setdefault(r["pair_label"], 0)
        counts_pair[r["pair_label"]] += 1
    assert counts_audio == {"bonafide": 2, "spoof": 2}
    assert counts_pair == {"matched_bonafide": 2, "generated_same_transcript": 2}

    # Every row carries both a lip and an audio feature path.
    for r in rows:
        assert r["lip_feature_path"]
        assert r["audio_feature_path"]


def test_validate_manifests_passes_on_clean_fixture(tmp_path, monkeypatch):
    from src.data.build_speech_manifests import (
        build_audio_spoof_manifest, build_visual_speech_manifest,
        build_fusion_speech_manifest, validate_manifests,
    )

    p = _fixture_tree(tmp_path)
    out_dir = tmp_path / "derived"
    build_audio_spoof_manifest(p["manifest"], p["splits"], p["tts"],
                               out_dir / "audio_spoof_manifest.csv",
                               ["elevenlabs", "google_tts"])
    build_visual_speech_manifest(p["manifest"], p["splits"], p["tts"],
                                 out_dir / "visual_speech_manifest.csv",
                                 ["elevenlabs", "google_tts"])
    build_fusion_speech_manifest(p["manifest"], p["splits"], p["tts"],
                                 out_dir / "fusion_speech_manifest.csv",
                                 ["elevenlabs", "google_tts"])
    # Skip on-disk-path existence check (clean unit; files are stubs only).
    monkeypatch.setenv("SPEECH_MANIFEST_SKIP_PATH_EXISTS", "1")
    issues = validate_manifests(out_dir, p["manifest"], p["splits"])
    assert issues == []


def test_validate_manifests_detects_split_leakage(tmp_path, monkeypatch):
    from src.data.build_speech_manifests import validate_manifests, write_manifest

    out_dir = tmp_path / "derived"
    # Build splits where 'dup' appears in both train and val.
    splits = tmp_path / "splits"
    _write_split_csv(splits / "train.csv", ["dup", "x"])
    _write_split_csv(splits / "val.csv", ["dup"])
    _write_split_csv(splits / "test.csv", ["y"])
    manifest = tmp_path / "manifest.csv"
    _write_manifest_csv(manifest, [("dup", "real"), ("x", "real"), ("y", "real")])

    # Write empty derived manifests to satisfy "files exist".
    for name in ("audio_spoof_manifest.csv", "visual_speech_manifest.csv",
                 "fusion_speech_manifest.csv"):
        write_manifest([], out_dir / name)

    monkeypatch.setenv("SPEECH_MANIFEST_SKIP_PATH_EXISTS", "1")
    issues = validate_manifests(out_dir, manifest, splits)
    assert any("dup" in s and "leakage" in s.lower() for s in issues)


def test_validate_manifests_detects_label_binary_mismatch(tmp_path, monkeypatch):
    from src.data.build_speech_manifests import (SCHEMA, validate_manifests,
                                                  write_manifest)

    p = _fixture_tree(tmp_path)
    out_dir = tmp_path / "derived"
    # Write a fusion manifest row with mismatched audio_label_binary.
    bad_row = {col: "" for col in SCHEMA}
    bad_row.update({
        "sample_id": "bad", "source_video_id": "src_a", "split": "train",
        "media_type": "pair", "source_folder": "real", "provider": "original",
        "audio_path": "", "video_path": "", "audio_feature_path": "x",
        "lip_feature_path": "y",
        "audio_label": "bonafide", "audio_label_binary": 1,   # mismatch!
        "video_label": "real", "video_label_binary": 0,
        "pair_label": "matched_bonafide", "pair_label_binary": 0,
    })
    write_manifest([bad_row], out_dir / "fusion_speech_manifest.csv")
    write_manifest([], out_dir / "audio_spoof_manifest.csv")
    write_manifest([], out_dir / "visual_speech_manifest.csv")

    monkeypatch.setenv("SPEECH_MANIFEST_SKIP_PATH_EXISTS", "1")
    issues = validate_manifests(out_dir, p["manifest"], p["splits"])
    assert any("audio_label_binary" in s for s in issues)


def test_validate_manifests_detects_duplicate_sample_ids(tmp_path, monkeypatch):
    from src.data.build_speech_manifests import (SCHEMA, validate_manifests,
                                                  write_manifest)

    p = _fixture_tree(tmp_path)
    out_dir = tmp_path / "derived"
    base = {col: "" for col in SCHEMA}
    base.update({
        "sample_id": "same", "source_video_id": "src_a", "split": "train",
        "media_type": "audio", "source_folder": "real", "provider": "elevenlabs",
        "audio_label": "spoof", "audio_label_binary": 1,
        "video_label": "na", "video_label_binary": "",
    })
    write_manifest([base, dict(base)], out_dir / "audio_spoof_manifest.csv")
    write_manifest([], out_dir / "visual_speech_manifest.csv")
    write_manifest([], out_dir / "fusion_speech_manifest.csv")

    monkeypatch.setenv("SPEECH_MANIFEST_SKIP_PATH_EXISTS", "1")
    issues = validate_manifests(out_dir, p["manifest"], p["splits"])
    assert any("duplicate sample_id" in s for s in issues)


def test_validate_manifests_detects_echomimic_audio_relabeled_spoof(tmp_path, monkeypatch):
    from src.data.build_speech_manifests import (SCHEMA, validate_manifests,
                                                  write_manifest)

    p = _fixture_tree(tmp_path)
    out_dir = tmp_path / "derived"
    bad = {col: "" for col in SCHEMA}
    bad.update({
        "sample_id": "src_b", "source_video_id": "src_b", "split": "val",
        "media_type": "video", "source_folder": "echomimic", "provider": "original",
        "audio_label": "spoof",  # forbidden!
        "audio_label_binary": 1,
        "video_label": "fake", "video_label_binary": 1,
    })
    write_manifest([bad], out_dir / "audio_spoof_manifest.csv")
    write_manifest([], out_dir / "visual_speech_manifest.csv")
    write_manifest([], out_dir / "fusion_speech_manifest.csv")

    monkeypatch.setenv("SPEECH_MANIFEST_SKIP_PATH_EXISTS", "1")
    issues = validate_manifests(out_dir, p["manifest"], p["splits"])
    assert any("echomimic" in s.lower() and "spoof" in s.lower() for s in issues)
