"""Tests for src/data/build_lipsync_pairs.py."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def test_schema_columns_are_exact_and_ordered():
    from src.data.build_lipsync_pairs import LIPSYNC_MANIFEST_SCHEMA

    assert LIPSYNC_MANIFEST_SCHEMA == (
        "pair_id", "split", "source_video_id", "lip_feature_path",
        "audio_sample_id", "audio_path", "audio_feature_path",
        "audio_provider", "audio_label",
        "sync_label", "sync_label_binary", "negative_type",
        "source_folder", "voice_id_or_name",
    )


def test_sync_label_binary_mapping_is_stable():
    from src.data.build_lipsync_pairs import SYNC_LABEL_BINARY

    assert SYNC_LABEL_BINARY == {"sync": 0, "async": 1}


def test_negative_types_tuple_is_exact():
    from src.data.build_lipsync_pairs import NEGATIVE_TYPES

    assert NEGATIVE_TYPES == (
        "generated_same_transcript",
        "mismatched_original",
        "mismatched_generated",
    )


_FUSION_FIELDS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path",
    "audio_label", "audio_label_binary",
    "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _matched(vid: str, split: str = "train", src: str = "real") -> dict:
    return {
        "sample_id": vid, "source_video_id": vid, "split": split,
        "media_type": "pair", "source_folder": src, "provider": "original",
        "voice_id_or_name": "",
        "audio_path": f"data/audio_wav/{src}/{vid}.wav",
        "video_path": f"data/raw/{src}/{vid}.mp4",
        "audio_feature_path": f"data/features/audio_wav2vec2/{vid}.npy",
        "lip_feature_path": f"data/features/lips/{vid}.npz",
        "audio_label": "bonafide", "audio_label_binary": "0",
        "video_label": "real", "video_label_binary": "0",
        "pair_label": "matched_bonafide", "pair_label_binary": "0",
    }


def _generated(vid: str, provider: str, voice: str, split: str = "train", src: str = "real") -> dict:
    sid = f"{provider}__{vid}__voice-{voice}"
    return {
        "sample_id": sid, "source_video_id": vid, "split": split,
        "media_type": "pair", "source_folder": src, "provider": provider,
        "voice_id_or_name": voice,
        "audio_path": f"data/tts_audio/{provider}/{src}/{sid}.mp3",
        "video_path": f"data/raw/{src}/{vid}.mp4",
        "audio_feature_path": f"data/features/audio_wav2vec2/{sid}.npy",
        "lip_feature_path": f"data/features/lips/{vid}.npz",
        "audio_label": "spoof", "audio_label_binary": "1",
        "video_label": "real", "video_label_binary": "0",
        "pair_label": "generated_same_transcript", "pair_label_binary": "1",
    }


def test_positives_are_emitted_once_per_matched_bonafide():
    from src.data.build_lipsync_pairs import build_pairs

    rows = [_matched("A"), _matched("B"), _generated("A", "elevenlabs", "v1")]

    pairs = build_pairs(rows, negatives_per_positive=0, splits=("train",), seed=42)

    positives = [p for p in pairs if p["sync_label"] == "sync"]
    assert {p["source_video_id"] for p in positives} == {"A", "B"}
    for p in positives:
        assert p["sync_label_binary"] == "0"
        assert p["negative_type"] == ""
        assert p["audio_provider"] == "original"
        assert p["audio_label"] == "bonafide"
        assert p["lip_feature_path"].endswith(".npz")
        assert p["audio_sample_id"] == p["source_video_id"]


def test_negatives_never_reuse_same_source_video_id():
    from src.data.build_lipsync_pairs import build_pairs

    rows = [
        _matched("A"), _matched("B"), _matched("C"),
        _generated("A", "elevenlabs", "v1"),
        _generated("B", "google_tts", "en-US-A"),
        _generated("C", "openai_tts", "alloy"),
    ]

    pairs = build_pairs(rows, negatives_per_positive=2, splits=("train",), seed=42)

    for p in pairs:
        if p["negative_type"] in ("mismatched_original", "mismatched_generated"):
            assert p["source_video_id"] != _origin_of(p["audio_sample_id"])


def _origin_of(sample_id: str) -> str:
    # bonafide sample_ids are bare video_ids; generated ids are provider__vid__voice-...
    if "__" not in sample_id:
        return sample_id
    _, vid, *_ = sample_id.split("__")
    return vid


def test_build_pairs_is_deterministic_under_seed():
    from src.data.build_lipsync_pairs import build_pairs

    rows = [_matched(f"V{i}") for i in range(6)] + [
        _generated(f"V{i}", "elevenlabs", "v1") for i in range(6)
    ]

    a = build_pairs(rows, negatives_per_positive=2, splits=("train",), seed=42)
    b = build_pairs(rows, negatives_per_positive=2, splits=("train",), seed=42)

    assert [r["pair_id"] for r in a] == [r["pair_id"] for r in b]
    assert [r["audio_sample_id"] for r in a] == [r["audio_sample_id"] for r in b]


def test_test_split_is_excluded_by_default():
    from src.data.build_lipsync_pairs import build_pairs

    rows = [_matched("A", split="train"), _matched("B", split="test")]

    pairs = build_pairs(rows, negatives_per_positive=0, seed=42)

    assert {p["split"] for p in pairs} == {"train"}
    assert all(p["source_video_id"] != "B" for p in pairs)


def test_negatives_stay_within_the_same_split():
    from src.data.build_lipsync_pairs import build_pairs

    rows = [
        _matched("A", split="train"), _matched("B", split="train"),
        _matched("C", split="val"),  _matched("D", split="val"),
        _generated("A", "elevenlabs", "v1", split="train"),
        _generated("C", "google_tts", "en-US-A", split="val"),
    ]

    pairs = build_pairs(rows, negatives_per_positive=1, splits=("train", "val"), seed=42)

    by_split: dict[str, set[str]] = {}
    for p in pairs:
        by_split.setdefault(p["split"], set()).add(p["audio_sample_id"])
    train_negatives = {p["audio_sample_id"] for p in pairs
                       if p["split"] == "train" and p["negative_type"] != ""}
    val_negatives = {p["audio_sample_id"] for p in pairs
                     if p["split"] == "val" and p["negative_type"] != ""}
    assert train_negatives.isdisjoint({"C", "D"})
    assert val_negatives.isdisjoint({"A", "B"})


def test_write_pair_manifest_uses_schema_header(tmp_path):
    from src.data.build_lipsync_pairs import (
        LIPSYNC_MANIFEST_SCHEMA,
        write_pair_manifest,
    )

    row = {c: f"v-{c}" for c in LIPSYNC_MANIFEST_SCHEMA}
    out = tmp_path / "m.csv"
    write_pair_manifest([row], out)

    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(LIPSYNC_MANIFEST_SCHEMA)
        got = list(reader)
    assert got == [row]


def test_write_pair_manifest_rejects_unknown_columns(tmp_path):
    from src.data.build_lipsync_pairs import write_pair_manifest

    with pytest.raises(ValueError, match="unknown column"):
        write_pair_manifest([{"pair_id": "x", "not_a_column": "?"}], tmp_path / "m.csv")


def test_write_provenance_records_counts(tmp_path):
    from src.data.build_lipsync_pairs import build_pairs, write_provenance

    rows = [_matched("A"), _matched("B"), _generated("A", "elevenlabs", "v1")]
    pairs = build_pairs(rows, negatives_per_positive=1, splits=("train",), seed=42)

    out = tmp_path / "prov.json"
    write_provenance(
        pairs, out,
        source_manifest=tmp_path / "src.csv",
        negatives_per_positive=1, seed=42,
    )

    data = json.loads(out.read_text())
    assert data["seed"] == 42
    assert data["negatives_per_positive"] == 1
    assert data["total"] == len(pairs)
    assert set(data["by_split"]) <= {"train"}
    assert "sync" in data["by_sync_label"]
    assert "async" in data["by_sync_label"]
    assert "" in data["by_negative_type"]  # positives
