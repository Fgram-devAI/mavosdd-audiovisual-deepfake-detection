"""Tests for src/data/apply_voice_split.py — split-column remapper."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest


_VIS_FIELDS = [
    "sample_id", "source_video_id", "split", "media_type", "source_folder",
    "provider", "voice_id_or_name", "audio_path", "video_path",
    "audio_feature_path", "lip_feature_path",
    "audio_label", "audio_label_binary",
    "video_label", "video_label_binary",
    "pair_label", "pair_label_binary",
]


def _row(sample_id: str, *, split: str, pair_label_binary: str = "0",
         provider: str = "original") -> dict:
    blank = {k: "" for k in _VIS_FIELDS}
    blank.update({
        "sample_id": sample_id,
        "source_video_id": sample_id,
        "split": split,
        "media_type": "pair",
        "source_folder": "real",
        "provider": provider,
        "audio_path": f"data/audio_wav/{sample_id}.wav",
        "video_path": f"data/raw/{sample_id}.mp4",
        "audio_feature_path": f"data/features/audio_wav2vec2/{sample_id}.npy",
        "lip_feature_path": f"data/features/lips/{sample_id}.npz",
        "audio_label": "bonafide",
        "audio_label_binary": "0",
        "video_label": "real",
        "video_label_binary": "0",
        "pair_label": "matched_bonafide",
        "pair_label_binary": pair_label_binary,
    })
    return blank


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_VIS_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_apply_voice_split_overwrites_only_split_column(tmp_path):
    from src.data.apply_voice_split import apply_voice_split

    source = tmp_path / "audio_voice.csv"
    target = tmp_path / "visual.csv"
    out = tmp_path / "visual_voice.csv"

    # source carries the *voice-disjoint* split per sample_id (and empty pair_label_binary)
    _write(source, [
        _row("a", split="val",   pair_label_binary=""),
        _row("b", split="train", pair_label_binary=""),
        _row("c", split="train", pair_label_binary=""),
    ])
    # target carries the *old* split + the populated pair_label_binary
    _write(target, [
        _row("a", split="train", pair_label_binary="0"),
        _row("b", split="val",   pair_label_binary="1"),
        _row("c", split="train", pair_label_binary="0"),
    ])

    apply_voice_split(source, target, out)

    out_rows = _read(out)
    by_id = {r["sample_id"]: r for r in out_rows}
    # split is rewritten from source
    assert by_id["a"]["split"] == "val"
    assert by_id["b"]["split"] == "train"
    assert by_id["c"]["split"] == "train"
    # pair_label_binary is preserved from target (NOT taken from source)
    assert by_id["a"]["pair_label_binary"] == "0"
    assert by_id["b"]["pair_label_binary"] == "1"
    assert by_id["c"]["pair_label_binary"] == "0"
    # column order preserved (writer must use target's header verbatim)
    assert out_rows[0].keys() == by_id["a"].keys()
    assert list(by_id["a"].keys())[:3] == ["sample_id", "source_video_id", "split"]


def test_apply_voice_split_preserves_all_non_split_columns_byte_identical(tmp_path):
    from src.data.apply_voice_split import apply_voice_split

    source = tmp_path / "audio_voice.csv"
    target = tmp_path / "visual.csv"
    out = tmp_path / "visual_voice.csv"

    _write(source, [_row("a", split="val")])
    target_row = _row("a", split="train", pair_label_binary="1", provider="elevenlabs")
    _write(target, [target_row])

    apply_voice_split(source, target, out)

    [out_row] = _read(out)
    for col in _VIS_FIELDS:
        if col == "split":
            continue
        assert out_row[col] == target_row[col], f"column {col} changed"
    assert out_row["split"] == "val"  # rewritten


def test_apply_voice_split_raises_when_target_sample_id_missing_from_source(tmp_path):
    from src.data.apply_voice_split import apply_voice_split

    source = tmp_path / "audio_voice.csv"
    target = tmp_path / "visual.csv"
    out = tmp_path / "visual_voice.csv"

    _write(source, [_row("a", split="train")])
    _write(target, [_row("a", split="train"), _row("b", split="val")])

    with pytest.raises(ValueError, match="missing.*source"):
        apply_voice_split(source, target, out)
    assert not out.exists(), "no output file should be written on validation failure"


def test_apply_voice_split_cli_writes_output(tmp_path):
    from src.data.apply_voice_split import main as cli_main

    source = tmp_path / "audio_voice.csv"
    target = tmp_path / "visual.csv"
    out = tmp_path / "visual_voice.csv"
    _write(source, [_row("a", split="val"), _row("b", split="train")])
    _write(target, [_row("a", split="train", pair_label_binary="0"),
                    _row("b", split="train", pair_label_binary="1")])

    rc = cli_main([
        "--source", str(source),
        "--target", str(target),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    rows = _read(out)
    by_id = {r["sample_id"]: r for r in rows}
    assert by_id["a"]["split"] == "val"
    assert by_id["b"]["split"] == "train"
    assert by_id["a"]["pair_label_binary"] == "0"
    assert by_id["b"]["pair_label_binary"] == "1"
