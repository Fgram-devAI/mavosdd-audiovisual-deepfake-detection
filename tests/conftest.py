"""Shared pytest fixtures for the ingestion test suite."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH; ingestion tests require it")


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def make_mp4_with_audio(ffmpeg_available, tmp_path: Path):
    def _make(name: str = "with_audio.mp4", duration: float = 1.0) -> Path:
        out = tmp_path / name
        _run_ffmpeg(
            [
                "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=64x64:rate=10",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                str(out),
            ]
        )
        return out
    return _make


@pytest.fixture
def make_mp4_without_audio(ffmpeg_available, tmp_path: Path):
    def _make(name: str = "no_audio.mp4", duration: float = 1.0) -> Path:
        out = tmp_path / name
        _run_ffmpeg(
            [
                "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=64x64:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                str(out),
            ]
        )
        return out
    return _make


@pytest.fixture
def make_corrupt_mp4(tmp_path: Path):
    def _make(name: str = "corrupt.mp4") -> Path:
        out = tmp_path / name
        out.write_bytes(b"not a valid mp4 \x00\x00\x00")
        return out
    return _make


@pytest.fixture
def redirect_data_root(monkeypatch, tmp_path: Path):
    """Point all src.common path constants at a tmp tree for the test."""
    from src import common

    data_root = tmp_path / "data"
    raw_dir = data_root / "raw"
    quar_dir = data_root / "quarantine"
    manifest = data_root / "manifest.csv"
    quar_log = data_root / "quarantine_log.csv"

    monkeypatch.setattr(common, "DATA_ROOT", data_root, raising=True)
    monkeypatch.setattr(common, "RAW_DIR", raw_dir, raising=True)
    monkeypatch.setattr(common, "QUARANTINE_DIR", quar_dir, raising=True)
    monkeypatch.setattr(common, "MANIFEST", manifest, raising=True)
    monkeypatch.setattr(common, "QUARANTINE_LOG", quar_log, raising=True)

    import src.data.download_subset as ds
    monkeypatch.setattr(ds, "RAW_DIR", raw_dir, raising=True)
    monkeypatch.setattr(ds, "QUARANTINE_DIR", quar_dir, raising=True)
    monkeypatch.setattr(ds, "MANIFEST", manifest, raising=True)
    monkeypatch.setattr(ds, "QUARANTINE_LOG", quar_log, raising=True)

    return {
        "data_root": data_root,
        "raw_dir": raw_dir,
        "quarantine_dir": quar_dir,
        "manifest": manifest,
        "quarantine_log": quar_log,
    }
