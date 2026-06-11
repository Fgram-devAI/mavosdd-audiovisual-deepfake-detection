"""cv2 probe must return a rejection reason or None for clean videos."""
from pathlib import Path

import pytest

from src.data.download_subset import has_audio_stream, probe_video


def test_clean_video_returns_no_reason(make_mp4_with_audio):
    path = make_mp4_with_audio()
    reason, duration, fps, n_frames = probe_video(path)
    assert reason is None
    assert duration > 0
    assert fps > 0
    assert n_frames > 0


def test_clean_no_audio_video_still_passes_cv2_probe(make_mp4_without_audio):
    path = make_mp4_without_audio()
    reason, _, _, _ = probe_video(path)
    assert reason is None  # cv2 cares about video frames only


def test_corrupt_file_is_unreadable(make_corrupt_mp4):
    path = make_corrupt_mp4()
    reason, duration, fps, n_frames = probe_video(path)
    assert reason in {"unreadable", "no_frames", "zero_fps"}
    # at least one of the metrics is non-positive
    assert duration == 0 or fps == 0 or n_frames == 0


def test_missing_file_is_unreadable(tmp_path: Path):
    reason, _, _, _ = probe_video(tmp_path / "nope.mp4")
    assert reason == "unreadable"


def test_video_with_audio_has_audio_stream(make_mp4_with_audio):
    assert has_audio_stream(make_mp4_with_audio()) is True


def test_video_without_audio_has_no_audio_stream(make_mp4_without_audio):
    assert has_audio_stream(make_mp4_without_audio()) is False


def test_corrupt_file_has_no_audio_stream(make_corrupt_mp4):
    assert has_audio_stream(make_corrupt_mp4()) is False


def test_missing_file_has_no_audio_stream(tmp_path):
    assert has_audio_stream(tmp_path / "nope.mp4") is False
