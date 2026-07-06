import csv
from pathlib import Path

import pytest


def test_summary_and_scores(monkeypatch, tmp_path):
    from scripts import score_generated_video_batch as m

    videos_dir = tmp_path / "vids"
    videos_dir.mkdir()
    (videos_dir / "a.mp4").write_bytes(b"")
    (videos_dir / "b.mp4").write_bytes(b"")

    audio_backend_dummy = {"backend": "wavlm_normalized"}
    sync_backend_dummy = {"backend": "syncnet"}
    video_av_backend_dummy = {"backend": "avhubert", "window_count": 25,
                              "window_policy": "center"}

    class DummyFusion:
        feature_columns = ("audio_fake_score", "video_av_fake_score", "sync_inconsistent_score")
        threshold = 0.5

        def predict_proba(self, X):
            import numpy as np
            return np.full(X.shape[0], 0.9, dtype=np.float32)

    def _fake_load(**kw):
        return audio_backend_dummy, sync_backend_dummy, video_av_backend_dummy, DummyFusion()

    def _fake_score(video, **kw):
        if video.name == "b.mp4":
            raise RuntimeError("boom")
        return {
            "audio_fake_score": 0.8,
            "sync_inconsistent_score": 0.2,
            "video_av_fake_score": 0.7,
        }

    monkeypatch.setattr(m, "load_all", _fake_load)
    monkeypatch.setattr(m, "extract_head_scores", _fake_score)

    csv_out = tmp_path / "scores.csv"
    md_out = tmp_path / "summary.md"
    rc = m.main([
        "--input-dir", str(videos_dir),
        "--batch-name", "unit_test_batch",
        "--audio-ckpt", str(tmp_path / "a.pt"),
        "--sync-ckpt", str(tmp_path / "s.pt"),
        "--video-av-ckpt", str(tmp_path / "v.pt"),
        "--fusion-ckpt", str(tmp_path / "f.pt"),
        "--out", str(csv_out),
        "--summary", str(md_out),
    ])
    assert rc == 0
    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r["video_path"].endswith("a.mp4")][0]
    failed = [r for r in rows if r["video_path"].endswith("b.mp4")][0]
    assert set(rows[0].keys()) == set(m.FIELDS)
    assert ok["status"] == "ok"
    assert ok["label_at_threshold"] == "fake"
    assert ok["batch_name"] == "unit_test_batch"
    assert failed["status"] == "failed"
    assert failed["final_fusion_score"] == ""
    assert failed["error"].startswith("RuntimeError:")

    text = md_out.read_text()
    for phrase in (
        "positive-only", "hit rate", "sync-consistency",
        "MAVOS-DD val", "unit_test_batch",
    ):
        assert phrase in text
