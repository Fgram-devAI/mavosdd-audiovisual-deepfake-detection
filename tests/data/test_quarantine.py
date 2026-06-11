"""Quarantine flow: move file to data/quarantine/<cls>/<id>.mp4 + append log row."""
import csv

from src.data.download_subset import quarantine_file


def test_quarantine_moves_file_and_logs_reason(redirect_data_root, make_corrupt_mp4):
    src_path = make_corrupt_mp4("v1.mp4")
    quarantine_file(src_path, video_id="v1", source_folder="real", reason="unreadable")

    target = redirect_data_root["quarantine_dir"] / "real" / "v1.mp4"
    assert target.exists()
    assert not src_path.exists()

    with redirect_data_root["quarantine_log"].open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"video_id": "v1", "source_folder": "real", "reason": "unreadable"}]


def test_quarantine_log_appends_header_only_once(redirect_data_root, make_corrupt_mp4):
    p1 = make_corrupt_mp4("a.mp4")
    p2 = make_corrupt_mp4("b.mp4")
    quarantine_file(p1, "a", "real", "unreadable")
    quarantine_file(p2, "b", "memo", "no_audio_stream")

    text = redirect_data_root["quarantine_log"].read_text().splitlines()
    assert text[0] == "video_id,source_folder,reason"
    assert text[1] == "a,real,unreadable"
    assert text[2] == "b,memo,no_audio_stream"
    assert len(text) == 3


def test_quarantine_creates_class_subdir(redirect_data_root, make_corrupt_mp4):
    src = make_corrupt_mp4("z.mp4")
    quarantine_file(src, "z", "echomimic", "zero_fps")
    assert (redirect_data_root["quarantine_dir"] / "echomimic" / "z.mp4").exists()
