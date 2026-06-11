"""Resume state: counts + done_ids from manifest, quarantined_ids from quarantine_log."""
import csv

from src.data.download_subset import load_existing_state


def _seed_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "relative_path", "source_folder", "binary_label",
                    "duration_s", "fps", "n_frames"])
        for r in rows:
            w.writerow(r)


def _seed_quarantine(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "source_folder", "reason"])
        for r in rows:
            w.writerow(r)


def test_empty_state_when_no_csvs(redirect_data_root):
    counts, done, quarantined = load_existing_state()
    assert counts == {"real": 0, "echomimic": 0, "memo": 0}
    assert done == set()
    assert quarantined == set()


def test_counts_from_manifest_rows(redirect_data_root):
    _seed_manifest(redirect_data_root["manifest"], [
        ("a", "data/raw/real/a.mp4", "real", 0, 4.0, 25.0, 100),
        ("b", "data/raw/real/b.mp4", "real", 0, 4.0, 25.0, 100),
        ("c", "data/raw/memo/c.mp4", "memo", 1, 4.0, 25.0, 100),
    ])
    counts, done, quarantined = load_existing_state()
    assert counts == {"real": 2, "echomimic": 0, "memo": 1}
    assert done == {"a", "b", "c"}
    assert quarantined == set()


def test_quarantined_ids_from_log(redirect_data_root):
    _seed_quarantine(redirect_data_root["quarantine_log"], [
        ("x", "real", "unreadable"),
        ("y", "memo", "no_audio_stream"),
    ])
    counts, done, quarantined = load_existing_state()
    assert counts == {"real": 0, "echomimic": 0, "memo": 0}
    assert done == set()
    assert quarantined == {"x", "y"}


def test_done_and_quarantined_coexist(redirect_data_root):
    _seed_manifest(redirect_data_root["manifest"], [
        ("a", "data/raw/real/a.mp4", "real", 0, 4.0, 25.0, 100),
    ])
    _seed_quarantine(redirect_data_root["quarantine_log"], [
        ("b", "real", "no_audio_stream"),
    ])
    counts, done, quarantined = load_existing_state()
    assert counts == {"real": 1, "echomimic": 0, "memo": 0}
    assert done == {"a"}
    assert quarantined == {"b"}
