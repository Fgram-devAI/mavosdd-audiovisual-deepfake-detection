"""Post-ingestion validation reports discrepancies; never mutates state."""
import csv

from src.common import CAPS, LABEL_MAP
from src.data.download_subset import validate_manifest


HEADER = ["video_id", "relative_path", "source_folder", "binary_label",
          "duration_s", "fps", "n_frames"]


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r)


def _make_row(video_id, source_folder, raw_dir, label=None, duration=4.0, fps=25.0, frames=100):
    rel = raw_dir / source_folder / f"{video_id}.mp4"
    rel.parent.mkdir(parents=True, exist_ok=True)
    rel.write_bytes(b"x")  # validation only checks existence, not validity
    return (video_id, str(rel), source_folder,
            LABEL_MAP[source_folder] if label is None else label,
            duration, fps, frames)


def _full_valid_manifest(redirect_data_root):
    rows = []
    raw_dir = redirect_data_root["raw_dir"]
    for cls, n in CAPS.items():
        for i in range(n):
            rows.append(_make_row(f"{cls}_{i:04d}", cls, raw_dir))
    _write_manifest(redirect_data_root["manifest"], rows)


def test_full_valid_manifest_has_no_issues(redirect_data_root):
    _full_valid_manifest(redirect_data_root)
    issues = validate_manifest()
    assert issues == []


def test_row_count_short_reports(redirect_data_root):
    raw_dir = redirect_data_root["raw_dir"]
    _write_manifest(redirect_data_root["manifest"],
                    [_make_row("r0", "real", raw_dir)])
    issues = validate_manifest()
    assert any("row count" in s.lower() for s in issues)
    assert any("real" in s.lower() and "under cap" in s.lower() for s in issues)


def test_duplicate_id_reports(redirect_data_root):
    raw_dir = redirect_data_root["raw_dir"]
    _full_valid_manifest(redirect_data_root)
    with redirect_data_root["manifest"].open("a", newline="") as f:
        csv.writer(f).writerow(_make_row("real_0000", "real", raw_dir))
    issues = validate_manifest()
    assert any("duplicate" in s.lower() for s in issues)


def test_missing_file_reports(redirect_data_root):
    _full_valid_manifest(redirect_data_root)
    (redirect_data_root["raw_dir"] / "real" / "real_0000.mp4").unlink()
    issues = validate_manifest()
    assert any("missing file" in s.lower() for s in issues)


def test_label_mismatch_reports(redirect_data_root):
    raw_dir = redirect_data_root["raw_dir"]
    _full_valid_manifest(redirect_data_root)
    # Overwrite manifest with a single wrong-label row to keep the assertion focused.
    bad = _make_row("bad_label", "real", raw_dir, label=1)
    with redirect_data_root["manifest"].open("a", newline="") as f:
        csv.writer(f).writerow(bad)
    issues = validate_manifest()
    assert any("label" in s.lower() for s in issues)


def test_non_positive_probe_reports(redirect_data_root):
    raw_dir = redirect_data_root["raw_dir"]
    _write_manifest(redirect_data_root["manifest"], [
        _make_row("z", "real", raw_dir, duration=0, fps=0, frames=0),
    ])
    issues = validate_manifest()
    assert any("probe" in s.lower() or "non-positive" in s.lower() for s in issues)


def test_quarantine_row_without_file_reports(redirect_data_root):
    _full_valid_manifest(redirect_data_root)
    qlog = redirect_data_root["quarantine_log"]
    qlog.parent.mkdir(parents=True, exist_ok=True)
    with qlog.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "source_folder", "reason"])
        w.writerow(["ghost", "real", "unreadable"])
    issues = validate_manifest()
    assert any("quarantine" in s.lower() and "ghost" in s for s in issues)


def test_invalid_quarantine_reason_reports(redirect_data_root):
    _full_valid_manifest(redirect_data_root)
    qdir = redirect_data_root["quarantine_dir"] / "real"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "x.mp4").write_bytes(b"x")
    qlog = redirect_data_root["quarantine_log"]
    qlog.parent.mkdir(parents=True, exist_ok=True)
    with qlog.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "source_folder", "reason"])
        w.writerow(["x", "real", "weird_reason"])
    issues = validate_manifest()
    assert any("reason" in s.lower() for s in issues)
