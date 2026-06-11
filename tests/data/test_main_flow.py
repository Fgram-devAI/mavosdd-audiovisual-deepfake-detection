"""End-to-end ingestion against a mocked HF stream."""
import csv

from src.common import CAPS
import src.data.download_subset as ds


def _record(video_id: str, cls: str, payload: bytes, lang: str = "english") -> dict:
    return {
        "language": lang,
        "generation_method": cls,
        "file_name": f"english/{cls}/{video_id}.mp4",
        "video": payload,
    }


def _patch_stream(monkeypatch, records):
    def fake_loader(_id, split, streaming):
        assert split == "train"
        assert streaming is True
        return iter(records)
    monkeypatch.setattr(ds, "load_dataset", fake_loader)


def _patch_caps(monkeypatch, caps):
    monkeypatch.setattr(ds, "CAPS", caps, raising=True)
    from src import common
    monkeypatch.setattr(common, "CAPS", caps, raising=True)


def test_accepted_videos_land_in_manifest(monkeypatch, redirect_data_root,
                                          make_mp4_with_audio):
    _patch_caps(monkeypatch, {"real": 1, "echomimic": 1, "memo": 1})
    clean = make_mp4_with_audio("clean.mp4").read_bytes()
    _patch_stream(monkeypatch, [
        _record("r1", "real", clean),
        _record("e1", "echomimic", clean),
        _record("m1", "memo", clean),
    ])

    ds.main()

    with redirect_data_root["manifest"].open() as f:
        rows = list(csv.DictReader(f))
    assert {(r["video_id"], r["source_folder"], int(r["binary_label"])) for r in rows} == {
        ("r1", "real", 0), ("e1", "echomimic", 1), ("m1", "memo", 1),
    }
    for r in rows:
        assert float(r["duration_s"]) > 0
        assert float(r["fps"]) > 0
        assert int(r["n_frames"]) > 0


def test_no_audio_video_is_quarantined_with_reason(monkeypatch, redirect_data_root,
                                                   make_mp4_with_audio, make_mp4_without_audio):
    _patch_caps(monkeypatch, {"real": 1, "echomimic": 0, "memo": 0})
    clean = make_mp4_with_audio("ok.mp4").read_bytes()
    silent = make_mp4_without_audio("silent.mp4").read_bytes()
    _patch_stream(monkeypatch, [
        _record("bad", "real", silent),
        _record("good", "real", clean),
    ])

    ds.main()

    with redirect_data_root["quarantine_log"].open() as f:
        qrows = list(csv.DictReader(f))
    assert qrows == [{"video_id": "bad", "source_folder": "real", "reason": "no_audio_stream"}]
    assert (redirect_data_root["quarantine_dir"] / "real" / "bad.mp4").exists()
    assert not (redirect_data_root["raw_dir"] / "real" / "bad.mp4").exists()

    with redirect_data_root["manifest"].open() as f:
        mrows = list(csv.DictReader(f))
    assert [r["video_id"] for r in mrows] == ["good"]


def test_corrupt_video_is_quarantined_unreadable(monkeypatch, redirect_data_root,
                                                 make_mp4_with_audio):
    _patch_caps(monkeypatch, {"real": 1, "echomimic": 0, "memo": 0})
    clean = make_mp4_with_audio("ok.mp4").read_bytes()
    _patch_stream(monkeypatch, [
        _record("rot", "real", b"\x00bad"),
        _record("good", "real", clean),
    ])

    ds.main()

    with redirect_data_root["quarantine_log"].open() as f:
        qrows = list(csv.DictReader(f))
    assert len(qrows) == 1
    assert qrows[0]["video_id"] == "rot"
    assert qrows[0]["reason"] in {"unreadable", "no_frames", "zero_fps"}


def test_out_of_scope_records_are_skipped(monkeypatch, redirect_data_root,
                                          make_mp4_with_audio):
    _patch_caps(monkeypatch, {"real": 1, "echomimic": 0, "memo": 0})
    clean = make_mp4_with_audio("ok.mp4").read_bytes()
    _patch_stream(monkeypatch, [
        _record("sp", "real", clean, lang="spanish"),
        _record("sg", "stylegan", clean),     # out-of-scope generator
        _record("ok", "real", clean),
    ])

    ds.main()

    with redirect_data_root["manifest"].open() as f:
        rows = list(csv.DictReader(f))
    assert [r["video_id"] for r in rows] == ["ok"]
    assert not redirect_data_root["quarantine_log"].exists()


def test_stream_breaks_at_saturation(monkeypatch, redirect_data_root,
                                     make_mp4_with_audio):
    _patch_caps(monkeypatch, {"real": 1, "echomimic": 0, "memo": 0})
    clean = make_mp4_with_audio("ok.mp4").read_bytes()
    consumed = []

    def loader(_id, split, streaming):
        for r in [_record("a", "real", clean), _record("b", "real", clean),
                  _record("c", "real", clean)]:
            consumed.append(r["file_name"])
            yield r
    monkeypatch.setattr(ds, "load_dataset", loader)

    ds.main()
    # second record should never be consumed because cap saturated after first.
    assert consumed == ["english/real/a.mp4"]


def test_resume_skips_already_done_and_quarantined(monkeypatch, redirect_data_root,
                                                   make_mp4_with_audio):
    _patch_caps(monkeypatch, {"real": 2, "echomimic": 0, "memo": 0})
    clean = make_mp4_with_audio("ok.mp4").read_bytes()

    # Seed prior state: one done, one quarantined.
    _patch_stream(monkeypatch, [
        _record("done", "real", clean),
    ])
    ds.main()
    # Mark a fake quarantined id by hand to assert resume honors it.
    with redirect_data_root["quarantine_log"].open("a", newline="") as f:
        csv.writer(f).writerow(["bad", "real", "no_audio_stream"])
    # If the log didn't exist, add the header.
    text = redirect_data_root["quarantine_log"].read_text().splitlines()
    if text[0] != "video_id,source_folder,reason":
        redirect_data_root["quarantine_log"].write_text(
            "video_id,source_folder,reason\nbad,real,no_audio_stream\n"
        )

    # Re-run: stream offers `done` (should be skipped) and `bad` (should be skipped)
    # and `new` (should be accepted to fill the second cap slot).
    _patch_stream(monkeypatch, [
        _record("done", "real", clean),
        _record("bad", "real", clean),
        _record("new", "real", clean),
    ])
    ds.main()

    with redirect_data_root["manifest"].open() as f:
        rows = list(csv.DictReader(f))
    assert {r["video_id"] for r in rows} == {"done", "new"}


def test_re_run_on_complete_set_is_noop(monkeypatch, redirect_data_root,
                                       make_mp4_with_audio):
    _patch_caps(monkeypatch, {"real": 1, "echomimic": 0, "memo": 0})
    clean = make_mp4_with_audio("ok.mp4").read_bytes()
    _patch_stream(monkeypatch, [_record("a", "real", clean)])
    ds.main()

    before = redirect_data_root["manifest"].read_text()
    # Empty stream on re-run; nothing to add, but main() must not crash.
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: iter([]))
    ds.main()
    assert redirect_data_root["manifest"].read_text() == before
