"""Tests for scripts/spike_avhubert.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _install_scripts_on_path():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def test_collect_environment_reports_python_and_torch_versions():
    _install_scripts_on_path()
    from scripts import spike_avhubert as s

    env = s.collect_environment()
    for key in ("python", "torch", "platform"):
        assert key in env
        assert isinstance(env[key], str) and env[key]


def test_run_spike_writes_blocker_json_and_md_when_backend_fails(tmp_path):
    _install_scripts_on_path()
    from scripts import spike_avhubert as s

    out_json = tmp_path / "task0.json"
    out_md = tmp_path / "task0.md"

    def _fail_load(*args, **kwargs):
        raise ImportError("fairseq is not importable in this env")

    with patch("scripts.spike_avhubert.load_avhubert", side_effect=_fail_load):
        rc = s.run_spike(
            checkpoint=Path("/nonexistent"),
            sample_pair_id="pos__dummy",
            out_json=out_json,
            out_md=out_md,
        )
    assert rc == 2
    payload = json.loads(out_json.read_text())
    assert payload["status"] == "blocked"
    assert "fairseq is not importable" in payload["blocker_trace"]
    assert "environment" in payload
    assert out_md.exists()
    md_text = out_md.read_text()
    assert "blocked" in md_text.lower()


def test_run_spike_writes_pass_json_with_shapes_when_backend_succeeds(tmp_path):
    _install_scripts_on_path()
    from scripts import spike_avhubert as s

    out_json = tmp_path / "task0.json"
    out_md = tmp_path / "task0.md"

    fake_adapter = MagicMock()
    fake_adapter.encode_visual.return_value = _FakeArray((10, 768))
    fake_adapter.encode_audio.return_value = _FakeArray((100, 768))
    fake_adapter.checkpoint_sha256 = "deadbeef" * 8

    with patch("scripts.spike_avhubert.load_avhubert", return_value=fake_adapter), \
         patch("scripts.spike_avhubert.load_sample_inputs", return_value=("pair_x", object(), object())):
        rc = s.run_spike(
            checkpoint=Path("/nonexistent"),
            sample_pair_id="pair_x",
            out_json=out_json,
            out_md=out_md,
        )
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["status"] == "passed"
    assert payload["input_pair_id"] == "pair_x"
    assert payload["visual_output_shape"] == [10, 768]
    assert payload["audio_output_shape"] == [100, 768]
    assert payload["checkpoint_sha256"] == "deadbeef" * 8
    assert "runtime_seconds" in payload


class _FakeArray:
    def __init__(self, shape):
        self.shape = shape


def test_main_exits_two_on_blocker(tmp_path, monkeypatch):
    _install_scripts_on_path()
    from scripts import spike_avhubert as s

    monkeypatch.setattr(s, "AVHUBERT_CKPT_PATH", tmp_path / "avhubert.pt")
    monkeypatch.setattr(s, "OUT_JSON", tmp_path / "task0.json")
    monkeypatch.setattr(s, "OUT_MD", tmp_path / "task0.md")

    def _fail_load(*args, **kwargs):
        raise RuntimeError("blocked")

    monkeypatch.setattr(s, "load_avhubert", _fail_load)
    rc = s.main([])
    assert rc == 2
