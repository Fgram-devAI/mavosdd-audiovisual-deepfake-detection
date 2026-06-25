"""Tests for src/predict.py — single-video real/fake scoring CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import predict


class TestArgparse:
    def test_missing_video_exits_nonzero(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            predict.main(["--checkpoint", str(tmp_path / "x.pt")])
        assert exc.value.code != 0

    def test_missing_checkpoint_exits_nonzero(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            predict.main(["--video", str(tmp_path / "x.mp4")])
        assert exc.value.code != 0

    def test_parser_has_all_spec_flags(self):
        parser = predict._build_parser()
        dests = {a.dest for a in parser._actions}
        assert {"video", "checkpoint", "device", "json", "threshold",
                "no_codec_match"}.issubset(dests)

    def test_threshold_default_is_half(self):
        parser = predict._build_parser()
        ns = parser.parse_args(["--video", "v.mp4", "--checkpoint", "c.pt"])
        assert ns.threshold == 0.5
        assert ns.json is False
        assert ns.no_codec_match is False
        assert ns.device == "auto"
