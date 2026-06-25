"""Tests for src/predict.py — single-video real/fake scoring CLI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


class TestCheckpointValidator:
    def test_missing_state_dict_raises(self):
        ckpt = {
            "modality": "audio", "backend": "wav2vec2",
            "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6},
        }
        with pytest.raises(ValueError, match="state_dict"):
            predict._validate_checkpoint(ckpt)

    def test_missing_norm_stats_raises(self):
        ckpt = {
            "state_dict": {}, "modality": "audio", "backend": "wav2vec2",
            "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
        }
        with pytest.raises(ValueError, match="norm_stats"):
            predict._validate_checkpoint(ckpt)

    def test_bad_modality_raises(self):
        ckpt = {
            "state_dict": {}, "modality": "speech", "backend": "wav2vec2",
            "model_hparams": {"modality": "speech", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6},
        }
        with pytest.raises(ValueError, match="modality"):
            predict._validate_checkpoint(ckpt)

    def test_non_visual_with_none_backend_raises(self):
        ckpt = {
            "state_dict": {}, "modality": "audio", "backend": None,
            "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6},
        }
        with pytest.raises(ValueError, match="backend"):
            predict._validate_checkpoint(ckpt)

    def test_visual_with_none_backend_is_ok(self):
        ckpt = {
            "state_dict": {}, "modality": "visual", "backend": None,
            "model_hparams": {"modality": "visual", "emb": 128, "dropout": 0.3},
            "norm_stats": {"eps": 1e-6, "lips_mean": [0.0] * 84, "lips_std": [1.0] * 84},
        }
        predict._validate_checkpoint(ckpt)  # no raise


class TestCodecMatchSignal:
    def test_visual_modality_always_skips(self):
        ckpt = {"modality": "visual", "audio_dir": None}
        assert predict._should_codec_match(ckpt, True) is False

    def test_no_codec_match_flag_skips(self):
        ckpt = {"modality": "audio", "audio_dir": "data/features/audio_wav2vec2_codec"}
        assert predict._should_codec_match(ckpt, False) is False

    def test_non_codec_audio_dir_skips(self):
        ckpt = {"modality": "audio", "audio_dir": "data/features/audio_wav2vec2"}
        assert predict._should_codec_match(ckpt, True) is False

    def test_codec_audio_dir_enables(self):
        ckpt = {"modality": "audio", "audio_dir": "data/features/audio_wav2vec2_codec"}
        assert predict._should_codec_match(ckpt, True) is True

    def test_fusion_codec_audio_dir_enables(self):
        ckpt = {"modality": "fusion", "audio_dir": "data/features/audio_hubert_codec"}
        assert predict._should_codec_match(ckpt, True) is True

    def test_missing_audio_dir_skips(self):
        ckpt = {"modality": "audio"}
        assert predict._should_codec_match(ckpt, True) is False


class TestNormalize:
    def test_returns_input_when_mean_is_none(self):
        x = np.ones((3, 4), dtype=np.float32)
        out = predict._normalize(x, None, np.ones(4, dtype=np.float32), 1e-6)
        assert out is x

    def test_returns_input_when_std_is_none(self):
        x = np.ones((3, 4), dtype=np.float32)
        out = predict._normalize(x, np.zeros(4, dtype=np.float32), None, 1e-6)
        assert out is x

    def test_centers_and_scales(self):
        x = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        mean = np.array([4.0, 6.0], dtype=np.float32)
        std = np.array([2.0, 2.0], dtype=np.float32)
        out = predict._normalize(x, mean, std, 0.0)
        np.testing.assert_allclose(out, np.array([[-1.0, -1.0], [1.0, 1.0]]))

    def test_eps_prevents_divide_by_zero(self):
        x = np.zeros((1, 2), dtype=np.float32)
        out = predict._normalize(x, np.zeros(2, dtype=np.float32),
                                  np.zeros(2, dtype=np.float32), 1e-6)
        assert np.isfinite(out).all()


class TestRenderers:
    AUDIO_RESULT = {
        "video": "v.mp4", "checkpoint": "best_audio_wav2vec2.pt",
        "modality": "audio", "backend": "wav2vec2",
        "audio_window_s": 4.0, "codec_matched": True,
        "lip_frames_present": None, "lip_frames_total": None,
        "face_detected": None,
        "p_spoof": 0.0421, "threshold": 0.5, "label": "bonafide",
    }
    VISUAL_RESULT = {
        "video": "v.mp4", "checkpoint": "best_visual.pt",
        "modality": "visual", "backend": None,
        "audio_window_s": None, "codec_matched": None,
        "lip_frames_present": 18, "lip_frames_total": 20,
        "face_detected": True,
        "p_spoof": 0.71, "threshold": 0.5, "label": "spoof",
    }

    def test_json_round_trip(self):
        s = predict._render_json(self.AUDIO_RESULT)
        parsed = json.loads(s)
        assert parsed == self.AUDIO_RESULT

    def test_human_audio_omits_lip_fields(self):
        s = predict._render_human(self.AUDIO_RESULT)
        assert "lip_frames" not in s
        assert "face_detected" not in s
        assert "modality=audio" in s and "backend=wav2vec2" in s
        assert "p_spoof=0.0421" in s and "label=bonafide" in s

    def test_human_visual_omits_audio_fields(self):
        s = predict._render_human(self.VISUAL_RESULT)
        assert "backend=" not in s
        assert "audio_window" not in s
        assert "codec_matched" not in s
        assert "lip_frames_present=18/20" in s
        assert "face_detected=True" in s
        assert "label=spoof" in s


class TestReconstructNormStats:
    def test_audio_only(self):
        d = {
            "audio_mean": np.ones(768, dtype=np.float32),
            "audio_std": np.full(768, 2.0, dtype=np.float32),
            "eps": 1e-5,
        }
        out = predict._reconstruct_norm_stats(d)
        assert out["eps"] == 1e-5
        assert out["audio_mean"].shape == (768,) and out["audio_std"].shape == (768,)
        assert out["lips_mean"] is None and out["lips_std"] is None

    def test_eps_defaults_to_1e6(self):
        out = predict._reconstruct_norm_stats({})
        assert out["eps"] == 1e-6
        assert out["audio_mean"] is None and out["lips_mean"] is None

    def test_fusion_has_both(self):
        d = {
            "audio_mean": [0.0] * 768, "audio_std": [1.0] * 768,
            "lips_mean": [0.0] * 84, "lips_std": [1.0] * 84,
            "eps": 1e-6,
        }
        out = predict._reconstruct_norm_stats(d)
        assert out["audio_mean"].dtype == np.float32
        assert out["lips_mean"].dtype == np.float32
        assert out["audio_mean"].shape == (768,)
        assert out["lips_mean"].shape == (84,)


import torch
from src.models.late_fusion import LateFusionClassifier


# ---- helpers: build minimal CPU-runnable checkpoints ----

def _build_checkpoint(tmp_path, modality: str, *, codec_dir: bool = True):
    """Build a randomly-initialized LateFusionClassifier checkpoint for `modality`."""
    model = LateFusionClassifier(modality, emb=128, p=0.3)
    norm = {"eps": 1e-6}
    if modality != "visual":
        norm["audio_mean"] = np.zeros(768, dtype=np.float32)
        norm["audio_std"] = np.ones(768, dtype=np.float32)
    if modality != "audio":
        norm["lips_mean"] = np.zeros(84, dtype=np.float32)
        norm["lips_std"] = np.ones(84, dtype=np.float32)
    audio_dir = None
    if modality != "visual":
        audio_dir = str(tmp_path / ("audio_wav2vec2_codec" if codec_dir else "audio_wav2vec2"))
    ckpt = {
        "state_dict": model.state_dict(),
        "modality": modality,
        "backend": "wav2vec2" if modality != "visual" else None,
        "audio_dir": audio_dir,
        "model_hparams": {"modality": modality, "emb": 128, "dropout": 0.3},
        "norm_stats": norm,
    }
    ckpt_path = tmp_path / f"ckpt_{modality}.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


class _StubBackend:
    """Stand-in for audio_backends.AudioEmbeddingBackend; no HF download."""
    def __init__(self, t: int = 199, dim: int = 768):
        self._t, self._dim = t, dim
    def encode(self, wave):
        # deterministic small embedding
        rng = np.random.RandomState(0)
        return rng.randn(self._t, self._dim).astype(np.float32)


@pytest.fixture
def stub_backend_loader(monkeypatch):
    """Patch audio_backends.load_backend at source — predict.py uses module-qualified calls,
    so the patch is picked up via the imported audio_backends module."""
    def _load(name, device):
        return _StubBackend()
    monkeypatch.setattr("src.features.audio_backends.load_backend", _load, raising=True)


@pytest.fixture
def stub_extract_lips(monkeypatch):
    """Patch extract_lips.extract_one at source. Returns deterministic (20,84) feats + ones mask."""
    def _extract_one(video_path, mesh):
        feats = np.random.RandomState(1).randn(20, 84).astype(np.float32)
        mask = np.ones(20, dtype=np.float32)
        return feats, mask
    monkeypatch.setattr("src.features.extract_lips.extract_one", _extract_one, raising=True)


class TestPredictVideoAudio:
    def test_audio_returns_full_result_dict(
        self, tmp_path, make_mp4_with_audio, stub_backend_loader,
    ):
        # Audio modality: real ffmpeg demux is OK (cheap, ~4s of testsrc+sine)
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "audio", codec_dir=False)  # disable codec stage
        out = predict.predict_video(video, ckpt, device="cpu")
        for k in ("video", "checkpoint", "modality", "backend",
                  "audio_window_s", "codec_matched",
                  "lip_frames_present", "lip_frames_total", "face_detected",
                  "p_spoof", "threshold", "label"):
            assert k in out, f"missing key {k!r}"
        assert out["modality"] == "audio"
        assert out["backend"] == "wav2vec2"
        assert out["audio_window_s"] == 4.0
        assert out["codec_matched"] is False
        assert out["lip_frames_present"] is None
        assert out["face_detected"] is None
        assert 0.0 <= out["p_spoof"] <= 1.0
        assert out["label"] in {"bonafide", "spoof"}


class TestPredictVideoVisual:
    def test_visual_returns_full_result_dict(
        self, tmp_path, make_mp4_with_audio, stub_extract_lips,
    ):
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "visual")
        out = predict.predict_video(video, ckpt, device="cpu")
        assert out["modality"] == "visual"
        assert out["backend"] is None
        assert out["audio_window_s"] is None
        assert out["codec_matched"] is None
        assert out["lip_frames_total"] == 20
        assert out["lip_frames_present"] == 20  # stub returns full mask
        assert out["face_detected"] is True
        assert 0.0 <= out["p_spoof"] <= 1.0


class TestPredictVideoFusion:
    def test_fusion_codec_match_runs_when_audio_dir_endswith_codec(
        self, tmp_path, make_mp4_with_audio,
        stub_backend_loader, stub_extract_lips,
    ):
        # Real ffmpeg codec round-trip; fast on 4s audio.
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "fusion", codec_dir=True)
        out = predict.predict_video(video, ckpt, device="cpu")
        assert out["modality"] == "fusion"
        assert out["codec_matched"] is True
        assert 0.0 <= out["p_spoof"] <= 1.0

    def test_fusion_no_codec_match_when_audio_dir_lacks_codec(
        self, tmp_path, make_mp4_with_audio,
        stub_backend_loader, stub_extract_lips,
    ):
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "fusion", codec_dir=False)
        out = predict.predict_video(video, ckpt, device="cpu")
        assert out["codec_matched"] is False


class TestPredictVideoNoFace:
    def test_visual_noface_still_scores(
        self, tmp_path, make_mp4_4s_black_no_audio,
    ):
        # Real MediaPipe on the black fixture — reliably yields no landmarks.
        video = make_mp4_4s_black_no_audio()
        ckpt = _build_checkpoint(tmp_path, "visual")
        out = predict.predict_video(video, ckpt, device="cpu")
        assert out["face_detected"] is False
        assert out["lip_frames_present"] == 0
        assert 0.0 <= out["p_spoof"] <= 1.0


class TestPredictVideoBadCheckpoint:
    def test_missing_norm_stats_raises_valueerror(
        self, tmp_path, make_mp4_with_audio,
    ):
        video = make_mp4_with_audio(duration=4.0)
        ckpt_path = tmp_path / "bad.pt"
        torch.save({"state_dict": {}, "modality": "audio",
                    "backend": "wav2vec2",
                    "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3}},
                   ckpt_path)
        with pytest.raises(ValueError, match="norm_stats"):
            predict.predict_video(video, ckpt_path, device="cpu")


class TestPredictVideoCorruptMp4:
    def test_audio_modality_wraps_ffmpeg_error(
        self, tmp_path, make_corrupt_mp4, stub_backend_loader,
    ):
        # spec §7: corrupt mp4 in audio path → RuntimeError("could not read video/audio from {path}: …")
        video = make_corrupt_mp4()
        ckpt = _build_checkpoint(tmp_path, "audio", codec_dir=False)
        with pytest.raises(RuntimeError, match="could not read video/audio"):
            predict.predict_video(video, ckpt, device="cpu")


class TestCLI:
    def test_cli_default_emits_human_line(
        self, tmp_path, capsys, make_mp4_with_audio, stub_backend_loader,
    ):
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "audio", codec_dir=False)
        rc = predict.main([
            "--video", str(video), "--checkpoint", str(ckpt), "--device", "cpu",
        ])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert "modality=audio" in out
        assert "p_spoof=" in out

    def test_cli_json_emits_parseable_dict(
        self, tmp_path, capsys, make_mp4_with_audio, stub_backend_loader,
    ):
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "audio", codec_dir=False)
        rc = predict.main([
            "--video", str(video), "--checkpoint", str(ckpt),
            "--device", "cpu", "--json",
        ])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["modality"] == "audio"
        assert 0.0 <= parsed["p_spoof"] <= 1.0

    def test_cli_threshold_flips_label(
        self, tmp_path, capsys, make_mp4_with_audio, stub_backend_loader,
    ):
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "audio", codec_dir=False)
        # threshold 0.0 → every prob ≥ 0 → label spoof
        rc = predict.main([
            "--video", str(video), "--checkpoint", str(ckpt),
            "--device", "cpu", "--threshold", "0.0", "--json",
        ])
        assert rc == 0
        parsed_low = json.loads(capsys.readouterr().out.strip())
        assert parsed_low["label"] == "spoof"
        # threshold 1.5 (above any sigmoid output) → label bonafide
        rc = predict.main([
            "--video", str(video), "--checkpoint", str(ckpt),
            "--device", "cpu", "--threshold", "1.5", "--json",
        ])
        assert rc == 0
        parsed_high = json.loads(capsys.readouterr().out.strip())
        assert parsed_high["label"] == "bonafide"

    def test_cli_no_codec_match_flag_sets_field_false(
        self, tmp_path, capsys, make_mp4_with_audio, stub_backend_loader,
    ):
        video = make_mp4_with_audio(duration=4.0)
        ckpt = _build_checkpoint(tmp_path, "audio", codec_dir=True)  # _codec dir → default ON
        rc = predict.main([
            "--video", str(video), "--checkpoint", str(ckpt),
            "--device", "cpu", "--no-codec-match", "--json",
        ])
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out.strip())
        assert parsed["codec_matched"] is False

    def test_cli_bad_checkpoint_returns_one_with_stderr_message(
        self, tmp_path, capsys, make_mp4_with_audio,
    ):
        # Spec §3: bad checkpoint → clean error, non-zero exit (no raw traceback).
        video = make_mp4_with_audio(duration=4.0)
        bad_ckpt = tmp_path / "bad.pt"
        torch.save({"state_dict": {}, "modality": "audio",
                    "backend": "wav2vec2",
                    "model_hparams": {"modality": "audio", "emb": 128, "dropout": 0.3}},
                   bad_ckpt)
        rc = predict.main([
            "--video", str(video), "--checkpoint", str(bad_ckpt), "--device", "cpu",
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "norm_stats" in captured.err
        assert captured.out.strip() == ""  # no human/json line on error

    def test_cli_corrupt_mp4_returns_one_with_stderr_message(
        self, tmp_path, capsys, make_corrupt_mp4, stub_backend_loader,
    ):
        # Spec §7: corrupt mp4 → wrapped RuntimeError, exit 1, stderr message.
        video = make_corrupt_mp4()
        ckpt = _build_checkpoint(tmp_path, "audio", codec_dir=False)
        rc = predict.main([
            "--video", str(video), "--checkpoint", str(ckpt), "--device", "cpu",
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "could not read video/audio" in captured.err
