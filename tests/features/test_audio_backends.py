"""Tests for src/features/audio_backends.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def test_list_backends_includes_wav2vec2():
    from src.features.audio_backends import list_backends

    names = list_backends()
    assert "wav2vec2" in names
    assert names == sorted(names)


def test_load_backend_rejects_unknown_name():
    from src.features.audio_backends import load_backend

    with pytest.raises(ValueError, match=r"^unknown audio backend:.*wav2vec2"):
        load_backend("definitely-not-a-backend", torch.device("cpu"))


def test_wav2vec2_backend_class_metadata():
    from src.features.audio_backends import Wav2Vec2Backend

    assert Wav2Vec2Backend.name == "wav2vec2"
    assert Wav2Vec2Backend.model_id == "facebook/wav2vec2-base-960h"
    assert Wav2Vec2Backend.sample_rate == 16000
    assert Wav2Vec2Backend.output_dim == 768


def test_load_backend_wav2vec2_instantiates_and_loads(monkeypatch):
    from src.features import audio_backends

    fake_processor = MagicMock(name="processor")
    fake_model = MagicMock(name="model")
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.parameters.return_value = []

    with patch.object(audio_backends, "Wav2Vec2Processor") as p_cls, \
         patch.object(audio_backends, "Wav2Vec2Model") as m_cls:
        p_cls.from_pretrained.return_value = fake_processor
        m_cls.from_pretrained.return_value = fake_model

        backend = audio_backends.load_backend("wav2vec2", torch.device("cpu"))

    assert isinstance(backend, audio_backends.Wav2Vec2Backend)
    p_cls.from_pretrained.assert_called_once_with("facebook/wav2vec2-base-960h")
    m_cls.from_pretrained.assert_called_once_with("facebook/wav2vec2-base-960h")
    fake_model.to.assert_called_once()
    fake_model.eval.assert_called_once()


def test_wav2vec2_encode_returns_time_by_dim_no_mean_pool():
    from src.features import audio_backends

    backend = audio_backends.Wav2Vec2Backend()
    # Inject mocks directly to bypass load() and avoid touching HF.
    backend._processor = MagicMock()
    backend._model = MagicMock()
    backend._device = torch.device("cpu")

    fake_input = MagicMock()
    fake_input.input_values = torch.zeros(1, 64000)
    backend._processor.return_value = fake_input

    fake_hidden = torch.randn(1, 199, 768)
    backend._model.return_value = MagicMock(last_hidden_state=fake_hidden)

    wave = np.zeros(64000, dtype=np.float32)
    out = backend.encode(wave)

    assert out.ndim == 2
    assert out.shape == (199, 768)
    assert out.dtype in (np.float16, np.float32)


def test_load_backend_returns_loaded_instance(monkeypatch):
    """Loaded backend must have its load() side-effects already applied."""
    from src.features import audio_backends

    calls: list[str] = []

    class FakeBackend(audio_backends.AudioEmbeddingBackend):
        name = "fake"
        model_id = "fake/fake"
        output_dim = 4

        def load(self, device):
            calls.append("load")
            self._device = device

        def encode(self, wave):
            return np.zeros((1, self.output_dim), dtype=np.float32)

    monkeypatch.setitem(audio_backends.BACKEND_REGISTRY, "fake", FakeBackend)

    backend = audio_backends.load_backend("fake", torch.device("cpu"))

    assert calls == ["load"]
    assert backend._device == torch.device("cpu")


def test_list_backends_includes_wavlm_and_hubert():
    from src.features.audio_backends import list_backends

    names = list_backends()
    assert "wav2vec2" in names
    assert "wavlm" in names
    assert "hubert" in names


def test_wavlm_backend_class_metadata():
    from src.features.audio_backends import WavLMBackend

    assert WavLMBackend.name == "wavlm"
    assert WavLMBackend.model_id == "microsoft/wavlm-base-plus"
    assert WavLMBackend.sample_rate == 16000
    assert WavLMBackend.output_dim == 768


def test_hubert_backend_class_metadata():
    from src.features.audio_backends import HubertBackend

    assert HubertBackend.name == "hubert"
    assert HubertBackend.model_id == "facebook/hubert-base-ls960"
    assert HubertBackend.sample_rate == 16000
    assert HubertBackend.output_dim == 768


def test_load_backend_wavlm_uses_auto_extractor_and_wavlm_model():
    from src.features import audio_backends

    fake_extractor = MagicMock(name="extractor")
    fake_model = MagicMock(name="model")
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.parameters.return_value = []

    with patch.object(audio_backends, "AutoFeatureExtractor") as e_cls, \
         patch.object(audio_backends, "WavLMModel") as m_cls:
        e_cls.from_pretrained.return_value = fake_extractor
        m_cls.from_pretrained.return_value = fake_model

        backend = audio_backends.load_backend("wavlm", torch.device("cpu"))

    assert isinstance(backend, audio_backends.WavLMBackend)
    e_cls.from_pretrained.assert_called_once_with("microsoft/wavlm-base-plus")
    m_cls.from_pretrained.assert_called_once_with("microsoft/wavlm-base-plus")


def test_load_backend_hubert_uses_auto_extractor_and_hubert_model():
    from src.features import audio_backends

    fake_extractor = MagicMock(name="extractor")
    fake_model = MagicMock(name="model")
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.parameters.return_value = []

    with patch.object(audio_backends, "AutoFeatureExtractor") as e_cls, \
         patch.object(audio_backends, "HubertModel") as m_cls:
        e_cls.from_pretrained.return_value = fake_extractor
        m_cls.from_pretrained.return_value = fake_model

        backend = audio_backends.load_backend("hubert", torch.device("cpu"))

    assert isinstance(backend, audio_backends.HubertBackend)
    e_cls.from_pretrained.assert_called_once_with("facebook/hubert-base-ls960")
    m_cls.from_pretrained.assert_called_once_with("facebook/hubert-base-ls960")


def test_wavlm_encode_returns_time_by_dim_no_mean_pool():
    from src.features import audio_backends

    backend = audio_backends.WavLMBackend()
    backend._processor = MagicMock()
    backend._model = MagicMock()
    backend._device = torch.device("cpu")

    fake_input = MagicMock()
    fake_input.input_values = torch.zeros(1, 64000)
    backend._processor.return_value = fake_input

    fake_hidden = torch.randn(1, 199, 768)
    backend._model.return_value = MagicMock(last_hidden_state=fake_hidden)

    out = backend.encode(np.zeros(64000, dtype=np.float32))

    assert out.shape == (199, 768)
    assert out.dtype in (np.float16, np.float32)


def test_hubert_encode_returns_time_by_dim_no_mean_pool():
    from src.features import audio_backends

    backend = audio_backends.HubertBackend()
    backend._processor = MagicMock()
    backend._model = MagicMock()
    backend._device = torch.device("cpu")

    fake_input = MagicMock()
    fake_input.input_values = torch.zeros(1, 64000)
    backend._processor.return_value = fake_input

    fake_hidden = torch.randn(1, 199, 768)
    backend._model.return_value = MagicMock(last_hidden_state=fake_hidden)

    out = backend.encode(np.zeros(64000, dtype=np.float32))

    assert out.shape == (199, 768)
    assert out.dtype in (np.float16, np.float32)
