"""Configurable frozen audio-encoder backends.

Each backend wraps a Hugging Face encoder behind a tiny, uniform interface:
``load(device)`` instantiates and freezes the model, ``encode(wave)`` returns
a ``(time, output_dim)`` numpy array. No mean pooling — temporal aggregation
is the model head's job.

Unit tests mock the HF classes; nothing on this module's import path should
download a pretrained model.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    HubertModel,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
)


class AudioEmbeddingBackend:
    """Base class. Subclasses set class attributes and implement load/encode."""

    name: ClassVar[str] = ""
    model_id: ClassVar[str] = ""
    sample_rate: ClassVar[int] = 16000
    output_dim: ClassVar[int] = 0

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._processor = None
        self._model = None

    def load(self, device: torch.device) -> None:
        raise NotImplementedError

    def encode(self, wave: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Wav2Vec2Backend(AudioEmbeddingBackend):
    name = "wav2vec2"
    model_id = "facebook/wav2vec2-base-960h"
    sample_rate = 16000
    output_dim = 768

    def load(self, device: torch.device) -> None:
        self._device = device
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self._model = Wav2Vec2Model.from_pretrained(self.model_id).to(device).eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, wave: np.ndarray) -> np.ndarray:
        inputs = self._processor(wave, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(self._device)
        hidden = self._model(input_values).last_hidden_state.squeeze(0)
        return hidden.cpu().numpy()


BACKEND_REGISTRY: dict[str, type[AudioEmbeddingBackend]] = {
    Wav2Vec2Backend.name: Wav2Vec2Backend,
}


def list_backends() -> list[str]:
    return sorted(BACKEND_REGISTRY)


def load_backend(name: str, device: torch.device) -> AudioEmbeddingBackend:
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown audio backend: {name!r}. Registered: {list_backends()}"
        )
    backend = cls()
    backend.load(device)
    return backend
