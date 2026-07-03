"""Tests for src/features/mouth_crop_extract.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


def test_specs_are_backend_specific():
    from src.features.mouth_crop_extract import SYNCNET_SPEC, AVHUBERT_SPEC

    assert SYNCNET_SPEC.stack_size == 5
    assert SYNCNET_SPEC.target_size == (96, 48)  # cv2 (W, H) → 48x96 crops
    assert SYNCNET_SPEC.color == "bgr"
    assert AVHUBERT_SPEC.stack_size == 25
    assert AVHUBERT_SPEC.target_size == (88, 88)
    assert AVHUBERT_SPEC.color == "gray"


def test_extract_returns_dtype_float16_and_expected_rank():
    from src.features.mouth_crop_extract import extract_mouth_crops, SYNCNET_SPEC

    with patch("src.features.mouth_crop_extract._open_video") as mo, \
         patch("src.features.mouth_crop_extract._detect_mouth_bbox") as md:
        frames = np.zeros((50, 480, 640, 3), dtype=np.uint8)
        mo.return_value = (frames, 25.0)
        md.return_value = (200, 200, 320, 300)
        out = extract_mouth_crops(Path("/dev/null"), SYNCNET_SPEC)
    assert out.dtype == np.float16
    assert out.ndim == 5
    assert out.shape[1] == SYNCNET_SPEC.stack_size


def test_extract_raises_mouth_detection_error_when_no_face(tmp_path):
    from src.features.mouth_crop_extract import (
        extract_mouth_crops, MouthDetectionError, SYNCNET_SPEC,
    )

    with patch("src.features.mouth_crop_extract._open_video") as mo, \
         patch("src.features.mouth_crop_extract._detect_mouth_bbox", return_value=None):
        mo.return_value = (np.zeros((50, 480, 640, 3), dtype=np.uint8), 25.0)
        with pytest.raises(MouthDetectionError) as exc:
            extract_mouth_crops(Path("/dev/null"), SYNCNET_SPEC)
    assert exc.value.stage == "face_detect"


def test_extract_gray_spec_produces_single_channel():
    from src.features.mouth_crop_extract import extract_mouth_crops, AVHUBERT_SPEC

    with patch("src.features.mouth_crop_extract._open_video") as mo, \
         patch("src.features.mouth_crop_extract._detect_mouth_bbox", return_value=(200, 200, 320, 300)):
        mo.return_value = (np.zeros((100, 480, 640, 3), dtype=np.uint8), 25.0)
        out = extract_mouth_crops(Path("/dev/null"), AVHUBERT_SPEC)
    # (N_windows, stack_size, C, H, W); gray => C == 1
    assert out.shape[2] == 1
    assert out.shape[3:] == AVHUBERT_SPEC.target_size


def test_extract_bgr_spec_produces_three_channels():
    from src.features.mouth_crop_extract import extract_mouth_crops, SYNCNET_SPEC

    with patch("src.features.mouth_crop_extract._open_video") as mo, \
         patch("src.features.mouth_crop_extract._detect_mouth_bbox", return_value=(200, 200, 320, 300)):
        mo.return_value = (np.zeros((100, 480, 640, 3), dtype=np.uint8), 25.0)
        out = extract_mouth_crops(Path("/dev/null"), SYNCNET_SPEC)
    assert out.shape[2] == 3
