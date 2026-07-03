"""Backend-agnostic mouth-crop extractor for pretrained AV-sync backends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class MouthDetectionError(RuntimeError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(f"[{stage}] {reason}")
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class MouthCropSpec:
    target_size: tuple[int, int]
    fps: int
    window_seconds: float
    stack_size: int
    color: str  # "rgb" | "bgr" | "gray"


SYNCNET_SPEC = MouthCropSpec(
    # cv2.resize takes (width, height), so this produces 48-row x 96-col crops
    # matching Wav2Lip's SyncNet_color contract (B, 15, 48, 96).
    target_size=(96, 48),
    fps=25,
    window_seconds=0.2,
    stack_size=5,
    color="bgr",
)
AVHUBERT_SPEC = MouthCropSpec(
    target_size=(88, 88),
    fps=25,
    window_seconds=1.0,
    stack_size=25,
    color="gray",
)


def _open_video(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MouthDetectionError("video_decode", f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise MouthDetectionError("video_decode", f"no frames in {path}")
    return np.stack(frames), fps


_FACE_CASCADE: "cv2.CascadeClassifier | None" = None


def _get_face_cascade() -> "cv2.CascadeClassifier":
    """Load and cache the OpenCV Haar frontal-face cascade.

    Uses the copy bundled with cv2, so no additional model download is needed
    and no OpenGL/Metal context is ever created — this is the CPU-only path
    that replaces the MediaPipe FaceMesh detector on macOS.
    """
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise MouthDetectionError("cascade_load", f"failed to load {path}")
        _FACE_CASCADE = cascade
    return _FACE_CASCADE


def _detect_mouth_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) mouth bbox or None if no face is detected.

    CPU-only Haar cascade face detection; the mouth region is estimated as a
    square patch centered horizontally on the face and vertically at ~78% of
    the face height, sized to ~60% of the face width. This preserves the
    bbox contract used by ``extract_mouth_crops`` while avoiding MediaPipe's
    OpenGL requirement (which crashes on headless macOS).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Histogram equalization materially improves Haar recall on low-contrast
    # or under-exposed talking-head clips (from ~15% to ~90% frame-hit rate
    # on the smoke sample) at negligible CPU cost.
    gray = cv2.equalizeHist(gray)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    cx = fx + fw // 2
    cy = fy + int(fh * 0.78)
    half = int(fw * 0.30)
    h, w = frame.shape[:2]
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _detect_reference_bbox(frames: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a stable mouth bbox by scanning a handful of candidate frames.

    Haar detection can miss individual frames (motion blur, head turn), so
    sample at several positions instead of trusting a single reference. The
    middle frame is tried first (matches the previous MediaPipe contract);
    remaining candidates are quartile/decile positions.
    """
    n = frames.shape[0]
    if n == 0:
        return None
    order = [
        n // 2,
        n // 4,
        3 * n // 4,
        n // 10,
        9 * n // 10,
        0,
        n - 1,
    ]
    seen: set[int] = set()
    for idx in order:
        if not (0 <= idx < n) or idx in seen:
            continue
        seen.add(idx)
        bbox = _detect_mouth_bbox(frames[idx])
        if bbox is not None:
            return bbox
    return None


def _resample_frames(frames: np.ndarray, src_fps: float, tgt_fps: int) -> np.ndarray:
    if src_fps <= 0:
        return frames
    n_src = frames.shape[0]
    n_tgt = int(round(n_src * tgt_fps / src_fps))
    if n_tgt <= 0:
        return frames
    idx = np.linspace(0, n_src - 1, num=n_tgt).round().astype(int)
    return frames[idx]


def _to_target(crop: np.ndarray, spec: MouthCropSpec) -> np.ndarray:
    resized = cv2.resize(crop, spec.target_size, interpolation=cv2.INTER_LINEAR)
    if spec.color == "gray":
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[..., None]
    if spec.color == "rgb":
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized


def extract_mouth_crops(video_path: Path, spec: MouthCropSpec) -> np.ndarray:
    frames, src_fps = _open_video(video_path)
    frames = _resample_frames(frames, src_fps, spec.fps)

    ref_bbox = _detect_reference_bbox(frames)
    if ref_bbox is None:
        raise MouthDetectionError("face_detect", f"no face detected in any candidate frame of {video_path}")
    x0, y0, x1, y1 = ref_bbox

    crops = np.stack([_to_target(f[y0:y1, x0:x1], spec) for f in frames])
    stack = spec.stack_size
    n_windows = crops.shape[0] // stack
    if n_windows == 0:
        raise MouthDetectionError("mouth_crop", f"fewer than {stack} usable frames")
    trimmed = crops[: n_windows * stack]
    trimmed = trimmed.reshape((n_windows, stack) + trimmed.shape[1:])
    trimmed = np.transpose(trimmed, (0, 1, 4, 2, 3))
    return trimmed.astype(np.float16) / 255.0
