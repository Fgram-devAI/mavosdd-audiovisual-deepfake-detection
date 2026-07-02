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
    target_size=(96, 96),
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


def _lip_landmark_ids() -> list[int]:
    """Return the deduplicated set of MediaPipe FaceMesh lip landmark indices.

    Uses the vetted ``FACEMESH_LIPS`` connection set from MediaPipe instead of
    the ad-hoc range ``61..88 + 291..318`` which pulls in non-lip landmarks.
    """
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

    return sorted({idx for edge in FACEMESH_LIPS for idx in edge})


def _detect_mouth_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) mouth bbox or None if no face is detected."""
    import mediapipe as mp

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as fm:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = fm.process(rgb)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    lip_ids = _lip_landmark_ids()
    xs = np.array([landmarks[i].x for i in lip_ids]) * frame.shape[1]
    ys = np.array([landmarks[i].y for i in lip_ids]) * frame.shape[0]
    cx = int(xs.mean())
    cy = int(ys.mean())
    half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.75)
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(frame.shape[1], cx + half)
    y1 = min(frame.shape[0], cy + half)
    return (x0, y0, x1, y1)


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

    ref_bbox = _detect_mouth_bbox(frames[len(frames) // 2])
    if ref_bbox is None:
        raise MouthDetectionError("face_detect", f"no face in reference frame of {video_path}")
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
