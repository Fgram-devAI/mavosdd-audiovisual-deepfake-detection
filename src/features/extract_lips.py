"""Video -> 20 frames @ 5 fps -> normalized FaceMesh lip landmarks + bbox."""
from __future__ import annotations

import csv

import cv2
import mediapipe as mp
import numpy as np

from src.common import AUDIO_SECONDS, FEAT_LIPS_DIR, MANIFEST, N_FRAMES

LIP_IDX = sorted(
    {
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    }
)
N_PTS = len(LIP_IDX)
FEAT_DIM = N_PTS * 2 + 4


def sample_indices(n_total: int, fps: float) -> list[int]:
    """Uniformly pick N_FRAMES indices across the first analysis window."""
    span = min(n_total, int(round(fps * AUDIO_SECONDS))) or 1
    return [min(int(round(i * span / N_FRAMES)), n_total - 1) for i in range(N_FRAMES)]


def extract_one(video_path: str, mesh) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    feats = np.zeros((N_FRAMES, FEAT_DIM), dtype=np.float32)
    mask = np.zeros(N_FRAMES, dtype=np.float32)

    for t, frame_idx in enumerate(sample_indices(max(n_total, 1), fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue

        landmarks = res.multi_face_landmarks[0].landmark
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in LIP_IDX], dtype=np.float32)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        diag = float(np.hypot(x1 - x0, y1 - y0)) + 1e-8
        norm = (pts - pts.mean(0)) / diag
        bbox = np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)

        feats[t] = np.concatenate([norm.flatten(), bbox])
        mask[t] = 1.0

    cap.release()
    return feats, mask


def main() -> None:
    FEAT_LIPS_DIR.mkdir(parents=True, exist_ok=True)
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    try:
        with MANIFEST.open(newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            out = FEAT_LIPS_DIR / f"{row['video_id']}.npz"
            if out.exists():
                continue
            feats, mask = extract_one(row["relative_path"], mesh)
            np.savez(out, feats=feats, mask=mask)
            if mask.sum() == 0:
                print(f"[NO-FACE] {row['video_id']} ({row['source_folder']})")
    finally:
        mesh.close()


if __name__ == "__main__":
    main()
