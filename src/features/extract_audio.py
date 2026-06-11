"""Video -> 16 kHz mono wave -> frozen Wav2Vec2 -> float16 .npy features."""
from __future__ import annotations

import csv
import subprocess
import tempfile

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.common import FEAT_AUDIO_DIR, MANIFEST, N_SAMPLES, SR, W2V_MODEL, device, set_seed


def demux_audio(video_path: str) -> np.ndarray:
    """Demux audio through ffmpeg, then crop or pad to the fixed analysis window."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                video_path,
                "-ac",
                "1",
                "-ar",
                str(SR),
                tmp.name,
            ],
            check=True,
        )
        wave, _ = librosa.load(tmp.name, sr=SR, mono=True)

    if len(wave) >= N_SAMPLES:
        start = (len(wave) - N_SAMPLES) // 2
        wave = wave[start : start + N_SAMPLES]
    else:
        wave = np.pad(wave, (0, N_SAMPLES - len(wave)))
    return wave.astype(np.float32)


@torch.no_grad()
def main() -> None:
    set_seed()
    dev = device()
    processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL)
    model = Wav2Vec2Model.from_pretrained(W2V_MODEL).to(dev).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    FEAT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        out = FEAT_AUDIO_DIR / f"{row['video_id']}.npy"
        if out.exists():
            continue
        try:
            wave = demux_audio(row["relative_path"])
            inputs = processor(wave, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
            hidden = model(inputs).last_hidden_state.squeeze(0)
            np.save(out, hidden.cpu().numpy().astype(np.float16))
        except Exception as exc:
            print(f"[FAIL] {row['video_id']}: {exc}")

    print(f"Audio store: {len(list(FEAT_AUDIO_DIR.glob('*.npy')))} arrays")


if __name__ == "__main__":
    main()
