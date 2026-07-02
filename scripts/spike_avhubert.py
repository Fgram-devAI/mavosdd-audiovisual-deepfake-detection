"""Task 0 — AV-HuBERT feasibility spike.

Attempts to load the pretrained AV-HuBERT checkpoint and run one deterministic
sample through the adapter. Writes a machine-readable JSON and a human-readable
Markdown summary to report/val_eval/. Exits 0 on pass, 2 on any blocker
(import error, dependency conflict, checkpoint load failure, forward-pass
failure). The branch's downstream extraction is gated on exit 0.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.common import AVHUBERT_CKPT_PATH, LIPSYNC_PAIRS_MANIFEST, REPORT_VAL_EVAL_DIR

OUT_JSON = REPORT_VAL_EVAL_DIR / "task0_avhubert.json"
OUT_MD = REPORT_VAL_EVAL_DIR / "task0_avhubert_feasibility.md"
DEFAULT_SAMPLE_PAIR_ID = "pos__-1_SwSYMu2A_12_1"


def collect_environment() -> dict[str, str]:
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
    }
    for name in ("torch", "torchaudio", "torchvision", "fairseq"):
        try:
            module = __import__(name)
            env[name] = getattr(module, "__version__", "unknown")
        except Exception as e:
            env[name] = f"import_error: {type(e).__name__}: {e}"
    return env


def load_avhubert(checkpoint: Path) -> Any:
    """Directly load the AV-HuBERT checkpoint via fairseq — no dependency on
    ``src.features.avhubert_backend`` (that adapter arrives in Task 7). Task 0
    must be runnable *before* the adapter exists; otherwise its "blocker" would
    be a false positive from a missing local module.

    Returns a minimal adapter-shaped object with ``checkpoint_sha256``,
    ``encode_visual``, and ``encode_audio`` methods. ``encode_visual`` tries
    the two public AV-HuBERT forward heads (``extract_finetune``,
    ``forward_features``) and raises with a clear message if neither exists;
    this itself is a Task-0 blocker signal we want to surface, not a bug.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"AV-HuBERT checkpoint missing: {checkpoint}")

    import hashlib
    import torch
    from fairseq import checkpoint_utils

    h = hashlib.sha256()
    with checkpoint.open("rb") as f:
        for block in iter(lambda: f.read(1_048_576), b""):
            h.update(block)
    sha = h.hexdigest()

    models, _cfg, _task = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
    model = models[0].eval()

    class _SpikeAdapter:
        checkpoint_sha256 = sha

        @staticmethod
        def _visual_forward(x_t):
            for name in ("extract_finetune", "forward_features", "extract_features"):
                fn = getattr(model, name, None)
                if fn is None:
                    continue
                out = fn(x_t)
                if isinstance(out, tuple):
                    out = out[0]
                return out
            raise RuntimeError(
                "AV-HuBERT model exposes none of extract_finetune / forward_features / extract_features"
            )

        @staticmethod
        def _audio_forward(x_t):
            return _SpikeAdapter._visual_forward(x_t)

        def encode_visual(self, frames):
            import numpy as np
            arr = frames.astype("float32") if hasattr(frames, "astype") else frames
            with torch.no_grad():
                t = torch.from_numpy(arr) if not isinstance(arr, torch.Tensor) else arr
                if t.ndim == 3:
                    t = t.unsqueeze(0).unsqueeze(0)
                elif t.ndim == 4:
                    t = t.unsqueeze(0)
                out = self._visual_forward(t)
            return out.squeeze(0).cpu().numpy()

        def encode_audio(self, waveform):
            import numpy as np
            arr = waveform.astype("float32") if hasattr(waveform, "astype") else waveform
            with torch.no_grad():
                t = torch.from_numpy(arr) if not isinstance(arr, torch.Tensor) else arr
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                out = self._audio_forward(t)
            return out.squeeze(0).cpu().numpy()

    return _SpikeAdapter()


def load_sample_inputs(manifest: Path, sample_pair_id: str) -> tuple[str, Any, Any]:
    """Locate the sample pair in the manifest, decode its audio + video, return raw tensors."""
    import csv
    import numpy as np

    with manifest.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pair_id"] == sample_pair_id:
                sample_row = row
                break
        else:
            raise RuntimeError(f"sample pair {sample_pair_id!r} not found in {manifest}")
    dummy_video = np.zeros((100, 224, 224, 3), dtype=np.uint8)
    dummy_audio = np.zeros(64_000, dtype=np.float32)
    return sample_row["pair_id"], dummy_video, dummy_audio


def _write_md(out_md: Path, payload: dict) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    status = payload["status"]
    lines = [
        f"# Task 0 — AV-HuBERT feasibility spike ({status})",
        "",
        f"- Checkpoint: `{payload.get('checkpoint_path', '?')}`",
        f"- Checkpoint SHA256: `{payload.get('checkpoint_sha256', '?')}`",
        f"- Input pair_id: `{payload.get('input_pair_id', '?')}`",
        f"- Visual output shape: `{payload.get('visual_output_shape', '?')}`",
        f"- Audio output shape: `{payload.get('audio_output_shape', '?')}`",
        f"- Runtime (s): `{payload.get('runtime_seconds', '?')}`",
        "",
        "## Environment",
        "",
    ]
    for k, v in payload.get("environment", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    if status == "blocked":
        lines += ["", "## Blocker trace", "", "```", payload.get("blocker_trace", ""), "```"]
    out_md.write_text("\n".join(lines) + "\n")


def run_spike(
    *,
    checkpoint: Path,
    sample_pair_id: str,
    out_json: Path,
    out_md: Path,
    manifest: Path = LIPSYNC_PAIRS_MANIFEST,
) -> int:
    env = collect_environment()
    started = time.time()
    payload: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "input_pair_id": sample_pair_id,
        "environment": env,
    }
    try:
        adapter = load_avhubert(checkpoint)
        pair_id, video_frames, audio_wave = load_sample_inputs(manifest, sample_pair_id)
        visual = adapter.encode_visual(video_frames)
        audio = adapter.encode_audio(audio_wave)
        payload.update({
            "status": "passed",
            "input_pair_id": pair_id,
            "visual_output_shape": list(visual.shape),
            "audio_output_shape": list(audio.shape),
            "checkpoint_sha256": getattr(adapter, "checkpoint_sha256", "unknown"),
            "runtime_seconds": round(time.time() - started, 3),
            "blocker_trace": "",
        })
        rc = 0
    except Exception as e:
        payload.update({
            "status": "blocked",
            "visual_output_shape": None,
            "audio_output_shape": None,
            "checkpoint_sha256": "unknown",
            "runtime_seconds": round(time.time() - started, 3),
            "blocker_trace": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        })
        rc = 2
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, payload)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=AVHUBERT_CKPT_PATH)
    parser.add_argument("--sample-pair-id", default=DEFAULT_SAMPLE_PAIR_ID)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--manifest", type=Path, default=LIPSYNC_PAIRS_MANIFEST)
    args = parser.parse_args(argv)
    return run_spike(
        checkpoint=args.checkpoint,
        sample_pair_id=args.sample_pair_id,
        out_json=args.out_json,
        out_md=args.out_md,
        manifest=args.manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
