"""Validation-only evaluator for lip-sync consistency checkpoints."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src import evaluate
from src.data.lipsync_pairs import (
    LipSyncPairDataset,
    make_lipsync_dataloader,
)
from src.models.lipsync_consistency import LipSyncConsistencyModel


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _per_async_group_recall(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    out: dict[str, float] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        y_g = y_true[m]
        if y_g.sum() == 0:
            continue
        out[g] = float((y_pred[m][y_g == 1] == 1).sum() / y_g.sum())
    return out


def _positive_sync_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true == 0
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == 0).sum() / mask.sum())


def evaluate_lipsync_checkpoint(
    ckpt_path: str | Path,
    *,
    split: str = "val",
    device: str = "auto",
    manifest: str | Path | None = None,
    audio_dir: Path | None = None,
    lips_dir: Path | None = None,
    batch_size: int = 32,
) -> dict:
    if split == "test":
        raise SystemExit(
            "Refusing to evaluate on test split for the lip-sync consistency branch. "
            "Test is locked until the final generated-video eval branch."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dev = _resolve_device(device)

    hp = ckpt.get("model_hparams", {})
    model = LipSyncConsistencyModel(
        audio_dim=hp.get("audio_dim", 768),
        lip_dim=hp.get("lip_dim", 84),
        emb_dim=hp.get("emb_dim", 128),
        gru_hidden=hp.get("gru_hidden", 96),
        mlp_hidden=hp.get("mlp_hidden", 128),
        dropout=hp.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    manifest_path = str(manifest) if manifest is not None else ckpt["manifest"]
    ds = LipSyncPairDataset(
        manifest_path=manifest_path, split=split, backend=ckpt["backend"],
        audio_dir=Path(audio_dir) if audio_dir is not None else Path(ckpt["audio_dir"]),
        lips_dir=Path(lips_dir) if lips_dir is not None else Path(ckpt["lips_dir"]),
    )
    loader = make_lipsync_dataloader(ds, batch_size=batch_size)

    ys: list[int] = []
    scores: list[float] = []
    providers: list[str] = []
    neg_types: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["audio"].to(dev),
                batch["lips"].to(dev),
                batch["lips_mask"].to(dev),
            )
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            scores.extend(probs)
            ys.extend(batch["label"].cpu().numpy().astype(int).tolist())
            providers.extend(m.get("audio_provider", "") for m in batch["metadata"])
            neg_types.extend(
                m.get("negative_type", "") or "positive" for m in batch["metadata"]
            )

    y = np.asarray(ys, dtype=int)
    s = np.asarray(scores, dtype=float)
    p = np.asarray(providers, dtype=object)
    result = evaluate.metric_battery(y, s, p, threshold=0.5)
    eer_thr = float(result["eer_threshold"])
    at_eer = evaluate.metric_battery(y, s, p, threshold=eer_thr)
    result["threshold_used"] = eer_thr
    for key in ("f1", "precision", "recall", "confusion", "per_provider_recall"):
        result[key] = at_eer[key]
    pred = (s >= eer_thr).astype(int)
    result["positive_class"] = "async_inconsistent_pair"
    result["per_negative_type_recall"] = _per_async_group_recall(
        y, pred, np.asarray(neg_types, dtype=object)
    )
    result["negative_type_counts"] = dict(Counter(neg_types))
    result["positive_sync_accuracy"] = _positive_sync_accuracy(y, pred)
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validation-only evaluation for a lip-sync consistency checkpoint."
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--audio-dir", type=Path, default=None)
    p.add_argument("--lips-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_lipsync_checkpoint(
            args.checkpoint, split=args.split, device=args.device,
            manifest=args.manifest, audio_dir=args.audio_dir,
            lips_dir=args.lips_dir, batch_size=args.batch_size,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"split={args.split} n={result['n']} "
        f"positive_class=async_inconsistent_pair "
        f"roc_auc={result['roc_auc']:.4f} eer={result['eer']:.4f} "
        f"threshold_used={result['threshold_used']:.4f} "
        f"f1={result['f1']:.4f}"
    )
    print(f"confusion={result['confusion']}")
    print(f"positive_sync_accuracy={result['positive_sync_accuracy']:.4f}")
    print(f"per_negative_type_recall={result['per_negative_type_recall']}")
    print(f"per_provider_recall={result['per_provider_recall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
