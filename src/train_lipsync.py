"""Training CLI for the lip-sync consistency head."""
from __future__ import annotations

import argparse
import csv
import dataclasses
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from tqdm.auto import tqdm

from src import common, evaluate
from src.data.lipsync_pairs import (
    LipSyncPairDataset,
    make_lipsync_dataloader,
)
from src.models.lipsync_consistency import LipSyncConsistencyModel


CODEC_DIRS: dict[str, Path] = {
    "wav2vec2": common.FEAT_AUDIO_WAV2VEC2_CODEC_DIR,
    "wavlm": common.FEAT_AUDIO_WAVLM_CODEC_DIR,
    "hubert": common.FEAT_AUDIO_HUBERT_CODEC_DIR,
}


@dataclass
class LipSyncRunConfig:
    backend: str
    manifest: Path
    audio_dir: Path
    lips_dir: Path
    run_name: str
    runs_dir: Path
    checkpoint_path: Path
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-2
    dropout: float = 0.3
    patience: int = 7
    device: str = "auto"
    seed: int = 42


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _per_async_group_recall(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    """Recall of the async (label=1) class within each negative-type group.

    Positive-sync rows (label=0, group='positive') are excluded because
    'recall of positives' is not meaningful when the positive class is async.
    Use ``positive_sync_accuracy`` (below) for the sync-side signal.
    """
    out: dict[str, float] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        y_g = y_true[m]
        if y_g.sum() == 0:
            continue
        out[g] = float((y_pred[m][y_g == 1] == 1).sum() / y_g.sum())
    return out


def _positive_sync_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of true-sync (label=0) rows the model correctly calls sync."""
    mask = y_true == 0
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == 0).sum() / mask.sum())


def _val_metric_battery(
    model: nn.Module, loader, device: torch.device,
) -> dict:
    ys: list[int] = []
    scores: list[float] = []
    providers: list[str] = []
    neg_types: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["audio"].to(device),
                batch["lips"].to(device),
                batch["lips_mask"].to(device),
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
    # metric_battery computes EER internally; call it once for AUC/EER,
    # then recompute the threshold-dependent metrics at the EER threshold
    # so F1/precision/recall/confusion/per-group breakdowns are all on the
    # same operating point rather than a hardcoded 0.5.
    battery = evaluate.metric_battery(y, s, p, threshold=0.5)
    eer_thr = float(battery["eer_threshold"])
    battery_eer = evaluate.metric_battery(y, s, p, threshold=eer_thr)
    battery["threshold_used"] = eer_thr
    for key in ("f1", "precision", "recall", "confusion", "per_provider_recall"):
        battery[key] = battery_eer[key]
    pred = (s >= eer_thr).astype(int)
    battery["positive_class"] = "async_inconsistent_pair"
    battery["per_negative_type_recall"] = _per_async_group_recall(
        y, pred, np.asarray(neg_types, dtype=object)
    )
    battery["negative_type_counts"] = dict(Counter(neg_types))
    battery["positive_sync_accuracy"] = _positive_sync_accuracy(y, pred)
    return battery


def run_lipsync_training(cfg: LipSyncRunConfig) -> dict:
    common.set_seed(cfg.seed)
    dev = _resolve_device(cfg.device)

    train_ds = LipSyncPairDataset(
        manifest_path=cfg.manifest, split="train",
        backend=cfg.backend, audio_dir=cfg.audio_dir, lips_dir=cfg.lips_dir,
    )
    val_ds = LipSyncPairDataset(
        manifest_path=cfg.manifest, split="val",
        backend=cfg.backend, audio_dir=cfg.audio_dir, lips_dir=cfg.lips_dir,
    )
    train_loader = make_lipsync_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = make_lipsync_dataloader(val_ds, batch_size=cfg.batch_size)

    model = LipSyncConsistencyModel(dropout=cfg.dropout).to(dev)
    n_params = model.param_count(trainable_only=True)
    assert n_params < 2_000_000, f"param budget exceeded: {n_params:,}"

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    runs_dir = cfg.runs_dir / cfg.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = runs_dir / "metrics.csv"

    best_auc = -1.0
    best_epoch = 0
    best_state: dict | None = None
    best_val_metrics: dict = {}
    no_improve = 0

    for epoch in tqdm(range(1, cfg.epochs + 1), desc=f"{cfg.run_name}", unit="ep"):
        model.train()
        seen = 0
        loss_sum = 0.0
        for batch in train_loader:
            labels = batch["label"].float().to(dev)
            optim.zero_grad()
            logits = model(
                batch["audio"].to(dev),
                batch["lips"].to(dev),
                batch["lips_mask"].to(dev),
            )
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            bsz = labels.size(0)
            loss_sum += float(loss) * bsz
            seen += bsz
        train_loss = loss_sum / max(seen, 1)

        val_metrics = _val_metric_battery(model, val_loader, dev)
        _append_metrics_csv(metrics_csv, {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_roc_auc": round(val_metrics["roc_auc"], 6),
            "val_eer": round(val_metrics["eer"], 6),
            "val_f1": round(val_metrics["f1"], 6),
        })

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > cfg.patience:
                break

    assert best_state is not None, "training produced no best checkpoint"

    model_hparams = {
        "audio_dim": 768,
        "lip_dim": 84,
        "emb_dim": 128,
        "gru_hidden": 96,
        "mlp_hidden": 128,
        "dropout": cfg.dropout,
    }
    ckpt = {
        "state_dict": best_state,
        "model_hparams": model_hparams,
        "backend": cfg.backend,
        "audio_dir": str(cfg.audio_dir),
        "lips_dir": str(cfg.lips_dir),
        "manifest": str(cfg.manifest),
        "val_metrics": best_val_metrics,
        "seed": cfg.seed,
        "positive_class": "async_inconsistent_pair",
    }
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, cfg.checkpoint_path)

    resolved = {}
    for fld in dataclasses.fields(cfg):
        v = getattr(cfg, fld.name)
        resolved[fld.name] = str(v) if isinstance(v, Path) else v
    with (runs_dir / "resolved_config.yaml").open("w") as f:
        yaml.safe_dump(resolved, f)

    return {"best_epoch": best_epoch, "best_val_metrics": best_val_metrics}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the lip-sync consistency head.")
    p.add_argument("--backend", choices=tuple(CODEC_DIRS.keys()), default="wavlm")
    p.add_argument("--manifest", type=Path,
                   default=common.LIPSYNC_PAIRS_MANIFEST)
    p.add_argument("--audio-dir", type=Path, default=None)
    p.add_argument("--lips-dir", type=Path, default=common.FEAT_LIPS_DIR)
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint-path", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--seed", type=int, default=common.SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audio_dir = args.audio_dir if args.audio_dir is not None else CODEC_DIRS[args.backend]
    run_name = args.run_name or f"lipsync_{args.backend}"
    checkpoint = args.checkpoint_path or Path("models/checkpoints") / f"best_{run_name}.pt"

    cfg = LipSyncRunConfig(
        backend=args.backend,
        manifest=args.manifest,
        audio_dir=audio_dir,
        lips_dir=args.lips_dir,
        run_name=run_name,
        runs_dir=args.runs_dir,
        checkpoint_path=checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
    result = run_lipsync_training(cfg)
    bm = result["best_val_metrics"]
    print(
        f"run={cfg.run_name} best_epoch={result['best_epoch']} "
        f"positive_class=async_inconsistent_pair "
        f"val_roc_auc={bm.get('roc_auc', float('nan')):.4f} "
        f"val_eer={bm.get('eer', float('nan')):.4f} "
        f"threshold_used={bm.get('threshold_used', 0.5):.4f}"
    )
    print(f"positive_sync_accuracy={bm.get('positive_sync_accuracy', float('nan')):.4f}")
    print(f"per_negative_type_recall={bm.get('per_negative_type_recall', {})}")
    print(f"per_provider_recall={bm.get('per_provider_recall', {})}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
