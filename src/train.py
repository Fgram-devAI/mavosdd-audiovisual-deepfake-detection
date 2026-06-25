"""Audio anti-spoof training harness over codec-matched embeddings."""
from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from src import common, evaluate
from src.data.feature_store import (
    AudioFeatureDataset,
    FusionFeatureDataset,
    NormalizationStats,
    VisualFeatureDataset,
    fit_normalization_stats,
    make_dataloader,
)
from src.models.late_fusion import LateFusionClassifier


CODEC_DIRS: dict[str, Path] = {
    "wav2vec2": common.FEAT_AUDIO_WAV2VEC2_CODEC_DIR,
    "wavlm": common.FEAT_AUDIO_WAVLM_CODEC_DIR,
    "hubert": common.FEAT_AUDIO_HUBERT_CODEC_DIR,
}

_DEFAULT_TRAINING = {
    "batch_size": 32,
    "max_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "early_stop_patience": 7,
    "dropout": 0.3,
    "max_trainable_params": 2_000_000,
}


def _load_training_defaults() -> dict:
    cfg_path = Path("config/default.yaml")
    if not cfg_path.exists():
        return dict(_DEFAULT_TRAINING)
    with cfg_path.open() as f:
        loaded = yaml.safe_load(f) or {}
    merged = dict(_DEFAULT_TRAINING)
    merged.update(loaded.get("training", {}))
    return merged


@dataclass
class RunConfig:
    modality: str
    backend: str
    manifest: Path
    audio_dir: Path
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    dropout: float
    patience: int
    device: str
    seed: int
    run_name: str
    runs_dir: Path
    checkpoint_path: Path
    max_trainable_params: int = 2_000_000


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_datasets(
    cfg: RunConfig,
    *,
    lips_dir: Path | None = None,
):
    """Return (train_ds, val_ds, NormalizationStats) for cfg.modality."""
    if cfg.modality == "audio":
        train_raw = AudioFeatureDataset(
            manifest_path=cfg.manifest, split="train",
            backend=cfg.backend, audio_dir=cfg.audio_dir,
        )
        stats = fit_normalization_stats(train_raw, modalities=("audio",))
        train_ds = AudioFeatureDataset(
            manifest_path=cfg.manifest, split="train",
            backend=cfg.backend, audio_dir=cfg.audio_dir, normalization=stats,
        )
        val_ds = AudioFeatureDataset(
            manifest_path=cfg.manifest, split="val",
            backend=cfg.backend, audio_dir=cfg.audio_dir, normalization=stats,
        )
        return train_ds, val_ds, stats

    if cfg.modality == "visual":
        train_raw = VisualFeatureDataset(
            manifest_path=cfg.manifest, split="train", lips_dir=lips_dir,
        )
        stats = fit_normalization_stats(train_raw, modalities=("lips",))
        train_ds = VisualFeatureDataset(
            manifest_path=cfg.manifest, split="train",
            lips_dir=lips_dir, normalization=stats,
        )
        val_ds = VisualFeatureDataset(
            manifest_path=cfg.manifest, split="val",
            lips_dir=lips_dir, normalization=stats,
        )
        return train_ds, val_ds, stats

    if cfg.modality == "fusion":
        train_raw = FusionFeatureDataset(
            manifest_path=cfg.manifest, split="train",
            backend=cfg.backend, audio_dir=cfg.audio_dir, lips_dir=lips_dir,
        )
        stats = fit_normalization_stats(train_raw, modalities=("audio", "lips"))
        train_ds = FusionFeatureDataset(
            manifest_path=cfg.manifest, split="train",
            backend=cfg.backend, audio_dir=cfg.audio_dir, lips_dir=lips_dir,
            normalization=stats,
        )
        val_ds = FusionFeatureDataset(
            manifest_path=cfg.manifest, split="val",
            backend=cfg.backend, audio_dir=cfg.audio_dir, lips_dir=lips_dir,
            normalization=stats,
        )
        return train_ds, val_ds, stats

    raise ValueError(f"unknown modality: {cfg.modality!r}")


def build_model(cfg: RunConfig) -> LateFusionClassifier:
    model = LateFusionClassifier(modality=cfg.modality, emb=128, p=cfg.dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < cfg.max_trainable_params, (
        f"param budget exceeded: {n_params:,} >= {cfg.max_trainable_params:,}"
    )
    return model


def _simulate_early_stop(val_aucs: list[float], patience: int) -> tuple[int, float]:
    """Return (epoch_at_stop, best_auc). Stops when no_improve epochs > patience."""
    best = -1.0
    best_epoch = 0
    no_improve = 0
    for epoch, auc in enumerate(val_aucs, start=1):
        if auc > best:
            best = auc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                return epoch, best
    return len(val_aucs), best


def _append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _forward_batch(
    model: nn.Module,
    batch: dict,
    modality: str,
    device: torch.device,
) -> torch.Tensor:
    if modality == "audio":
        return model(batch["audio"].to(device))
    if modality == "visual":
        lips = batch["lips"].to(device)
        mask = batch["lips_mask"].to(device)
        return model(None, lips, mask)
    if modality == "fusion":
        audio = batch["audio"].to(device)
        lips = batch["lips"].to(device)
        mask = batch["lips_mask"].to(device)
        return model(audio, lips, mask)
    raise ValueError(f"unknown modality: {modality!r}")


def _val_metric_battery(
    model: nn.Module, loader, device: torch.device, modality: str,
) -> dict:
    ys: list[int] = []
    scores: list[float] = []
    providers: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = _forward_batch(model, batch, modality, device)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            scores.extend(probs)
            ys.extend(batch["label"].cpu().numpy().astype(int).tolist())
            providers.extend(m.get("provider", "") for m in batch["metadata"])
    return evaluate.metric_battery(
        np.asarray(ys, dtype=int),
        np.asarray(scores, dtype=float),
        np.asarray(providers, dtype=object),
    )


def run_training(cfg: RunConfig, *, lips_dir: Path | None = None) -> dict:
    common.set_seed(cfg.seed)
    dev = _resolve_device(cfg.device)

    train_ds, val_ds, stats = build_datasets(cfg, lips_dir=lips_dir)
    train_loader = make_dataloader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = make_dataloader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    runs_dir = cfg.runs_dir / cfg.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = runs_dir / "metrics.csv"

    best_auc = -1.0
    best_epoch = 0
    best_state: dict | None = None
    best_val_metrics: dict = {}
    epochs_without_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            labels = batch["label"].float().to(dev)
            optim.zero_grad()
            logits = _forward_batch(model, batch, cfg.modality, dev)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            bsz = labels.size(0)
            train_loss += float(loss) * bsz
            n_seen += bsz
        train_loss = train_loss / max(n_seen, 1)

        val_metrics = _val_metric_battery(model, val_loader, dev, cfg.modality)
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
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve > cfg.patience:
                break

    assert best_state is not None, "training produced no best checkpoint"

    norm_stats: dict = {"eps": stats.eps}
    if stats.audio_mean is not None and stats.audio_std is not None:
        norm_stats["audio_mean"] = stats.audio_mean
        norm_stats["audio_std"] = stats.audio_std
    if stats.lips_mean is not None and stats.lips_std is not None:
        norm_stats["lips_mean"] = stats.lips_mean
        norm_stats["lips_std"] = stats.lips_std

    ckpt = {
        "state_dict": best_state,
        "modality": cfg.modality,
        "backend": cfg.backend if cfg.modality != "visual" else None,
        "audio_dir": str(cfg.audio_dir) if cfg.modality != "visual" else None,
        "model_hparams": {
            "modality": cfg.modality,
            "emb": 128,
            "dropout": cfg.dropout,
        },
        "norm_stats": norm_stats,
        "val_metrics": best_val_metrics,
        "seed": cfg.seed,
        "manifest": str(cfg.manifest),
    }
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, cfg.checkpoint_path)

    resolved_cfg = runs_dir / "resolved_config.yaml"
    with resolved_cfg.open("w") as f:
        yaml.safe_dump(_runconfig_to_dict(cfg), f)

    return {"best_epoch": best_epoch, "best_val_metrics": best_val_metrics}


def _runconfig_to_dict(cfg: RunConfig) -> dict:
    out = {}
    for fld in dataclasses.fields(cfg):
        v = getattr(cfg, fld.name)
        out[fld.name] = str(v) if isinstance(v, Path) else v
    return out


def _build_parser() -> argparse.ArgumentParser:
    defaults = _load_training_defaults()
    p = argparse.ArgumentParser(description="Train audio anti-spoof baseline.")
    p.add_argument("--modality", choices=("audio", "visual", "fusion"), default="audio")
    p.add_argument("--backend", choices=tuple(CODEC_DIRS.keys()), default="wav2vec2")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="Override audio feature dir (default: codec-matched store for backend).")
    p.add_argument("--lips-dir", type=Path, default=None,
                   help="Override lip feature dir (default: data/features/lips/).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint-path", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=defaults["max_epochs"])
    p.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    p.add_argument("--lr", type=float, default=defaults["lr"])
    p.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    p.add_argument("--dropout", type=float, default=defaults["dropout"])
    p.add_argument("--patience", type=int, default=defaults["early_stop_patience"])
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    defaults = _load_training_defaults()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.modality == "audio":
        manifest = args.manifest if args.manifest is not None \
            else common.AUDIO_SPOOF_MANIFEST_VOICE_SPLIT
    elif args.modality == "visual":
        manifest = args.manifest if args.manifest is not None \
            else common.VISUAL_SPEECH_MANIFEST_VOICE_SPLIT
    else:  # fusion
        manifest = args.manifest if args.manifest is not None \
            else common.FUSION_SPEECH_MANIFEST_VOICE_SPLIT

    if args.modality == "visual":
        audio_dir = args.audio_dir if args.audio_dir is not None else Path("data/_unused_visual")
    else:
        audio_dir = args.audio_dir if args.audio_dir is not None else CODEC_DIRS[args.backend]

    if args.run_name is not None:
        run_name = args.run_name
    elif args.modality == "visual":
        run_name = "visual_bigru"
    elif args.modality == "fusion":
        run_name = f"fusion_{args.backend}_codec"
    else:
        run_name = f"audio_{args.backend}_codec"

    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    elif args.modality == "visual":
        checkpoint_path = Path("models/checkpoints/best_visual.pt")
    elif args.modality == "fusion":
        checkpoint_path = Path("models/checkpoints") / f"best_fusion_{args.backend}.pt"
    else:
        checkpoint_path = Path("models/checkpoints") / f"best_audio_{args.backend}.pt"

    cfg = RunConfig(
        modality=args.modality,
        backend=args.backend,
        manifest=manifest,
        audio_dir=audio_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        run_name=run_name,
        runs_dir=args.runs_dir,
        checkpoint_path=checkpoint_path,
        max_trainable_params=defaults["max_trainable_params"],
    )
    result = run_training(cfg, lips_dir=args.lips_dir)
    bm = result["best_val_metrics"]
    print(
        f"run={cfg.run_name} best_epoch={result['best_epoch']} "
        f"val_roc_auc={bm.get('roc_auc', float('nan')):.4f} "
        f"val_eer={bm.get('eer', float('nan')):.4f}"
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
