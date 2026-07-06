"""Train video-level real-vs-fake classifier over pretrained AV embeddings."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src import common
from src.data.lipsync_pretrained_dataset import SYNC_FEATURE_DIM
from src.data.video_av_dataset import VideoAVDataset, make_dataloader
from src.models.lipsync_pretrained_head import LipSyncPretrainedHead
from src.train_lipsync_pretrained import _roc_auc, resolve_backend


@dataclass
class VideoAVTrainConfig:
    backend: str
    manifest: Path
    visual_dir: Path
    audio_dir: Path
    failures_csv: Path | None
    run_name: str
    runs_dir: Path
    out: Path
    embed_dim: int
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.3
    patience: int = 5
    hidden: int = 128
    device: str = "cpu"
    seed: int = 42


def train(config: VideoAVTrainConfig) -> None:
    common.set_seed(config.seed)
    train_ds = VideoAVDataset(
        manifest=config.manifest,
        split="train",
        visual_dir=config.visual_dir,
        audio_dir=config.audio_dir,
        failures_csv=config.failures_csv,
    )
    val_ds = VideoAVDataset(
        manifest=config.manifest,
        split="val",
        visual_dir=config.visual_dir,
        audio_dir=config.audio_dir,
        failures_csv=config.failures_csv,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"empty split: train={len(train_ds)} val={len(val_ds)}; "
            "build the manifest and extract AV embeddings first"
        )

    train_loader = make_dataloader(train_ds, batch_size=config.batch_size, shuffle=True,
                                   num_workers=0, seed=config.seed)
    val_loader = make_dataloader(val_ds, batch_size=config.batch_size, shuffle=False,
                                 num_workers=0, seed=config.seed)

    device = torch.device(config.device)
    head = LipSyncPretrainedHead(
        sync_feature_dim=SYNC_FEATURE_DIM,
        embed_dim=config.embed_dim,
        hidden=config.hidden,
        dropout=config.dropout,
    ).to(device)
    assert head.param_count() < 500_000, f"budget exceeded: {head.param_count():,}"

    opt = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    run_dir = config.runs_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_roc_auc"])

    best_auc = -1.0
    patience = 0
    epoch_bar = tqdm(range(1, config.epochs + 1), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        head.train()
        total = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"train e{epoch}", unit="batch", leave=False):
            sf = batch["sync_features"].to(device)
            pv = batch["pooled_visual"].to(device)
            pa = batch["pooled_audio"].to(device)
            y = batch["label"].to(device)
            logits = head(sf, pv, pa)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * y.numel()
            n += y.numel()
        train_loss = total / max(n, 1)

        head.eval()
        vtotal = 0.0
        vn = 0
        scores: list[float] = []
        labels: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val e{epoch}", unit="batch", leave=False):
                sf = batch["sync_features"].to(device)
                pv = batch["pooled_visual"].to(device)
                pa = batch["pooled_audio"].to(device)
                y = batch["label"].to(device)
                logits = head(sf, pv, pa)
                loss = loss_fn(logits, y)
                vtotal += float(loss.item()) * y.numel()
                vn += y.numel()
                scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels.extend(y.cpu().numpy().tolist())
        val_loss = vtotal / max(vn, 1)
        val_auc = _roc_auc(np.asarray(scores), np.asarray(labels))

        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_auc:.6f}"])

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_auc=f"{val_auc:.4f}",
            best_auc=f"{max(best_auc, val_auc):.4f}",
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            config.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": head.state_dict(),
                "config": vars(config),
                "epoch": epoch,
                "val_roc_auc": val_auc,
                "positive_class": "fake_video",
            }, config.out)
        else:
            patience += 1
            if patience >= config.patience:
                break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("syncnet", "avhubert"), required=True)
    parser.add_argument("--manifest", type=Path, default=common.VIDEO_AV_MANIFEST)
    parser.add_argument("--visual-dir", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--failures-csv", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    visual_dir, audio_dir, failures_csv, embed_dim = resolve_backend(args.backend)
    fcsv = args.failures_csv or failures_csv
    cfg = VideoAVTrainConfig(
        backend=args.backend,
        manifest=args.manifest,
        visual_dir=args.visual_dir or visual_dir,
        audio_dir=args.audio_dir or audio_dir,
        failures_csv=fcsv if fcsv.exists() else None,
        run_name=args.run_name,
        runs_dir=args.runs_dir,
        out=args.out,
        embed_dim=embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        hidden=args.hidden,
        device=args.device,
        seed=args.seed,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
