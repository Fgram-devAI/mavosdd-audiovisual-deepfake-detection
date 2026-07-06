"""Train the final-fusion logistic regression and tiny MLP over score CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from tqdm import tqdm

from src import common
from src.data.final_fusion_dataset import DEFAULT_FEATURE_COLUMNS, FinalFusionDataset
from src.models.final_fusion import FinalFusionLogReg, FinalFusionMLP


def train_logreg(*, train_ds: FinalFusionDataset, val_ds: FinalFusionDataset,
                 out: Path) -> FinalFusionLogReg:
    model = FinalFusionLogReg.fit(
        X_train=train_ds.X, y_train=train_ds.y,
        X_val=val_ds.X, y_val=val_ds.y,
        feature_columns=train_ds.feature_columns,
    )
    model.save(out)
    return model


def _standardize(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    denom = np.where(scale == 0, 1.0, scale)
    return ((X.astype(np.float64) - mean) / denom).astype(np.float32)


def train_mlp(
    *,
    train_ds: FinalFusionDataset,
    val_ds: FinalFusionDataset,
    out: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden: int,
    patience: int,
    seed: int,
    device: str,
) -> dict:
    common.set_seed(seed)
    mean = train_ds.X.mean(axis=0)
    std = train_ds.X.std(axis=0)
    scale = np.where(std < 1e-8, 1.0, std)

    X_tr = torch.from_numpy(_standardize(train_ds.X, mean, scale))
    y_tr = torch.from_numpy(train_ds.y.astype(np.float32))
    X_va = torch.from_numpy(_standardize(val_ds.X, mean, scale))
    y_va_np = val_ds.y

    torch_device = torch.device(device)
    mlp = FinalFusionMLP(input_dim=X_tr.shape[1], hidden=hidden, dropout=dropout).to(torch_device)
    n = mlp.param_count()
    assert n < 50_000, f"mlp budget exceeded: {n:,}"

    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    best_threshold = 0.5
    wait = 0
    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", unit="epoch")
    for _ in epoch_bar:
        mlp.train()
        opt.zero_grad()
        logits = mlp(X_tr.to(torch_device))
        loss = loss_fn(logits, y_tr.to(torch_device))
        loss.backward()
        opt.step()

        mlp.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(mlp(X_va.to(torch_device))).cpu().numpy()
        if len(set(y_va_np.tolist())) < 2:
            val_auc = float("nan")
        else:
            val_auc = float(roc_auc_score(y_va_np, val_scores))
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
            fpr, tpr, thr = roc_curve(y_va_np, val_scores)
            fnr = 1.0 - tpr
            idx = int(np.argmin(np.abs(fnr - fpr)))
            best_threshold = float(thr[idx])
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        epoch_bar.set_postfix(val_auc=f"{val_auc:.4f}", best=f"{best_auc:.4f}")

    if best_state is None:
        best_state = mlp.state_dict()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "feature_columns": list(train_ds.feature_columns),
        "input_dim": int(X_tr.shape[1]),
        "hidden": int(hidden),
        "dropout": float(dropout),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "threshold": float(best_threshold),
        "val_roc_auc": float(best_auc),
        "seed": int(seed),
        "model_kind": "final_fusion_mlp_v1",
    }, out)
    return {"val_roc_auc": float(best_auc), "threshold": float(best_threshold)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--val-scores", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["logreg", "mlp"],
                        choices=["logreg", "mlp"])
    parser.add_argument("--logreg-out", type=Path, default=common.CKPT_FINAL_FUSION_LOGREG)
    parser.add_argument("--mlp-out", type=Path, default=common.CKPT_FINAL_FUSION_MLP)
    parser.add_argument("--feature-columns", nargs="+", default=list(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    feat = tuple(args.feature_columns)
    train_ds = FinalFusionDataset(score_csv=args.train_scores, split="train", feature_columns=feat)
    val_ds = FinalFusionDataset(score_csv=args.val_scores, split="val", feature_columns=feat)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty split: train={len(train_ds)} val={len(val_ds)}")

    if "logreg" in args.models:
        m = train_logreg(train_ds=train_ds, val_ds=val_ds, out=args.logreg_out)
        print(f"logreg val_roc_auc={m.val_roc_auc:.4f} threshold={m.threshold:.4f} -> {args.logreg_out}")
    if "mlp" in args.models:
        res = train_mlp(
            train_ds=train_ds, val_ds=val_ds, out=args.mlp_out,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            dropout=args.dropout, hidden=args.hidden, patience=args.patience,
            seed=args.seed, device=args.device,
        )
        print(f"mlp val_roc_auc={res['val_roc_auc']:.4f} threshold={res['threshold']:.4f} -> {args.mlp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
