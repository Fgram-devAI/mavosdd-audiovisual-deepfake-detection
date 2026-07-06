"""Final-fusion models: logistic-regression wrapper, tiny MLP, and rule baselines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _get(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    return float(v)


def rule_score_audio_only(row: dict) -> float:
    v = _get(row, "audio_fake_score")
    if v is None:
        raise ValueError("audio_fake_score missing")
    return v


def rule_score_video_av_only(row: dict) -> float:
    v = _get(row, "video_av_fake_score")
    if v is None:
        raise ValueError("video_av_fake_score missing")
    return v


def rule_score_sync_only(row: dict) -> float:
    v = _get(row, "sync_inconsistent_score")
    if v is None:
        raise ValueError("sync_inconsistent_score missing")
    return v


def rule_score_max_audio_video_av(row: dict) -> float:
    return max(rule_score_audio_only(row), rule_score_video_av_only(row))


def rule_score_max_available(row: dict) -> float:
    values = [_get(row, k) for k in ("audio_fake_score", "video_av_fake_score",
                                     "sync_inconsistent_score")]
    values = [v for v in values if v is not None]
    if not values:
        raise ValueError("no non-missing head scores")
    return max(values)


@dataclass
class FinalFusionLogReg:
    feature_columns: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    val_roc_auc: float

    @classmethod
    def fit(
        cls,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_columns: tuple[str, ...],
    ) -> "FinalFusionLogReg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, roc_curve

        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(max_iter=1000, random_state=42).fit(
            scaler.transform(X_train), y_train,
        )
        val_scores = model.predict_proba(scaler.transform(X_val))[:, 1]
        roc = float(roc_auc_score(y_val, val_scores)) if len(set(y_val.tolist())) > 1 else float("nan")
        fpr, tpr, thr = roc_curve(y_val, val_scores)
        fnr = 1.0 - tpr
        idx = int(np.argmin(np.abs(fnr - fpr)))
        threshold = float(thr[idx])
        return cls(
            feature_columns=tuple(feature_columns),
            scaler_mean=scaler.mean_.astype(np.float64),
            scaler_scale=scaler.scale_.astype(np.float64),
            coef=model.coef_[0].astype(np.float64),
            intercept=float(model.intercept_[0]),
            threshold=threshold,
            val_roc_auc=roc,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X.astype(np.float64) - self.scaler_mean) / np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
        logit = Z @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logit))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "feature_columns": list(self.feature_columns),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "threshold": float(self.threshold),
            "val_roc_auc": float(self.val_roc_auc),
            "model_kind": "final_fusion_logreg_v1",
        }, path)

    @classmethod
    def load(cls, path: Path) -> "FinalFusionLogReg":
        state = torch.load(path, map_location="cpu")
        return cls(
            feature_columns=tuple(state["feature_columns"]),
            scaler_mean=np.asarray(state["scaler_mean"], dtype=np.float64),
            scaler_scale=np.asarray(state["scaler_scale"], dtype=np.float64),
            coef=np.asarray(state["coef"], dtype=np.float64),
            intercept=float(state["intercept"]),
            threshold=float(state["threshold"]),
            val_roc_auc=float(state["val_roc_auc"]),
        )


class FinalFusionMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def param_count(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
