"""Val-only evaluator + comparison writer for the pretrained AV-consistency head."""
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src import common
from src.data.lipsync_pretrained_dataset import (
    SYNC_FEATURE_DIM,
    LipSyncPretrainedDataset,
    make_dataloader,
)
from src.models.lipsync_pretrained_head import LipSyncPretrainedHead
from src.train_lipsync_pretrained import resolve_backend


class EvaluationRefusedError(RuntimeError):
    pass


def _eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2), float(thr[idx])


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    providers: list[str],
    negative_types: list[str],
) -> dict:
    from sklearn.metrics import (
        confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    )

    roc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    eer, thr = _eer(scores, labels)
    preds = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    f1 = float(f1_score(labels, preds, zero_division=0))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    sync_accuracy = float(
        (preds[labels == 0] == 0).mean() if (labels == 0).any() else float("nan")
    )

    per_negative: dict[str, float] = {}
    for neg in ("generated_same_transcript", "mismatched_generated", "mismatched_original"):
        mask = np.array([nt == neg and labels[i] == 1 for i, nt in enumerate(negative_types)])
        if mask.any():
            per_negative[neg] = float((preds[mask] == 1).mean())
        else:
            per_negative[neg] = float("nan")

    per_provider: dict[str, float] = {}
    for prov in ("elevenlabs", "google_tts", "openai_tts", "original"):
        mask = np.array([
            p == prov and labels[i] == 1 for i, p in enumerate(providers)
        ])
        if mask.any():
            per_provider[prov] = float((preds[mask] == 1).mean())
        else:
            per_provider[prov] = float("nan")

    # Sanity guardrail: per_provider["original"] must equal
    # per_negative_type_recall["mismatched_original"] — both measure the same
    # rows (label==1 AND original audio, which by construction is mismatched_original).
    if not (np.isnan(per_negative["mismatched_original"]) or np.isnan(per_provider["original"])):
        assert abs(per_provider["original"] - per_negative["mismatched_original"]) < 1e-6, (
            f"guardrail: per_provider[original]={per_provider['original']} != "
            f"per_negative_type[mismatched_original]={per_negative['mismatched_original']}"
        )

    return {
        "roc_auc": roc,
        "eer": eer,
        "threshold_used": thr,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sync_accuracy": sync_accuracy,
        "per_negative_type_recall": per_negative,
        "per_provider_recall": per_provider,
    }


def format_val_line(
    metrics: dict,
    *,
    split: str,
    n_manifest: int,
    n_evaluated: int,
    n_excluded: int,
    excluded_by_reason: dict,
    positive_class: str,
    partial: bool,
) -> str:
    header = (
        f"split={split} n_manifest={n_manifest} n_evaluated={n_evaluated} "
        f"n_excluded={n_excluded} excluded_by_reason={excluded_by_reason} "
        f"partial_evaluation={'true' if partial else 'false'} "
        f"positive_class={positive_class} "
        f"roc_auc={metrics['roc_auc']:.4f} eer={metrics['eer']:.4f} "
        f"threshold_used={metrics['threshold_used']:.4f} f1={metrics['f1']:.4f}"
    )
    conf = f"confusion={metrics['confusion']}"
    sync = f"sync_accuracy={metrics['sync_accuracy']:.4f}"
    per_neg = f"per_negative_type_recall={metrics['per_negative_type_recall']}"
    per_prov = f"per_provider_recall={metrics['per_provider_recall']}"
    return "\n".join([header, conf, sync, per_neg, per_prov]) + "\n"


def write_val_file(path: Path, content: str, *, partial: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "# WARNING: partial_evaluation=true — see excluded_by_reason\n" if partial else ""
    path.write_text(prefix + content)


def evaluate_backend(
    *,
    checkpoint: Path,
    backend: str,
    manifest: Path,
    split: str,
    allow_partial: bool,
    out: Path,
    device: str = "cpu",
) -> int:
    # Test split is locked — refuse immediately before any I/O.
    if split == "test":
        raise ValueError("test split is locked; refuse to evaluate on test")

    visual_dir, audio_dir, failures_csv, embed_dim = resolve_backend(backend)
    fcsv = failures_csv if failures_csv.exists() else None
    ds = LipSyncPretrainedDataset(
        manifest=manifest, split=split, backend=backend,
        visual_dir=visual_dir, audio_dir=audio_dir, failures_csv=fcsv,
    )

    with manifest.open() as f:
        reader = csv.DictReader(f)
        all_split_rows = [r for r in reader if r["split"] == split]
    n_manifest = len(all_split_rows)
    n_evaluated = len(ds)
    n_excluded = n_manifest - n_evaluated
    reasons: Counter = Counter()

    if n_excluded > 0:
        excluded_ids = ds.excluded_pair_ids
        for row in all_split_rows:
            if row["pair_id"] not in excluded_ids:
                continue
            vid = row["source_video_id"]
            aid = row["audio_sample_id"]
            vpath = visual_dir / f"{vid}.npy"
            apath = audio_dir / f"{aid}.npy"
            if not vpath.exists():
                reasons["missing_visual"] += 1
            elif not apath.exists():
                reasons["missing_audio"] += 1
            else:
                reasons["extraction_failure"] += 1
        if not allow_partial:
            raise EvaluationRefusedError(
                f"partial evaluation refused: n_manifest={n_manifest} "
                f"n_evaluated={n_evaluated} excluded_by_reason={dict(reasons)}; "
                "pass --allow-partial to write partial metrics"
            )

    state = torch.load(checkpoint, map_location=device)
    cfg = state.get("config", {})
    hidden = cfg.get("hidden", 128)
    dropout = cfg.get("dropout", 0.3)
    head = LipSyncPretrainedHead(
        sync_feature_dim=SYNC_FEATURE_DIM,
        embed_dim=embed_dim,
        hidden=hidden,
        dropout=dropout,
    ).to(device)
    head.load_state_dict(state["state_dict"])
    head.eval()

    dl = make_dataloader(ds, batch_size=64, shuffle=False, num_workers=0, seed=42)
    scores: list[float] = []
    labels: list[float] = []
    providers: list[str] = []
    negatives: list[str] = []
    with torch.no_grad():
        for batch in dl:
            sf = batch["sync_features"].to(device)
            pv = batch["pooled_visual"].to(device)
            pa = batch["pooled_audio"].to(device)
            logits = head(sf, pv, pa)
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["sync_label"].cpu().numpy().tolist())
            providers.extend(batch["audio_provider"])
            negatives.extend(batch["negative_type"])

    # Fail-fast guard 1: zero evaluated rows (even with --allow-partial).
    if n_evaluated == 0:
        raise EvaluationRefusedError(
            f"n_evaluated=0 after exclusions; refuse to write metrics even with "
            f"--allow-partial (n_manifest={n_manifest}, excluded_by_reason={dict(reasons)})"
        )

    # Fail-fast guard 2: single label class (even with --allow-partial).
    unique_labels = set(int(x) for x in labels)
    if unique_labels != {0, 1}:
        raise EvaluationRefusedError(
            f"missing a label class in evaluated rows (present={sorted(unique_labels)}); "
            f"refuse to write metrics even with --allow-partial"
        )

    m = compute_metrics(np.asarray(scores), np.asarray(labels), providers, negatives)
    content = format_val_line(
        m, split=split, n_manifest=n_manifest, n_evaluated=n_evaluated,
        n_excluded=n_excluded, excluded_by_reason=dict(reasons),
        positive_class="async_inconsistent_pair",
        partial=(n_excluded > 0),
    )
    write_val_file(out, content, partial=(n_excluded > 0))
    return 0


_LITERAL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$")


def _parse_val_text(text: str) -> dict:
    if text.strip().lower().startswith("n/a"):
        return {"status": "blocked", "raw": text.strip()}
    out: dict = {}
    for line in text.strip().splitlines():
        if line.startswith("#"):
            continue
        for token in line.split():
            m = _LITERAL_RE.match(token)
            if not m:
                continue
            key, raw = m.group(1), m.group(2)
            try:
                out[key] = ast.literal_eval(raw)
            except Exception:
                out[key] = raw
        for key in ("confusion", "per_negative_type_recall", "per_provider_recall"):
            if line.startswith(f"{key}="):
                out[key] = ast.literal_eval(line[len(key) + 1:])
    out.setdefault("status", "ok")
    return out


def emit_comparison(
    *,
    wavlm_val: Path,
    syncnet_val: Path,
    avhubert_val: Path,
    out: Path,
) -> int:
    wavlm = _parse_val_text(wavlm_val.read_text())
    syncnet = _parse_val_text(syncnet_val.read_text())
    avhubert = _parse_val_text(avhubert_val.read_text())

    def _cell(d: dict, key: str, subkey: str | None = None) -> str:
        if d.get("status") != "ok":
            return "blocked"
        if subkey is None:
            v = d.get(key)
        else:
            v = d.get(key, {}).get(subkey)
        if v is None:
            return "?"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    baseline_original = wavlm.get("per_negative_type_recall", {}).get("mismatched_original")
    rows = [
        ("WavLM+BiGRU baseline", wavlm),
        ("Pretrained SyncNet", syncnet),
        ("AV-HuBERT", avhubert),
    ]
    header = (
        "| Model | ROC-AUC | EER | F1 | sync_accuracy | mismatched_original recall | "
        "elevenlabs | google_tts | openai_tts | original |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for name, d in rows:
        body += "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            name,
            _cell(d, "roc_auc"),
            _cell(d, "eer"),
            _cell(d, "f1"),
            _cell(d, "sync_accuracy"),
            _cell(d, "per_negative_type_recall", "mismatched_original"),
            _cell(d, "per_provider_recall", "elevenlabs"),
            _cell(d, "per_provider_recall", "google_tts"),
            _cell(d, "per_provider_recall", "openai_tts"),
            _cell(d, "per_provider_recall", "original"),
        )

    honest_read = (
        "\n## Honest read\n\n"
        f"WavLM+BiGRU baseline `mismatched_original` recall: `{baseline_original}`. "
        "Compare the SyncNet and AV-HuBERT rows above against this number — a real "
        "improvement means the pretrained backend actually learned real-audio-vs-real-mouth "
        "consistency rather than TTS-fingerprint shortcuts. If overall ROC-AUC is high "
        "but `mismatched_original` recall is not materially above the baseline, the head "
        "class-shifted to the same shortcut. Any row marked `blocked` was gated by Task 0; "
        "see `task0_avhubert_feasibility.md`.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Pretrained SyncNet vs AV-HuBERT comparison\n\n" + header + body + honest_read)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-comparison", action="store_true")
    parser.add_argument("--backend", choices=("syncnet", "avhubert"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=common.LIPSYNC_PAIRS_MANIFEST)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wavlm-val", type=Path)
    parser.add_argument("--syncnet-val", type=Path)
    parser.add_argument("--avhubert-val", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.emit_comparison:
        return emit_comparison(
            wavlm_val=args.wavlm_val, syncnet_val=args.syncnet_val,
            avhubert_val=args.avhubert_val, out=args.out,
        )

    if not (args.backend and args.checkpoint):
        parser.error("--backend and --checkpoint required unless --emit-comparison")
    return evaluate_backend(
        checkpoint=args.checkpoint, backend=args.backend,
        manifest=args.manifest, split=args.split,
        allow_partial=args.allow_partial, out=args.out, device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
