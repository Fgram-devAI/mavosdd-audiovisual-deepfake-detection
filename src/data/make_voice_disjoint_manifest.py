"""CLI: build a voice-disjoint derived manifest for the spoof-detection task.

The source manifest (default: codec-matched audio spoof manifest) splits rows by
the parent native video. TTS voices fan out across many native videos, so the
same synthetic voice appears in train, val, and test — letting any audio
classifier shortcut to voice-identity recognition instead of generic spoof
detection. Empirically, 100% of val spoof rows use a voice that was in train.

This CLI rewrites the ``split`` column under a voice-disjoint protocol:

* **Spoof rows** are split so each ``(provider, voice_id_or_name)`` lives in
  exactly one of train / val / test. The assignment is greedy bin-packing on
  row counts, stratified per provider, deterministic under ``--seed``.
* **Bonafide rows** keep their original ``split`` (they have no voice axis).
* Each split is required to contain at least one voice per provider so val /
  test never lose a provider entirely.

The output has the same SCHEMA as the input; only ``split`` may differ.
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src import common

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def collect_voice_row_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Return {provider: {voice: row_count}} for spoof rows only."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r.get("audio_label") != "spoof":
            continue
        prov = r.get("provider", "")
        voice = r.get("voice_id_or_name", "")
        out[prov][voice] += 1
    return {p: dict(c) for p, c in out.items()}


def split_voices_for_provider(
    voice_counts: dict[str, int],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    """Greedy bin-pack voices into splits matching row-count ratios.

    Sorts voices by row count descending then assigns each to the split whose
    remaining row-count budget is largest. ``rng`` only breaks ties between
    voices of equal count, so the assignment is reproducible under a fixed
    seed. Raises ValueError if any split would receive zero voices.
    """
    total_rows = sum(voice_counts.values())
    budgets = {s: ratios[s] * total_rows for s in SPLITS}

    voices = list(voice_counts.items())
    rng.shuffle(voices)
    voices.sort(key=lambda kv: kv[1], reverse=True)

    assignment: dict[str, str] = {}
    used_rows = {s: 0 for s in SPLITS}
    for voice, count in voices:
        target = max(SPLITS, key=lambda s: budgets[s] - used_rows[s])
        assignment[voice] = target
        used_rows[target] += count

    voices_per_split = Counter(assignment.values())
    missing = [s for s in SPLITS if voices_per_split[s] == 0]
    if missing:
        raise ValueError(
            f"provider has no voices in split(s) {missing}; "
            f"input voice_counts={voice_counts} budgets={budgets}"
        )
    return assignment


def assign_voice_splits(
    voice_counts_by_provider: dict[str, dict[str, int]],
    ratios: dict[str, float],
    seed: int,
) -> dict[tuple[str, str], str]:
    """Stratify per provider and return {(provider, voice): split}."""
    out: dict[tuple[str, str], str] = {}
    for prov, voice_counts in sorted(voice_counts_by_provider.items()):
        rng = random.Random(f"{seed}|{prov}")
        per_prov = split_voices_for_provider(voice_counts, ratios, rng)
        for voice, split in per_prov.items():
            out[(prov, voice)] = split
    return out


def rewrite_rows(
    rows: list[dict],
    voice_split: dict[tuple[str, str], str],
) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        new = dict(r)
        if r.get("audio_label") == "spoof":
            key = (r.get("provider", ""), r.get("voice_id_or_name", ""))
            if key not in voice_split:
                raise KeyError(
                    f"spoof row {r.get('sample_id')} has unmapped voice {key}"
                )
            new["split"] = voice_split[key]
        out.append(new)
    return out


def summarize(rows: list[dict]) -> str:
    by_split_label: Counter = Counter()
    by_split_provider: Counter = Counter()
    for r in rows:
        s = r.get("split", "")
        by_split_label[(s, r.get("audio_label", ""))] += 1
        by_split_provider[(s, r.get("provider", ""))] += 1
    lines = []
    for s in SPLITS:
        bona = by_split_label[(s, "bonafide")]
        spoof = by_split_label[(s, "spoof")]
        lines.append(f"  {s:5s} bonafide={bona:4d} spoof={spoof:4d}")
        for prov in sorted({p for (sp, p) in by_split_provider if sp == s}):
            lines.append(f"           provider={prov:14s} rows={by_split_provider[(s, prov)]}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_voice_disjoint_manifest",
        description="Rewrite split column under voice-disjoint protocol.",
    )
    p.add_argument("--manifest", type=Path,
                   default=common.AUDIO_SPOOF_MANIFEST_CODEC_MATCHED)
    p.add_argument("--out-manifest", type=Path,
                   default=common.AUDIO_SPOOF_MANIFEST_VOICE_SPLIT)
    p.add_argument("--seed", type=int, default=common.SEED)
    p.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    p.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"])
    p.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {total}")

    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"manifest={args.manifest} rows={len(rows)}")

    voice_counts = collect_voice_row_counts(rows)
    voice_split = assign_voice_splits(voice_counts, ratios, args.seed)

    print(f"distinct voices per provider:")
    for prov, vs in sorted(voice_counts.items()):
        per_split = Counter(voice_split[(prov, v)] for v in vs)
        print(f"  {prov:14s} total_voices={len(vs):3d} per_split={dict(per_split)}")

    rewritten = rewrite_rows(rows, voice_split)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rewritten)

    print(f"wrote {args.out_manifest}")
    print("row counts per split:")
    print(summarize(rewritten))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
