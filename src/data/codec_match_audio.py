"""CLI: match audio codec footprint across bonafide and spoof rows.

For bonafide rows the source WAV is round-tripped through MP3 at a codec spec
sampled deterministically per-row from the spoof codec distribution; for spoof
rows the MP3 is decoded straight to WAV. All outputs are 16 kHz mono PCM WAV
keyed by sample_id, and a derived manifest is written with audio_path columns
repointed at the new tree so the existing embedding extractors can consume it.

Run::

    python -m src.data.codec_match_audio \\
        --manifest data/derived/audio_spoof_manifest.csv \\
        --out-dir data/audio_wav_codec_matched \\
        --out-manifest data/derived/audio_spoof_manifest_codec_matched.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from src import common

logger = logging.getLogger(__name__)

# Codec parameters of each TTS provider's MP3 output. These were probed from
# the on-disk files (ffprobe) and are stable per provider. Bonafide rows are
# randomly assigned one of these specs in proportion to the spoof row count
# per provider, so the codec footprint becomes label-independent.
PROVIDER_CODEC: dict[str, tuple[int, str]] = {
    "elevenlabs": (44100, "128k"),
    "google_tts": (24000, "64k"),
    "elevenlabs_sts": (44100, "128k"),
}

TARGET_SR = 16000


def _run_ffmpeg(args: list[str]) -> None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        capture_output=True, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace").strip())


def encode_mp3(wav_in: Path, mp3_out: Path, sr: int, bitrate: str) -> None:
    _run_ffmpeg([
        "-i", str(wav_in),
        "-ar", str(sr), "-ac", "1",
        "-c:a", "libmp3lame", "-b:a", bitrate,
        str(mp3_out),
    ])


def decode_to_wav16k(audio_in: Path, wav_out: Path) -> None:
    _run_ffmpeg([
        "-i", str(audio_in),
        "-ar", str(TARGET_SR), "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_out),
    ])


def codec_for_bonafide(
    sample_id: str, provider_weights: dict[str, int]
) -> tuple[str, int, str]:
    """Return (provider, sr, bitrate) for a bonafide row, sampled from the spoof distribution.

    SHA-1 of the sample_id picks a deterministic point on the cumulative
    provider-count line, so the marginal codec distribution over bonafide
    converges to the spoof distribution regardless of iteration order.
    """
    h = int(hashlib.sha1(sample_id.encode()).hexdigest(), 16)
    total = sum(provider_weights.values())
    if total == 0:
        raise ValueError("provider_weights is empty; no spoof rows in manifest")
    point = h % total
    cum = 0
    for prov in sorted(provider_weights):
        cum += provider_weights[prov]
        if point < cum:
            sr, br = PROVIDER_CODEC[prov]
            return prov, sr, br
    raise AssertionError("unreachable")


def process_row(
    row: dict,
    out_dir: Path,
    provider_weights: dict[str, int],
    overwrite: bool,
    tmpdir: Path,
) -> str:
    sid = row["sample_id"]
    out_path = out_dir / f"{sid}.wav"
    if out_path.exists() and not overwrite:
        return "skipped"
    in_path_str = row.get("audio_path") or ""
    if not in_path_str:
        tqdm.write(f"[FAIL] {sid}: audio_path is empty")
        return "failed"
    in_path = Path(in_path_str)
    if not in_path.exists():
        tqdm.write(f"[FAIL] {sid}: input missing {in_path}")
        return "failed"

    try:
        if row.get("audio_label") == "bonafide":
            _, sr, br = codec_for_bonafide(sid, provider_weights)
            mp3_tmp = tmpdir / f"{sid}.mp3"
            encode_mp3(in_path, mp3_tmp, sr, br)
            decode_to_wav16k(mp3_tmp, out_path)
            mp3_tmp.unlink(missing_ok=True)
        else:
            decode_to_wav16k(in_path, out_path)
        return "written"
    except Exception as exc:  # noqa: BLE001 — extractor must continue past per-row errors
        tqdm.write(f"[FAIL] {sid}: {exc}")
        if out_path.exists():
            out_path.unlink()
        return "failed"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codec_match_audio",
        description="Round-trip bonafide WAVs through MP3 to match the spoof codec footprint.",
    )
    p.add_argument("--manifest", type=Path, default=common.AUDIO_SPOOF_MANIFEST)
    p.add_argument("--out-dir", type=Path,
                   default=common.AUDIO_WAV_CODEC_MATCHED_DIR)
    p.add_argument("--out-manifest", type=Path,
                   default=common.AUDIO_SPOOF_MANIFEST_CODEC_MATCHED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N manifest rows (for smoke tests).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report planned codec assignments per row and exit without ffmpeg calls.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    provider_weights: dict[str, int] = {}
    bona_count = spoof_count = 0
    for r in rows:
        lbl = r.get("audio_label")
        if lbl == "spoof":
            prov = r.get("provider", "")
            provider_weights[prov] = provider_weights.get(prov, 0) + 1
            spoof_count += 1
        elif lbl == "bonafide":
            bona_count += 1
    unknown = [p for p in provider_weights if p not in PROVIDER_CODEC]
    if unknown:
        raise SystemExit(f"unknown provider(s) in manifest, add to PROVIDER_CODEC: {unknown}")

    print(f"manifest={args.manifest} rows={len(rows)} "
          f"bonafide={bona_count} spoof={spoof_count}")
    print(f"spoof codec distribution: {provider_weights}")

    if args.dry_run:
        plan: dict[tuple[int, str], int] = {}
        for r in rows:
            if r.get("audio_label") == "bonafide":
                _, sr, br = codec_for_bonafide(r["sample_id"], provider_weights)
                plan[(sr, br)] = plan.get((sr, br), 0) + 1
        print(f"dry-run bonafide codec plan: {plan}")
        return 0

    counts = {"written": 0, "skipped": 0, "failed": 0}
    with tempfile.TemporaryDirectory(prefix="codec-match-") as td:
        tmpdir = Path(td)
        for row in tqdm(rows, desc="codec-match", unit="row"):
            counts[process_row(row, args.out_dir, provider_weights, args.overwrite, tmpdir)] += 1

    fieldnames = list(rows[0].keys()) if rows else []
    with args.out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            new = dict(r)
            new["audio_path"] = str(args.out_dir / f"{r['sample_id']}.wav")
            w.writerow(new)

    print(
        f"out_dir={args.out_dir} out_manifest={args.out_manifest} "
        f"written={counts['written']} skipped={counts['skipped']} failed={counts['failed']}"
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
