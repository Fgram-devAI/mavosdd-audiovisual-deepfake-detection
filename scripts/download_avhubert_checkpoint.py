"""Fetch the pretrained AV-HuBERT checkpoint (Meta fairseq).

Manual placement fallback:
    Place the checkpoint file at ``<repo>/models/checkpoints/avhubert_pretrained/avhubert_base.pt``
    with SHA256 matching ``EXPECTED_SHA256`` below, then re-run this script.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

from src.common import CKPT_AVHUBERT_DIR, AVHUBERT_CKPT_PATH

# Pinned URL for the Meta fairseq AV-HuBERT base checkpoint.
# Source: https://facebookresearch.github.io/av_hubert/ ("AV-HuBERT Base",
# LRS3, no finetuning).
DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/base_lrs3_iter5.pt"
# Update EXPECTED_SHA256 to the actual hash after the first successful fetch;
# operators can verify by running `shasum -a 256 <path>` on the downloaded file.
EXPECTED_SHA256 = "94f26cf6789b356b42e0a2555593b2b3495f07171161cdf770ba0aa21e13d110"
_SHA_PLACEHOLDER_PREFIX = "REPLACE_WITH_ACTUAL_SHA256"


def _sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(
    dest: Path,
    *,
    expected_sha256: str,
    url: str,
    chunk_size: int = 1_048_576,
    allow_placeholder_sha: bool = False,
) -> str | None:
    """Download ``url`` to ``dest``, verify SHA256, remove partial on mismatch.

    Returns the computed SHA256 when bootstrapping an unpinned checkpoint.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        actual_existing = _sha256(dest)
        if actual_existing == expected_sha256:
            return None
        if expected_sha256.startswith(_SHA_PLACEHOLDER_PREFIX):
            return actual_existing
        raise RuntimeError(
            f"sha256 mismatch for existing {dest}: expected {expected_sha256}, "
            f"got {actual_existing}. Delete or replace the file, or pass the "
            "correct --expected-sha256."
        )
    if expected_sha256.startswith(_SHA_PLACEHOLDER_PREFIX) and not allow_placeholder_sha:
        raise RuntimeError(
            "EXPECTED_SHA256 is still a placeholder. Pass --bootstrap-sha256 "
            "for the first download, then commit the printed SHA256 pin."
        )
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(url) as resp, tmp.open("wb") as out:
            try:
                total = int(resp.headers.get("Content-Length") or 0) or None
            except (TypeError, ValueError):
                total = None
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar:
                while True:
                    block = resp.read(chunk_size)
                    if not block:
                        break
                    out.write(block)
                    bar.update(len(block))
        actual = _sha256(tmp)
        if expected_sha256.startswith(_SHA_PLACEHOLDER_PREFIX) and allow_placeholder_sha:
            tmp.replace(dest)
            return actual
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {dest}: expected {expected_sha256}, got {actual}. "
                f"Place the file manually at {dest} and re-run."
            )
        tmp.replace(dest)
        return None
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the pretrained AV-HuBERT checkpoint. Idempotent (skips when the "
            f"file at {AVHUBERT_CKPT_PATH} already matches EXPECTED_SHA256). "
            "Manual placement: drop the file at that path with the expected SHA256 and re-run."
        ),
    )
    parser.add_argument("--url", default=DOWNLOAD_URL)
    parser.add_argument("--dest", type=Path, default=AVHUBERT_CKPT_PATH)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument(
        "--bootstrap-sha256",
        action="store_true",
        help=(
            "Allow the placeholder EXPECTED_SHA256 on the first download, keep the "
            "file, and print the computed SHA256 so it can be pinned."
        ),
    )
    args = parser.parse_args(argv)
    try:
        actual = download(
            args.dest,
            expected_sha256=args.expected_sha256,
            url=args.url,
            allow_placeholder_sha=args.bootstrap_sha256,
        )
    except Exception as e:
        print(f"[download_avhubert_checkpoint] failed: {e}", file=sys.stderr)
        return 1
    if actual:
        print(
            "[download_avhubert_checkpoint] bootstrap SHA256 computed; "
            f"update EXPECTED_SHA256 to: {actual}"
        )
    print(f"[download_avhubert_checkpoint] ok: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
