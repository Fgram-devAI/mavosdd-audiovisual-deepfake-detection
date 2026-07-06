"""Fetch the pretrained SyncNet checkpoint (Prajwal/Wav2Lip lineage).

Manual placement fallback:
    Place the checkpoint file at ``<repo>/models/checkpoints/syncnet_pretrained/syncnet.pt``
    with SHA256 matching ``EXPECTED_SHA256`` below, then re-run this script.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

from src.common import CKPT_SYNCNET_DIR, SYNCNET_CKPT_PATH

# The original upstream README exposes the Wav2Lip weights through a Google
# Drive folder, which urllib cannot download reliably without an interactive
# gdown flow, and the folder's contents have been reshuffled to remove the
# standalone SyncNet expert. camenduru/Wav2Lip mirrors the original checkpoints
# (including lipsync_expert.pth, 197 MB) on the HuggingFace CDN over plain
# HTTPS, so we pin that as the default direct URL.
DOWNLOAD_URL = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/lipsync_expert.pth"
UPSTREAM_WEIGHTS_PAGE = "https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
# Update EXPECTED_SHA256 to the actual hash after the first successful fetch;
# operators can verify by running `shasum -a 256 <path>` on the downloaded file.
EXPECTED_SHA256 = "9b9936c721696446eeed353032cab242a8cf0eed8c46cde540366f6ae5493be5"
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
    if not url:
        raise RuntimeError(
            "No stable direct SyncNet checkpoint URL is pinned. Download the "
            f"Wav2Lip/SyncNet expert weights from {UPSTREAM_WEIGHTS_PAGE} "
            "(upstream filename inside the Google Drive folder is "
            f"'lipsync_expert.pth'), place the file at {dest} (rename if "
            "needed), then re-run this script with --expected-sha256 set to "
            "the file's SHA256 to verify manual placement. If you have a "
            "direct file URL, pass it with --url and --bootstrap-sha256."
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
            "Fetch the pretrained SyncNet checkpoint. Idempotent (skips when the "
            f"file at {SYNCNET_CKPT_PATH} already matches EXPECTED_SHA256). "
            "Manual placement: download the Wav2Lip/SyncNet expert checkpoint "
            f"from {UPSTREAM_WEIGHTS_PAGE} (upstream filename inside the Google "
            "Drive folder is 'lipsync_expert.pth'), drop it at that path "
            "(renaming as needed), and re-run. Pass --url only when you have a "
            "direct file URL."
        ),
    )
    parser.add_argument("--url", default=DOWNLOAD_URL)
    parser.add_argument("--dest", type=Path, default=SYNCNET_CKPT_PATH)
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
        print(f"[download_syncnet_checkpoint] failed: {e}", file=sys.stderr)
        return 1
    if actual:
        print(
            "[download_syncnet_checkpoint] bootstrap SHA256 computed; "
            f"update EXPECTED_SHA256 to: {actual}"
        )
    print(f"[download_syncnet_checkpoint] ok: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
