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

from src.common import CKPT_SYNCNET_DIR, SYNCNET_CKPT_PATH

# Pinned URL for the Prajwal/Wav2Lip SyncNet expert checkpoint.
DOWNLOAD_URL = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EQRvqxOS4jRLu2YrHK5Q3ycBAZY2S6mM6BUxNsQvY-p6Bw"
# Update EXPECTED_SHA256 to the actual hash after the first successful fetch;
# operators can verify by running `shasum -a 256 <path>` on the downloaded file.
EXPECTED_SHA256 = "REPLACE_WITH_ACTUAL_SHA256_AFTER_FIRST_DOWNLOAD"


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
) -> None:
    """Download ``url`` to ``dest``, verify SHA256, remove partial on mismatch."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _sha256(dest) == expected_sha256:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(url) as resp, tmp.open("wb") as out:
            while True:
                block = resp.read(chunk_size)
                if not block:
                    break
                out.write(block)
        actual = _sha256(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {dest}: expected {expected_sha256}, got {actual}. "
                f"Place the file manually at {dest} and re-run."
            )
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the pretrained SyncNet checkpoint. Idempotent (skips when the "
            f"file at {SYNCNET_CKPT_PATH} already matches EXPECTED_SHA256). "
            "Manual placement: drop the file at that path with the expected SHA256 and re-run."
        ),
    )
    parser.add_argument("--url", default=DOWNLOAD_URL)
    parser.add_argument("--dest", type=Path, default=SYNCNET_CKPT_PATH)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = parser.parse_args(argv)
    try:
        download(args.dest, expected_sha256=args.expected_sha256, url=args.url)
    except Exception as e:
        print(f"[download_syncnet_checkpoint] failed: {e}", file=sys.stderr)
        return 1
    print(f"[download_syncnet_checkpoint] ok: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
