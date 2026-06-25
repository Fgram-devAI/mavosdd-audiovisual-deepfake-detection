"""Generate Coqui XTTS-v2 audio from transcript JSON files (local, free).

XTTS-v2 is a multilingual zero-shot voice-clone model. We use its 58 built-in
studio voices in deterministic round-robin so the voice-disjoint split logic
can confine each voice to one split, matching the ElevenLabs/Google-TTS
pipelines.

Native 24 kHz WAV output. Resampled to 16 kHz by codec-match before training.

First-run model download is ~1.8 GB into the Coqui cache. The accept-license
prompt is suppressed via the COQUI_TOS_AGREED env var, which is set by this
script before importing TTS.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


# The 58 built-in XTTS-v2 studio voices (English-friendly subset documented by
# Coqui as the "studio_speakers" pool). Rotating through these gives stable,
# voice-disjoint-friendly identifiers.
DEFAULT_XTTS_VOICES = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Andrew Chipper",
    "Badr Odhiambo",
    "Dionisio Schuyler",
    "Royston Min",
    "Viktor Eka",
    "Abrahan Mack",
    "Adde Michal",
    "Baldur Sanjin",
    "Craig Gutsy",
    "Damien Black",
    "Gilberto Mathias",
    "Ilkin Urbano",
    "Kazuhiko Atallah",
    "Ludvig Milivoj",
    "Suad Qasim",
    "Torcull Diarmuid",
    "Viktor Menelaos",
    "Zacharie Aimilios",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def import_xtts():
    # Coqui's license-prompt blocks scripts; pre-accept before import.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    try:
        from TTS.api import TTS  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: TTS (Coqui). Install with:\n"
            "  pip install 'TTS>=0.22.0'\n"
            "Requires Python 3.9-3.11."
        ) from exc
    return TTS


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_transcript(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text())
    text = str(payload.get("transcript", "")).strip()
    source_folder = str(payload.get("source_folder", path.parent.name)).strip().lower()
    video_id = str(payload.get("video_id", path.stem)).strip()
    if not text:
        return None
    return {
        "path": path,
        "source_folder": source_folder,
        "video_id": video_id,
        "text": text,
        "audio_path": payload.get("audio_path"),
    }


def collect_transcripts(
    transcript_dir: Path,
    source_folders: list[str] | None,
    limit: int | None,
    max_chars: int | None,
) -> list[dict[str, Any]]:
    allowed = {folder.lower() for folder in source_folders or []}
    records: list[dict[str, Any]] = []
    selected_chars = 0
    for path in sorted(transcript_dir.glob("*/*.json")):
        record = read_transcript(path)
        if record is None:
            continue
        if allowed and record["source_folder"] not in allowed:
            continue
        text_chars = len(record["text"])
        if max_chars is not None and selected_chars + text_chars > max_chars:
            break
        records.append(record)
        selected_chars += text_chars
        if limit is not None and len(records) >= limit:
            break
    return records


def load_voice_names(raw: list[str] | None, voice_file: Path | None) -> list[str]:
    names = list(DEFAULT_XTTS_VOICES)
    if voice_file:
        names = [
            line.strip()
            for line in voice_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if raw:
        names = [n.strip() for n in raw if n.strip()]
    deduped = list(dict.fromkeys(names))
    if not deduped:
        raise SystemExit("No XTTS voice names configured.")
    return deduped


def assign_voice(idx: int, voice_names: list[str]) -> str:
    return voice_names[idx % len(voice_names)]


def voice_slug(voice_name: str) -> str:
    return voice_name.lower().replace(" ", "-")


def output_path_for(out_dir: Path, record: dict[str, Any], voice_name: str) -> Path:
    filename = f"{record['video_id']}__voice-{voice_slug(voice_name)}.wav"
    return out_dir / "coqui_xtts" / record["source_folder"] / filename


def existing_output_for(out_dir: Path, record: dict[str, Any]) -> Path | None:
    folder = out_dir / "coqui_xtts" / record["source_folder"]
    matches = sorted(folder.glob(f"{record['video_id']}__voice-*.wav"))
    return matches[0] if matches else None


def synthesize_xtts(
    tts: Any,
    text: str,
    output_path: Path,
    voice_name: str,
    language: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=text,
        speaker=voice_name,
        language=language,
        file_path=str(output_path),
    )


def write_manifest(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    manifest_path = out_dir / "coqui_xtts_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Coqui XTTS-v2 audio from transcript JSON files.")
    parser.add_argument("--env-file", default=".env", type=Path)
    parser.add_argument("--transcript-dir", default="data/transcripts/google_stt_v2", type=Path)
    parser.add_argument("--out-dir", default="data/tts_audio", type=Path)
    parser.add_argument(
        "--source-folder",
        action="append",
        default=None,
        help="Restrict to source folder/class. Defaults to real. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--voice-name", action="append", default=None, help="Override voice pool. Repeatable.")
    parser.add_argument("--voice-file", type=Path, default=None, help="Optional newline-delimited voice names.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    args = parser.parse_args()

    load_env_file(args.env_file)
    voice_names = load_voice_names(args.voice_name, args.voice_file)
    source_folders = args.source_folder if args.source_folder is not None else ["real"]
    records = collect_transcripts(args.transcript_dir, source_folders, args.limit, args.max_chars)

    source_counts: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()
    total_chars = 0
    for idx, record in enumerate(records):
        source_counts[record["source_folder"]] += 1
        voice_counts[assign_voice(idx, voice_names)] += 1
        total_chars += len(record["text"])

    print(
        f"Selected {len(records)} transcripts | {total_chars} chars | "
        f"{len(voice_names)} voices | sources={dict(source_counts)} | "
        f"local compute only (no API cost)"
    )
    if records:
        print(f"Voice spread: {dict(voice_counts)}")
    if args.estimate_only:
        return 0

    TTS = import_xtts()
    device = pick_device(args.device)
    print(f"Loading {args.model_name} on device={device} (first run downloads ~1.8 GB)...")
    tts = TTS(args.model_name).to(device)

    created = 0
    skipped = 0
    failed = 0
    manifest_records: list[dict[str, Any]] = []
    for idx, record in enumerate(tqdm(records, desc="coqui xtts", unit="file")):
        voice_name = assign_voice(idx, voice_names)
        existing = existing_output_for(args.out_dir, record) if not args.overwrite else None
        if existing is not None:
            skipped += 1
            output_path = existing
        else:
            output_path = output_path_for(args.out_dir, record, voice_name)
            try:
                synthesize_xtts(
                    tts=tts,
                    text=record["text"],
                    output_path=output_path,
                    voice_name=voice_name,
                    language=args.language,
                )
                created += 1
            except Exception as exc:
                failed += 1
                tqdm.write(f"[FAIL] {record['path']}: {exc}")
                continue

        manifest_records.append(
            {
                "source_folder": record["source_folder"],
                "video_id": record["video_id"],
                "provider": "coqui_xtts",
                "model": args.model_name,
                "voice_name": voice_name,
                "language": args.language,
                "text_chars": len(record["text"]),
                "transcript_path": str(record["path"]),
                "original_audio_path": record["audio_path"],
                "synthetic_audio_path": str(output_path),
            }
        )

    manifest_path = write_manifest(args.out_dir, manifest_records)
    print(
        "Coqui XTTS-v2 complete: "
        f"created={created} skipped={skipped} failed={failed} selected={len(records)} "
        f"chars={total_chars} manifest={manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
