"""Generate OpenAI TTS audio from transcript JSON files.

Defaults to source-folder real and rotates through the OpenAI voice pool.
Use --estimate-only first to count selected characters before spending OpenAI
credit. Native WAV output keeps codec consistent with bonafide.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_OPENAI_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]


# $0.015 / 1k characters for tts-1, $0.030 / 1k for tts-1-hd (Jan 2026 pricing).
PRICE_PER_MILLION_CHARS = {
    "tts-1": 15.0,
    "tts-1-hd": 30.0,
    "gpt-4o-mini-tts": 0.60,
}


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


def import_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install with:\n  pip install openai>=1.0.0"
        ) from exc
    return OpenAI


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
    names = list(DEFAULT_OPENAI_VOICES)
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
        raise SystemExit("No OpenAI TTS voice names configured.")
    return deduped


def assign_voice(idx: int, voice_names: list[str]) -> str:
    return voice_names[idx % len(voice_names)]


def output_path_for(out_dir: Path, record: dict[str, Any], voice_name: str) -> Path:
    filename = f"{record['video_id']}__voice-{voice_name}.wav"
    return out_dir / "openai_tts" / record["source_folder"] / filename


def existing_output_for(out_dir: Path, record: dict[str, Any]) -> Path | None:
    folder = out_dir / "openai_tts" / record["source_folder"]
    matches = sorted(folder.glob(f"{record['video_id']}__voice-*.wav"))
    return matches[0] if matches else None


def synthesize_openai(
    client: Any,
    text: str,
    output_path: Path,
    voice_name: str,
    model: str,
    response_format: str,
    speed: float,
    timeout: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice_name,
        input=text,
        response_format=response_format,
        speed=speed,
        timeout=timeout,
    ) as response:
        response.stream_to_file(str(output_path))


def write_manifest(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    manifest_path = out_dir / "openai_tts_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenAI TTS audio from transcript JSON files.")
    parser.add_argument("--env-file", default=".env", type=Path)
    parser.add_argument("--transcript-dir", default="data/transcripts/google_stt_v2", type=Path)
    parser.add_argument("--out-dir", default="data/tts_audio", type=Path)
    parser.add_argument(
        "--source-folder",
        action="append",
        default=["real"],
        help="Restrict to source folder/class. Defaults to real. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--voice-name", action="append", default=None, help="Override voice pool. Repeatable.")
    parser.add_argument("--voice-file", type=Path, default=None, help="Optional newline-delimited voice names.")
    parser.add_argument("--model", default="tts-1", choices=("tts-1", "tts-1-hd", "gpt-4o-mini-tts"))
    parser.add_argument("--response-format", default="wav", choices=("wav", "mp3", "flac", "opus", "aac", "pcm"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    load_env_file(args.env_file)
    voice_names = load_voice_names(args.voice_name, args.voice_file)
    records = collect_transcripts(args.transcript_dir, args.source_folder, args.limit, args.max_chars)

    source_counts: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()
    total_chars = 0
    for idx, record in enumerate(records):
        source_counts[record["source_folder"]] += 1
        voice_counts[assign_voice(idx, voice_names)] += 1
        total_chars += len(record["text"])
    price = PRICE_PER_MILLION_CHARS.get(args.model, 15.0)
    estimate = (total_chars / 1_000_000.0) * price

    print(
        f"Selected {len(records)} transcripts | {total_chars} chars | "
        f"{len(voice_names)} voices | model={args.model} | "
        f"sources={dict(source_counts)} | estimated cost ${estimate:.4f}"
    )
    if records:
        print(f"Voice spread: {dict(voice_counts)}")
    if args.estimate_only:
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to .env or export it before running."
        )
    OpenAI = import_openai_client()
    client = OpenAI(api_key=api_key)

    created = 0
    skipped = 0
    failed = 0
    manifest_records: list[dict[str, Any]] = []
    for idx, record in enumerate(tqdm(records, desc="openai tts", unit="file")):
        voice_name = assign_voice(idx, voice_names)
        existing = existing_output_for(args.out_dir, record) if not args.overwrite else None
        if existing is not None:
            skipped += 1
            voice_name = existing.stem.rsplit("__voice-", 1)[-1]
            output_path = existing
        else:
            output_path = output_path_for(args.out_dir, record, voice_name)
            try:
                synthesize_openai(
                    client=client,
                    text=record["text"],
                    output_path=output_path,
                    voice_name=voice_name,
                    model=args.model,
                    response_format=args.response_format,
                    speed=args.speed,
                    timeout=args.timeout,
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
                "provider": "openai_tts",
                "model": args.model,
                "voice_name": voice_name,
                "text_chars": len(record["text"]),
                "transcript_path": str(record["path"]),
                "original_audio_path": record["audio_path"],
                "synthetic_audio_path": str(output_path),
            }
        )

    manifest_path = write_manifest(args.out_dir, manifest_records)
    print(
        "OpenAI TTS complete: "
        f"created={created} skipped={skipped} failed={failed} selected={len(records)} "
        f"chars={total_chars} manifest={manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
