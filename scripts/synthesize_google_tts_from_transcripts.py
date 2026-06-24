"""Generate Google Cloud Text-to-Speech audio from transcript JSON files.

Defaults to source-folder real and rotates through a small pool of en-US voices.
Use --estimate-only first to count selected characters before spending Google
Cloud credit.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_GOOGLE_VOICES = [
    "en-US-Neural2-A",
    "en-US-Neural2-C",
    "en-US-Neural2-D",
    "en-US-Neural2-E",
    "en-US-Neural2-F",
    "en-US-Neural2-G",
    "en-US-Neural2-H",
    "en-US-Neural2-I",
    "en-US-Neural2-J",
]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without overriding existing env."""
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


def import_google_tts():
    try:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import texttospeech
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: google-cloud-texttospeech. Install requirements with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return texttospeech, DefaultCredentialsError


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


def load_voice_names(raw_voice_names: list[str] | None, voice_file: Path | None) -> list[str]:
    voice_names = list(DEFAULT_GOOGLE_VOICES)
    if voice_file:
        voice_names = [
            line.strip()
            for line in voice_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if raw_voice_names:
        voice_names = [voice_name.strip() for voice_name in raw_voice_names if voice_name.strip()]
    deduped = list(dict.fromkeys(voice_names))
    if not deduped:
        raise SystemExit("No Google TTS voice names configured.")
    return deduped


def assign_voice(record_index: int, voice_names: list[str]) -> str:
    return voice_names[record_index % len(voice_names)]


def output_path_for(out_dir: Path, record: dict[str, Any], voice_name: str) -> Path:
    filename = f"{record['video_id']}__voice-{voice_name}.mp3"
    return out_dir / "google_tts" / record["source_folder"] / filename


def existing_output_for(out_dir: Path, record: dict[str, Any]) -> Path | None:
    folder = out_dir / "google_tts" / record["source_folder"]
    matches = sorted(folder.glob(f"{record['video_id']}__voice-*.mp3"))
    return matches[0] if matches else None


def synthesize_google(
    client: Any,
    texttospeech: Any,
    text: str,
    output_path: Path,
    voice_name: str,
    language_code: str,
    speaking_rate: float,
    pitch: float,
    timeout: float,
) -> None:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch,
    )
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
        timeout=timeout,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.audio_content)


def write_manifest(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    manifest_path = out_dir / "google_tts_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Google TTS audio from transcript JSON files.")
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
    parser.add_argument("--language-code", default="en-US")
    parser.add_argument("--speaking-rate", type=float, default=1.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--price-per-million-chars", type=float, default=16.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    load_env_file(args.env_file)
    voice_names = load_voice_names(args.voice_name, args.voice_file)
    records = collect_transcripts(args.transcript_dir, args.source_folder, args.limit, args.max_chars)

    source_counts: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()
    total_chars = 0
    for index, record in enumerate(records):
        source_counts[record["source_folder"]] += 1
        voice_counts[assign_voice(index, voice_names)] += 1
        total_chars += len(record["text"])
    estimate = (total_chars / 1_000_000.0) * args.price_per_million_chars

    print(
        f"Selected {len(records)} transcripts | {total_chars} chars | "
        f"{len(voice_names)} voices | sources={dict(source_counts)} | "
        f"estimated Neural2 cost ${estimate:.4f}"
    )
    if records:
        print(f"Voice spread: {dict(voice_counts)}")
    if args.estimate_only:
        return 0

    texttospeech, default_credentials_error = import_google_tts()
    try:
        client = texttospeech.TextToSpeechClient()
    except default_credentials_error as exc:
        raise SystemExit(
            "Google Application Default Credentials were not found.\n\n"
            "For local development, run:\n"
            "  gcloud auth application-default login\n\n"
            "Or put a service-account key path in .env:\n"
            "  GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json\n"
        ) from exc

    created = 0
    skipped = 0
    failed = 0
    manifest_records: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(records, desc="google tts", unit="file")):
        voice_name = assign_voice(index, voice_names)
        output_path = existing_output_for(args.out_dir, record) if not args.overwrite else None
        if output_path is not None:
            skipped += 1
            voice_name = output_path.stem.rsplit("__voice-", 1)[-1]
        else:
            output_path = output_path_for(args.out_dir, record, voice_name)
            try:
                synthesize_google(
                    client=client,
                    texttospeech=texttospeech,
                    text=record["text"],
                    output_path=output_path,
                    voice_name=voice_name,
                    language_code=args.language_code,
                    speaking_rate=args.speaking_rate,
                    pitch=args.pitch,
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
                "provider": "google_tts",
                "voice_name": voice_name,
                "language_code": args.language_code,
                "text_chars": len(record["text"]),
                "transcript_path": str(record["path"]),
                "original_audio_path": record["audio_path"],
                "synthetic_audio_path": str(output_path),
            }
        )

    manifest_path = write_manifest(args.out_dir, manifest_records)
    print(
        "Google TTS complete: "
        f"created={created} skipped={skipped} failed={failed} selected={len(records)} "
        f"chars={total_chars} manifest={manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
