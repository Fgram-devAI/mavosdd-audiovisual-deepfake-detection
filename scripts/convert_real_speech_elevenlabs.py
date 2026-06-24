"""Create speech-to-speech variants from real WAV audio with ElevenLabs.

This is separate from transcript-based TTS:
    TTS: text transcript -> synthetic speech
    STS: original real speech audio -> converted synthetic speech

Use --estimate-only first. Outputs are resumable and skipped unless
--overwrite is used.
"""
from __future__ import annotations

import argparse
import json
import os
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_ELEVENLABS_VOICE_IDS = [
    "hpp4J3VqNfWAUOO0d1Us",
    "CwhRBWXzGAHq8TQ4Fs17",
    "EXAVITQu4vr4xnSDxMaL",
    "FGY2WhTYpPnrIDTdsKH5",
    "IKne3meq5aSn9XLyUdCD",
    "JBFqnCBsd6RMkjVDRZzb",
    "N2lVS1w4EtoT3dr4eOWO",
    "SAz9YHcvj6GT2YYXdXww",
    "SOYHLrjzK2X1ezoPC6cr",
    "TX3LPaxmHKxFdv7VOQHJ",
    "Xb7hH8MSUJpSbSDYk0k2",
    "bIHbv24MWmeRgasZH58o",
    "cjVigY5qzO86Huf0OWal",
    "nPczCjzI2devNBz1zQrb",
    "onwK4e9ZLuTAKqWW03F9",
    "pqHfZKP75CvOlQylNhV4",
    "pNInz6obpgDQGcFmaJgB",
    "pFZP5JQG7iQjIQuC4Bku",
]


class ElevenLabsError(RuntimeError):
    def __init__(self, status_code: int, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


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


def import_requests():
    try:
        import requests
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: requests. Install requirements with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return requests


def load_voice_ids(raw_voice_ids: list[str] | None, voice_file: Path | None) -> list[str]:
    voice_ids = list(DEFAULT_ELEVENLABS_VOICE_IDS)
    if voice_file:
        voice_ids = [
            line.strip()
            for line in voice_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if raw_voice_ids:
        voice_ids = [voice_id.strip() for voice_id in raw_voice_ids if voice_id.strip()]
    seen = set()
    deduped = []
    for voice_id in voice_ids:
        if voice_id not in seen:
            deduped.append(voice_id)
            seen.add(voice_id)
    if not deduped:
        raise SystemExit("No ElevenLabs voice ids configured.")
    return deduped


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def collect_wavs(
    wav_dir: Path,
    source_folders: list[str] | None,
    limit: int | None,
    max_seconds: float | None,
) -> list[Path]:
    allowed = {folder.lower() for folder in source_folders or []}
    selected = []
    selected_seconds = 0.0
    for path in sorted(wav_dir.glob("*/*.wav")):
        source_folder = path.parent.name.lower()
        if allowed and source_folder not in allowed:
            continue
        seconds = wav_seconds(path)
        if max_seconds is not None and selected_seconds + seconds > max_seconds:
            break
        selected.append(path)
        selected_seconds += seconds
        if limit is not None and len(selected) >= limit:
            break
    return selected


def assign_voice(record_index: int, voice_ids: list[str]) -> str:
    return voice_ids[record_index % len(voice_ids)]


def next_voice_id(start_index: int, voice_ids: list[str], disabled_voice_ids: set[str]) -> tuple[str, int] | None:
    for offset in range(len(voice_ids)):
        index = (start_index + offset) % len(voice_ids)
        voice_id = voice_ids[index]
        if voice_id not in disabled_voice_ids:
            return voice_id, index
    return None


def output_path_for(out_dir: Path, wav_dir: Path, wav_path: Path, voice_id: str) -> Path:
    rel = wav_path.relative_to(wav_dir)
    filename = f"{rel.stem}__voice-{voice_id}.mp3"
    return out_dir / "elevenlabs_sts" / rel.parent / filename


def convert_speech_elevenlabs(
    requests: Any,
    wav_path: Path,
    output_path: Path,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    timeout: float,
    stability: float,
    similarity_boost: float,
) -> None:
    url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}"
    with wav_path.open("rb") as audio:
        response = requests.post(
            url,
            headers={
                "xi-api-key": api_key,
                "accept": "audio/mpeg",
            },
            params={"output_format": output_format},
            data={
                "model_id": model_id,
                "voice_settings": json.dumps(
                    {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                    }
                ),
            },
            files={"audio": (wav_path.name, audio, "audio/wav")},
            timeout=timeout,
        )
    if response.status_code >= 400:
        code = None
        message = response.text[:500]
        try:
            detail = response.json().get("detail", {})
            if isinstance(detail, dict):
                code = detail.get("code")
                message = detail.get("message", message)
        except ValueError:
            pass
        raise ElevenLabsError(response.status_code, f"ElevenLabs HTTP {response.status_code}: {message}", code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def write_manifest(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    manifest_path = out_dir / "sts_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ElevenLabs speech-to-speech variants from WAV files.")
    parser.add_argument("--env-file", default=".env", type=Path)
    parser.add_argument("--wav-dir", default="data/audio_wav", type=Path)
    parser.add_argument("--out-dir", default="data/tts_audio", type=Path)
    parser.add_argument(
        "--source-folder",
        action="append",
        default=["real"],
        help="Restrict to source folder/class. Defaults to real. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--voice-id", action="append", default=None)
    parser.add_argument("--voice-file", type=Path, default=None)
    parser.add_argument("--elevenlabs-model", default="eleven_multilingual_sts_v2")
    parser.add_argument("--elevenlabs-output-format", default="mp3_44100_128")
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-retry-voice", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    voice_ids = load_voice_ids(args.voice_id, args.voice_file)
    wavs = collect_wavs(args.wav_dir, args.source_folder, args.limit, args.max_seconds)

    source_counts = Counter(path.parent.name for path in wavs)
    voice_counts = Counter(assign_voice(index, voice_ids) for index, _ in enumerate(wavs))
    total_seconds = sum(wav_seconds(path) for path in wavs)
    print(
        f"Selected {len(wavs)} WAV files | {total_seconds / 60.0:.2f} minutes | "
        f"{len(voice_ids)} voices | sources={dict(source_counts)}"
    )
    if wavs:
        print(f"Voice spread: {dict(voice_counts)}")
    if args.estimate_only:
        return 0

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Set ELEVENLABS_API_KEY in .env or the environment.")
    requests = import_requests()

    created = 0
    skipped = 0
    failed = 0
    disabled_voice_ids: set[str] = set()
    manifest_records: list[dict[str, Any]] = []

    for index, wav_path in enumerate(tqdm(wavs, desc="elevenlabs sts", unit="file")):
        voice_choice = next_voice_id(index, voice_ids, disabled_voice_ids)
        if voice_choice is None:
            failed += 1
            tqdm.write(f"[FAIL] {wav_path}: no usable ElevenLabs voices remain")
            continue
        voice_id, voice_index = voice_choice
        output_path = output_path_for(args.out_dir, args.wav_dir, wav_path, voice_id)
        if output_path.exists() and not args.overwrite:
            skipped += 1
        else:
            while True:
                try:
                    convert_speech_elevenlabs(
                        requests=requests,
                        wav_path=wav_path,
                        output_path=output_path,
                        api_key=api_key,
                        voice_id=voice_id,
                        model_id=args.elevenlabs_model,
                        output_format=args.elevenlabs_output_format,
                        timeout=args.timeout,
                        stability=args.stability,
                        similarity_boost=args.similarity_boost,
                    )
                    created += 1
                    break
                except ElevenLabsError as exc:
                    if exc.code == "paid_plan_required" and not args.no_retry_voice:
                        disabled_voice_ids.add(voice_id)
                        tqdm.write(f"[VOICE SKIP] {voice_id}: {exc}")
                        voice_choice = next_voice_id(voice_index + 1, voice_ids, disabled_voice_ids)
                        if voice_choice is None:
                            failed += 1
                            tqdm.write(f"[FAIL] {wav_path}: no usable ElevenLabs voices remain")
                            break
                        voice_id, voice_index = voice_choice
                        output_path = output_path_for(args.out_dir, args.wav_dir, wav_path, voice_id)
                        continue
                    failed += 1
                    tqdm.write(f"[FAIL] {wav_path}: {exc}")
                    break
                except Exception as exc:
                    failed += 1
                    tqdm.write(f"[FAIL] {wav_path}: {exc}")
                    break
            if failed and not output_path.exists():
                continue

        manifest_records.append(
            {
                "source_folder": wav_path.parent.name,
                "video_id": wav_path.stem,
                "provider": "elevenlabs",
                "method": "speech_to_speech",
                "voice_id": voice_id,
                "model_id": args.elevenlabs_model,
                "output_format": args.elevenlabs_output_format,
                "duration_seconds": wav_seconds(wav_path),
                "original_audio_path": str(wav_path),
                "synthetic_audio_path": str(output_path),
            }
        )

    manifest_path = write_manifest(args.out_dir, manifest_records)
    print(
        "ElevenLabs STS complete: "
        f"created={created} skipped={skipped} failed={failed} selected={len(wavs)} "
        f"minutes={total_seconds / 60.0:.2f} disabled_voices={len(disabled_voice_ids)} "
        f"manifest={manifest_path}"
    )
    if disabled_voice_ids:
        print(f"Disabled voices: {sorted(disabled_voice_ids)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
