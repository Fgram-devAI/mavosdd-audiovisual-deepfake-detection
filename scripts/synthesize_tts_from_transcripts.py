"""Generate ElevenLabs synthetic speech from transcript JSON files.

The script is resumable and skips existing outputs unless --overwrite is used.
Voice assignment is deterministic: transcripts are sorted, then assigned across
the configured voice pool in round-robin order.

Use --estimate-only before spending API credits.
"""
from __future__ import annotations

import argparse
import json
import os
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


def import_requests():
    try:
        import requests
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: requests. Install requirements with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return requests


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


def elevenlabs_headers(api_key: str) -> dict[str, str]:
    return {
        "xi-api-key": api_key,
        "accept": "application/json",
    }


def get_available_voices(requests: Any, api_key: str, timeout: float) -> list[dict[str, Any]]:
    response = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers=elevenlabs_headers(api_key),
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"ElevenLabs voices HTTP {response.status_code}: {response.text[:500]}")
    voices = response.json().get("voices", [])
    if not isinstance(voices, list):
        raise RuntimeError("ElevenLabs voices response did not contain a voices list.")
    return voices


def free_tier_voice_ids(voices: list[dict[str, Any]]) -> list[str]:
    selected = []
    for voice in voices:
        category = str(voice.get("category", "")).lower()
        sharing = voice.get("sharing") or {}
        if category in {"premade", "cloned", "generated", "professional"} and not sharing:
            voice_id = voice.get("voice_id")
            if voice_id:
                selected.append(str(voice_id))
    return selected


def print_voices(voices: list[dict[str, Any]]) -> None:
    for voice in voices:
        labels = voice.get("labels") or {}
        label_text = ", ".join(f"{key}={value}" for key, value in sorted(labels.items()))
        print(
            f"{voice.get('voice_id')}\t"
            f"{voice.get('name')}\t"
            f"category={voice.get('category')}\t"
            f"{label_text}"
        )


def assign_voice(record_index: int, voice_ids: list[str]) -> str:
    return voice_ids[record_index % len(voice_ids)]


def output_path_for(out_dir: Path, record: dict[str, Any], voice_id: str) -> Path:
    filename = f"{record['video_id']}__voice-{voice_id}.mp3"
    return out_dir / "elevenlabs" / record["source_folder"] / filename


def existing_output_for(out_dir: Path, record: dict[str, Any]) -> Path | None:
    folder = out_dir / "elevenlabs" / record["source_folder"]
    matches = sorted(folder.glob(f"{record['video_id']}__voice-*.mp3"))
    return matches[0] if matches else None


def probe_output_path_for(out_dir: Path, voice_id: str) -> Path:
    return out_dir / "voice_probe" / f"{voice_id}.mp3"


def synthesize_elevenlabs(
    requests: Any,
    text: str,
    output_path: Path,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    timeout: float,
    stability: float,
    similarity_boost: float,
) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(
        url,
        headers={
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        },
        params={"output_format": output_format},
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        },
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
    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def next_voice_id(start_index: int, voice_ids: list[str], disabled_voice_ids: set[str]) -> tuple[str, int] | None:
    for offset in range(len(voice_ids)):
        index = (start_index + offset) % len(voice_ids)
        voice_id = voice_ids[index]
        if voice_id not in disabled_voice_ids:
            return voice_id, index
    return None


def is_quota_error(exc: ElevenLabsError) -> bool:
    code = (exc.code or "").lower()
    message = str(exc).lower()
    markers = ("quota", "credit", "limit", "billing", "subscription")
    return exc.status_code == 429 or any(marker in code or marker in message for marker in markers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ElevenLabs TTS audio from transcript JSON files.")
    parser.add_argument("--env-file", default=".env", type=Path)
    parser.add_argument("--transcript-dir", default="data/transcripts/google_stt_v2", type=Path)
    parser.add_argument("--out-dir", default="data/tts_audio", type=Path)
    parser.add_argument(
        "--source-folder",
        action="append",
        default=None,
        help="Restrict to a source folder/class. Repeatable. Defaults to every transcript folder.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None, help="Stop selection before exceeding this text budget.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--voice-id", action="append", default=None, help="Override voice pool. Repeatable.")
    parser.add_argument("--voice-file", type=Path, default=None, help="Optional newline-delimited voice id file.")
    parser.add_argument("--list-voices", action="store_true", help="Print voices available to this API key and exit.")
    parser.add_argument(
        "--probe-voices",
        action="store_true",
        help="Test each configured voice with a tiny TTS request and exit.",
    )
    parser.add_argument("--probe-text", default="Voice probe.", help="Text used by --probe-voices.")
    parser.add_argument(
        "--use-account-voices",
        action="store_true",
        help="Use available account/premade voices instead of the project voice-id pool.",
    )
    parser.add_argument("--elevenlabs-model", default="eleven_multilingual_v2")
    parser.add_argument("--elevenlabs-output-format", default="mp3_44100_128")
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument(
        "--no-retry-voice",
        action="store_true",
        help="Do not retry a transcript with another voice when a voice is unavailable.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if (args.list_voices or args.use_account_voices or args.probe_voices) and not api_key:
        raise SystemExit("Set ELEVENLABS_API_KEY in .env or the environment.")

    requests = None
    if args.list_voices or args.use_account_voices:
        requests = import_requests()
        voices = get_available_voices(requests, api_key, args.timeout)
        if args.list_voices:
            print_voices(voices)
            return 0
        voice_ids = free_tier_voice_ids(voices)
        if not voice_ids:
            raise SystemExit(
                "No usable non-library voices were found for this account. "
                "Run --list-voices to inspect what ElevenLabs exposes."
            )
    else:
        voice_ids = load_voice_ids(args.voice_id, args.voice_file)

    if args.probe_voices:
        if requests is None:
            requests = import_requests()
        usable = []
        unusable = []
        for voice_id in tqdm(voice_ids, desc="voice probe", unit="voice"):
            try:
                synthesize_elevenlabs(
                    requests=requests,
                    text=args.probe_text,
                    output_path=probe_output_path_for(args.out_dir, voice_id),
                    api_key=api_key,
                    voice_id=voice_id,
                    model_id=args.elevenlabs_model,
                    output_format=args.elevenlabs_output_format,
                    timeout=args.timeout,
                    stability=args.stability,
                    similarity_boost=args.similarity_boost,
                )
                usable.append(voice_id)
            except Exception as exc:
                unusable.append((voice_id, str(exc)))
        print("\nUsable voices:")
        for voice_id in usable:
            print(voice_id)
        print("\nUnusable voices:")
        for voice_id, reason in unusable:
            print(f"{voice_id}\t{reason}")
        print(f"\nSummary: usable={len(usable)} unusable={len(unusable)} total={len(voice_ids)}")
        return 1 if not usable else 0

    records = collect_transcripts(
        transcript_dir=args.transcript_dir,
        source_folders=args.source_folder,
        limit=args.limit,
        max_chars=args.max_chars,
    )

    selected = []
    source_counts: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()
    total_chars = 0
    for index, record in enumerate(records):
        voice_id = assign_voice(index, voice_ids)
        selected.append((index, record, voice_id))
        source_counts[record["source_folder"]] += 1
        voice_counts[voice_id] += 1
        total_chars += len(record["text"])

    print(
        f"Selected {len(selected)} transcripts | {total_chars} chars | "
        f"{len(voice_ids)} voices | sources={dict(source_counts)}"
    )
    if selected:
        print(f"Voice spread: {dict(voice_counts)}")
    if args.estimate_only:
        return 0

    if not api_key:
        raise SystemExit("Set ELEVENLABS_API_KEY in .env or the environment.")

    if requests is None:
        requests = import_requests()
    created = 0
    skipped = 0
    failed = 0
    disabled_voice_ids: set[str] = set()
    manifest_records: list[dict[str, Any]] = []
    stopped_for_quota = False

    for record_index, record, planned_voice_id in tqdm(selected, desc="elevenlabs tts", unit="file"):
        existing_output_path = None if args.overwrite else existing_output_for(args.out_dir, record)
        if existing_output_path is not None:
            skipped += 1
            output_path = existing_output_path
            voice_id = output_path.stem.rsplit("__voice-", 1)[-1]
        else:
            output_path = None
            voice_id = planned_voice_id

        voice_choice = next_voice_id(record_index, voice_ids, disabled_voice_ids)
        if existing_output_path is None and voice_choice is None:
            failed += 1
            tqdm.write(f"[FAIL] {record['path']}: no usable ElevenLabs voices remain")
            continue
        if existing_output_path is None:
            voice_id, voice_index = voice_choice
            output_path = output_path_for(args.out_dir, record, voice_id)
            while True:
                try:
                    synthesize_elevenlabs(
                        requests=requests,
                        text=record["text"],
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
                            tqdm.write(f"[FAIL] {record['path']}: no usable ElevenLabs voices remain")
                            break
                        voice_id, voice_index = voice_choice
                        output_path = output_path_for(args.out_dir, record, voice_id)
                        continue
                    if is_quota_error(exc):
                        stopped_for_quota = True
                        tqdm.write(f"[STOP] ElevenLabs quota/credit limit reached: {exc}")
                        break
                    failed += 1
                    tqdm.write(f"[FAIL] {record['path']}: {exc}")
                    break
                except Exception as exc:
                    failed += 1
                    tqdm.write(f"[FAIL] {record['path']}: {exc}")
                    break
            else:
                continue
            if failed and not output_path.exists():
                continue
            if stopped_for_quota:
                break

        manifest_records.append(
            {
                "source_folder": record["source_folder"],
                "video_id": record["video_id"],
                "provider": "elevenlabs",
                "voice_id": voice_id,
                "planned_voice_id": planned_voice_id,
                "model_id": args.elevenlabs_model,
                "output_format": args.elevenlabs_output_format,
                "text_chars": len(record["text"]),
                "transcript_path": str(record["path"]),
                "original_audio_path": record["audio_path"],
                "synthetic_audio_path": str(output_path),
            }
        )
        if stopped_for_quota:
            break

    manifest_path = write_manifest(args.out_dir, manifest_records)
    print(
        "ElevenLabs TTS complete: "
        f"created={created} skipped={skipped} failed={failed} selected={len(selected)} "
        f"chars={total_chars} disabled_voices={len(disabled_voice_ids)} "
        f"stopped_for_quota={stopped_for_quota} manifest={manifest_path}"
    )
    if disabled_voice_ids:
        print(f"Disabled voices: {sorted(disabled_voice_ids)}")
    return 2 if stopped_for_quota else (1 if failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
