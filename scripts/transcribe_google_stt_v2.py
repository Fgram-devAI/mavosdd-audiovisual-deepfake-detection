"""Transcribe exported WAV files with Google Speech-to-Text V2.

Prerequisites:
    pip install google-cloud-speech
    gcloud auth application-default login
    echo "GOOGLE_CLOUD_PROJECT=<your-project-id>" >> .env

Usage:
    python scripts/transcribe_google_stt_v2.py --limit 5
    python scripts/transcribe_google_stt_v2.py --project-id my-gcp-project
    python scripts/transcribe_google_stt_v2.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path
from typing import Any

from tqdm import tqdm


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


def import_google_speech():
    try:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import speech_v2
        from google.cloud.speech_v2.types import cloud_speech
        from google.protobuf.json_format import MessageToDict
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: google-cloud-speech. Install it with:\n"
            "  pip install google-cloud-speech"
        ) from exc
    return speech_v2, cloud_speech, MessageToDict, DefaultCredentialsError


def recognizer_name(project_id: str, location: str, recognizer: str) -> str:
    if recognizer.startswith("projects/"):
        return recognizer
    return f"projects/{project_id}/locations/{location}/recognizers/{recognizer}"


def transcript_from_response(response: Any) -> tuple[str, list[dict[str, Any]]]:
    segments = []
    texts = []
    for result in response.results:
        if not result.alternatives:
            continue
        alt = result.alternatives[0]
        text = alt.transcript.strip()
        if text:
            texts.append(text)
        segments.append(
            {
                "transcript": text,
                "confidence": float(getattr(alt, "confidence", 0.0)),
            }
        )
    return " ".join(texts).strip(), segments


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def consolidate_jsonl(out_dir: Path, jsonl_path: Path) -> int:
    files = sorted(p for p in out_dir.glob("*/*.json") if p.name != jsonl_path.name)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as f:
        for path in files:
            f.write(path.read_text().strip() + "\n")
    return len(files)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def filter_wav_files(wav_files: list[Path], source_folders: list[str] | None) -> list[Path]:
    if not source_folders:
        return wav_files
    allowed = {folder.lower() for folder in source_folders}
    return [path for path in wav_files if path.parent.name.lower() in allowed]


def transcribe_one(
    client: Any,
    cloud_speech: Any,
    message_to_dict: Any,
    wav_path: Path,
    output_path: Path,
    recognizer: str,
    language_codes: list[str],
    model: str | None,
    timeout: float,
) -> None:
    features = cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True)
    config_kwargs: dict[str, Any] = {
        "auto_decoding_config": cloud_speech.AutoDetectDecodingConfig(),
        "language_codes": language_codes,
        "features": features,
    }
    if model:
        config_kwargs["model"] = model

    config = cloud_speech.RecognitionConfig(**config_kwargs)
    request = cloud_speech.RecognizeRequest(
        recognizer=recognizer,
        config=config,
        content=wav_path.read_bytes(),
    )
    response = client.recognize(request=request, timeout=timeout)
    transcript, segments = transcript_from_response(response)
    response_dict = message_to_dict(response._pb, preserving_proto_field_name=True)

    source_folder = wav_path.parent.name
    payload = {
        "audio_path": str(wav_path),
        "source_folder": source_folder,
        "video_id": wav_path.stem,
        "recognizer": recognizer,
        "language_codes": language_codes,
        "model": model,
        "transcript": transcript,
        "segments": segments,
        "response": response_dict,
    }
    write_json(output_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch transcribe WAV files with Google STT V2.")
    parser.add_argument("--env-file", default=".env", type=Path, help="Load Google env vars from this file.")
    parser.add_argument("--wav-dir", default="data/audio_wav", type=Path)
    parser.add_argument("--out-dir", default="data/transcripts/google_stt_v2", type=Path)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--location", default="global")
    parser.add_argument("--recognizer", default="_", help="Recognizer id, '_' default recognizer, or full path.")
    parser.add_argument("--language-code", action="append", dest="language_codes", default=None)
    parser.add_argument("--model", default=None, help="Optional STT V2 model name, e.g. latest_long.")
    parser.add_argument(
        "--source-folder",
        action="append",
        default=None,
        help="Restrict to a source folder/class. Repeatable, e.g. --source-folder echomimic --source-folder memo.",
    )
    parser.add_argument("--estimate-only", action="store_true", help="Print selected duration/cost estimate and exit.")
    parser.add_argument("--price-per-minute", type=float, default=0.016)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--jsonl", default=None, type=Path)
    args = parser.parse_args()

    load_env_file(args.env_file)
    project_id = args.project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if not project_id:
        raise SystemExit("Set --project-id or GOOGLE_CLOUD_PROJECT.")

    speech_v2, cloud_speech, message_to_dict, default_credentials_error = import_google_speech()
    try:
        client = speech_v2.SpeechClient()
    except default_credentials_error as exc:
        raise SystemExit(
            "Google Application Default Credentials were not found.\n\n"
            "For local development, run:\n"
            "  gcloud auth application-default login\n"
            f"  gcloud config set project {project_id}\n\n"
            "Or put a service-account key path in .env:\n"
            "  GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json\n"
            f"  GOOGLE_CLOUD_PROJECT={project_id}\n"
        ) from exc
    rec_name = recognizer_name(project_id, args.location, args.recognizer)
    language_codes = args.language_codes or ["en-US"]

    wav_files = filter_wav_files(sorted(args.wav_dir.glob("*/*.wav")), args.source_folder)
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    total_seconds = sum(wav_seconds(path) for path in wav_files)
    estimated_cost = (total_seconds / 60.0) * args.price_per_minute
    print(
        f"Selected {len(wav_files)} WAV files | "
        f"{total_seconds / 60.0:.2f} minutes | "
        f"estimated standard STT cost ${estimated_cost:.2f} "
        f"at ${args.price_per_minute:.3f}/minute"
    )
    if args.estimate_only:
        return 0

    created = 0
    skipped = 0
    failed = 0
    for wav_path in tqdm(wav_files, desc="google stt", unit="file"):
        rel = wav_path.relative_to(args.wav_dir)
        output_path = args.out_dir / rel.with_suffix(".json")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            transcribe_one(
                client=client,
                cloud_speech=cloud_speech,
                message_to_dict=message_to_dict,
                wav_path=wav_path,
                output_path=output_path,
                recognizer=rec_name,
                language_codes=language_codes,
                model=args.model,
                timeout=args.timeout,
            )
            created += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"[FAIL] {wav_path}: {exc}")

    jsonl_path = args.jsonl or (args.out_dir / "transcripts.jsonl")
    total_jsonl = consolidate_jsonl(args.out_dir, jsonl_path)
    print(
        "Google STT complete: "
        f"created={created} skipped={skipped} failed={failed} selected={len(wav_files)} "
        f"jsonl_rows={total_jsonl} jsonl={jsonl_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
