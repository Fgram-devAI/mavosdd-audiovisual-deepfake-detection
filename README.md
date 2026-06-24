# Audiovisual Deepfake Detection

Late-fusion detector for MAVOS-DD videos, combining frozen Wav2Vec2 audio
features with MediaPipe lip-landmark motion features.

The pipeline is intentionally feature-first:

1. Stream and verify a capped 1,000-video MAVOS-DD subset.
2. Extract frozen Wav2Vec2 audio embeddings and MediaPipe lip-landmark sequences.
3. Train a small late-fusion neural classifier on serialized feature arrays only.
4. Evaluate once on the locked test split and package a reproducible predictor.

## Quick Start

Create a Python 3.10 virtual environment.

macOS / Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Smoke-test the model module:

```bash
python -m src.models.late_fusion
```

## Fetch The MAVOS-DD Subset

The ingestion command builds the capped audiovisual subset from the Hugging Face
MAVOS-DD repository. It lists remote filenames, filters only
`english/{real,echomimic,memo}/*.mp4`, sorts that candidate list, and downloads
until the accepted-video caps are reached:

```text
real        500
echomimic   250
memo        250
total      1000
```

Run:

macOS / Linux:

```bash
source .venv/bin/activate
python -m src.data.download_subset
python -m src.data.download_subset --validate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.data.download_subset
python -m src.data.download_subset --validate
```

Useful monitoring commands while fetching:

```bash
du -sh data data/raw data/raw/real data/raw/echomimic data/raw/memo data/quarantine
find data/raw -type f | wc -l
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/manifest.csv")
print(df["source_folder"].value_counts())
print("total accepted:", len(df))
PY
```

The fetch is deterministic for a fixed MAVOS-DD repository state: candidates are
sorted before caps are applied, and rejected files do not count toward caps.
`data/manifest.csv` is generated locally as the record of the accepted subset,
but it stays gitignored with raw videos, quarantine logs, extracted features,
checkpoints, and run artifacts. Keeping the manifest out of Git lets fresh clones
run the same fetch command instead of treating a committed manifest as already
downloaded local data.

## Split And Extract Features

After `python -m src.data.download_subset --validate` prints `VALIDATION OK`,
create the local train/validation/test split files:

```bash
python -m src.data.make_splits
```

Then extract audio and lip-motion feature artifacts:

```bash
python -m src.features.extract_audio
python -m src.features.extract_lips
```

Check extraction counts:

macOS / Linux:

```bash
find data/features/audio -name '*.npy' | wc -l
find data/features/lips -name '*.npz' | wc -l
```

Windows PowerShell:

```powershell
(Get-ChildItem data/features/audio -Filter *.npy).Count
(Get-ChildItem data/features/lips -Filter *.npz).Count
```

Expected count for both modalities is `1000`.

## Transcribe WAV Audio With Google STT V2

The optional transcription step turns local WAV files into JSON transcripts for
later synthetic speech experiments with Google TTS, ElevenLabs, or another voice
generation provider.

First export WAV files from the raw videos:

```bash
python scripts/export_wav.py
```

Then configure Google Cloud authentication. For local development with the
Google Cloud CLI:

```bash
gcloud auth application-default login
echo 'GOOGLE_CLOUD_PROJECT="<your-gcp-project-id>"' >> .env
```

For a service-account key file:

```bash
cat >> .env <<'EOF'
GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
GOOGLE_CLOUD_PROJECT="<your-gcp-project-id>"
EOF
```

Windows PowerShell equivalents:

```powershell
gcloud auth application-default login
Add-Content .env 'GOOGLE_CLOUD_PROJECT="<your-gcp-project-id>"'
```

or:

```powershell
Add-Content .env 'GOOGLE_APPLICATION_CREDENTIALS="C:\absolute\path\to\service-account.json"'
Add-Content .env 'GOOGLE_CLOUD_PROJECT="<your-gcp-project-id>"'
```

Make sure the Google Cloud project has billing enabled and the Speech-to-Text
API enabled. The transcription script automatically loads `.env` by default.
Then run a small smoke test:

```bash
python scripts/transcribe_google_stt_v2.py --limit 5
```

Run all WAV files:

```bash
python scripts/transcribe_google_stt_v2.py
```

Run only the synthetic classes planned for TTS augmentation:

```bash
python scripts/transcribe_google_stt_v2.py \
  --source-folder echomimic \
  --source-folder memo \
  --model latest_long
```

Outputs are written locally under:

```text
data/transcripts/google_stt_v2/
```

`data/transcripts/` is gitignored because transcripts may contain dataset speech
content and are generated artifacts.

## Generate ElevenLabs TTS Alternatives

Use the Google STT transcripts to generate synthetic speech alternatives with
ElevenLabs. The script rotates deterministically through the project voice pool
so repeated runs assign the same transcript to the same voice as long as the
input transcripts and voice list stay unchanged.

Add the ElevenLabs API key to `.env`:

```bash
cat >> .env <<'EOF'
ELEVENLABS_API_KEY="<your-elevenlabs-api-key>"
EOF
```

Windows PowerShell:

```powershell
Add-Content .env 'ELEVENLABS_API_KEY="<your-elevenlabs-api-key>"'
```

Estimate selected transcript characters before spending API credits:

```bash
python scripts/synthesize_tts_from_transcripts.py --estimate-only
```

If ElevenLabs rejects library voices on a free plan, inspect the voices available
to your API key:

```bash
python scripts/synthesize_tts_from_transcripts.py --list-voices
```

Then run with voices from your account instead of the project voice pool, if
needed:

```bash
python scripts/synthesize_tts_from_transcripts.py --use-account-voices --limit 2
```

Generate a capped batch that stays under a character budget:

```bash
python scripts/synthesize_tts_from_transcripts.py --max-chars 10000
```

Generate all available transcripts:

```bash
python scripts/synthesize_tts_from_transcripts.py
```

Restrict to the planned synthetic classes:

```bash
python scripts/synthesize_tts_from_transcripts.py \
  --source-folder echomimic \
  --source-folder memo
```

Outputs are written locally under:

```text
data/tts_audio/
```

`data/tts_audio/` is gitignored because generated speech is a local artifact.

## Generate ElevenLabs Speech-To-Speech Alternatives

For real clips, you can also preserve the original timing and delivery while
changing the voice with ElevenLabs speech-to-speech. This uses the exported WAV
files, not transcripts.

Estimate the real-audio selection:

```bash
python scripts/convert_real_speech_elevenlabs.py --estimate-only
```

Run a tiny paid smoke test:

```bash
python scripts/convert_real_speech_elevenlabs.py --limit 2
```

Run a capped batch by audio duration:

```bash
python scripts/convert_real_speech_elevenlabs.py --max-seconds 600
```

Outputs are written locally under:

```text
data/tts_audio/elevenlabs_sts/real/
```

## Generate Google TTS For Real Transcripts

If ElevenLabs credits are exhausted, use the existing Google STT transcripts
with Google Cloud Text-to-Speech. By default this runs only `real` transcripts
and rotates through several `en-US-Neural2-*` voices.

Estimate selected characters:

```bash
python scripts/synthesize_google_tts_from_transcripts.py --estimate-only
```

Run a small smoke test:

```bash
python scripts/synthesize_google_tts_from_transcripts.py --limit 5
```

Run all available real transcripts:

```bash
python scripts/synthesize_google_tts_from_transcripts.py
```

Outputs are written locally under:

```text
data/tts_audio/google_tts/real/
```

## Match Audio Codec Footprint (Anti-Leakage)

Bonafide rows on disk are clean 16 kHz PCM WAVs; every generated spoof row is a
lossy MP3 (ElevenLabs at 44.1 kHz / 128 kbps, Google TTS at 24 kHz / 64 kbps).
That 100% format/label correlation lets any mel-input model shortcut to a
WAV-vs-MP3 codec discriminator and ignore the actual TTS artifacts. The
codec-match step round-trips each bonafide WAV through MP3 — at a codec spec
sampled deterministically per row from the spoof provider distribution — and
decodes every row back to a 16 kHz mono WAV, so codec history becomes
label-independent.

Requires `ffmpeg` and `libmp3lame` on PATH (`brew install ffmpeg` on macOS).

```bash
python -m src.data.codec_match_audio
```

Defaults read `data/derived/audio_spoof_manifest.csv` and write:

```text
data/audio_wav_codec_matched/{sample_id}.wav
data/derived/audio_spoof_manifest_codec_matched.csv
```

The new manifest has the same SCHEMA as the input with `audio_path` repointed
at the new tree. Both paths are gitignored.

### Re-extract Audio Embeddings From The Codec-Matched WAVs

After codec-match the existing `data/features/audio_{wav2vec2,wavlm,hubert}/`
stores are stale — they were extracted from the leaky originals. Re-run each
backend against the codec-matched manifest with `--overwrite`:

```bash
python -m src.features.extract_audio_embeddings --backend wav2vec2 --manifest data/derived/audio_spoof_manifest_codec_matched.csv --overwrite
python -m src.features.extract_audio_embeddings --backend wavlm    --manifest data/derived/audio_spoof_manifest_codec_matched.csv --overwrite
python -m src.features.extract_audio_embeddings --backend hubert   --manifest data/derived/audio_spoof_manifest_codec_matched.csv --overwrite
```

Each backend writes 2171 `.npy` files into its backend-specific default
directory; pass `--device cuda` or `--device mps` if a GPU is available.

### Re-extract Mel-Spectrograms From The Codec-Matched WAVs

The mel-CNN baseline in `notebooks/01_mel_cnn_baseline.ipynb` is the most
codec-sensitive head in the project — its near-perfect val ROC-AUC on the
original features was the trigger for this whole step. Re-extract mel into
the same `data/features/audio_mel/` directory and re-run the notebook end to
end as a diagnostic; a large AUC drop confirms the fix neutralized the
shortcut.

```bash
python -m src.features.extract_mel --manifest data/derived/audio_spoof_manifest_codec_matched.csv --overwrite
```

See `docs/workflow.md` for the phase-based implementation guide.
