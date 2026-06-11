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

See `docs/workflow.md` for the phase-based implementation guide.
