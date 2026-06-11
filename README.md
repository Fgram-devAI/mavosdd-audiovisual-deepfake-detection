# Audiovisual Deepfake Detection

Late-fusion detector for MAVOS-DD videos, combining frozen Wav2Vec2 audio
features with MediaPipe lip-landmark motion features.

The pipeline is intentionally feature-first:

1. Stream and verify a capped 1,000-video MAVOS-DD subset.
2. Extract frozen Wav2Vec2 audio embeddings and MediaPipe lip-landmark sequences.
3. Train a small late-fusion neural classifier on serialized feature arrays only.
4. Evaluate once on the locked test split and package a reproducible predictor.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.models.late_fusion
```

See `docs/workflow.md` for the phase-based implementation guide.
