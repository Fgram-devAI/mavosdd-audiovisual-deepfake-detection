from pathlib import Path
import random

import numpy as np
import torch

SEED = 42

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
QUARANTINE_DIR = DATA_ROOT / "quarantine"
FEAT_AUDIO_DIR = DATA_ROOT / "features" / "audio"
FEAT_LIPS_DIR = DATA_ROOT / "features" / "lips"
SPLITS_DIR = DATA_ROOT / "splits"
MANIFEST = DATA_ROOT / "manifest.csv"
QUARANTINE_LOG = DATA_ROOT / "quarantine_log.csv"
AUDIO_WAV_DIR = DATA_ROOT / "audio_wav"
TTS_AUDIO_DIR = DATA_ROOT / "tts_audio"
FEAT_AUDIO_GEN_DIR = DATA_ROOT / "features" / "audio_generated"
FEAT_AUDIO_WAV2VEC2_DIR = DATA_ROOT / "features" / "audio_wav2vec2"
FEAT_AUDIO_WAVLM_DIR = DATA_ROOT / "features" / "audio_wavlm"
FEAT_AUDIO_HUBERT_DIR = DATA_ROOT / "features" / "audio_hubert"
FEAT_AUDIO_MEL_DIR = DATA_ROOT / "features" / "audio_mel"
DERIVED_DIR = DATA_ROOT / "derived"
AUDIO_SPOOF_MANIFEST = DERIVED_DIR / "audio_spoof_manifest.csv"
VISUAL_SPEECH_MANIFEST = DERIVED_DIR / "visual_speech_manifest.csv"
FUSION_SPEECH_MANIFEST = DERIVED_DIR / "fusion_speech_manifest.csv"

CAPS = {"real": 500, "echomimic": 250, "memo": 250}
LABEL_MAP = {"real": 0, "echomimic": 1, "memo": 1}

SR = 16_000
AUDIO_SECONDS = 4.0
N_SAMPLES = int(SR * AUDIO_SECONDS)
LIP_FPS = 5
N_FRAMES = int(LIP_FPS * AUDIO_SECONDS)
W2V_MODEL = "facebook/wav2vec2-base-960h"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
