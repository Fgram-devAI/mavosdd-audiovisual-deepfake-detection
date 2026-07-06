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
FEAT_AUDIO_WAV2VEC2_CODEC_DIR = DATA_ROOT / "features" / "audio_wav2vec2_codec"
FEAT_AUDIO_WAVLM_CODEC_DIR = DATA_ROOT / "features" / "audio_wavlm_codec"
FEAT_AUDIO_HUBERT_CODEC_DIR = DATA_ROOT / "features" / "audio_hubert_codec"
DERIVED_DIR = DATA_ROOT / "derived"
AUDIO_SPOOF_MANIFEST = DERIVED_DIR / "audio_spoof_manifest.csv"
AUDIO_SPOOF_MANIFEST_CODEC_MATCHED = DERIVED_DIR / "audio_spoof_manifest_codec_matched.csv"
AUDIO_SPOOF_MANIFEST_VOICE_SPLIT = DERIVED_DIR / "audio_spoof_manifest_voice_split.csv"
VISUAL_SPEECH_MANIFEST = DERIVED_DIR / "visual_speech_manifest.csv"
FUSION_SPEECH_MANIFEST = DERIVED_DIR / "fusion_speech_manifest.csv"
VISUAL_SPEECH_MANIFEST_VOICE_SPLIT = DERIVED_DIR / "visual_speech_manifest_voice_split.csv"
FUSION_SPEECH_MANIFEST_VOICE_SPLIT = DERIVED_DIR / "fusion_speech_manifest_voice_split.csv"
LIPSYNC_PAIRS_MANIFEST = DERIVED_DIR / "lipsync_pairs_manifest.csv"
LIPSYNC_PAIRS_PROVENANCE = DERIVED_DIR / "lipsync_pairs_manifest.provenance.json"
VIDEO_AV_MANIFEST = DERIVED_DIR / "video_av_manifest.csv"
AUDIO_WAV_CODEC_MATCHED_DIR = DATA_ROOT / "audio_wav_codec_matched"

FEAT_SYNCNET_VISUAL_DIR = DATA_ROOT / "features" / "syncnet_visual"
FEAT_SYNCNET_AUDIO_DIR = DATA_ROOT / "features" / "syncnet_audio"
FEAT_AVHUBERT_VISUAL_DIR = DATA_ROOT / "features" / "avhubert_visual"
FEAT_AVHUBERT_AUDIO_DIR = DATA_ROOT / "features" / "avhubert_audio"

CKPT_DIR = Path("models/checkpoints")
CKPT_SYNCNET_DIR = CKPT_DIR / "syncnet_pretrained"
CKPT_AVHUBERT_DIR = CKPT_DIR / "avhubert_pretrained"
SYNCNET_CKPT_PATH = CKPT_SYNCNET_DIR / "syncnet.pt"
AVHUBERT_CKPT_PATH = CKPT_AVHUBERT_DIR / "avhubert_base.pt"

SYNCNET_FAILURES_CSV = DATA_ROOT / "features" / "syncnet_extraction_failures.csv"
AVHUBERT_FAILURES_CSV = DATA_ROOT / "features" / "avhubert_extraction_failures.csv"

REPORT_DIR = Path("report")
REPORT_VAL_EVAL_DIR = REPORT_DIR / "val_eval"

EXTRACTION_FAILURE_FIELDS = (
    "sample_id",
    "stage",
    "error_type",
    "error_message",
    "timestamp",
)

CAPS = {
    "real": 2500,
    "echomimic": 600,
    "memo": 400,
    "liveportrait": 314,
    "sonic": 335,
}
LABEL_MAP = {
    "real": 0,
    "echomimic": 1,
    "memo": 1,
    "liveportrait": 1,
    "sonic": 1,
}

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
