from pathlib import Path

from src import common


def test_quarantine_log_under_data_root():
    assert common.QUARANTINE_LOG == common.DATA_ROOT / "quarantine_log.csv"
    assert isinstance(common.QUARANTINE_LOG, Path)


def test_derived_paths_under_data_root():
    assert common.DERIVED_DIR == common.DATA_ROOT / "derived"
    assert common.AUDIO_SPOOF_MANIFEST == common.DERIVED_DIR / "audio_spoof_manifest.csv"
    assert common.VISUAL_SPEECH_MANIFEST == common.DERIVED_DIR / "visual_speech_manifest.csv"
    assert common.FUSION_SPEECH_MANIFEST == common.DERIVED_DIR / "fusion_speech_manifest.csv"


def test_tts_and_generated_paths_under_data_root():
    assert common.TTS_AUDIO_DIR == common.DATA_ROOT / "tts_audio"
    assert common.AUDIO_WAV_DIR == common.DATA_ROOT / "audio_wav"
    assert common.FEAT_AUDIO_GEN_DIR == common.DATA_ROOT / "features" / "audio_generated"


def test_backend_feature_dirs_under_data_features():
    assert common.FEAT_AUDIO_WAV2VEC2_DIR == common.DATA_ROOT / "features" / "audio_wav2vec2"
    assert common.FEAT_AUDIO_WAVLM_DIR == common.DATA_ROOT / "features" / "audio_wavlm"
    assert common.FEAT_AUDIO_HUBERT_DIR == common.DATA_ROOT / "features" / "audio_hubert"


def test_mel_feature_dir_under_data_features():
    assert common.FEAT_AUDIO_MEL_DIR == common.DATA_ROOT / "features" / "audio_mel"


def test_codec_matched_backend_feature_dirs_under_data_features():
    assert common.FEAT_AUDIO_WAV2VEC2_CODEC_DIR == common.DATA_ROOT / "features" / "audio_wav2vec2_codec"
    assert common.FEAT_AUDIO_WAVLM_CODEC_DIR == common.DATA_ROOT / "features" / "audio_wavlm_codec"
    assert common.FEAT_AUDIO_HUBERT_CODEC_DIR == common.DATA_ROOT / "features" / "audio_hubert_codec"
