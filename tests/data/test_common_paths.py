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


def test_voice_split_visual_and_fusion_paths_under_derived():
    assert common.VISUAL_SPEECH_MANIFEST_VOICE_SPLIT == (
        common.DERIVED_DIR / "visual_speech_manifest_voice_split.csv"
    )
    assert common.FUSION_SPEECH_MANIFEST_VOICE_SPLIT == (
        common.DERIVED_DIR / "fusion_speech_manifest_voice_split.csv"
    )


def test_syncnet_paths_are_under_data_features():
    from src import common
    assert common.FEAT_SYNCNET_VISUAL_DIR == common.DATA_ROOT / "features" / "syncnet_visual"
    assert common.FEAT_SYNCNET_AUDIO_DIR == common.DATA_ROOT / "features" / "syncnet_audio"


def test_avhubert_paths_are_under_data_features():
    from src import common
    assert common.FEAT_AVHUBERT_VISUAL_DIR == common.DATA_ROOT / "features" / "avhubert_visual"
    assert common.FEAT_AVHUBERT_AUDIO_DIR == common.DATA_ROOT / "features" / "avhubert_audio"


def test_pretrained_checkpoint_paths_are_under_models_checkpoints():
    from src import common
    assert common.CKPT_SYNCNET_DIR == Path("models/checkpoints/syncnet_pretrained")
    assert common.CKPT_AVHUBERT_DIR == Path("models/checkpoints/avhubert_pretrained")
    assert common.SYNCNET_CKPT_PATH == common.CKPT_SYNCNET_DIR / "syncnet.pt"
    assert common.AVHUBERT_CKPT_PATH == common.CKPT_AVHUBERT_DIR / "avhubert_base.pt"


def test_extraction_failure_schema_is_canonical():
    from src import common
    assert common.EXTRACTION_FAILURE_FIELDS == (
        "sample_id",
        "stage",
        "error_type",
        "error_message",
        "timestamp",
    )


def test_extraction_failure_paths_are_backend_specific():
    from src import common
    assert common.SYNCNET_FAILURES_CSV == common.DATA_ROOT / "features" / "syncnet_extraction_failures.csv"
    assert common.AVHUBERT_FAILURES_CSV == common.DATA_ROOT / "features" / "avhubert_extraction_failures.csv"


def test_report_val_eval_dir():
    from src import common
    assert common.REPORT_VAL_EVAL_DIR == Path("report") / "val_eval"
