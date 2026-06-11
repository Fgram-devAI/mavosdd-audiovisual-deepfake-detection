"""Filter rules: language + generator (metadata-first, path-prefix fallback)."""
import pytest

from src.data.download_subset import classify


@pytest.mark.parametrize("method,cls", [("real", "real"), ("echomimic", "echomimic"), ("memo", "memo")])
def test_metadata_match_each_class(method, cls):
    record = {"language": "English", "generation_method": method, "file_name": f"x/{method}/abc.mp4"}
    assert classify(record) == cls


def test_metadata_match_is_case_insensitive():
    record = {"language": "ENGLISH", "generation_method": "EchoMimic", "file_name": "x/y/z.mp4"}
    assert classify(record) == "echomimic"


def test_non_english_is_rejected():
    record = {"language": "Spanish", "generation_method": "real", "file_name": "spanish/real/a.mp4"}
    assert classify(record) is None


def test_out_of_scope_generator_is_rejected():
    record = {"language": "english", "generation_method": "stylegan", "file_name": "english/stylegan/a.mp4"}
    assert classify(record) is None


def test_path_prefix_fallback_when_metadata_missing():
    record = {"file_name": "english/memo/clip_42.mp4"}
    assert classify(record) == "memo"


def test_path_field_alias_works():
    record = {"path": "english/real/clip.mp4"}
    assert classify(record) == "real"


def test_method_field_alias_works():
    record = {"language": "english", "method": "memo", "file_name": "x.mp4"}
    assert classify(record) == "memo"


def test_no_match_returns_none():
    record = {"language": "english", "generation_method": "", "file_name": "x.mp4"}
    assert classify(record) is None
