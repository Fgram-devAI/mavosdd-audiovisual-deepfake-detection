"""First-record schema inspection emits warnings when fields are missing."""
import logging

from src.data.download_subset import inspect_schema


def test_full_schema_emits_no_warning(caplog):
    record = {
        "language": "english",
        "generation_method": "real",
        "file_name": "english/real/x.mp4",
        "video": b"\x00",
    }
    with caplog.at_level(logging.WARNING):
        inspect_schema(record)
    assert caplog.records == []


def test_missing_language_warns(caplog):
    record = {"generation_method": "real", "file_name": "english/real/x.mp4", "video": b""}
    with caplog.at_level(logging.WARNING):
        inspect_schema(record)
    assert any("language" in r.getMessage() for r in caplog.records)


def test_missing_generator_warns(caplog):
    record = {"language": "english", "file_name": "english/real/x.mp4", "video": b""}
    with caplog.at_level(logging.WARNING):
        inspect_schema(record)
    assert any("generation_method" in r.getMessage() or "method" in r.getMessage()
               for r in caplog.records)


def test_method_alias_satisfies_generator_check(caplog):
    record = {"language": "english", "method": "real", "file_name": "english/real/x.mp4", "video": b""}
    with caplog.at_level(logging.WARNING):
        inspect_schema(record)
    assert not any("generation_method" in r.getMessage() or "method" in r.getMessage()
                   for r in caplog.records)


def test_local_path_payload_satisfies_required_file_check(caplog):
    record = {
        "language": "english",
        "generation_method": "real",
        "file_name": "english/real/x.mp4",
        "local_path": "/tmp/x.mp4",
    }
    with caplog.at_level(logging.WARNING):
        inspect_schema(record)
    assert caplog.records == []


def test_deferred_hub_payload_satisfies_required_file_check(caplog):
    record = {
        "language": "english",
        "generation_method": "real",
        "file_name": "english/real/x.mp4",
        "path": "english/real/x.mp4",
        "repo_id": "unibuc-cs/MAVOS-DD",
    }
    with caplog.at_level(logging.WARNING):
        inspect_schema(record)
    assert caplog.records == []


def test_missing_payload_raises():
    import pytest
    with pytest.raises(KeyError):
        inspect_schema({"language": "english", "generation_method": "real", "file_name": "x"})
