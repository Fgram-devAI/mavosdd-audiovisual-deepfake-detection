"""Tests for scripts/download_syncnet_checkpoint.py."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _install_scripts_on_path():
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def test_download_is_idempotent_when_file_exists_with_matching_sha(tmp_path):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    payload = b"fake-checkpoint-bytes"
    dest = tmp_path / "syncnet.pt"
    dest.write_bytes(payload)
    sha = _sha256(payload)

    with patch("scripts.download_syncnet_checkpoint.urlopen") as m:
        dl.download(dest, expected_sha256=sha, url="https://example.invalid/x")
        assert m.call_count == 0


def test_download_writes_file_and_verifies_sha(tmp_path):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    payload = b"fresh-download-bytes"
    dest = tmp_path / "syncnet.pt"
    sha = _sha256(payload)

    fake_response = MagicMock()
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False
    fake_response.read.side_effect = [payload, b""]

    with patch("scripts.download_syncnet_checkpoint.urlopen", return_value=fake_response):
        dl.download(dest, expected_sha256=sha, url="https://example.invalid/x")

    assert dest.exists()
    assert dest.read_bytes() == payload


def test_download_removes_partial_file_on_sha_mismatch(tmp_path):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    payload = b"wrong-bytes"
    dest = tmp_path / "syncnet.pt"

    fake_response = MagicMock()
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False
    fake_response.read.side_effect = [payload, b""]

    with patch("scripts.download_syncnet_checkpoint.urlopen", return_value=fake_response):
        with pytest.raises(RuntimeError, match=r"sha256 mismatch"):
            dl.download(dest, expected_sha256="0" * 64, url="https://example.invalid/x")

    assert not dest.exists()


def test_download_bootstrap_placeholder_keeps_file_and_returns_sha(tmp_path):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    payload = b"fresh-download-bytes"
    dest = tmp_path / "syncnet.pt"
    sha = _sha256(payload)

    fake_response = MagicMock()
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False
    fake_response.read.side_effect = [payload, b""]

    with patch("scripts.download_syncnet_checkpoint.urlopen", return_value=fake_response):
        actual = dl.download(
            dest,
            expected_sha256="REPLACE_WITH_ACTUAL_SHA256_AFTER_FIRST_DOWNLOAD",
            url="https://example.invalid/x",
            allow_placeholder_sha=True,
        )

    assert actual == sha
    assert dest.exists()
    assert dest.read_bytes() == payload


def test_download_refuses_placeholder_without_bootstrap(tmp_path):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    with pytest.raises(RuntimeError, match="placeholder"):
        dl.download(
            tmp_path / "syncnet.pt",
            expected_sha256="REPLACE_WITH_ACTUAL_SHA256_AFTER_FIRST_DOWNLOAD",
            url="https://example.invalid/x",
        )


def test_main_exits_zero_when_file_already_present(tmp_path, monkeypatch, capsys):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    payload = b"fake"
    sha = _sha256(payload)
    dest = tmp_path / "syncnet.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)

    monkeypatch.setattr(dl, "SYNCNET_CKPT_PATH", dest)
    monkeypatch.setattr(dl, "EXPECTED_SHA256", sha)

    rc = dl.main([])
    assert rc == 0


def test_main_exits_nonzero_on_download_failure(tmp_path, monkeypatch):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    dest = tmp_path / "syncnet.pt"
    monkeypatch.setattr(dl, "SYNCNET_CKPT_PATH", dest)
    monkeypatch.setattr(dl, "EXPECTED_SHA256", "0" * 64)

    def _boom(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr(dl, "urlopen", _boom)
    rc = dl.main([])
    assert rc != 0
    assert not dest.exists()


def test_default_url_points_to_huggingface_mirror():
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    assert dl.DOWNLOAD_URL == (
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/"
        "lipsync_expert.pth"
    )
    assert "Wav2Lip#getting-the-weights" in dl.UPSTREAM_WEIGHTS_PAGE


def test_help_message_documents_manual_placement(capsys):
    _install_scripts_on_path()
    from scripts import download_syncnet_checkpoint as dl

    with pytest.raises(SystemExit) as exc:
        dl.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "manual placement" in out.lower() or "SYNCNET_CKPT_PATH" in out or "SHA256" in out
