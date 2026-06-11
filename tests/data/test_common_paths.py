from pathlib import Path

from src import common


def test_quarantine_log_under_data_root():
    assert common.QUARANTINE_LOG == common.DATA_ROOT / "quarantine_log.csv"
    assert isinstance(common.QUARANTINE_LOG, Path)
