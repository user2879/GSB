from pathlib import Path

import pytest

from mvcc_lab.errors import FieldError_, PortError, TimeoutError_
from mvcc_lab.executor import ReplayExecutor

ROOT = Path(__file__).parents[1]
SPEC = ROOT / "samples" / "sample_spec.json"


def test_batch_receipt_roundtrip():
    with ReplayExecutor(SPEC) as executor:
        receipt = executor.send_batch(
            "b0", snapshot_seq=3, reader_txn_id="T1",
            query_ids=["Q1"], row_keys=["R-A"],
        )
    assert receipt["batch_id"] == "b0"
    assert receipt["query_ids"] == ["Q1"]
    assert receipt["rule_version"].startswith("mvcc-rules-")
    assert receipt["summaries"][0]["visible_version"] == 1


def test_port_busy_raises_e_port():
    with pytest.raises(PortError) as info:
        with ReplayExecutor(SPEC, injection="port_busy"):
            pass
    assert info.value.code == "E_PORT"
    assert info.value.field == "port"


def test_silent_hold_raises_real_timeout():
    # 接受连接但不响应 -> 真实 socket 超时，无人为 sleep。
    with pytest.raises(TimeoutError_) as info:
        with ReplayExecutor(SPEC, injection="silent_hold"):
            pass
    assert info.value.code == "E_TIMEOUT"


def test_silent_reset_raises_timeout_code():
    with pytest.raises(TimeoutError_) as info:
        with ReplayExecutor(SPEC, injection="silent"):
            pass
    assert info.value.code == "E_TIMEOUT"


def test_missing_field_raises_e_field():
    with ReplayExecutor(SPEC, injection="missing_field") as executor:
        with pytest.raises(FieldError_) as info:
            executor.send_batch(
                "bx", 3, "T1", ["Q1"], ["R-A"],
            )
    assert info.value.code == "E_FIELD"
    assert info.value.field == "rule_version"
