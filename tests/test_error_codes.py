"""要求 8：8 个业务错误码定义齐全且异常链路可用。"""
from __future__ import annotations

import pytest

from syncbench.errors import (
    ALL_CODES,
    E_BUDGET_EXCEEDED,
    E_CH_FORMAT,
    E_DEPENDENCY_STALE,
    E_DRIFT_EXCEED,
    E_GAP_TOO_LONG,
    E_PULSE_MISSING,
    E_SNAPSHOT_STALE,
    E_SYNC_MISMATCH,
    BusinessError,
)


def test_all_eight_codes_defined():
    expected = {
        E_CH_FORMAT, E_PULSE_MISSING, E_DRIFT_EXCEED, E_GAP_TOO_LONG,
        E_SYNC_MISMATCH, E_DEPENDENCY_STALE, E_SNAPSHOT_STALE,
        E_BUDGET_EXCEEDED,
    }
    assert set(ALL_CODES) == expected
    assert len(ALL_CODES) == 8


def test_business_error_serialization():
    err = BusinessError(E_DRIFT_EXCEED, "boom", channel=1, segment=2,
                        detail={"k": "v"})
    d = err.to_dict()
    assert d["code"] == E_DRIFT_EXCEED
    assert d["channel"] == 1 and d["segment"] == 2
    assert d["detail"] == {"k": "v"}


def test_dependency_stale_raised_on_live_read(tmp_path):
    from syncbench.engine import Engine
    from syncbench.config import default_config
    eng = Engine(str(tmp_path), default_config())
    eng.prepare_media()
    with pytest.raises(BusinessError) as exc:
        eng.graph.get_live("drift:0")
    assert exc.value.code == E_DEPENDENCY_STALE
