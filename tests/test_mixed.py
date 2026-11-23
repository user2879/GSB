"""E_MIXED：热更新与请求交错时的混装拒绝路径。"""

import pytest

from gsb.backend_host import BackendHost, BackendStaleError
from gsb.errors import GSBError
from gsb.spec import load_spec

SAMPLE_DIR = "samples"


def test_stale_exec_rejected_by_backend():
    """热更新后以旧参数版本派发 -> 后端拒绝并报告冲突版本序号。"""
    spec = load_spec(SAMPLE_DIR)
    host = BackendHost(spec.backends["b-gray-a"])
    host.start()
    try:
        assert host.request({"cmd": "ping"}, 2000)["param_version"] == 1
        host.hot_update(999)
        assert host.request({"cmd": "ping"}, 2000)["param_version"] == 2
        with pytest.raises(BackendStaleError) as exc:
            host.request(
                {
                    "cmd": "exec",
                    "batch_id": "batch-x",
                    "route": "gray",
                    "sample_ids": ["q-1"],
                    "param_version": 1,
                },
                2000,
            )
        assert exc.value.bound_version == 1
        assert exc.value.current_version == 2
    finally:
        host.stop()


def test_e_mixed_lists_batch_and_conflicting_versions():
    from gsb.errors import e_mixed

    err = e_mixed("混装", batch_id="batch-x", versions=sorted({1, 2}))
    assert err.code == "E_MIXED"
    assert err.context["batch_id"] == "batch-x"
    assert err.context["versions"] == [1, 2]
    assert isinstance(err, GSBError)
