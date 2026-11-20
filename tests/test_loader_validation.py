"""强校验：E_SPEC、E_ORDER、E_PRIORITY 与首个出错对象定位。"""
import json
import pytest

from quota_arbiter.errors import OrderError, PriorityError, SpecError
from quota_arbiter.loader import load_requests, load_spec


def _write(tmp_path, name, obj):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        if name.endswith(".ndjson"):
            fh.write("\n".join(json.dumps(x, ensure_ascii=False) for x in obj))
        else:
            json.dump(obj, fh, ensure_ascii=False)
    return str(path)


VALID_SPEC = {
    "tenants": [
        {"tid": "A", "quota": 100, "concurrency_cap": 2, "burst": 10},
        {"tid": "B", "quota": 50, "concurrency_cap": 1, "burst": 0},
    ],
    "pools": [
        {"pid": "P1", "capacity": 30, "priority_order": ["A", "B"]},
    ],
}


def _req(rid, tid, arrival, demand=5, priority=0, pid="P1"):
    return {"rid": rid, "tid": tid, "pid": pid, "demand": demand,
            "arrival": arrival, "priority": priority}


def test_valid_spec_and_requests(tmp_path):
    spec = _write(tmp_path, "spec.json", VALID_SPEC)
    req = _write(tmp_path, "r.ndjson", [_req("r1", "A", 0), _req("r2", "B", 1)])
    tenants, pools = load_spec(spec)
    requests = load_requests(req, tenants, pools)
    assert len(tenants) == 2 and len(pools) == 1 and len(requests) == 2


def test_duplicate_tenant_id_espec(tmp_path):
    spec_obj = {
        "tenants": [
            {"tid": "A", "quota": 1, "concurrency_cap": 1, "burst": 0},
            {"tid": "A", "quota": 2, "concurrency_cap": 1, "burst": 0},
        ],
        "pools": [{"pid": "P1", "capacity": 1, "priority_order": ["A"]}],
    }
    spec = _write(tmp_path, "spec.json", spec_obj)
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.code == "E_SPEC"
    assert exc.value.ref_id == "A" and exc.value.field == "tid"


def test_non_positive_quota_espec(tmp_path):
    spec_obj = {
        "tenants": [{"tid": "A", "quota": 0, "concurrency_cap": 1, "burst": 0}],
        "pools": [{"pid": "P1", "capacity": 1, "priority_order": ["A"]}],
    }
    spec = _write(tmp_path, "spec.json", spec_obj)
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.field == "quota"


def test_priority_duplicate_espec(tmp_path):
    spec_obj = {
        "tenants": [{"tid": "A", "quota": 1, "concurrency_cap": 1, "burst": 0}],
        "pools": [{"pid": "P1", "capacity": 1, "priority_order": ["A", "A"]}],
    }
    spec = _write(tmp_path, "spec.json", spec_obj)
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.field == "priority_order"


def test_priority_missing_tenant_epriority(tmp_path):
    spec_obj = {
        "tenants": [{"tid": "A", "quota": 1, "concurrency_cap": 1, "burst": 0}],
        "pools": [{"pid": "P1", "capacity": 1, "priority_order": ["A", "GHOST"]}],
    }
    spec = _write(tmp_path, "spec.json", spec_obj)
    with pytest.raises(PriorityError) as exc:
        load_spec(spec)
    assert exc.value.code == "E_PRIORITY"
    assert exc.value.ref_id == "P1"
    assert exc.value.detail["missing_tid"] == "GHOST"


def test_same_tenant_arrival_duplicate_espec(tmp_path):
    spec = _write(tmp_path, "spec.json", VALID_SPEC)
    req = _write(tmp_path, "r.ndjson", [_req("r1", "A", 5), _req("r2", "A", 5)])
    tenants, pools = load_spec(spec)
    with pytest.raises(SpecError) as exc:
        load_requests(req, tenants, pools)
    assert exc.value.ref_id == "r2" and exc.value.field == "arrival"


def test_global_order_regression_eorder(tmp_path):
    spec = _write(tmp_path, "spec.json", VALID_SPEC)
    req = _write(tmp_path, "r.ndjson",
                 [_req("r1", "A", 10), _req("r2", "B", 3)])
    tenants, pools = load_spec(spec)
    with pytest.raises(OrderError) as exc:
        load_requests(req, tenants, pools, check_global_order=True)
    assert exc.value.code == "E_ORDER" and exc.value.ref_id == "r2"


def test_unknown_tenant_in_request_espec(tmp_path):
    spec = _write(tmp_path, "spec.json", VALID_SPEC)
    req = _write(tmp_path, "r.ndjson", [_req("r1", "Z", 0)])
    tenants, pools = load_spec(spec)
    with pytest.raises(SpecError) as exc:
        load_requests(req, tenants, pools)
    assert exc.value.ref_id == "r1" and exc.value.field == "tid"
