import json

import pytest

from chaos.errors import E_SPEC, ChaosError
from chaos.spec import load_spec

BASE = {
    "name": "t", "seed": 7,
    "budget": {"max_total_injections": 2, "max_parallel": 1,
               "max_observe_ms": 500},
    "sampling": {"interval_ms": 20, "min_samples": 2, "scale": 10000},
    "services": [
        {"id": "a", "host": "127.0.0.1", "port": 12001, "control_port": 12002,
         "fixed_latency_ms": 1, "error_rate_ppm": 0},
        {"id": "b", "host": "127.0.0.1", "port": 12003, "control_port": 12004,
         "fixed_latency_ms": 1, "error_rate_ppm": 100},
    ],
    "injections": [
        {"type": "latency", "target": "a", "duration_ms": 100,
         "observe_ms": 100, "params": {"delay_ms": 10}},
    ],
}


def write(tmp_path, doc):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_valid_spec_loads(tmp_path):
    spec = load_spec(write(tmp_path, BASE))
    assert spec.seed == 7
    assert len(spec.services) == 2
    assert spec.budget.max_parallel == 1


def expect_fail(tmp_path, mutate, field_prefix):
    doc = json.loads(json.dumps(BASE))
    mutate(doc)
    with pytest.raises(ChaosError) as exc:
        load_spec(write(tmp_path, doc))
    assert exc.value.code == E_SPEC
    assert exc.value.field.startswith(field_prefix)


def test_bad_seed(tmp_path):
    expect_fail(tmp_path, lambda d: d.update(seed="x"), "$.seed")


def test_bad_host(tmp_path):
    expect_fail(tmp_path, lambda d: d["services"][0].update(host="0.0.0.0"),
                "$.services[0].host")


def test_port_out_of_range(tmp_path):
    expect_fail(tmp_path, lambda d: d["services"][0].update(port=70000),
                "$.services[0].port")


def test_duplicate_service_id(tmp_path):
    expect_fail(tmp_path, lambda d: d["services"][1].update(id="a"),
                "$.services[1].id")


def test_duplicate_port(tmp_path):
    expect_fail(tmp_path, lambda d: d["services"][1].update(port=12001),
                "$.services[1].port")


def test_bad_injection_type(tmp_path):
    expect_fail(tmp_path, lambda d: d["injections"][0].update(type="flood"),
                "$.injections[0].type")


def test_unknown_target(tmp_path):
    expect_fail(tmp_path, lambda d: d["injections"][0].update(target="zz"),
                "$.injections[0].target")


def test_nonpositive_duration(tmp_path):
    expect_fail(tmp_path, lambda d: d["injections"][0].update(duration_ms=0),
                "$.injections[0].duration_ms")


def test_bad_budget(tmp_path):
    expect_fail(tmp_path, lambda d: d["budget"].update(max_parallel=0),
                "$.budget.max_parallel")


def test_corrupt_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"seed": 1, ')
    with pytest.raises(ChaosError) as exc:
        load_spec(str(path))
    assert exc.value.code == E_SPEC
    assert exc.value.field == "$"
