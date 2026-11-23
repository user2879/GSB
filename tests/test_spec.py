import json

import pytest

from gsb.errors import GSBError
from gsb.spec import load_spec

SAMPLE_DIR = "samples"


def _write(tmp_path, name, data):
    path = tmp_path / name
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_backend(**overrides):
    base = {
        "backend_id": "b1",
        "host": "127.0.0.1",
        "port": 18001,
        "model_dim": 4,
        "hidden_dim": 8,
        "seed": 7,
        "quant_profile": "int8",
        "format_version": "v1",
    }
    base.update(overrides)
    return base


def _make_spec(tmp_path, backends, rules, requests):
    _write(tmp_path, "backends.json", {"backends": backends})
    _write(tmp_path, "rules.json", {"rules": rules})
    _write(tmp_path, "requests.ndjson", requests)


def _rule(**overrides):
    base = {
        "rule_id": "r1",
        "stable_backend": "b1",
        "gray_backend": "b2",
        "weight": 5000,
        "rule_version": 1,
    }
    base.update(overrides)
    return base


REQUESTS = (
    '{"request_id": "q1", "batch_id": "ba", "client_key": "c1", "rule_id": "r1"}\n'
)


def test_samples_load_ok():
    spec = load_spec(SAMPLE_DIR)
    assert len(spec.backends) == 4
    assert len(spec.rules) == 2
    assert len(spec.requests) == 12
    assert [b.batch_id for b in spec.batches] == [
        "batch-1",
        "batch-2",
        "batch-3",
        "batch-4",
    ]


@pytest.mark.parametrize(
    "override, field",
    [
        ({"host": "localhost"}, "host"),
        ({"host": "127.0.0.2"}, "host"),
        ({"port": 0}, "port"),
        ({"port": "18001"}, "port"),
        ({"model_dim": 0}, "model_dim"),
        ({"model_dim": -3}, "model_dim"),
        ({"hidden_dim": 1.5}, "hidden_dim"),
        ({"seed": "x"}, "seed"),
        ({"seed": True}, "seed"),
        ({"quant_profile": "fp32"}, "quant_profile"),
        ({"format_version": "v9"}, "format_version"),
    ],
)
def test_backend_field_violations(tmp_path, override, field):
    backends = [_valid_backend(), _valid_backend(backend_id="b2", port=18002)]
    backends[0].update(override)
    _make_spec(tmp_path, backends, [_rule()], REQUESTS)
    with pytest.raises(GSBError) as exc:
        load_spec(tmp_path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["field"] == field
    assert exc.value.context["owner"] == "b1"


def test_duplicate_backend_id(tmp_path):
    backends = [_valid_backend(), _valid_backend(port=18002)]
    _make_spec(tmp_path, backends, [_rule()], REQUESTS)
    with pytest.raises(GSBError) as exc:
        load_spec(tmp_path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["field"] == "backend_id"


def test_duplicate_port(tmp_path):
    backends = [_valid_backend(), _valid_backend(backend_id="b2")]
    _make_spec(tmp_path, backends, [_rule()], REQUESTS)
    with pytest.raises(GSBError) as exc:
        load_spec(tmp_path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["field"] == "port"
    assert exc.value.context["owner"] == "b2"


@pytest.mark.parametrize(
    "override, field",
    [
        ({"weight": -1}, "weight"),
        ({"weight": 10001}, "weight"),
        ({"weight": 0.5}, "weight"),
        ({"stable_backend": "nope"}, "stable_backend"),
        ({"gray_backend": "nope"}, "gray_backend"),
        ({"rule_version": 0}, "rule_version"),
    ],
)
def test_rule_violations(tmp_path, override, field):
    backends = [_valid_backend(), _valid_backend(backend_id="b2", port=18002)]
    rule = _rule()
    rule.update(override)
    _make_spec(tmp_path, backends, [rule], REQUESTS)
    with pytest.raises(GSBError) as exc:
        load_spec(tmp_path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["field"] == field
    assert exc.value.context["owner"] == "r1"


def test_weight_boundary_ok(tmp_path):
    backends = [_valid_backend(), _valid_backend(backend_id="b2", port=18002)]
    _make_spec(
        tmp_path,
        backends,
        [_rule(weight=0), _rule(rule_id="r2", weight=10000)],
        REQUESTS,
    )
    spec = load_spec(tmp_path)
    assert spec.rules["r1"].weight == 0
    assert spec.rules["r2"].weight == 10000


def test_request_unknown_rule(tmp_path):
    backends = [_valid_backend(), _valid_backend(backend_id="b2", port=18002)]
    bad = '{"request_id": "q9", "batch_id": "ba", "client_key": "c", "rule_id": "rx"}\n'
    _make_spec(tmp_path, backends, [_rule()], bad)
    with pytest.raises(GSBError) as exc:
        load_spec(tmp_path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["field"] == "rule_id"
    assert exc.value.context["owner"] == "q9"
