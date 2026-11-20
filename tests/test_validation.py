import json

import pytest

from gateway.errors import ValidationError
from gateway.loader import load_declaration


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def test_valid_declaration_loads(sample):
    decl = load_declaration(sample["backends"], sample["rules"],
                            sample["requests"])
    assert decl.param_dim == 4
    assert [b.model_id for b in decl.backends] == ["m0", "m1"]
    assert [r.seq for r in decl.requests] == list(range(8))


def test_duplicate_model_id_rejected(sample, tmp_path):
    backs = json.loads(open(sample["backends"]).read())
    backs["backends"].append(dict(backs["backends"][0], port=19999))
    path = _write(tmp_path / "bad.json", backs)
    with pytest.raises(ValidationError) as exc:
        load_declaration(path, sample["rules"], sample["requests"])
    assert exc.value.path == "backends[2].model_id"


def test_non_loopback_host_rejected(sample, tmp_path):
    backs = json.loads(open(sample["backends"]).read())
    backs["backends"][0]["host"] = "0.0.0.0"
    path = _write(tmp_path / "bad.json", backs)
    with pytest.raises(ValidationError) as exc:
        load_declaration(path, sample["rules"], sample["requests"])
    assert exc.value.path == "backends[0].host"


def test_weight_out_of_range_rejected(sample, tmp_path):
    rules = json.loads(open(sample["rules"]).read())
    rules["rules"][0]["canary_weight_bp"] = 10001
    path = _write(tmp_path / "bad.json", rules)
    with pytest.raises(ValidationError) as exc:
        load_declaration(sample["backends"], path, sample["requests"])
    assert exc.value.path == "rules[0].canary_weight_bp"


def test_non_finite_param_rejected(sample, tmp_path):
    backs = json.loads(open(sample["backends"]).read())
    backs["backends"][1]["params"][2] = float("nan")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(backs).replace("NaN", "1e999"))
    with pytest.raises(ValidationError) as exc:
        load_declaration(str(path), sample["rules"], sample["requests"])
    assert exc.value.path == "backends[1].params[2]"


def test_sequence_gap_rejected(sample, tmp_path):
    lines = open(sample["requests"]).read().splitlines()
    lines[3] = json.dumps({"seq": 9, "client_key": "c:x",
                           "inputs": [0.5] * 4, "expected_stage": 0})
    path = tmp_path / "bad.ndjson"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValidationError) as exc:
        load_declaration(sample["backends"], sample["rules"], str(path))
    assert exc.value.path == "requests"


def test_input_dim_mismatch_rejected(sample, tmp_path):
    lines = open(sample["requests"]).read().splitlines()
    lines[0] = json.dumps({"seq": 0, "client_key": "c:x",
                           "inputs": [0.5] * 3, "expected_stage": 0})
    path = tmp_path / "bad.ndjson"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValidationError) as exc:
        load_declaration(sample["backends"], sample["rules"], str(path))
    assert exc.value.path == "requests[0].inputs"


def test_zero_batch_size_rejected(sample, tmp_path):
    backs = json.loads(open(sample["backends"]).read())
    backs["backends"][0]["batch_size"] = 0
    path = _write(tmp_path / "bad.json", backs)
    with pytest.raises(ValidationError) as exc:
        load_declaration(path, sample["rules"], sample["requests"])
    assert exc.value.path == "backends[0].batch_size"
