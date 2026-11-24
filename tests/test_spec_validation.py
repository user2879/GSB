import copy
import json

import pytest

from receipt_ledger.errors import SpecError
from receipt_ledger.spec import load_spec


def _write(tmp_path, workers, requests, rules):
    paths = {}
    for name, payload in (("workers", workers), ("requests", requests),
                          ("rules", rules)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)
    return paths


def _base():
    workers = [{
        "worker_id": "w1", "host": "127.0.0.1", "port": 39201,
        "model_dim": 4, "hidden_dim": 3, "seed": 7,
        "quant": "fp32", "param_format": "pf1",
    }]
    requests = [{
        "seq": 1, "client_key": "k1", "vector": [1, 2, 3, 4],
        "arrival_ms": 0,
    }]
    rules = [{
        "rule_id": "r1", "worker_ids": ["w1"], "batch_size": 1,
        "param_version": 1, "receipt_formats": ["r3"],
        "retry_policy": "immediate", "max_retries": 1,
        "socket_timeout_ms": 1000,
    }]
    return workers, requests, rules


def test_valid_load(tmp_path):
    paths = _write(tmp_path, *_base())
    spec = load_spec(paths["workers"], paths["requests"], paths["rules"])
    assert spec.workers[0].worker_id == "w1"


@pytest.mark.parametrize("mutator,field", [
    (lambda w, r, u: w[0].update(host="10.0.0.1"), "host"),
    (lambda w, r, u: w[0].update(port=80), "port"),
    (lambda w, r, u: w[0].update(model_dim=0), "model_dim"),
    (lambda w, r, u: w[0].update(seed="x"), "seed"),
    (lambda w, r, u: w[0].update(quant="fp99"), "quant"),
    (lambda w, r, u: w[0].update(param_format="pf9"), "param_format"),
    (lambda w, r, u: r[0].update(seq=1), "seq"),
    (lambda w, r, u: r[0].update(vector=[1, 2]), "vector"),
    (lambda w, r, u: u[0].update(rule_id="r1"), "rule_id"),
    (lambda w, r, u: u[0].update(worker_ids=["nope"]), "worker_ids"),
])
def test_first_violation_reported(tmp_path, mutator, field):
    workers, requests, rules = _base()
    if field == "rule_id":
        rules.append(copy.deepcopy(rules[0]))
    if field == "seq":
        requests.append(copy.deepcopy(requests[0]))
    mutator(workers, requests, rules)
    paths = _write(tmp_path, workers, requests, rules)
    with pytest.raises(SpecError) as exc:
        load_spec(paths["workers"], paths["requests"], paths["rules"])
    assert exc.value.code == "E_SPEC"
    assert exc.value.location["field"] == field


def test_duplicate_worker_id_and_port(tmp_path):
    workers, requests, rules = _base()
    workers.append({**workers[0], "worker_id": "w2", "port": 39201})
    paths = _write(tmp_path, workers, requests, rules)
    with pytest.raises(SpecError) as exc:
        load_spec(paths["workers"], paths["requests"], paths["rules"])
    assert exc.value.location["field"] == "port"

    workers, requests, rules = _base()
    workers.append({**workers[0], "worker_id": "w1", "port": 39202})
    paths = _write(tmp_path, workers, requests, rules)
    with pytest.raises(SpecError) as exc:
        load_spec(paths["workers"], paths["requests"], paths["rules"])
    assert exc.value.location["field"] == "worker_id"


def test_bool_rejected_as_integer(tmp_path):
    workers, requests, rules = _base()
    workers[0]["hidden_dim"] = True
    paths = _write(tmp_path, workers, requests, rules)
    with pytest.raises(SpecError) as exc:
        load_spec(paths["workers"], paths["requests"], paths["rules"])
    assert exc.value.location["field"] == "hidden_dim"
