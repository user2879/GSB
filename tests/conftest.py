import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIM = 4


def _backends(tmp_path, n=2):
    backs = []
    for i in range(n):
        backs.append({
            "model_id": f"m{i}", "host": "127.0.0.1", "port": 19000 + i,
            "params": [0.1 * (i + 1)] * DIM,
            "max_concurrency": 1 + (i % 3), "batch_size": 2,
        })
    path = tmp_path / "backends.json"
    path.write_text(json.dumps({"backends": backs}))
    return str(path)


def _rules(tmp_path, weight=5000, batch_size=2):
    rules = {"rules": [{
        "rule_id": "r1", "prefix": "c:",
        "stable_backend": "m0", "canary_backend": "m1",
        "canary_weight_bp": weight, "batch_size": batch_size,
    }]}
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules))
    return str(path)


def _requests(tmp_path, count=8):
    lines = []
    for seq in range(count):
        lines.append(json.dumps({
            "seq": seq, "client_key": f"c:client-{seq % 3}",
            "inputs": [0.5] * DIM, "expected_stage": seq // 2,
        }))
    path = tmp_path / "requests.ndjson"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def sample(tmp_path):
    return {
        "backends": _backends(tmp_path),
        "rules": _rules(tmp_path),
        "requests": _requests(tmp_path),
        "db": str(tmp_path / "ledger.db"),
    }
