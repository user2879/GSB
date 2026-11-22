import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faultdiag.spec import DomainSpec, InstanceSpec

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def domains():
    return {"dom-a": DomainSpec("dom-a", 1, 300),
            "dom-b": DomainSpec("dom-b", 2, 500)}


@pytest.fixture
def instances():
    return {"s-a1": InstanceSpec("s-a1", "dom-a", 1, 1),
            "s-a2": InstanceSpec("s-a2", "dom-a", 1, 2),
            "s-b1": InstanceSpec("s-b1", "dom-b", 1, 1),
            "s-b2": InstanceSpec("s-b2", "dom-b", 1, 2)}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def spec_files(tmp_path):
    write_json(tmp_path / "domains.json",
               [{"id": "dom-a", "isolation_level": 1, "recovery_cooldown_ms": 300},
                {"id": "dom-b", "isolation_level": 2, "recovery_cooldown_ms": 500}])
    write_json(tmp_path / "instances.json",
               [{"id": "s-a1", "domain": "dom-a", "weight": 1, "replica": 1},
                {"id": "s-a2", "domain": "dom-a", "weight": 1, "replica": 2},
                {"id": "s-b1", "domain": "dom-b", "weight": 1, "replica": 1},
                {"id": "s-b2", "domain": "dom-b", "weight": 1, "replica": 2}])
    events = [
        {"seq": 1, "ts_ms": 0, "instance": "s-a1", "fault": "crash"},
        {"seq": 2, "ts_ms": 100, "instance": "s-a2", "fault": "recover"},
        {"seq": 3, "ts_ms": 200, "instance": "s-b1", "fault": "crash"},
        {"seq": 4, "ts_ms": 350, "instance": "s-a1", "fault": "recover"},
        {"seq": 5, "ts_ms": 400, "instance": "s-b2", "fault": "crash"},
        {"seq": 6, "ts_ms": 600, "instance": "s-a1", "fault": "crash"},
        {"seq": 7, "ts_ms": 700, "instance": "s-a1", "fault": "recover"},
        {"seq": 8, "ts_ms": 800, "instance": "s-b1", "fault": "recover"},
        {"seq": 9, "ts_ms": 900, "instance": "s-b2", "fault": "recover"},
        {"seq": 10, "ts_ms": 1000, "instance": "s-a2", "fault": "degrade"},
    ]
    (tmp_path / "events.ndjson").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return tmp_path
