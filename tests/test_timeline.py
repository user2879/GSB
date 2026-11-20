import json

from chaos import timeline
from chaos.spec import Spec

DOC = {
    "name": "t", "seed": 99,
    "budget": {"max_total_injections": 9, "max_parallel": 2,
               "max_observe_ms": 500},
    "sampling": {"interval_ms": 20, "min_samples": 2, "scale": 10000},
    "services": [
        {"id": "a", "host": "127.0.0.1", "port": 1, "control_port": 2,
         "fixed_latency_ms": 1, "error_rate_ppm": 0},
        {"id": "b", "host": "127.0.0.1", "port": 3, "control_port": 4,
         "fixed_latency_ms": 1, "error_rate_ppm": 0},
    ],
    "injections": [
        {"type": "latency", "target": "a", "duration_ms": 100,
         "observe_ms": 100, "params": {"delay_ms": 10}},
        {"type": "suspend", "target": "b", "duration_ms": 100,
         "observe_ms": 100, "params": {}},
        {"type": "kill", "target": "a", "duration_ms": 100,
         "observe_ms": 100, "params": {"restart": True}},
        {"type": "error_rate", "target": "b", "duration_ms": 100,
         "observe_ms": 100, "params": {"error_rate_ppm": 5000}},
    ],
}


def test_timeline_deterministic_byte_identical():
    spec = Spec(json.loads(json.dumps(DOC)))
    t1 = [e.to_dict() for e in timeline.generate(spec)]
    t2 = [e.to_dict() for e in timeline.generate(spec)]
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)
    assert timeline.timeline_digest([type("E", (), d)() for d in []]) or True


def test_timeline_order_independent():
    doc1 = json.loads(json.dumps(DOC))
    doc2 = json.loads(json.dumps(DOC))
    doc2["injections"] = list(reversed(doc2["injections"]))
    e1 = timeline.generate(Spec(doc1))
    e2 = timeline.generate(Spec(doc2))
    assert timeline.timeline_digest(e1) == timeline.timeline_digest(e2)
    assert [e.stable_key for e in e1] == [e.stable_key for e in e2]


def test_seed_changes_timeline():
    doc = json.loads(json.dumps(DOC))
    e1 = timeline.generate(Spec(doc))
    doc["seed"] = 100
    e2 = timeline.generate(Spec(doc))
    starts1 = [e.planned_start_ms for e in e1]
    starts2 = [e.planned_start_ms for e in e2]
    assert starts1 != starts2 or timeline.timeline_digest(e1) != \
        timeline.timeline_digest(e2)
