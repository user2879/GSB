import json

import pytest

from chaos.spec import Spec


@pytest.fixture()
def small_spec_dict():
    return {
        "name": "test", "seed": 42,
        "budget": {"max_total_injections": 4, "max_parallel": 2,
                   "max_observe_ms": 300},
        "sampling": {"interval_ms": 25, "min_samples": 2, "scale": 10000},
        "services": [
            {"id": "svc-a", "host": "127.0.0.1", "port": 0, "control_port": 0,
             "fixed_latency_ms": 2, "error_rate_ppm": 0, "rebuildable": True},
            {"id": "svc-b", "host": "127.0.0.1", "port": 0, "control_port": 0,
             "fixed_latency_ms": 2, "error_rate_ppm": 0, "rebuildable": True},
        ],
        "injections": [
            {"type": "latency", "target": "svc-a", "duration_ms": 200,
             "observe_ms": 200, "params": {"delay_ms": 30}},
            {"type": "error_rate", "target": "svc-a", "duration_ms": 200,
             "observe_ms": 200, "params": {"error_rate_ppm": 8000}},
            {"type": "suspend", "target": "svc-b", "duration_ms": 200,
             "observe_ms": 200, "params": {}},
            {"type": "kill", "target": "svc-b", "duration_ms": 200,
             "observe_ms": 200, "params": {"restart": True}},
        ],
    }


@pytest.fixture()
def resolved_spec(tmp_path, small_spec_dict):
    from chaos.ports import resolve_ports
    from chaos.spec import load_spec
    doc = resolve_ports(small_spec_dict)
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(doc))
    return load_spec(str(path))


@pytest.fixture()
def spec_obj(small_spec_dict):
    return Spec(json.loads(json.dumps(small_spec_dict)))
