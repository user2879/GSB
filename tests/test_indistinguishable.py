import json
import tempfile
import unittest
from pathlib import Path

from degprop.engine import DiagnosticsEngine
from degprop.loader import load_topology

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp: Path, name: str, payload) -> Path:
    path = tmp / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class IndistinguishablePlanTest(unittest.TestCase):
    def test_capacity_diff_inside_judgment_bound_marks_indistinguishable(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            topology = {
                "topology_version": "topo-indist-v1",
                "session_params": {"scale_exponent": 2, "global_concurrency": 2,
                                   "window_size": 5, "socket_timeout_ms": 200,
                                   "max_retries": 0},
                "domains": [
                    {"domain_id": "dom-a", "in_flight_quota": 1},
                    {"domain_id": "dom-b", "in_flight_quota": 1},
                    {"domain_id": "dom-c", "in_flight_quota": 1},
                ],
                "services": [
                    {"service_id": "svc-a", "domain_id": "dom-a",
                     "host": "127.0.0.1", "port": 19201,
                     "degrade_strategy": "discard", "capacity_limit": 50,
                     "seed": 32},
                    {"service_id": "svc-b", "domain_id": "dom-b",
                     "host": "127.0.0.1", "port": 19202,
                     "degrade_strategy": "cache_copy", "capacity_limit": 3,
                     "seed": 2},
                ],
                "edges": [
                    {"upstream": "svc-a", "downstream": "svc-b",
                     "propagation_mode": "forced"},
                ],
            }
            events = "\n".join(
                json.dumps({"seq": i, "arrival_ms": i * 10,
                            "upstream": "svc-a", "downstream": "svc-b",
                            "client_key": f"k{i}"})
                for i in range(1, 6)
            )
            topo_path = _write(tmp, "topo.json", topology)
            events_path = tmp / "events.ndjson"
            events_path.write_text(events, encoding="utf-8")
            ledger_path = tmp / "session.db"
            with DiagnosticsEngine(topo_path, events_path, ledger_path) as engine:
                report = engine.run_batch(engine.events, 1)
            plans = report.propagation.domain_plans
            # svc-b 触发后 forced 传播 svc-a；dom-b 容量 3，分摊差落入 J=200。
            self.assertTrue(any(p.indistinguishable for p in plans.values()))
            target = plans["dom-b"]
            judgment = target.judgment["capacity"]
            self.assertLessEqual(judgment["abs_diff_fixed"],
                                 judgment["judgment_bound_fixed"])
            # 决胜结果不依赖枚举顺序：local 容量更大应稳定胜出。
            self.assertEqual(target.source_plan, "local_priority")


if __name__ == "__main__":
    unittest.main()
