import json
import tempfile
import unittest
from pathlib import Path

from degprop.errors import CapacityOverflowError, CycleError, TopologyError
from degprop.loader import load_events, load_topology, topology_digest

ROOT = Path(__file__).resolve().parents[1]
TOPO = ROOT / "samples" / "topology.json"
EVENTS = ROOT / "samples" / "events.ndjson"


class LoaderTest(unittest.TestCase):
    def test_loads_valid_sample(self):
        topology = load_topology(TOPO)
        events = load_events(EVENTS, topology)
        self.assertEqual(len(events), 16)
        self.assertEqual(topology.order_index[0], "svc-gateway")
        self.assertEqual(len(topology_digest(topology)), 64)

    def _write(self, doc: dict) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="utf-8")
        json.dump(doc, handle)
        handle.close()
        return Path(handle.name)

    def test_duplicate_service_id_rejected(self):
        doc = json.loads(TOPO.read_text(encoding="utf-8"))
        doc["services"][1]["service_id"] = doc["services"][0]["service_id"]
        path = self._write(doc)
        with self.assertRaises(TopologyError) as ctx:
            load_topology(path)
        self.assertEqual(ctx.exception.code, "E_TOPO")
        self.assertEqual(ctx.exception.field, "service_id")
        self.assertIsNotNone(ctx.exception.line)

    def test_non_loopback_host_rejected(self):
        doc = json.loads(TOPO.read_text(encoding="utf-8"))
        doc["services"][0]["host"] = "10.0.0.1"
        with self.assertRaises(TopologyError) as ctx:
            load_topology(self._write(doc))
        self.assertEqual(ctx.exception.field, "host")

    def test_duplicate_port_rejected(self):
        doc = json.loads(TOPO.read_text(encoding="utf-8"))
        doc["services"][1]["port"] = doc["services"][0]["port"]
        with self.assertRaises(TopologyError) as ctx:
            load_topology(self._write(doc))
        self.assertEqual(ctx.exception.field, "port")

    def test_missing_domain_reference_rejected(self):
        doc = json.loads(TOPO.read_text(encoding="utf-8"))
        doc["services"][0]["domain_id"] = "dom-ghost"
        with self.assertRaises(TopologyError) as ctx:
            load_topology(self._write(doc))
        self.assertEqual(ctx.exception.field, "domain_id")

    def test_cycle_reports_e_cycle(self):
        doc = json.loads(TOPO.read_text(encoding="utf-8"))
        doc["edges"].append({"upstream": "svc-db", "downstream": "svc-gateway",
                             "propagation_mode": "forced"})
        with self.assertRaises(CycleError) as ctx:
            load_topology(self._write(doc))
        self.assertEqual(ctx.exception.code, "E_CYCLE")

    def test_capacity_overflow_reports_e_capacity(self):
        doc = json.loads(TOPO.read_text(encoding="utf-8"))
        doc["services"][0]["capacity_limit"] = 2**62
        with self.assertRaises(CapacityOverflowError) as ctx:
            load_topology(self._write(doc))
        self.assertEqual(ctx.exception.code, "E_CAPACITY")

    def test_non_positive_capacity_rejected(self):
        doc = json.loads(TOPO.read_text(encoding="utf-8"))
        doc["services"][0]["capacity_limit"] = 0
        with self.assertRaises(TopologyError) as ctx:
            load_topology(self._write(doc))
        self.assertEqual(ctx.exception.field, "capacity_limit")

    def test_event_gap_rejected(self):
        topology = load_topology(TOPO)
        lines = EVENTS.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace('"seq": 2', '"seq": 9')
        handle = tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False,
                                             encoding="utf-8")
        handle.write("\n".join(lines))
        handle.close()
        with self.assertRaises(TopologyError) as ctx:
            load_events(handle.name, topology)
        self.assertEqual(ctx.exception.field, "seq")

    def test_arrival_not_monotonic_rejected(self):
        topology = load_topology(TOPO)
        lines = EVENTS.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace('"arrival_ms": 20', '"arrival_ms": 5')
        handle = tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False,
                                             encoding="utf-8")
        handle.write("\n".join(lines))
        handle.close()
        with self.assertRaises(TopologyError) as ctx:
            load_events(handle.name, topology)
        self.assertEqual(ctx.exception.field, "arrival_ms")


if __name__ == "__main__":
    unittest.main()
