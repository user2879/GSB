import os
import unittest

from repairbench.netprobe import probe_ports
from repairbench.spec import load_spec

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "groups.json")


class TestNetProbe(unittest.TestCase):
    def test_loopback_ports_listenable(self):
        result = probe_ports(load_spec(SAMPLE))
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["ports"]), 5)
        for entry in result["ports"]:
            self.assertTrue(entry["address"].startswith("127.0.0.1:"))


if __name__ == "__main__":
    unittest.main()
