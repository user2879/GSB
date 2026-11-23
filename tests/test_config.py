import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccobs.config import load_peer_script, load_session_config
from ccobs.errors import E_CONFIG, BizError

BASE_CFG = {
    "initial_seq": 1000, "cwnd_max": 65535, "mss": 1460, "scale_factor": 3,
    "session_budget": 64, "batch_size": 4, "window_batches": 2,
    "dup_ack_limit": 3, "peer_port": 39211,
}
BASE_SCRIPT = {"entries": [
    {"event_seq": 1000 + i, "action": "ack", "delay_ms": 1} for i in range(4)
]}


def write(tmp, name, obj):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return path


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def load(self, cfg=None, script=None):
        cfg_path = write(self.tmp, "c.json", cfg or BASE_CFG)
        config = load_session_config(cfg_path)
        if script is None:
            return config
        script_path = write(self.tmp, "s.json", script)
        return config, load_peer_script(script_path, config)

    def test_valid_loads(self):
        config, script = self.load(script=BASE_SCRIPT)
        self.assertEqual(config.unit, 1000)
        self.assertEqual(len(script.entries), 4)
        self.assertEqual(script.entries[0].event_seq, 1000)

    def test_scale_factor_range(self):
        for bad in (-1, 7, True, "3"):
            cfg = dict(BASE_CFG, scale_factor=bad)
            with self.assertRaises(BizError) as ctx:
                self.load(cfg=cfg)
            self.assertEqual(ctx.exception.code, E_CONFIG)
            self.assertIn("field=scale_factor", str(ctx.exception))

    def test_negative_delay_rejected(self):
        script = {"entries": [
            {"event_seq": 1000, "action": "ack", "delay_ms": -1}]}
        with self.assertRaises(BizError) as ctx:
            self.load(script=script)
        self.assertEqual(ctx.exception.code, E_CONFIG)
        self.assertIn("entries[0]", str(ctx.exception))

    def test_non_continuous_seq_rejected_at_first_bad_entry(self):
        script = {"entries": [
            {"event_seq": 1000, "action": "ack"},
            {"event_seq": 1002, "action": "ack"},
        ]}
        with self.assertRaises(BizError) as ctx:
            self.load(script=script)
        self.assertIn("entry=entries[1]", str(ctx.exception))
        self.assertIn("expected=1001", str(ctx.exception))

    def test_duplicate_seq_rejected(self):
        script = {"entries": [
            {"event_seq": 1000, "action": "ack"},
            {"event_seq": 1000, "action": "ack"},
        ]}
        with self.assertRaises(BizError) as ctx:
            self.load(script=script)
        self.assertEqual(ctx.exception.code, E_CONFIG)

    def test_unknown_action_rejected(self):
        script = {"entries": [{"event_seq": 1000, "action": "explode"}]}
        with self.assertRaises(BizError) as ctx:
            self.load(script=script)
        self.assertIn("field=action", str(ctx.exception))

    def test_budget_exceeded_rejected(self):
        cfg = dict(BASE_CFG, session_budget=2)
        with self.assertRaises(BizError) as ctx:
            self.load(cfg=cfg, script=BASE_SCRIPT)
        self.assertIn("budget=2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
