import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qos.compiler import PlanCompiler
from qos.errors import ShapeError
from qos.ruleset import build_graph, load_rules
from qos.shaper import (deterministic_packets, is_port_free,
                        match_rule, run_virtual, verify_throughput)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class ShaperTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(SAMPLES / "rules.json")
        self.graph = build_graph(self.rules)
        self.plan = PlanCompiler.full_recompile(self.rules, self.graph)

    def test_longest_prefix_match(self):
        entry = match_rule(self.plan.entries, 14001)
        self.assertEqual(entry["rule_id"], "R6_GRAND")  # 前缀 1400 长于 140/14
        entry = match_rule(self.plan.entries, 14099)
        self.assertEqual(entry["rule_id"], "R3_CHILD")
        self.assertIsNone(match_rule(self.plan.entries, 99999))

    def test_virtual_simulation_respects_commit_rate(self):
        # 到达间隔 100ms：平均到达速率远低于承诺速率，突发由令牌桶吸收。
        packets = deterministic_packets(30, 1024, [14000, 14100, 15000])
        result = run_virtual(self.plan.entries, packets,
                             self.plan.plan_version, arrival_gap_us=100_000)
        report = verify_throughput(result, self.plan.entries)
        self.assertTrue(report)
        # 每个观测都包含要求字段
        rec = result.records()[0]
        for key in ("timestamp_us", "rule_id", "sent_bytes", "delay_us"):
            self.assertIn(key, rec)
        # 存在被整形器延迟的包（令牌桶欠债机制生效）
        self.assertTrue(any(o.delay_us > 0 for o in result.observations))

    def test_deterministic_observations(self):
        packets = deterministic_packets(20, 512, [14000])
        r1 = run_virtual(self.plan.entries, packets, 1, 50_000)
        r2 = run_virtual(self.plan.entries, packets, 1, 50_000)
        self.assertEqual(r1.records(), r2.records())

    def test_e_shape_when_throughput_exceeds_commit(self):
        # 构造一个极小速率规则 + 高频大包，验证违规能被 E_SHAPE 捕获
        from qos.models import Rule
        rule = Rule("R_TIGHT", "127.0.0.1", (199,), 1, 64, 0)
        graph = build_graph([rule])
        plan = PlanCompiler.full_recompile([rule], graph)
        packets = deterministic_packets(50, 4096, [19900])
        result = run_virtual(plan.entries, packets, plan.plan_version,
                             arrival_gap_us=1_000)
        with self.assertRaises(ShapeError) as ctx:
            verify_throughput(result, plan.entries, tolerance_ratio=0.0)
        self.assertEqual(ctx.exception.code, "E_SHAPE")
        self.assertIn("committed_bytes_per_sec", ctx.exception.details)
        self.assertGreater(ctx.exception.details["actual_bytes_per_sec"],
                           ctx.exception.details["committed_bytes_per_sec"])

    def test_real_loopback_tcp_subprocess(self):
        shaper_port, sink_port = 49251, 49252
        if not (is_port_free("127.0.0.1", shaper_port)
                and is_port_free("127.0.0.1", sink_port)):
            self.skipTest("回环高位端口被占用")
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(
                self.plan.payload()["entries"]), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "traffic_sim.py"), "run",
                 "--plan", str(plan_path),
                 "--shaper-port", str(shaper_port),
                 "--sink-port", str(sink_port),
                 "--count", "12", "--size", "256",
                 "--ports", "14000", "15000", "--gap-ms", "5",
                 "--out-dir", str(Path(tmp) / "out")],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = json.loads((Path(tmp) / "out" / "traffic_summary.json")
                                 .read_text())
            self.assertEqual(summary["sink"]["frames"], 12)
            self.assertEqual(summary["sink"]["total_bytes"], 12 * 256)
            obs = json.loads((Path(tmp) / "out" / "observations.json")
                             .read_text())
            self.assertTrue(all(o["rule_id"] for o in obs))


if __name__ == "__main__":
    unittest.main()
