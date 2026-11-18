import unittest
from pathlib import Path

from qos.canonical import canonical_bytes
from qos.changeset import ChangeSet, apply_changeset
from qos.compiler import PlanCompiler
from qos.errors import IncrementalError
from qos.ruleset import build_graph, load_rules

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(SAMPLES / "rules.json")
        self.graph = build_graph(self.rules)

    def test_deterministic_priority_order_and_bytes(self):
        p1 = PlanCompiler.full_recompile(self.rules, self.graph)
        p2 = PlanCompiler.full_recompile(self.rules, self.graph)
        self.assertEqual(canonical_bytes(p1.payload()),
                         canonical_bytes(p2.payload()))
        self.assertEqual(p1.plan_hash, p2.plan_hash)
        ids = [e["rule_id"] for e in p1.entries]
        # 更具体的长前缀（如 R6 的 1400）排在短前缀（R1 的 14）之前
        self.assertLess(ids.index("R6_GRAND"), ids.index("R1_BASE"))

    def test_monotonic_plan_version_and_full_replacement(self):
        compiler = PlanCompiler()
        all_ids = [r.rule_id for r in self.rules]
        v1 = compiler.compile(self.rules, self.graph, all_ids,
                              next_plan_version=1)
        v2 = compiler.compile(self.rules, self.graph, all_ids,
                              next_plan_version=2)
        self.assertEqual(v1.plan_version, 1)
        self.assertEqual(v2.plan_version, 2)
        with self.assertRaises(ValueError):
            compiler.compile(self.rules, self.graph, all_ids,
                             next_plan_version=2)

    def test_incremental_reuses_unchanged_byte_identical_artifacts(self):
        compiler = PlanCompiler()
        all_ids = [r.rule_id for r in self.rules]
        first = compiler.compile(self.rules, self.graph, all_ids,
                                 next_plan_version=1)
        first_bytes = {rid: canonical_bytes(entry)
                       for rid, entry in
                       ((e["rule_id"], e) for e in first.entries)}
        first_versions = dict(first.artifact_versions)

        cs = ChangeSet.from_dict(__import__("json").loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        applied = apply_changeset(self.rules, cs)
        second = compiler.compile(applied.rules, applied.graph,
                                  applied.invalidation_sequence,
                                  next_plan_version=2)
        # 复用规则产物逐字节一致且产物版本号不变
        for rid in second.reused:
            entry = next(e for e in second.entries if e["rule_id"] == rid)
            self.assertEqual(canonical_bytes(entry), first_bytes[rid])
            self.assertEqual(second.artifact_versions[rid],
                             first_versions[rid])
        # 被修改规则产物版本号递增
        self.assertEqual(second.artifact_versions["R1_BASE"], 2)
        self.assertIn("R1_BASE", second.recompiled)
        self.assertNotIn("R2_BASE", second.recompiled)

    def test_incremental_equals_full_compile(self):
        compiler = PlanCompiler()
        all_ids = [r.rule_id for r in self.rules]
        compiler.compile(self.rules, self.graph, all_ids, next_plan_version=1)
        cs = ChangeSet.from_dict(__import__("json").loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        applied = apply_changeset(self.rules, cs)
        incremental = compiler.compile(applied.rules, applied.graph,
                                       applied.invalidation_sequence,
                                       next_plan_version=2)
        report = PlanCompiler.assert_incremental_matches_full(
            incremental, applied.rules, applied.graph)
        self.assertTrue(report["ok"])
        self.assertEqual(report["reused_count"], 3)   # R2/R5/R8 支系
        self.assertEqual(report["recompiled_count"], 6)

    def test_e_incr_reports_first_diff_rule(self):
        # 手工污染增量计划的某条产物哈希，验证 E_INCR 定位能力
        plan = PlanCompiler.full_recompile(self.rules, self.graph)
        plan.artifact_hashes["R2_BASE"] = "deadbeef"
        with self.assertRaises(IncrementalError) as ctx:
            PlanCompiler.assert_incremental_matches_full(
                plan, self.rules, self.graph)
        self.assertEqual(ctx.exception.code, "E_INCR")
        self.assertEqual(ctx.exception.details["first_diff_rule"], "R2_BASE")


if __name__ == "__main__":
    unittest.main()
