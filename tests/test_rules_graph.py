import unittest
from pathlib import Path

from qos.errors import DependError, RuleError
from qos.ruleset import load_rules, ruleset_version
from qos.changeset import ChangeSet, apply_changeset
from qos.errors import RefError

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class RuleLoadingTests(unittest.TestCase):
    def test_load_valid_rules(self):
        rules = load_rules(SAMPLES / "rules.json")
        self.assertEqual(len(rules), 9)
        self.assertEqual(len({r.rule_id for r in rules}), 9)

    def test_corrupt_file_is_rejected_with_e_rule(self):
        with self.assertRaises(RuleError) as ctx:
            load_rules(SAMPLES / "rules_corrupt.json")
        self.assertEqual(ctx.exception.code, "E_RULE")
        self.assertIn("line", ctx.exception.details)

    def test_invalid_fields_are_all_reported_and_rejected(self):
        with self.assertRaises(RuleError) as ctx:
            load_rules(SAMPLES / "rules_invalid_fields.json")
        violations = ctx.exception.details["violations"]
        rule_ids = {v["rule_id"] for v in violations}
        paths = {v["field_path"] for v in violations}
        self.assertIn("R_BAD_ADDR", rule_ids)
        self.assertIn("R_BAD_PORT", rule_ids)
        self.assertIn("R_BAD_QUEUE", rule_ids)
        self.assertIn("R_BAD_RATE", rule_ids)
        self.assertTrue(any("shaping" in path for path in paths))

    def test_missing_upstream_reference_reports_field_path(self):
        import json
        from qos.models import parse_rules, validate_rule_dicts
        raw = json.loads((SAMPLES / "rules_invalid_fields.json").read_text())
        # 仅保留字段合法但引用不存在幽灵上游的场景
        clean = [item for item in raw if item["rule_id"] == "R_OK"]
        clean.append({
            "rule_id": "R_GHOST_REF",
            "match": {"dst_addr": "127.0.0.1", "port_prefixes": [180]},
            "shaping": {"rate_kbps": 100, "burst_bytes": 2048,
                        "queue_limit_packets": 4},
            "upstream": ["R_GHOST"], "actions": []})
        self.assertEqual(validate_rule_dicts(clean), [])
        rules = parse_rules(clean)
        from qos.graph import DependencyGraph
        graph = DependencyGraph(rules)
        refs = graph.validate_references()
        self.assertTrue(any(v.field_path == "upstream" for v in refs))
        with self.assertRaises(RuleError):
            from qos.ruleset import build_graph
            build_graph(rules)

    def test_cycle_raises_e_depend(self):
        with self.assertRaises(DependError) as ctx:
            load_rules(SAMPLES / "rules_cycle.json")
        self.assertEqual(ctx.exception.code, "E_DEPEND")
        self.assertEqual(set(ctx.exception.details["cycle_nodes"]),
                         {"R_CYC_A", "R_CYC_B", "R_CYC_C"})

    def test_ruleset_version_deterministic(self):
        r1 = load_rules(SAMPLES / "rules.json")
        r2 = load_rules(SAMPLES / "rules.json")
        self.assertEqual(ruleset_version(r1), ruleset_version(r2))


class InvalidationTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(SAMPLES / "rules.json")

    def test_modify_root_invalidates_all_downstream_only(self):
        cs = ChangeSet.from_dict(__import__("json").loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        result = apply_changeset(self.rules, cs)
        seq = result.invalidation_sequence
        # R1 及其全部下游：R3,R4,R6,R7,R9；不含 R2/R5/R8 支系
        expected = {"R1_BASE", "R3_CHILD", "R4_CHILD", "R6_GRAND",
                    "R7_GRAND", "R9_SIBLING"}
        self.assertEqual(set(seq), expected)
        self.assertNotIn("R2_BASE", seq)
        self.assertNotIn("R8_GRAND", seq)
        # 拓扑序：被失效者的上游必须排在前面
        position = {rid: i for i, rid in enumerate(seq)}
        self.assertLess(position["R1_BASE"], position["R3_CHILD"])
        self.assertLess(position["R3_CHILD"], position["R6_GRAND"])
        # 每步都有触发源
        self.assertTrue(all(t.triggered_by for t in result.trace))
        r6 = next(t for t in result.trace if t.rule_id == "R6_GRAND")
        self.assertEqual(r6.via_edge, "R3_CHILD->R6_GRAND")

    def test_delete_referenced_rule_is_refused(self):
        cs = ChangeSet(delete_rule_ids=["R1_BASE"])
        with self.assertRaises(RefError) as ctx:
            apply_changeset(self.rules, cs)
        self.assertEqual(ctx.exception.code, "E_REF")
        self.assertIn("R1_BASE", ctx.exception.details["blocked"])

    def test_delete_leaf_succeeds_and_propagates(self):
        cs = ChangeSet(delete_rule_ids=["R8_GRAND"])
        result = apply_changeset(self.rules, cs)
        self.assertNotIn("R8_GRAND", [r.rule_id for r in result.rules])
        self.assertEqual(result.change_summary["deleted"], ["R8_GRAND"])

    def test_remove_edge_revalidates(self):
        cs = ChangeSet(remove_edges=[{"from_rule": "R6_GRAND",
                                      "upstream": "R3_CHILD"}])
        result = apply_changeset(self.rules, cs)
        ids = {r.rule_id for r in result.rules}
        self.assertIn("R6_GRAND", ids)
        r6 = next(r for r in result.rules if r.rule_id == "R6_GRAND")
        self.assertEqual(r6.upstream, ())


if __name__ == "__main__":
    unittest.main()
