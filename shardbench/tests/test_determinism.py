import json
import unittest

from shardbench.config import load_config
from shardbench.engine import compile_all

from .test_validation import sample_paths


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_byte_identical(self):
        config = load_config(*sample_paths())
        first = compile_all(config)
        for _ in range(5):
            again = compile_all(config)
            self.assertEqual(
                json.dumps(first, ensure_ascii=False, sort_keys=True),
                json.dumps(again, ensure_ascii=False, sort_keys=True),
            )

    def test_unit_processing_order_independent(self):
        config = load_config(*sample_paths())
        canonical = compile_all(config)
        # 打乱拓扑顺序后手工按依赖解析，结果仍须一致。
        reversed_order = list(reversed(config.topo_order))
        from shardbench.router import Router
        from shardbench.engine import _merge_plan

        router = Router(config)
        compiled = {}
        remaining = list(reversed_order)
        while remaining:
            progressed = False
            for unit_id in list(remaining):
                deps = config.units[unit_id].depends_on
                if all(dep in compiled for dep in deps):
                    plan = router.compile(unit_id)
                    compiled[unit_id] = _merge_plan(config, unit_id, plan, compiled)
                    remaining.remove(unit_id)
                    progressed = True
            self.assertTrue(progressed, "依赖解析未能推进，测试自身有误")
        shuffled = {unit_id: plan.to_dict() for unit_id, plan in compiled.items()}
        for unit_id in canonical:
            self.assertEqual(
                canonical[unit_id]["targets"], shuffled[unit_id]["targets"]
            )
            self.assertEqual(
                canonical[unit_id]["summary"], shuffled[unit_id]["summary"]
            )

    def test_hash_independent_of_dict_traversal(self):
        from shardbench.determinism import stable_hash

        self.assertEqual(stable_hash("user_id=42"), stable_hash("user_id=42"))
        # 相同内容不同插入顺序的集合，稳定排序后摘要一致。
        from shardbench.determinism import stable_targets

        self.assertEqual(
            stable_targets(["db1.t0", "db0.t0"]),
            stable_targets(["db0.t0", "db1.t0"]),
        )


if __name__ == "__main__":
    unittest.main()
