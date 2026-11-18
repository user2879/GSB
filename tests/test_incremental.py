"""要求4：增量更新、逐字节产物不变、依赖失效传播。"""
import unittest

from ccdplatform.compiler import Compiler, item_fingerprint
from ccdplatform.fixedpoint import GridSpec
from ccdplatform.model import (
    ConfigItem,
    Derivation,
    DispatchStrategy,
    ManifestVersion,
)

STRATEGY = DispatchStrategy(batch_size=2, timeout_ms=1000, max_retries=1)


def mv(version, raw_items):
    items = []
    for key, value, refs in raw_items:
        derivation = Derivation(tuple(refs)) if refs else None
        items.append(
            ConfigItem(key=key, value=value, targets=("n1",), derivation=derivation)
        )
    return ManifestVersion(version=version, items=tuple(items), strategy=STRATEGY)


class IncrementalTest(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler(GridSpec(exponent=3))

    def test_unchanged_item_byte_identical(self):
        v1 = mv(1, [
            ("a", 1.0, None),
            ("b", 2.0, [("a", 1.0)]),
            ("c", 3.0, None),
        ])
        r1 = self.compiler.compile_version(v1)
        bytes_a_v1 = r1.canonical["a"]
        bytes_b_v1 = r1.canonical["b"]

        # 只改 c：a、b 必须逐字节不变且不重新编译
        v2 = mv(2, [
            ("a", 1.0, None),
            ("b", 2.0, [("a", 1.0)]),
            ("c", 9.0, None),
        ])
        r2 = self.compiler.compile_version(v2)
        self.assertEqual(r2.unchanged, ["a", "b"])
        self.assertEqual(r2.recompiled, ["c"])
        self.assertEqual(r2.canonical["a"], bytes_a_v1)
        self.assertEqual(r2.canonical["b"], bytes_b_v1)

    def test_change_propagates_invalidation(self):
        v1 = mv(1, [
            ("a", 1.0, None),
            ("b", 2.0, [("a", 1.0)]),
            ("c", 3.0, [("b", 1.0)]),
        ])
        self.compiler.compile_version(v1)
        v2 = mv(2, [
            ("a", 1.5, None),  # 改基础项
            ("b", 2.0, [("a", 1.0)]),
            ("c", 3.0, [("b", 1.0)]),
        ])
        r2 = self.compiler.compile_version(v2)
        self.assertEqual(r2.recompiled, ["a", "b", "c"])
        self.assertIn("b", r2.invalidated)
        self.assertIn("c", r2.invalidated)  # 间接下游失效

    def test_fingerprint_detects_coef_change(self):
        i1 = ConfigItem("b", 2.0, ("n1",), Derivation((("a", 1.0),)))
        i2 = ConfigItem("b", 2.0, ("n1",), Derivation((("a", 2.0),)))
        self.assertNotEqual(item_fingerprint(i1), item_fingerprint(i2))

    def test_deleted_referenced_item_refused(self):
        from ccdplatform.errors import ReferenceError

        v1 = mv(1, [
            ("a", 1.0, None),
            ("b", 2.0, [("a", 1.0)]),
        ])
        self.compiler.compile_version(v1)
        v2 = mv(2, [("b", 2.0, [("a", 1.0)])])  # 删除仍被引用的 a
        with self.assertRaises(ReferenceError) as ctx:
            self.compiler.compile_version(v2)
        self.assertEqual(ctx.exception.code, "E_REF")


if __name__ == "__main__":
    unittest.main()
