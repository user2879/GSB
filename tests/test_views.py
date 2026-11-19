import tempfile
import unittest
from pathlib import Path

from dwb_diag.engine import DiagnosticEngine, build_demo_views
from dwb_diag.errors import DependCycleError, ViewError
from dwb_diag.views import ViewCatalog, ViewDef, ViewManager
from dwb_diag.fault_injection import inject_dependency_cycle


def fresh_engine(root: Path) -> DiagnosticEngine:
    eng = DiagnosticEngine(root)
    build_demo_views(eng.catalog)
    eng.catalog.save()
    eng.views = ViewManager(eng.catalog, eng.storage.keyspace)
    return eng


class TestViewPropagation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = fresh_engine(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalidation_topological_and_order_independent(self):
        self.eng.submit({"user:1": 5, "order:1": 50})
        seq_a = self.eng.views.propagate_invalidation({0, 1})
        seq_b = self.eng.views.propagate_invalidation({1, 0})
        self.assertEqual(seq_a, seq_b)
        # rollup_all 必须在其依赖之后出现
        if "rollup_all" in seq_a:
            deps = {"cnt_user", "cnt_order"}
            if deps & set(seq_a):
                self.assertGreater(
                    seq_a.index("rollup_all"),
                    max(seq_a.index(d) for d in deps & set(seq_a)),
                )

    def test_versions_increment_and_rebuild(self):
        self.eng.submit({"user:1": 5})
        # 再提交一次以获得新的失效周期, 记录真实受影响页
        changed_pages = {
            self.eng.storage.keyspace.page_for(key)
            for key in ["user:1", "user:9"]
        }
        before = {n: s.version for n, s in self.eng.views.states.items()}
        sequence = self.eng.views.propagate_invalidation(changed_pages)
        after = self.eng.views.states
        changed = [n for n in sequence if after[n].version == before[n] + 1]
        self.assertEqual(changed, sequence)
        self.assertTrue(changed)
        rebuilt = self.eng.views.rebuild_all()
        self.assertEqual(set(rebuilt), set(changed))
        manifest = self.eng.verify_views()
        names = [row["name"] for row in manifest]
        self.assertEqual(names, sorted(names))
        self.assertTrue(set(changed).issubset(set(names)))

    def test_view_content_based_on_main_pages(self):
        self.eng.submit({"user:1": 3, "user:2": 8, "order:1": 500})
        self.eng.views.rebuild_graph()
        self.eng.views.rebuild_all()
        cnt = self.eng.views.states["cnt_user"].content
        self.assertEqual(cnt["count"], 2)
        rng = self.eng.views.states["range_low"].content
        self.assertTrue(all(v <= 9 for _, v in rng["items"]))

    def test_dependency_cycle_raises_e_depend(self):
        self.eng.submit({"user:1": 1})
        inject_dependency_cycle(self.eng.layout, ("rollup_all", "cnt_user"))
        catalog = ViewCatalog.load(self.eng.layout.views_file)
        from dwb_diag.keyspace import Keyspace

        vm = ViewManager(catalog, Keyspace(self.eng.storage.main))
        with self.assertRaises(DependCycleError) as ctx:
            vm.propagate_invalidation({0})
        self.assertEqual(ctx.exception.code, "E_DEPEND")
        self.assertTrue(ctx.exception.detail["cycle"])

    def test_view_version_not_incremented_is_e_view(self):
        self.eng.submit({"user:1": 1})
        page = self.eng.storage.keyspace.page_for("user:1")
        sequence = self.eng.views.propagate_invalidation({page})
        self.assertTrue(sequence)
        # 构造一个已参与失效却版本未递增的视图: 回退版本号并保留 stale
        first_view = next(
            name
            for name in self.eng.views.topological_views()
            if self.eng.views.states[name].version > 0
        )
        target = self.eng.views.states[first_view]
        target.version = 0
        target.stale = True
        target.digest = "x"
        with self.assertRaises(ViewError) as ctx:
            self.eng.verify_views()
        self.assertEqual(ctx.exception.code, "E_VIEW")
        self.assertIn(ctx.exception.detail["view"], sequence)
        self.assertEqual(ctx.exception.detail["field"], "version")


if __name__ == "__main__":
    unittest.main()
