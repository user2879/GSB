import os
import tempfile
import unittest

from snapshot_migration.chain import load_chain
from snapshot_migration.diagnostics import cursor_from_row, repack_range
from snapshot_migration.engine import MigrationEngine
from snapshot_migration.errors import (BoundError, CleanError, CursorError,
                                       InheritError)
from snapshot_migration.fragments import explicit_split_sizes
from snapshot_migration.recovery import (inject_bound_fault,
                                        inject_clean_fault,
                                        inject_inherit_fault, replay,
                                        verify_integrity)

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples",
                      "snapshot_chain.json")


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.chain = load_chain(SAMPLE)
        self.work = tempfile.mkdtemp()
        self.db = os.path.join(self.work, "store.sqlite3")
        self.engine = MigrationEngine(self.db, self.chain)
        for gen in self.chain.generations:
            self.engine.migrate_generation(gen.generation_id)

    def tearDown(self):
        self.engine.close()

    def all(self, generation_id):
        return self.engine.repo.active_fragments(generation_id)


class FragmentPackingTests(unittest.TestCase):
    def test_split_sizes_balanced_and_in_capacity(self):
        self.assertEqual(explicit_split_sizes(6, 3, 5), [3, 3])
        self.assertEqual(explicit_split_sizes(8, 3, 5), [4, 4])
        self.assertEqual(explicit_split_sizes(11, 3, 5), [4, 4, 3])

    def test_infeasible_range_raises_bound(self):
        with self.assertRaises(BoundError):
            explicit_split_sizes(11, 6, 7)


class MigrationTests(EngineTestCase):
    def test_fragments_cover_each_generation(self):
        for gen in self.chain.generations:
            fragments = self.all(gen.generation_id)
            active = gen.active_ranges
            self.assertEqual({f.range_id for f in fragments},
                             {r.range_id for r in active})
            for item in active:
                items = sorted(
                    (f for f in fragments if f.range_id == item.range_id),
                    key=lambda f: f.ordinal)
                self.assertEqual(items[0].lower, item.lower)
                self.assertEqual(items[-1].upper, item.upper)
                self.assertEqual(tuple(k for f in items for k in f.keys),
                                 item.keys)

    def test_inheritance_is_byte_identical(self):
        child = next(f for f in self.all("G1") if f.range_id == "R1"
                     and f.ordinal == 0)
        parent = next(f for f in self.all("G0") if f.range_id == "R1"
                      and f.ordinal == 0)
        self.assertEqual(child.digest, parent.digest)
        self.assertEqual(child.inherited_from, "G0")
        self.assertEqual(child.source_fragment_id, parent.fragment_id)

    def test_rewrite_dereferences_parent_version(self):
        ledger = self.engine.repo.migration_ledger("G3")
        rewrite = next(row for row in ledger if row["range_id"] == "R4")
        self.assertEqual(rewrite["disposition"], "rewrite")
        self.assertTrue(rewrite["evidence"]["regenerated"])

    def test_inherit_with_divergent_keys_raises_inherit(self):
        from snapshot_migration.model import Generation, KeyRange, SnapshotChain
        gens = []
        for gen in self.chain.generations:
            if gen.generation_id != "G2":
                gens.append(gen)
                continue
            ranges = tuple(
                KeyRange(r.range_id, r.lower, r.upper,
                         tuple(list(r.keys) + ["k0000099"]) if r.range_id == "R1"
                         else r.keys, r.action, r.tombstone)
                for r in gen.ranges)
            gens.append(Generation(gen.generation_id, gen.seq, gen.parent_id,
                                   ranges, gen.rules))
        broken = SnapshotChain(self.chain.title, self.chain.limits,
                               tuple(gens), self.chain.range_universe)
        db = os.path.join(self.work, "broken.sqlite3")
        engine = MigrationEngine(db, broken)
        engine.migrate_generation("G0")
        engine.migrate_generation("G1")
        with self.assertRaises(InheritError):
            engine.migrate_generation("G2")
        engine.close()


class CleanupTests(EngineTestCase):
    def test_view_blocks_then_allows_cleanup(self):
        view = self.engine.open_read_view("v", "G0")
        deferred = self.engine.run_cleanup()
        self.assertTrue(all(item["action"] == "deferred"
                            for item in deferred if item["range_id"] == "R2"))
        view.close()
        self.engine.run_cleanup()
        cleaned = [row for row in self.engine.repo.cleanup_ledger()
                   if row["range_id"] == "R2"]
        self.assertEqual(len(cleaned), 1)
        # idempotent
        self.engine.run_cleanup()
        self.assertEqual(
            len([row for row in self.engine.repo.cleanup_ledger()
                 if row["range_id"] == "R2"]), 1)


class CursorTests(EngineTestCase):
    def test_pagination_has_no_gap_or_duplicate(self):
        with self.engine.open_read_view("v", "G3") as view:
            paged = []
            cursor = None
            while True:
                page = view.scan(limit=4, cursor=cursor)
                paged.extend(page["rows"])
                if page["next_cursor"] is None:
                    break
                cursor = _cursor(page)
            self.assertEqual([r["key"] for r in paged],
                             [cell[1] for cell in view.plan])

    def test_benign_repack_keeps_cursor_valid(self):
        with self.engine.open_read_view("v", "G3") as view:
            page = view.scan(limit=4)
            cursor = _cursor(page)
            repack_range(self.engine, "G3", "R1", destructive=False)
            next_page = view.scan(limit=4, cursor=cursor)
            self.assertTrue(next_page["rows"])

    def test_destructive_repack_invalidates_cursor(self):
        with self.engine.open_read_view("v", "G3") as view:
            full = view.iterate_all(4)
            repack_range(self.engine, "G3", "R1", destructive=True)
            with self.assertRaises(CursorError) as ctx:
                view.scan(limit=4, cursor=cursor_from_row(full[0]))
            self.assertEqual(ctx.exception.code, "E_CURSOR")
            self.assertEqual(ctx.exception.fragment_id,
                             "frag:G3:R1:000")

    def test_same_generation_byte_identical_with_and_without_activity(self):
        with self.engine.open_read_view("a", "G3") as view:
            before = view.iterate_all(4)
        repack_range(self.engine, "G3", "R4", destructive=False)
        with self.engine.open_read_view("b", "G3") as view:
            after = view.iterate_all(4)
        self.assertEqual(
            [(r["range_id"], r["key"]) for r in before],
            [(r["range_id"], r["key"]) for r in after])


def _cursor(page):
    last = page["rows"][-1]
    return cursor_from_row(last)


class SnapshotAndRecoveryTests(EngineTestCase):
    def test_snapshot_replace_and_replay_match(self):
        snapshot = os.path.join(self.work, "snapshot.json")
        self.engine.run_cleanup()
        previous = self.engine.save_snapshot(snapshot)
        reopened_path = os.path.join(self.work, "reopened.sqlite3")
        reopened = MigrationEngine(reopened_path, self.chain)
        diff = reopened.replace_from_chain(previous)
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        result = replay(reopened)
        self.assertTrue(result.ok)
        self.assertIsNone(result.first_mismatch)
        reopened.close()

    def test_three_faults_are_detected(self):
        inject_bound_fault(self.engine, "G0", "frag:G0:R2:000")
        with self.assertRaises(BoundError):
            verify_integrity(self.engine)

    def test_inherit_fault_detected(self):
        inject_inherit_fault(self.engine, "G1", "frag:G1:R1:000")
        with self.assertRaises(InheritError):
            verify_integrity(self.engine)

    def test_clean_fault_detected(self):
        self.engine.repo.upsert_generation(
            "G-X", 99, "G0", "digest", committed=False)
        inject_clean_fault(self.engine, "G-X")
        with self.assertRaises(CleanError):
            verify_integrity(self.engine)


if __name__ == "__main__":
    unittest.main()
