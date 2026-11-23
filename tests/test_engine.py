import json
import os
import shutil
import unittest

import numpy as np

from hotblock_diag.config import Command
from hotblock_diag.engine import DiagnosticEngine
from hotblock_diag.errors import (E_QBLOCK_BUDGET_EXHAUSTED,
                                  E_SCORE_BUDGET_EXHAUSTED,
                                  E_SELECT_UNCONVERGED, FailureList)
from hotblock_diag.injection import InjectionPlan
from tests._harness import (blocks_for, layout_for, make_fixedpoint,
                            make_pack, make_prefilter, temp_workspace)


def build_engine(workspace: str, pack_path: str, pre=None, fp=None,
                 failures=None, fresh=True):
    pre = pre or make_prefilter()
    fp = fp or make_fixedpoint()
    failures = failures or FailureList()
    engine = DiagnosticEngine(workspace, pre, fp, InjectionPlan(),
                              failures, fresh=fresh)
    assert engine.load_pack(pack_path)
    assert engine.build_layout()
    return engine


class SelectionTests(unittest.TestCase):
    def test_feedback_converges_and_bitmaps_stable(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        engine = build_engine(os.path.join(ws, "run"), pack)
        report = engine.run_to_completion()
        self.assertEqual(report["status"], "converged")
        self.assertTrue(report["converged"])
        self.assertEqual(report["rounds_completed"], len(report["bitmap_trajectory"]))
        self.assertEqual(report["bitmap_trajectory"][-1],
                         report["bitmap_trajectory"][-2])
        for qb_summary in report["bitmap_summary"].values():
            self.assertLessEqual(qb_summary["count"], engine.pre.candidates)

    def test_unconverged_code_when_max_rounds_is_one(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        fp = make_fixedpoint(max_rounds=1)
        engine = build_engine(os.path.join(ws, "run"), pack, fp=fp)
        report = engine.run_to_completion()
        self.assertEqual(report["status"], "unconverged")
        self.assertIn(E_SELECT_UNCONVERGED, report["failures"][0]["code"])
        self.assertEqual(len(report["bitmap_trajectory"]), 1)

    def test_tie_break_is_kb_ascending(self):
        ws = temp_workspace()
        pack, tensors = make_pack(ws)
        # Force identical key blocks so every block-pair score ties.
        tensors["key"][:] = 0.25
        from hotblock_diag.tensorpack import write_pack
        tied_pack = os.path.join(ws, "tied.bin")
        write_pack(tied_pack, tensors, order=["query", "key", "value"])
        pre = make_prefilter(candidates=2, score_floor_ratio=0.0)
        engine = build_engine(os.path.join(ws, "run"), tied_pack, pre=pre)
        engine.run_to_completion()
        for qb, kb_list in engine.state.bitmaps[-1].items():
            self.assertEqual(kb_list, sorted(kb_list)[:pre.candidates])


class InvalidationTests(unittest.TestCase):
    def test_qblock_replacement_only_invalidates_dependent_artifacts(self):
        ws = temp_workspace()
        pack, tensors = make_pack(ws)
        run_dir = os.path.join(ws, "run")
        engine = build_engine(run_dir, pack)
        report = engine.run_to_completion()
        full_recomputed = set(report["cache_audit"]["recomputed"])
        self.assertEqual(len(full_recomputed), 16)

        replacement = np.zeros((4, 8), dtype=np.float32).reshape(-1).tolist()
        engine2 = DiagnosticEngine(run_dir, make_prefilter(), make_fixedpoint(),
                                   InjectionPlan(), FailureList(), fresh=False)
        assert engine2.load_pack(pack)
        assert engine2.build_layout()
        engine2.apply_edits([Command("replace_qblock", 1, {"data": replacement})])
        report2 = engine2.run_to_completion()
        reused = set(report2["cache_audit"]["reused"])
        recomputed = set(report2["cache_audit"]["recomputed"])
        self.assertTrue(reused)
        self.assertTrue(recomputed.isdisjoint(reused))
        self.assertEqual(len(recomputed), engine2.state.layout.k_blocks)
        self.assertEqual(len(reused) + len(recomputed),
                         engine2.state.layout.q_blocks * engine2.state.layout.k_blocks)
        self.assertLess(len(recomputed), len(full_recomputed))

    def test_prefilter_change_keeps_block_pair_operators(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        run_dir = os.path.join(ws, "run")
        engine = build_engine(run_dir, pack)
        engine.run_to_completion()
        pre = make_prefilter()
        engine2 = DiagnosticEngine(run_dir, pre, make_fixedpoint(),
                                   InjectionPlan(), FailureList(), fresh=False)
        assert engine2.load_pack(pack)
        assert engine2.build_layout()
        engine2.apply_edits([
            Command("update_prefilter", -1,
                    {"params": {"score_floor_ratio": 0.2, "temperature": 2.0}})
        ])
        report = engine2.run_to_completion()
        self.assertEqual(len(report["cache_audit"]["reused"]), 16)
        self.assertEqual(report["cache_audit"]["recomputed"], [])


class DeterminismTests(unittest.TestCase):
    def test_two_runs_byte_identical_reports(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        first = build_engine(os.path.join(ws, "a"), pack).run_to_completion()
        second = build_engine(os.path.join(ws, "b"), pack).run_to_completion()
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
