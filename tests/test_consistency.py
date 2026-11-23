import copy
import json
import os
import unittest

import numpy as np

from hotblock_diag.config import GEAR_FRAC, Command
from hotblock_diag.engine import DiagnosticEngine
from hotblock_diag.errors import FailureList
from hotblock_diag.injection import InjectionPlan
from hotblock_diag.tensorpack import write_pack
from tests._harness import (make_fixedpoint, make_pack, make_prefilter,
                            temp_workspace)


COMPARABLE = (
    "block_pairs", "bitmap_trajectory", "bitmap_summary", "hot_table",
    "failures", "status", "converged", "rounds_completed",
)


def _edits(tensors, dim, rng):
    q_data = rng.normal(size=(4, dim)).astype(np.float32).reshape(-1).tolist()
    k_data = rng.normal(size=(4, dim)).astype(np.float32).reshape(-1).tolist()
    return [
        Command("replace_qblock", 1, {"data": q_data}),
        Command("replace_kblock", 2, {"data": k_data}),
        Command("update_prefilter", -1,
                {"params": {"score_floor_ratio": 0.35, "temperature": 1.4}}),
        Command("set_quant", -1, {"quant_gear": "q12"}),
        Command("set_hot_capacity", -1, {"capacity": 5}),
    ], q_data, k_data


def _apply_final_state(tensors, pre, fp, q_data, k_data, dim):
    tensors["query"][4:8] = np.asarray(q_data, dtype=np.float32).reshape(4, dim)
    tensors["key"][8:12] = np.asarray(k_data, dtype=np.float32).reshape(4, dim)
    pre.score_floor_ratio = 0.35
    pre.temperature = 1.4
    fp.quant_gear = "q12"
    fp.frac_bits = GEAR_FRAC["q12"]
    fp.hot_capacity = 5


class IncrementalConsistencyTests(unittest.TestCase):
    def test_incremental_chain_matches_blank_rebuild(self):
        ws = temp_workspace()
        initial_pack, tensors = make_pack(ws, seq_q=16, seq_k=16, dim=8)
        rng = np.random.default_rng(909)
        dim = 8
        pre = make_prefilter()
        fp = make_fixedpoint()
        run_dir = os.path.join(ws, "incr")

        engine = DiagnosticEngine(run_dir, pre, fp, InjectionPlan(),
                                  FailureList(), fresh=True)
        assert engine.load_pack(initial_pack)
        assert engine.build_layout()
        engine.run_to_completion()

        commands, q_data, k_data = _edits(tensors, dim, rng)
        engine.apply_edits(commands)
        incremental = engine.run_to_completion()

        final_tensors = copy.deepcopy(tensors)
        final_pre = make_prefilter()
        final_fp = make_fixedpoint()
        _apply_final_state(final_tensors, final_pre, final_fp,
                           q_data, k_data, dim)
        final_pack = os.path.join(ws, "final.bin")
        write_pack(final_pack, final_tensors,
                   order=["query", "key", "value"])
        full = DiagnosticEngine(os.path.join(ws, "full"), final_pre, final_fp,
                                InjectionPlan(), FailureList(), fresh=True)
        assert full.load_pack(final_pack)
        assert full.build_layout()
        full_report = full.run_to_completion()

        for field in COMPARABLE:
            self.assertEqual(
                incremental[field], full_report[field],
                msg=f"field {field} differs between incremental and full run",
            )

    def test_budget_paused_then_resume_matches_unconstrained(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        constrained_fp = make_fixedpoint(batch_budget=6, qblock_quota=3)
        engine = DiagnosticEngine(os.path.join(ws, "paused"), make_prefilter(),
                                  constrained_fp, InjectionPlan(),
                                  FailureList(), fresh=True)
        assert engine.load_pack(pack)
        assert engine.build_layout()
        paused = engine.run_to_completion(max_batches=1)
        self.assertEqual(paused["status"], "budget_paused")
        resumed = engine.run_to_completion()
        self.assertEqual(resumed["status"], "converged")

        full_fp = make_fixedpoint(batch_budget=100, qblock_quota=100)
        full = DiagnosticEngine(os.path.join(ws, "full"), make_prefilter(),
                                full_fp, InjectionPlan(), FailureList(),
                                fresh=True)
        assert full.load_pack(pack)
        assert full.build_layout()
        full_report = full.run_to_completion()
        for field in ("block_pairs", "bitmap_trajectory", "bitmap_summary",
                      "hot_table", "converged", "rounds_completed"):
            self.assertEqual(resumed[field], full_report[field])
        resumed_keys = [(d["qb"], d["kb"])
                        for d in resumed["budget"]["deduction_sequence"]]
        full_keys = [(d["qb"], d["kb"])
                     for d in full_report["budget"]["deduction_sequence"]]
        self.assertEqual(set(resumed_keys), set(full_keys))
        # Every batch honors the per-qblock quota; the concatenated deduction
        # sequence is stable within each batch and covers the full plan.
        batches = resumed["budget"]["deduction_sequence"]
        per_q: dict[int, int] = {}
        for item in batches:
            per_q[item["qb"]] = per_q.get(item["qb"], 0) + 1
        self.assertTrue(all(count <= engine.fp.qblock_quota * 4
                            for count in per_q.values()))
        self.assertEqual(full_keys, sorted(full_keys))


if __name__ == "__main__":
    unittest.main()
