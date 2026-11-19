import numpy as np
import pytest

from spmm import datasets
from spmm.checks import compare_runs
from spmm.config import EngineConfig
from spmm.csr import (DISTRIBUTIONS, append_rows, delete_rows,
                      fingerprint_inputs, replace_rows)
from spmm.engine import IncrementalEngine
from spmm.errors import E_BUDGET_ZERO, E_RECOMPUTE_PENDING


def _cfg(**kw):
    base = dict(row_block_size=8, nnz_chunk_size=8, alpha=1.25, beta=0.5,
                precision="fp32", accumulation_order="asc",
                recompute_budget=1000, max_workers=2,
                snapshot_commit_interval=2)
    base.update(kw)
    return EngineConfig.from_dict(base)


def _sample(tmp_path, dist):
    data_dir = tmp_path / "data"
    datasets.generate_all(data_dir)
    a, b, c, _ = datasets.load_sample(data_dir, dist)
    return a, b, c


def _full_rebuild(tmp_path, name, a, b, c, cfg):
    engine = IncrementalEngine(tmp_path / name)
    return engine.full_build(a, b, c, cfg)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
@pytest.mark.parametrize("block_size", [8, 6])
def test_edit_sequence_matches_full_rebuild(tmp_path, dist, block_size):
    a, b, c = _sample(tmp_path, dist)
    cfg = _cfg(row_block_size=block_size)
    engine = IncrementalEngine(tmp_path / f"state_{dist}_{block_size}")
    engine.full_build(a, b, c, cfg)
    rng = np.random.default_rng(42)
    k = a.n_cols
    steps = []
    repl = {r: datasets.random_row(rng, k) for r in [3, 10, 27]}
    steps.append((replace_rows(a, repl), c, cfg, {"replaced_rows": sorted(repl)}))
    a2 = delete_rows(steps[-1][0], [5])
    steps.append((a2, np.delete(c, 5, axis=0), cfg, {"deleted_rows": [5]}))
    new_rows = [datasets.random_row(rng, k) for _ in range(2)]
    a3 = append_rows(steps[-1][0], new_rows)
    c3 = np.concatenate([steps[-1][1],
                         rng.standard_normal((2, c.shape[1])).astype(np.float32)])
    steps.append((a3, c3, cfg, {"appended_rows": 2}))
    cfg_blk = _cfg(row_block_size=block_size, nnz_chunk_size=4)
    steps.append((steps[-1][0], steps[-1][1], cfg_blk, {}))
    cfg_ab = _cfg(row_block_size=block_size, nnz_chunk_size=4, alpha=0.75, beta=1.5)
    steps.append((steps[-1][0], steps[-1][1], cfg_ab, {}))
    cfg_fp = _cfg(row_block_size=block_size, nnz_chunk_size=4, alpha=0.75,
                  beta=1.5, precision="fp64")
    steps.append((steps[-1][0], steps[-1][1], cfg_fp, {}))
    cfg_ord = _cfg(row_block_size=block_size, nnz_chunk_size=4, alpha=0.75,
                   beta=1.5, precision="fp64", accumulation_order="abs_desc")
    steps.append((steps[-1][0], steps[-1][1], cfg_ord, {}))
    prev_a, prev_c, prev_cfg = a, c, cfg
    for idx, (a_new, c_new, cfg_new, edits) in enumerate(steps):
        res = engine.incremental_run(
            a_new, b, c_new, cfg_new, edits=edits,
            base_input_fp=fingerprint_inputs(prev_a, b, prev_c),
            base_config_fp=prev_cfg.fingerprint())
        while res.report["pending_blocks"]:
            res = engine.incremental_run(
                a_new, b, c_new, cfg_new, edits={},
                base_input_fp=fingerprint_inputs(a_new, b, c_new),
                base_config_fp=cfg_new.fingerprint())
        ref = _full_rebuild(tmp_path, f"rebuild_{dist}_{block_size}_{idx}",
                            a_new, b, c_new, cfg_new)
        assert compare_runs(res, ref) == {}
        prev_a, prev_c, prev_cfg = a_new, c_new, cfg_new


def test_precise_invalidation_on_row_replace(tmp_path):
    a, b, c = _sample(tmp_path, "uniform")
    cfg = _cfg(row_block_size=8)
    engine = IncrementalEngine(tmp_path / "state")
    engine.full_build(a, b, c, cfg)
    rng = np.random.default_rng(1)
    repl = {r: datasets.random_row(rng, a.n_cols) for r in [3, 10]}
    a2 = replace_rows(a, repl)
    res = engine.incremental_run(
        a2, b, c, cfg, edits={"replaced_rows": [3, 10]},
        base_input_fp=fingerprint_inputs(a, b, c),
        base_config_fp=cfg.fingerprint())
    assert res.report["invalidated_blocks"] == [0, 1]
    assert res.report["reused_blocks"] == [2, 3, 4, 5, 6, 7]
    assert res.report["recomputed_blocks"] == [0, 1]


def test_delete_row_invalidates_from_deleted_block_onward(tmp_path):
    a, b, c = _sample(tmp_path, "uniform")
    cfg = _cfg(row_block_size=8)
    engine = IncrementalEngine(tmp_path / "state")
    engine.full_build(a, b, c, cfg)
    a2 = delete_rows(a, [12])
    c2 = np.delete(c, 12, axis=0)
    res = engine.incremental_run(
        a2, b, c2, cfg, edits={"deleted_rows": [12]},
        base_input_fp=fingerprint_inputs(a, b, c),
        base_config_fp=cfg.fingerprint())
    assert res.report["invalidated_blocks"] == [1, 2, 3, 4, 5, 6, 7]
    assert res.report["reused_blocks"] == [0]


def test_budget_scheduling_and_pending_queue(tmp_path):
    a, b, c = _sample(tmp_path, "uniform")
    cfg = _cfg(row_block_size=8, recompute_budget=2)
    engine = IncrementalEngine(tmp_path / "state")
    engine.full_build(a, b, c, cfg)
    rng = np.random.default_rng(2)
    rows = [0, 9, 17, 25, 33]
    repl = {r: datasets.random_row(rng, a.n_cols) for r in rows}
    a2 = replace_rows(a, repl)
    res1 = engine.incremental_run(
        a2, b, c, cfg, edits={"replaced_rows": rows},
        base_input_fp=fingerprint_inputs(a, b, c),
        base_config_fp=cfg.fingerprint())
    assert res1.report["recomputed_blocks"] == [0, 1]
    assert res1.report["pending_blocks"] == [2, 3, 4]
    assert E_RECOMPUTE_PENDING in res1.report["codes"]
    deduction = res1.report["budget"]["deduction_sequence"]
    assert [d["block"] for d in deduction] == [0, 1]
    assert [d["remaining"] for d in deduction] == [1, 0]
    res2 = engine.incremental_run(
        a2, b, c, cfg, edits={},
        base_input_fp=fingerprint_inputs(a2, b, c),
        base_config_fp=cfg.fingerprint())
    assert res2.report["recomputed_blocks"] == [2, 3]
    assert res2.report["pending_blocks"] == [4]
    res3 = engine.incremental_run(
        a2, b, c, cfg, edits={},
        base_input_fp=fingerprint_inputs(a2, b, c),
        base_config_fp=cfg.fingerprint())
    assert res3.report["recomputed_blocks"] == [4]
    assert res3.report["pending_blocks"] == []
    ref = _full_rebuild(tmp_path, "rebuild", a2, b, c, cfg)
    assert compare_runs(res3, ref) == {}


def test_budget_zero_marks_all_pending(tmp_path):
    a, b, c = _sample(tmp_path, "uniform")
    cfg = _cfg(row_block_size=8, recompute_budget=0)
    engine = IncrementalEngine(tmp_path / "state")
    engine.full_build(a, b, c, _cfg(row_block_size=8))
    rng = np.random.default_rng(3)
    repl = {0: datasets.random_row(rng, a.n_cols)}
    a2 = replace_rows(a, repl)
    res = engine.incremental_run(
        a2, b, c, cfg, edits={"replaced_rows": [0]},
        base_input_fp=fingerprint_inputs(a, b, c),
        base_config_fp=_cfg(row_block_size=8).fingerprint())
    assert res.report["recomputed_blocks"] == []
    assert res.report["pending_blocks"] == [0]
    assert E_BUDGET_ZERO in res.report["codes"]
    assert E_RECOMPUTE_PENDING in res.report["codes"]
