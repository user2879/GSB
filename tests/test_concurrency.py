import numpy as np

from spmm import datasets
from spmm.checks import compare_runs
from spmm.config import EngineConfig
from spmm.csr import fingerprint_inputs, replace_rows
from spmm.engine import IncrementalEngine


def _cfg(workers):
    return EngineConfig(row_block_size=8, nnz_chunk_size=8, alpha=1.25,
                        beta=0.5, precision="fp32", accumulation_order="asc",
                        recompute_budget=1000, max_workers=workers,
                        snapshot_commit_interval=3)


def test_parallel_matches_serial_bitwise(tmp_path):
    datasets.generate_all(tmp_path / "data")
    a, b, c, _ = datasets.load_sample(tmp_path / "data", "powerlaw")
    rng = np.random.default_rng(5)
    repl = {r: datasets.random_row(rng, a.n_cols) for r in [1, 9, 20, 33, 47]}
    a2 = replace_rows(a, repl)
    results = {}
    for workers in (1, 4):
        cfg = _cfg(workers)
        engine = IncrementalEngine(tmp_path / f"state_{workers}")
        engine.full_build(a, b, c, cfg)
        res = engine.incremental_run(
            a2, b, c, cfg, edits={"replaced_rows": sorted(repl)},
            base_input_fp=fingerprint_inputs(a, b, c),
            base_config_fp=cfg.fingerprint())
        results[workers] = res
    serial, parallel = results[1], results[4]
    assert compare_runs(serial, parallel) == {}
    for key in ("invalidated_blocks", "reused_blocks", "recomputed_blocks",
                "pending_blocks"):
        assert serial.report[key] == parallel.report[key]
    seq_s = [d["block"] for d in serial.report["budget"]["deduction_sequence"]]
    seq_p = [d["block"] for d in parallel.report["budget"]["deduction_sequence"]]
    assert seq_s == seq_p
