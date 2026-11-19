"""子进程辅助：带人工延迟的全量/增量运行，用于信号退出与强杀恢复测试。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from spmm import datasets
from spmm.config import EngineConfig
from spmm.csr import fingerprint_inputs, replace_rows
from spmm.engine import IncrementalEngine


def main():
    state_dir = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    sleep_ms = int(os.environ.get("SPMM_BLOCK_SLEEP_MS", "50"))
    a, b, c, _ = datasets.load_sample(Path(__file__).resolve().parents[1] / "data",
                                      "uniform")
    cfg = EngineConfig(row_block_size=4, nnz_chunk_size=8, alpha=1.25, beta=0.5,
                       precision="fp32", accumulation_order="asc",
                       recompute_budget=10 ** 6, max_workers=1,
                       snapshot_commit_interval=1)
    engine = IncrementalEngine(state_dir, block_sleep_ms=sleep_ms,
                               install_signals=True)
    if mode == "full":
        engine.full_build(a, b, c, cfg)
    else:
        rng = np.random.default_rng(99)
        repl = {r: datasets.random_row(rng, a.n_cols)
                for r in range(0, a.n_rows, 3)}
        a_new = replace_rows(a, repl)
        engine.incremental_run(
            a_new, b, c, cfg,
            edits={"replaced_rows": sorted(repl)},
            base_input_fp=fingerprint_inputs(a, b, c),
            base_config_fp=cfg.fingerprint(),
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
