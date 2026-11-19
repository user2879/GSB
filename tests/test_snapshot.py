import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from spmm import datasets
from spmm.checks import compare_runs
from spmm.config import EngineConfig
from spmm.csr import fingerprint_inputs, replace_rows
from spmm.engine import IncrementalEngine
from spmm.errors import E_SNAPSHOT_STALE, SpmmError
from spmm.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**kw):
    base = dict(row_block_size=8, nnz_chunk_size=8, alpha=1.25, beta=0.5,
                precision="fp32", accumulation_order="asc",
                recompute_budget=1000, max_workers=1,
                snapshot_commit_interval=2)
    base.update(kw)
    return EngineConfig.from_dict(base)


def _sample(tmp_path):
    datasets.generate_all(tmp_path / "data")
    a, b, c, _ = datasets.load_sample(tmp_path / "data", "uniform")
    return a, b, c


def test_stale_snapshot_rejected(tmp_path):
    a, b, c = _sample(tmp_path)
    cfg = _cfg()
    engine = IncrementalEngine(tmp_path / "state")
    engine.full_build(a, b, c, cfg)
    with pytest.raises(SpmmError) as exc:
        engine.incremental_run(a, b, c, cfg, edits={},
                               base_input_fp="0" * 32,
                               base_config_fp=cfg.fingerprint())
    assert exc.value.code == E_SNAPSHOT_STALE
    cfg2 = _cfg(alpha=9.0)
    with pytest.raises(SpmmError) as exc:
        engine.incremental_run(a, b, c, cfg2, edits={},
                               base_input_fp=fingerprint_inputs(a, b, c),
                               base_config_fp=cfg2.fingerprint())
    assert exc.value.code == E_SNAPSHOT_STALE


def test_tmp_residue_ignored_after_forced_kill(tmp_path):
    a, b, c = _sample(tmp_path)
    cfg = _cfg()
    engine = IncrementalEngine(tmp_path / "state")
    engine.full_build(a, b, c, cfg)
    snap = load_snapshot(tmp_path / "state")
    assert snap is not None
    (tmp_path / "state" / "snapshot.json.tmp").write_text("garbage")
    (tmp_path / "state" / "blocks.npz.tmp").write_bytes(b"garbage")
    snap2 = load_snapshot(tmp_path / "state")
    assert snap2[0]["version"] == snap[0]["version"]


def test_sigterm_exits_zero_after_commit(tmp_path):
    _sample(tmp_path)
    state_dir = tmp_path / "state_sig"
    env = dict(os.environ, SPMM_BLOCK_SLEEP_MS="40")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "signal_child.py"),
         str(state_dir), "full"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.4)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, f"stdout={out} stderr={err}"
    snap = load_snapshot(state_dir)
    assert snap is not None
    assert snap[0]["run_in_progress"] is True


def test_sigkill_resume_matches_full_rebuild(tmp_path):
    a, b, c = _sample(tmp_path)
    cfg = EngineConfig(row_block_size=4, nnz_chunk_size=8, alpha=1.25, beta=0.5,
                       precision="fp32", accumulation_order="asc",
                       recompute_budget=10 ** 6, max_workers=1,
                       snapshot_commit_interval=1)
    state_dir = tmp_path / "state_kill"
    IncrementalEngine(state_dir).full_build(a, b, c, cfg)
    rng = np.random.default_rng(99)
    repl = {r: datasets.random_row(rng, a.n_cols) for r in range(0, a.n_rows, 3)}
    a_new = replace_rows(a, repl)
    env = dict(os.environ, SPMM_BLOCK_SLEEP_MS="50")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "signal_child.py"),
         str(state_dir), "incr"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    proc.kill()
    proc.communicate(timeout=30)
    snap = load_snapshot(state_dir)
    assert snap is not None and snap[0]["run_in_progress"] is True
    engine = IncrementalEngine(state_dir)
    res = engine.incremental_run(
        a_new, b, c, cfg,
        edits={"replaced_rows": sorted(repl)},
        base_input_fp=fingerprint_inputs(a, b, c),
        base_config_fp=cfg.fingerprint())
    ref = IncrementalEngine(tmp_path / "rebuild").full_build(a_new, b, c, cfg)
    assert compare_runs(res, ref) == {}
