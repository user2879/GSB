"""完整示例：全量构建 -> 行级编辑序列 -> 增量重算 -> 与全量重建逐元素核对。"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spmm import datasets
from spmm.checks import compare_runs
from spmm.config import EngineConfig
from spmm.csr import (DISTRIBUTIONS, append_rows, delete_rows,
                      fingerprint_inputs, replace_rows)
from spmm.engine import IncrementalEngine
from spmm.errors import E_SNAPSHOT_STALE, SpmmError

ROOT = Path(__file__).resolve().parents[1]


def _fresh_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def build_edit_sequence(a, c, cfg, seed):
    """构造覆盖全部编辑类型的确定性编辑序列。"""
    rng = np.random.default_rng(seed + 100)
    k = a.n_cols
    steps = []

    repl_rows = [3, 10, 27]
    repl = {r: datasets.random_row(rng, k) for r in repl_rows}
    steps.append(("replace_rows", replace_rows(a, repl), c, cfg,
                  {"replaced_rows": repl_rows}))

    a2 = steps[-1][1]
    steps.append(("delete_row", delete_rows(a2, [5]), np.delete(c, 5, axis=0), cfg,
                  {"deleted_rows": [5]}))

    a3 = steps[-1][1]
    new_rows = [datasets.random_row(rng, k) for _ in range(2)]
    c_app = np.concatenate([steps[-1][2],
                            rng.standard_normal((2, c.shape[1])).astype(np.float32)])
    steps.append(("append_rows", append_rows(a3, new_rows), c_app, cfg,
                  {"appended_rows": 2}))

    cfg_blk = EngineConfig.from_dict({**cfg.to_dict(), "row_block_size": 6})
    steps.append(("row_block_size", steps[-1][1], steps[-1][2], cfg_blk, {}))

    cfg_ab = EngineConfig.from_dict({**cfg_blk.to_dict(), "alpha": 0.75, "beta": 1.5})
    steps.append(("alpha_beta", steps[-1][1], steps[-1][2], cfg_ab, {}))

    cfg_fp64 = EngineConfig.from_dict({**cfg_ab.to_dict(), "precision": "fp64"})
    steps.append(("precision", steps[-1][1], steps[-1][2], cfg_fp64, {}))

    cfg_ord = EngineConfig.from_dict({**cfg_fp64.to_dict(),
                                      "accumulation_order": "abs_desc"})
    steps.append(("accumulation_order", steps[-1][1], steps[-1][2], cfg_ord, {}))
    return steps


def run_distribution(dist, a, b, c, cfg, report_dir, state_root):
    report_dir = _fresh_dir(report_dir)
    state_dir = _fresh_dir(state_root / dist)
    engine = IncrementalEngine(state_dir)
    res = engine.full_build(a, b, c, cfg,
                            report_path=report_dir / "run_00_full.json",
                            extra={"distribution": dist})
    record = {
        "distribution": dist,
        "full_build": {"blocks": res.report["recomputed_count"],
                       "snapshot_version": res.report["snapshot_version"]},
        "steps": [],
        "all_equivalent": True,
    }
    prev_a, prev_c, prev_cfg = a, c, cfg
    steps = build_edit_sequence(a, c, cfg, datasets.SEEDS[dist])
    for idx, (name, a_new, c_new, cfg_new, edits) in enumerate(steps, start=1):
        base_ifp = fingerprint_inputs(prev_a, b, prev_c)
        base_cfp = prev_cfg.fingerprint()
        res_inc = engine.incremental_run(
            a_new, b, c_new, cfg_new, edits=edits,
            base_input_fp=base_ifp, base_config_fp=base_cfp,
            report_path=report_dir / f"run_{idx:02d}_{name}.json",
            extra={"distribution": dist, "step": name},
        )
        drained = 0
        while res_inc.report["pending_blocks"]:
            drained += 1
            res_inc = engine.incremental_run(
                a_new, b, c_new, cfg_new, edits={},
                base_input_fp=fingerprint_inputs(a_new, b, c_new),
                base_config_fp=cfg_new.fingerprint(),
                report_path=report_dir / f"run_{idx:02d}_{name}_drain{drained}.json",
                extra={"distribution": dist, "step": f"{name}_drain{drained}"},
            )
        rebuild_dir = _fresh_dir(state_root / f"{dist}_rebuild_{idx:02d}")
        res_full = IncrementalEngine(rebuild_dir).full_build(a_new, b, c_new, cfg_new)
        diffs = compare_runs(res_inc, res_full)
        ok = not diffs
        record["all_equivalent"] = record["all_equivalent"] and ok
        record["steps"].append({
            "step": name,
            "invalidated_blocks": res_inc.report["invalidated_blocks"],
            "reused_blocks": res_inc.report["reused_blocks"],
            "recomputed_blocks": res_inc.report["recomputed_blocks"],
            "drain_runs": drained,
            "codes": res_inc.report["codes"],
            "equivalent_to_full_rebuild": ok,
            "diffs": diffs,
        })
        prev_a, prev_c, prev_cfg = a_new, c_new, cfg_new
    return record


def snapshot_self_check(dist, a, b, c, cfg, state_root):
    """快照失效拒绝与残留临时文件忽略的自检。"""
    state_dir = _fresh_dir(state_root / f"{dist}_snapcheck")
    engine = IncrementalEngine(state_dir)
    engine.full_build(a, b, c, cfg)
    outcome = {}
    try:
        engine.incremental_run(a, b, c, cfg, edits={},
                               base_input_fp="0" * 32,
                               base_config_fp=cfg.fingerprint())
        outcome["stale_rejected"] = False
    except SpmmError as exc:
        outcome["stale_rejected"] = exc.code == E_SNAPSHOT_STALE
    for tmp in state_dir.glob("*.tmp"):
        tmp.unlink()
    (state_dir / "snapshot.json.tmp").write_text("garbage", encoding="utf-8")
    (state_dir / "blocks.npz.tmp").write_bytes(b"garbage")
    res = engine.incremental_run(a, b, c, cfg, edits={},
                                 base_input_fp=fingerprint_inputs(a, b, c),
                                 base_config_fp=cfg.fingerprint())
    outcome["tmp_residue_ignored"] = res.report["recomputed_count"] == 0
    return outcome


def main():
    cfg = EngineConfig.from_json(ROOT / "configs" / "default.json")
    data_dir = ROOT / cfg.data_dir
    report_root = _fresh_dir(ROOT / cfg.report_dir)
    state_root = _fresh_dir(ROOT / "state")
    datasets.generate_all(data_dir)
    summary = {"distributions": {}, "snapshot_checks": {}, "equivalence_passed": True}
    for dist in DISTRIBUTIONS:
        a, b, c, meta = datasets.load_sample(data_dir, dist)
        record = run_distribution(dist, a, b, c, cfg, report_root / dist, state_root)
        summary["distributions"][dist] = record
        summary["equivalence_passed"] = (
            summary["equivalence_passed"] and record["all_equivalent"])
        summary["snapshot_checks"][dist] = snapshot_self_check(
            dist, a, b, c, cfg, state_root)
    with open(report_root / "demo_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "equivalence_passed": summary["equivalence_passed"],
        "snapshot_checks": summary["snapshot_checks"],
        "report": str(report_root / "demo_report.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["equivalence_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
