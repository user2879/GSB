"""异常注入隔离：按注入配置触发异常样例并映射业务码。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import EngineConfig
from .csr import CSRMatrix, fingerprint_inputs, replace_rows
from .engine import IncrementalEngine
from .errors import E_BUDGET_ZERO, SpmmError


def _fresh_dir(root, name):
    path = Path(root) / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _tampered_csr(a, **overrides):
    return CSRMatrix(
        indptr=overrides.get("indptr", a.indptr.copy()),
        indices=overrides.get("indices", a.indices.copy()),
        values=overrides.get("values", a.values.copy()),
        shape=a.shape,
    )


def _run_one(inj, a, b, c, base_config_dict, work_root):
    kind = inj["kind"]
    cfg = EngineConfig.from_dict(base_config_dict)
    state_dir = _fresh_dir(work_root, inj["id"])
    engine = IncrementalEngine(state_dir)
    if kind == "nonfinite_value":
        target = inj.get("target", "A")
        if target == "A":
            values = a.values.copy()
            values[0] = np.float32("nan")
            a2 = _tampered_csr(a, values=values)
            engine.full_build(a2, b, c, cfg)
        else:
            b2 = b.copy()
            b2[0, 0] = np.float32("inf")
            engine.full_build(a, b2, c, cfg)
        return None
    if kind == "shape_mismatch":
        engine.full_build(a, b[:-1], c, cfg)
        return None
    if kind == "block_param_out_of_range":
        bad = dict(base_config_dict)
        bad[inj["param"]] = inj["value"]
        engine.full_build(a, b, c, EngineConfig.from_dict(bad))
        return None
    if kind == "csr_structure":
        indptr = a.indptr.copy()
        if a.n_rows >= 2:
            indptr[1] = indptr[2] + 1
        engine.full_build(_tampered_csr(a, indptr=indptr), b, c, cfg)
        return None
    if kind == "budget_zero":
        engine.full_build(a, b, c, cfg)
        rng = np.random.default_rng(7)
        cols = np.sort(rng.choice(a.n_cols, size=3, replace=False))
        vals = rng.standard_normal(3).astype(np.float32)
        a2 = replace_rows(a, {0: (cols, vals)})
        zero_cfg = EngineConfig.from_dict({**base_config_dict, "recompute_budget": 0})
        res = engine.incremental_run(
            a2, b, c, zero_cfg,
            edits={"replaced_rows": [0]},
            base_input_fp=fingerprint_inputs(a, b, c),
            base_config_fp=cfg.fingerprint(),
        )
        if E_BUDGET_ZERO in res.report["codes"]:
            return E_BUDGET_ZERO
        return None
    raise ValueError(f"未知注入类型: {kind}")


def run_injections(injections, a, b, c, base_config_dict, work_root):
    """逐项执行注入样例，记录命中情况与业务码映射。"""
    results = []
    for inj in injections:
        actual = None
        detail = ""
        violations = []
        try:
            actual = _run_one(inj, a, b, c, base_config_dict, work_root)
        except SpmmError as exc:
            actual = exc.code
            detail = exc.message
            violations = exc.violations
        hit = actual == inj["expected_code"]
        results.append({
            "id": inj["id"],
            "kind": inj["kind"],
            "expected_code": inj["expected_code"],
            "actual_code": actual,
            "hit": hit,
            "detail": detail,
            "violations": violations,
        })
    return {
        "total": len(results),
        "hits": sum(1 for r in results if r["hit"]),
        "injections": results,
    }


def load_injection_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["injections"]
