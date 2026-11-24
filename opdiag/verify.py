"""增量运行 vs 从空白全量重建的逐字段一致性验证。

对同一份"最终配置+输入"，比较两条路径：
  A. 增量路径：从空白开始施加一串编辑，每步运行一次（含缓存/快照/预算）
  B. 全量路径：清空状态，按最终配置一次性运行（足够大预算，单批完成）
比较逐节点输出张量、量化整数/残差/步长、顺序标识、舍入次数、融合差值、
复用/重算清单（在最终一步）与失败清单顺序。
"""

from __future__ import annotations

import copy
import json
import os
import shutil

import numpy as np

from .engine import Engine
from .hashing import fingerprint_array


def _run(chain, inputs_desc, cfg, injections, workdir, budget=None):
    cfg = copy.deepcopy(cfg)
    if budget is not None:
        cfg["element_budget"] = budget
    eng = Engine(chain, inputs_desc, cfg, injections or [],
                 base_dir=workdir)
    art = eng.run()
    return eng, art


def _product_vector(products):
    """把每个节点产物转成可逐字段比较的稳定字典。"""
    out = {}
    for nid, p in sorted(products.items()):
        out[nid] = {
            "output_fp": fingerprint_array(np.asarray(p["output"])),
            "q_fp": fingerprint_array(np.asarray(p["quant"]["q"])),
            "residual_fp":
                fingerprint_array(np.asarray(p["quant"]["residual"])),
            "step_fp": fingerprint_array(
                np.atleast_1d(np.asarray(p["quant"]["step"]))),
            "dequant_fp":
                fingerprint_array(np.asarray(p["quant"]["dequant"])),
            "order_fp": fingerprint_array(np.asarray(p["order_ids"])),
            "round_fp":
                fingerprint_array(np.asarray(p["round_counts"])),
            "cache_key": p["cache_key"],
            "fusion": None if not p.get("fusion") else {
                k: p["fusion"][k] for k in
                ("max_abs_diff", "over_tolerance",
                 "over_tolerance_count", "attribution")},
        }
    return out


def compare_reports(inc_report: dict, full_report: dict) -> list[str]:
    """比较报告中与编辑历史无关、应逐字段一致的部分。"""
    diffs: list[str] = []

    def map_nodes(rep):
        return {n["node_id"]: n for n in rep["nodes"]
                if n["status"] in ("recomputed", "reused")}

    a, b = map_nodes(inc_report), map_nodes(full_report)
    if set(a) != set(b):
        diffs.append(f"completed node set differs: {set(a) ^ set(b)}")
    for nid in sorted(set(a) & set(b)):
        na, nb = a[nid], b[nid]
        for field in ("cache_key", "output_fingerprint"):
            if na.get(field) != nb.get(field):
                diffs.append(f"node {nid} field {field} differs")
        for field in ("step_fingerprint", "residual_fingerprint"):
            if na["quant"].get(field) != nb["quant"].get(field):
                diffs.append(f"node {nid} quant {field} differs")

    # 失败清单：代码 + 节点 + 注入 + 顺序
    fa = [(f["topo_index"], f["node_id"], f["code"], f["injection_id"])
          for f in inc_report["failures"]
          if f["code"] != "E_RECOMPUTE_BUDGET_PENDING"]
    fb = [(f["topo_index"], f["node_id"], f["code"], f["injection_id"])
          for f in full_report["failures"]
          if f["code"] != "E_RECOMPUTE_BUDGET_PENDING"]
    if fa != fb:
        diffs.append(f"failure list differs:\n  inc={fa}\n  full={fb}")

    # 融合诊断（按 transpose 节点对齐）
    fa_d = {d["transpose"]: d for d in inc_report["fusion_diagnostics"]}
    fb_d = {d["transpose"]: d for d in full_report["fusion_diagnostics"]}
    if set(fa_d) != set(fb_d):
        diffs.append("fusion diagnostic node set differs")
    for t in sorted(set(fa_d) & set(fb_d)):
        for k in ("max_abs_diff", "over_tolerance",
                  "over_tolerance_count", "attribution"):
            if fa_d[t].get(k) != fb_d[t].get(k):
                diffs.append(f"fusion {t} field {k} differs")
    return diffs


def run_consistency_probe(base_chain, inputs_desc, base_cfg, injections,
                          workroot, edits) -> dict:
    """edits: 作用于 chain/cfg 的可调用列表，逐步施加，最后全量重建比对。"""
    os.makedirs(workroot, exist_ok=True)
    inc_dir = os.path.join(workroot, "incremental")
    full_dir = os.path.join(workroot, "full_rebuild")
    shutil.rmtree(inc_dir, ignore_errors=True)
    shutil.rmtree(full_dir, ignore_errors=True)

    chain = copy.deepcopy(base_chain)
    cfg = copy.deepcopy(base_cfg)
    step_reports = []

    def run_to_drain(ch, cf):
        # 小预算制造多批/快照/复用；同目录反复续跑直到待办清空
        last = None
        for _ in range(50):
            eng = Engine(ch, inputs_desc, cf, injections or [],
                         base_dir=inc_dir)
            last = eng.run()
            if not last.report["pending_list"]:
                break
        return eng, last

    # 初始运行（多批）
    eng, art = run_to_drain(chain, cfg)
    step_reports.append(art.report)

    for i, edit in enumerate(edits):
        chain, cfg = edit(chain, cfg)
        # 同目录续跑，触发缓存失效传播与快照恢复
        eng, art = run_to_drain(chain, cfg)
        step_reports.append(art.report)

    inc_products = _product_vector(eng.products)
    inc_report = art.report

    # 全量重建：独立空白目录、大预算单批、同样的最终配置
    full_cfg = copy.deepcopy(cfg)
    full_eng, full_art = _run(chain, inputs_desc, full_cfg, injections,
                              full_dir, budget=10 ** 9)
    full_products = _product_vector(full_eng.products)

    diffs = []
    if set(inc_products) != set(full_products):
        diffs.append(
            f"product node set differs: {set(inc_products) ^ set(full_products)}")
    for nid in sorted(set(inc_products) & set(full_products)):
        if inc_products[nid] != full_products[nid]:
            for k in inc_products[nid]:
                if inc_products[nid][k] != full_products[nid][k]:
                    diffs.append(f"node {nid} product field {k} differs")
    diffs += compare_reports(inc_report, full_art.report)
    return {
        "consistent": not diffs,
        "diffs": diffs,
        "steps": len(step_reports),
        "final_reuse_list": inc_report["reuse_list"],
        "final_recompute_list": inc_report["recompute_list"],
        "full_recompute_list": full_art.report["recompute_list"],
    }
