"""核心执行引擎。

职责：确定性样本绑定、拓扑执行、节点级缓存失效传播、重算预算调度、
线程池并发（不超 max_workers）、阶段化快照提交、异常注入隔离与融合诊断。
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import IMPLEMENTATION_VERSION, dmath
from .cache import node_cache_key
from .config import normalize_config
from .errors import (
    E_FUSION_TOLERANCE, E_INJ_ACTIVATION_PARAM, E_INJ_DUPLICATE_PERM,
    E_INJ_IMPL_MISSING, E_INJ_NONFINITE, E_INPLACE_CONFLICT,
    E_RECOMPUTE_BUDGET_PENDING, E_UPSTREAM_FAILED, Failure,
)
from .fixedpoint import quantize
from .graph import build_graph
from .hashing import canonical_hash, fingerprint_array
from . import ops
from .storage import SnapshotStore, build_config_hash

STAGES = ("validate", "compute", "quantize")


@dataclass
class RunArtifacts:
    report: dict
    graph: object
    products: dict = field(default_factory=dict)


class Engine:
    def __init__(self, chain, inputs_desc, config_raw, injections=None,
                 base_dir="."):
        self.chain = chain
        self.inputs_desc = inputs_desc or {}
        self.cfg, enum_failures = normalize_config(config_raw)
        self.injections = injections or []
        self.base_dir = os.path.abspath(base_dir)
        out = os.path.join(self.base_dir, self.cfg["output_dir"])
        self.store = SnapshotStore(out)
        self.failures: list[Failure] = []
        for ef in enum_failures:
            self.failures.append(Failure(-1, "<config>", "validate",
                                         ef["code"], ef["message"],
                                         detail={"field": ef["field"]}))
        self.config_hash = build_config_hash(chain, self.inputs_desc, self.cfg)
        self.inputs: dict[str, np.ndarray] = {}
        self.products: dict[str, dict] = {}
        self.reused: list[str] = []
        self.recomputed: list[str] = []
        self.pending: list[str] = []
        self.injection_hits: list[dict] = []
        self.fusion_results: list[dict] = []
        self.events: list[dict] = []
        self.manifest = None
        self.version = 0

    # ---------- 注入 ----------
    def _injection(self, node_id, stage, fault):
        for inj in self.injections:
            if inj.get("node_id") == node_id and inj.get("stage") == stage \
                    and inj.get("fault") == fault:
                return inj
        return None

    def _hit(self, inj, node_id, stage, code, message):
        self.injection_hits.append({
            "injection_id": inj["id"], "node_id": node_id, "stage": stage,
            "fault": inj["fault"], "code": code, "message": message,
        })
        self.failures.append(Failure(
            self.graph.nodes[node_id].topo_index if node_id in
            self.graph.nodes else -1, node_id, stage, code, message,
            injection_id=inj["id"], detail={"injection": inj}))

    # ---------- 确定性输入 ----------
    def _materialize_inputs(self):
        for name, desc in self.inputs_desc.items():
            seed = int(desc.get("seed", 0))
            shape = tuple(desc["shape"])
            dt = np.float64 if self.cfg["precision"] == "fp64" \
                else np.float32
            rng = np.random.default_rng(seed)
            dist = desc.get("distribution", "uniform")
            lo = float(desc.get("low", -2.0))
            hi = float(desc.get("high", 2.0))
            if dist == "normal":
                arr = rng.standard_normal(shape).astype(dt)
            else:
                arr = rng.uniform(lo, hi, size=shape).astype(dt)
            self.inputs[name] = arr

    # ---------- 恢复判定 ----------
    def _resume_state(self):
        manifest = self.store.load_manifest()
        if manifest is None:
            self.events.append({"type": "cold_start", "reason": "no_manifest"})
            return
        if manifest.get("config_hash") != self.config_hash:
            self.events.append({
                "type": "config_hash_mismatch_full_scan",
                "old": manifest.get("config_hash"), "new": self.config_hash})
            self.version = 0
            return
        self.manifest = manifest
        self.version = int(manifest.get("global_version", 0))
        self.events.append({"type": "resume", "global_version": self.version})

    # ---------- 主入口 ----------
    def run(self) -> RunArtifacts:
        self._materialize_inputs()
        input_shapes = {k: v.shape for k, v in self.inputs.items()}
        self.graph = build_graph(self.chain, input_shapes, self.injections)
        for f in self.graph.failures:
            self.failures.append(f)
        self._resume_state()

        # 校验期注入命中回填（duplicate_perm / activation_param）
        for f in self.graph.failures:
            if f.injection_id and f.code in ("E_PERM_INVALID",
                                             "E_ACTIVATION_PARAM"):
                fault = "duplicate_perm" if f.code == "E_PERM_INVALID" \
                    else "activation_param"
                self.injection_hits.append({
                    "injection_id": f.injection_id, "node_id": f.node_id,
                    "stage": "validate", "fault": fault, "code": f.code,
                    "message": f.message})

        self._execute_budgeted()
        self.failures.sort(key=lambda f: (f.topo_index, f.node_id,
                                          f.stage, f.code))
        report = self._build_report()
        return RunArtifacts(report=report, graph=self.graph,
                            products=self.products)

    # ---------- 预算调度 ----------
    def _node_cost(self, nid):
        st = self.graph.nodes[nid]
        if st.shape is None:
            return 0
        return int(np.prod(st.shape))

    def _execute_budgeted(self):
        order = self.graph.order
        failed_nodes = {f.node_id for f in self.failures}
        budget = int(self.cfg["element_budget"])
        remaining = budget
        runnable: list[str] = []
        observed_max_concurrency = 0

        for nid in order:
            st = self.graph.nodes[nid]
            # 校验期已失败：自身与下游隔离，不影响其他合法分支
            if not st.valid or nid in failed_nodes:
                if not (not st.valid):
                    self.failures.append(Failure(
                        st.topo_index, nid, "compute", E_UPSTREAM_FAILED,
                        "skipped due to earlier failure"))
                failed_nodes.add(nid)
                continue
            if st.producer is not None and st.producer in failed_nodes:
                self.failures.append(Failure(
                    st.topo_index, nid, "compute", E_UPSTREAM_FAILED,
                    f"upstream {st.producer} failed"))
                failed_nodes.add(nid)
                continue

            # 缓存复用优先：未失效节点直接复用，不消耗重算预算、不重算
            key = self._compute_key(nid)
            cached = self._try_reuse(nid, key)
            if cached is not None:
                self.products[nid] = cached
                self.reused.append(nid)
                if cached.get("fusion") is not None and not any(
                        d["transpose"] == nid
                        for d in self.fusion_results):
                    self.fusion_results.append(cached["fusion"])
                continue

            cost = self._node_cost(nid)
            if cost > remaining:
                self.pending.append(nid)
                self.failures.append(Failure(
                    st.topo_index, nid, "schedule",
                    E_RECOMPUTE_BUDGET_PENDING,
                    f"cost {cost} exceeds remaining budget {remaining}",
                    detail={"cost": cost, "remaining": remaining}))
                failed_nodes.add(nid)
                for later in order[order.index(nid) + 1:]:
                    lst = self.graph.nodes[later]
                    if later not in failed_nodes:
                        self.pending.append(later)
                        self.failures.append(Failure(
                            lst.topo_index, later, "schedule",
                            E_RECOMPUTE_BUDGET_PENDING,
                            "blocked behind budget-pending node"))
                        failed_nodes.add(later)
                break
            remaining -= cost
            runnable.append(nid)

        max_workers = min(int(self.cfg["max_workers"]),
                          max(1, len(runnable)))
        if runnable:
            layers = self._group_layers(runnable)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for layer in layers:
                    observed_max_concurrency = max(
                        observed_max_concurrency, min(len(layer), max_workers))
                    # 同层互不依赖，可安全并发；携带稳定排序键
                    futures = {
                        pool.submit(self._execute_node, nid):
                        (nid, self._sort_key(nid)) for nid in layer}
                    layer_results = {}
                    for fut in futures:
                        nid, _ = futures[fut]
                        layer_results[nid] = fut.result()
                    # 该层整体完成后立即回填，下一层才能取到上游产物
                    for nid in sorted(layer_results,
                                      key=lambda n:
                                      (self.graph.nodes[n].topo_index, n)):
                        res = layer_results[nid]
                        if res is None or res.get("failed"):
                            failed_nodes.add(nid)
                            continue
                        self.products[nid] = res
                        self.recomputed.append(nid)
            # 运行期失败节点的下游隔离
            for nid in order:
                st = self.graph.nodes[nid]
                if nid in self.products or nid in failed_nodes:
                    continue
                if st.producer is not None and st.producer in failed_nodes:
                    self.failures.append(Failure(
                        st.topo_index, nid, "compute", E_UPSTREAM_FAILED,
                        f"upstream {st.producer} failed at runtime"))
                    failed_nodes.add(nid)
        historical = 0
        if self.manifest:
            historical = int(self.manifest.get("observed_max_concurrency", 0))
        self._observed_max_concurrency = max(historical,
                                             observed_max_concurrency)
        # 复用/重算/待办清单统一按拓扑序位次+节点ID稳定排列
        self.recomputed.sort(key=lambda n:
                             (self.graph.nodes[n].topo_index, n))
        self.reused.sort(key=lambda n:
                         (self.graph.nodes[n].topo_index, n))
        self.pending.sort(key=lambda n:
                          (self.graph.nodes[n].topo_index, n))
        self._commit_snapshots()

    def _group_layers(self, runnable):
        """把可运行节点按依赖分层，同层互不依赖可并发。"""
        runset = set(runnable)
        depth = {}
        for nid in runnable:
            p = self.graph.nodes[nid].producer
            depth[nid] = 0 if p not in runset else depth[p] + 1
        layers: dict[int, list[str]] = {}
        for nid, d in depth.items():
            layers.setdefault(d, []).append(nid)
        return [sorted(layers[d], key=lambda n:
                       (self.graph.nodes[n].topo_index, n))
                for d in sorted(layers)]

    def _sort_key(self, nid):
        st = self.graph.nodes[nid]
        return f"{st.topo_index:08d}|{nid}|compute"

    # ---------- 缓存键 / 复用 ----------
    def _input_fingerprints(self, nid):
        st = self.graph.nodes[nid]
        if st.producer is not None:
            prod = self.products.get(st.producer)
            if prod is None:
                return None
            return [prod["output_fingerprint"]]
        arr = self.inputs.get(st.input_ref)
        return [fingerprint_array(arr)] if arr is not None else None

    def _compute_key(self, nid):
        st = self.graph.nodes[nid]
        fps = self._input_fingerprints(nid) or []
        return node_cache_key(st.spec, fps, self.cfg["precision"],
                              self.cfg["bit_width"], self.cfg["quant_mode"])

    def _try_reuse(self, nid, key):
        loaded = self.store.load_product(nid)
        if loaded is None:
            return None
        if loaded["meta"].get("cache_key") != key:
            return None
        product = {
            "output": loaded["output"], "order_ids": loaded["order_ids"],
            "round_counts": loaded["round_counts"],
            "output_fingerprint": loaded["meta"]["output_fingerprint"],
            "cache_key": key, "reused": True,
            "quant": {
                "q": loaded["q"], "step": loaded["step"],
                "dequant": loaded["dequant"],
                "residual": loaded["residual"],
                "bit_width": loaded["meta"]["quant_meta"]["bit_width"],
                "mode": loaded["meta"]["quant_meta"]["mode"],
                "qmax": loaded["meta"]["quant_meta"]["qmax"]},
            "meta": loaded["meta"].get("meta", {}),
            "fusion": loaded["meta"].get("fusion"),
        }
        return product

    # ---------- 单节点执行（线程任务） ----------
    def _execute_node(self, nid):
        st = self.graph.nodes[nid]
        try:
            # 实现版本缺失注入
            inj = self._injection(nid, "compute", "impl_version_missing")
            if inj is not None:
                self._hit(inj, nid, "compute", E_INJ_IMPL_MISSING,
                          "injected missing operator implementation version")
                return {"failed": True}

            # 运行期上游缺失保护（注入隔离 / 运行期失败级联）
            if st.producer is not None and st.producer not in self.products:
                self.failures.append(Failure(
                    st.topo_index, nid, "compute", E_UPSTREAM_FAILED,
                    f"upstream {st.producer} product unavailable"))
                return {"failed": True}
            # 取输入数组（copy 保证原地写回不破坏其他节点引用）
            if st.producer is not None:
                x = self.products[st.producer]["output"].copy()
            else:
                x = self.inputs[st.input_ref].copy()

            # 非有限元素注入（compute 阶段）
            inj_nf = self._injection(nid, "compute", "nonfinite")
            if inj_nf is not None:
                bad = x.copy()
                bad.reshape(-1)[0] = np.array(np.nan, dtype=x.dtype)
                self._hit(inj_nf, nid, "compute", E_INJ_NONFINITE,
                          "injected non-finite tensor element")
                return {"failed": True, "poisoned": bad}
            if not np.all(np.isfinite(x)):
                self.failures.append(Failure(
                    st.topo_index, nid, "compute", E_INJ_NONFINITE,
                    "non-finite element detected at compute stage"))
                return {"failed": True}

            inplace = bool(st.spec.get("inplace", False))
            if st.op in ("relu", "gelu", "sigmoid", "tanh"):
                if inplace:
                    # 原地写回：直接在输入缓冲上逐元素更新（形状不变）。
                    # x 本身是私有 copy，fan-out/转置情形已在校验期拒绝。
                    y, rounds = ops.activate_inplace(
                        st.op, x, st.spec.get("params", {}),
                        self.cfg["precision"])
                else:
                    y, rounds = ops.run_activation(
                        st.op, x, st.spec.get("params", {}),
                        self.cfg["precision"])
                order_ids = ops.lexicographic_order_ids(y.shape)
            else:
                y = ops.transpose(x, st.perm)
                order_ids = ops.lexicographic_order_ids(y.shape)
                rounds = np.zeros(y.shape, dtype=np.int64)

            # 原地写回运行期保护
            if st.spec.get("inplace", False):
                conflict = self._inplace_conflict(nid)
                if conflict:
                    self.failures.append(Failure(
                        st.topo_index, nid, "compute", E_INPLACE_CONFLICT,
                        conflict))
                    return {"failed": True}

            # quantize 阶段注入：非有限
            inj_q = self._injection(nid, "quantize", "nonfinite")
            if inj_q is not None:
                self._hit(inj_q, nid, "quantize", E_INJ_NONFINITE,
                          "injected non-finite element before quantize")
                return {"failed": True}

            quant = quantize(y, int(self.cfg["bit_width"]),
                             self.cfg["quant_mode"])
            out_fp = fingerprint_array(y)
            product = {
                "output": y, "order_ids": order_ids, "round_counts": rounds,
                "output_fingerprint": out_fp,
                "cache_key": self._compute_key(nid), "reused": False,
                "quant": {k: quant[k] for k in
                          ("q", "step", "dequant", "residual", "bit_width",
                           "mode", "qmax")},
                "meta": {
                    "node_id": nid, "op": st.op,
                    "precision": self.cfg["precision"],
                    "impl_version": IMPLEMENTATION_VERSION,
                    "shape": list(y.shape),
                    "approximations": dmath.approx_metadata(),
                    "inplace": bool(st.spec.get("inplace", False)),
                    "sort_key": self._sort_key(nid),
                },
            }
            if st.op == "transpose":
                product["meta"]["perm"] = list(st.perm)
            # 融合诊断
            self._fusion_diagnostic(nid, x, product)
            return product
        except Exception as exc:  # 隔离：单节点异常不拖垮整体
            self.failures.append(Failure(
                st.topo_index, nid, "compute", "E_RUNTIME", repr(exc)))
            return {"failed": True}

    def _inplace_conflict(self, nid):
        st = self.graph.nodes[nid]
        if st.producer is not None and \
                len(self.graph.nodes[st.producer].consumers) > 1:
            return "write-back would break another node's input reference"
        if st.op == "transpose":
            return "transpose changes shape, cannot write back in place"
        return None

    # ---------- 融合诊断 ----------
    def _fusion_diagnostic(self, nid, transpose_input, product):
        """transpose_input 即激活节点的输出（本转置节点的输入）。"""
        if not self.cfg["enable_fusion"]:
            return
        cand = next((c for c in self.graph.fusion_candidates
                     if c["transpose"] == nid), None)
        if cand is None:
            return
        act_id = cand["activation"]
        act_st = self.graph.nodes[act_id]
        act_product = self.products.get(act_id)
        if act_product is None:
            return
        # 激活前原始输入
        if act_st.producer is not None:
            origin = self.products[act_st.producer]["output"]
        else:
            origin = self.inputs[act_st.input_ref]
        perm = product["meta"]["perm"]
        params = act_st.spec.get("params", {})
        unfused = ops.run_unfused(act_st.op, params, origin, perm,
                                  self.cfg["precision"])
        fused = ops.run_fused(act_st.op, params, origin, perm,
                              self.cfg["precision"])
        tol = float(self.cfg["tolerance"])
        diff = np.abs(fused["output"] - unfused["output"])
        over = diff > tol
        result = {
            "activation": act_id, "transpose": nid,
            "unfused_path": unfused["path"], "fused_path": fused["path"],
            "unfused_order_ids_fingerprint":
                fingerprint_array(unfused["order_ids"]),
            "fused_order_ids_fingerprint":
                fingerprint_array(fused["order_ids"]),
            "unfused_round_counts": unfused["round_counts"].tolist(),
            "fused_round_counts": fused["round_counts"].tolist(),
            "max_abs_diff": float(np.max(diff)),
            "over_tolerance": bool(np.any(over)),
            "over_tolerance_count": int(np.sum(over)),
            "over_tolerance_elements": np.argwhere(over).tolist()[:50],
            "attribution": "fused_path" if np.any(over) else None,
            "tolerance": tol,
        }
        if np.any(over):
            self.failures.append(Failure(
                self.graph.nodes[nid].topo_index, nid, "compute",
                E_FUSION_TOLERANCE,
                f"fusion diff exceeds tolerance at "
                f"{int(np.sum(over))} elements",
                detail={"max_abs_diff": float(np.max(diff))}))
        self.fusion_results.append(result)
        product["fusion"] = result

    # ---------- 快照提交 ----------
    def _commit_snapshots(self):
        interval = int(self.cfg["snapshot_interval"])
        completed_since = 0
        completed = list(self.recomputed)
        for i, nid in enumerate(completed, start=1):
            self.version += 1
            self.store.save_product_staging(
                nid, self.products[nid], self.version)
            completed_since += 1
            if completed_since >= interval:
                self._flush_manifest()
                completed_since = 0
        self._flush_manifest()

    def _flush_manifest(self):
        done = sorted(self.products.keys(),
                      key=lambda n: (self.graph.nodes[n].topo_index, n))
        product_versions = {}
        for nid in done:
            loaded = self.store.load_product(nid)
            product_versions[nid] = loaded["meta"]["version"] \
                if loaded else self.version
        manifest = {
            "global_version": self.version,
            "config_hash": self.config_hash,
            "completed_nodes": done,
            "product_versions": product_versions,
            "reused": list(self.reused),
            "recomputed": list(self.recomputed),
            "pending": list(self.pending),
            "observed_max_concurrency":
                getattr(self, "_observed_max_concurrency", 0),
        }
        self.store.commit(manifest)
        self.manifest = manifest

    # ---------- 报告 ----------
    def _build_report(self):
        nodes_report = []
        for nid in self.graph.order:
            st = self.graph.nodes[nid]
            prod = self.products.get(nid)
            entry = {
                "node_id": nid, "op": st.op,
                "topo_index": st.topo_index, "shape":
                list(st.shape) if st.shape else None,
                "status": ("recomputed" if nid in self.recomputed else
                           "reused" if nid in self.reused else
                           "pending" if nid in self.pending else "failed"),
                "sort_key": self._sort_key(nid),
                "inplace": bool(st.spec.get("inplace", False)),
            }
            if prod is not None:
                entry["cache_key"] = prod["cache_key"]
                entry["output_fingerprint"] = prod["output_fingerprint"]
                entry["quant"] = {
                    "bit_width": prod["quant"]["bit_width"],
                    "mode": prod["quant"]["mode"],
                    "step_fingerprint":
                        fingerprint_array(np.atleast_1d(
                            np.asarray(prod["quant"]["step"]))),
                    "residual_fingerprint":
                        fingerprint_array(prod["quant"]["residual"]),
                    "integer_dtype": str(prod["quant"]["q"].dtype),
                }
                entry["meta"] = prod["meta"]
                if prod.get("fusion"):
                    entry["fusion"] = {
                        k: v for k, v in prod["fusion"].items()
                        if k not in ("unfused_round_counts",
                                     "fused_round_counts")}
            nodes_report.append(entry)

        report = {
            "component": "opdiag",
            "impl_version": IMPLEMENTATION_VERSION,
            "config_hash": self.config_hash,
            "global_version": self.version,
            "precision": self.cfg["precision"],
            "bit_width": self.cfg["bit_width"],
            "quant_mode": self.cfg["quant_mode"],
            "tolerance": self.cfg["tolerance"],
            "max_workers": int(self.cfg["max_workers"]),
            "observed_max_concurrency":
                getattr(self, "_observed_max_concurrency", 0),
            "element_budget": int(self.cfg["element_budget"]),
            "events": self.events,
            "nodes": nodes_report,
            "reuse_list": list(self.reused),
            "recompute_list": list(self.recomputed),
            "pending_list": list(self.pending),
            "fusion_diagnostics": self.fusion_results,
            "injection_hits": self.injection_hits,
            "failures": [f.to_dict() for f in self.failures],
            "failure_codes": sorted({f.code for f in self.failures}),
            "topology": [self.graph.nodes[n].to_public()
                         for n in self.graph.order],
        }
        return report
