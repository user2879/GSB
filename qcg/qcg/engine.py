"""Budget-constrained deterministic engine.

The engine deliberately separates *planning* from *execution*:

* planning walks nodes in stable topological order and produces an ordered,
  fully deterministic ledger (reuse / compute / eviction / budget events),
  so thread timing can never affect budgets or ordering;
* execution runs the planned compute tasks through a thread pool, but every
  task is independent and side-effect free, and all outputs are committed in
  topological order — hence concurrent and serial runs are field-identical.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .converter import hash_sidecar
from .dequant import dequantize
from .errors import (FaultRecord, E_NODE_BUDGET_EXHAUSTED,
                     E_GLOBAL_BUDGET_PENDING, E_RECOMPUTE_PENDING)
from .fingerprint import (compute_cache_key, downstream_closure,
                          graph_content_hash, tensor_fingerprint)
from .injection import (matching, FAULT_PACK_TRUNCATION,
                        FAULT_NONFINITE_SCALE, FAULT_BITWIDTH_OVERFLOW,
                        FAULT_SIDECAR_TAMPER)
from .kernel import integer_conv
from .models import Graph, Node, NODE_INPUT, NODE_CONV
from .product_store import ProductStore, Product, block_bytes
from .quant_pack import read_pack
from .snapshot import SnapshotManager, STAGES
from .validator import validate


@dataclass
class NodePlan:
    node_id: str
    topo_index: int
    sort_key: str
    action: str                 # "reuse" | "compute"
    macs: int = 0
    recompute: bool = False
    reload_block: bool = False
    evicted: List[str] = field(default_factory=list)
    pending_code: Optional[str] = None
    upstream: List[str] = field(default_factory=list)


@dataclass
class Plan:
    ordered: List[NodePlan]
    budget_events: List[dict]
    reused: List[str]
    recomputed: List[str]
    pending: List[dict]
    evictions: List[dict]
    reloads: List[dict]
    remaining_global_budget: int
    remaining_batch_budget: int
    remaining_node_budget: Dict[str, int]


def estimate_macs(node: Node, input_shape) -> int:
    if node.kind != NODE_CONV or input_shape is None:
        return 0
    in_h, in_w = input_shape
    out_h = (in_h + 2 * node.padding - node.kernel) // node.stride + 1
    out_w = (in_w + 2 * node.padding - node.kernel) // node.stride + 1
    return max(out_h, 0) * max(out_w, 0) * node.kernel * node.kernel


class Planner:
    """Deterministic single-threaded planner producing the execution ledger."""

    def __init__(self, graph: Graph, cfg: Config, store: ProductStore,
                 tensor_shapes: Dict[str, tuple],
                 invalid_ids: List[str], round_no: int):
        self.graph = graph
        self.cfg = cfg
        self.store = store
        self.tensor_shapes = tensor_shapes
        self.round_no = round_no
        self.topo = graph.topo_order()
        self.invalid = set(invalid_ids)
        self.persisted = store.persisted_keys()
        self.global_left = cfg.global_mac_budget
        self.batch_left = cfg.batch_recompute_budget
        self.node_left = {n.node_id: cfg.node_mac_quota for n in self.topo}
        self.resident: List[str] = []
        self.resident_bytes = 0
        self.evictions: List[dict] = []
        self.reloads: List[dict] = []
        self.budget_events: List[dict] = []
        self.pending: List[dict] = []

    def plan(self) -> Plan:
        ordered: List[NodePlan] = []
        reused, recomputed = [], []
        index = {n.node_id: i for i, n in enumerate(self.topo)}
        blocked: set = set()
        for idx, node in enumerate(self.topo):
            np_item = NodePlan(
                node_id=node.node_id, topo_index=idx,
                sort_key="%04d:%s" % (idx, node.node_id),
                action="compute", upstream=list(node.inputs))
            need_recompute = node.node_id in self.invalid or \
                node.node_id not in self.persisted
            upstream_blocked = any(p in blocked for p in node.inputs)
            if upstream_blocked:
                np_item.pending_code = E_GLOBAL_BUDGET_PENDING
                np_item.action = "compute"
                blocked.add(node.node_id)
                self.pending.append(
                    {"node_id": node.node_id,
                     "code": E_GLOBAL_BUDGET_PENDING,
                     "sort_key": np_item.sort_key,
                     "reason": "upstream_pending"})
                ordered.append(np_item)
                continue
            if not need_recompute:
                np_item.action = "reuse"
                reused.append(node.node_id)
                np_item.reload_block = self._touch_residency(node)
                if np_item.reload_block:
                    self.reloads.append(
                        {"node_id": node.node_id, "round": self.round_no,
                         "reason": "block_evicted_reload"})
                ordered.append(np_item)
                continue
            macs = estimate_macs(node, self.tensor_shapes.get(
                self._input_id(node)))
            np_item.macs = macs
            # node quota check
            if macs > self.node_left[node.node_id]:
                np_item.pending_code = E_NODE_BUDGET_EXHAUSTED
                blocked.add(node.node_id)
                self.pending.append({"node_id": node.node_id,
                                     "code": E_NODE_BUDGET_EXHAUSTED,
                                     "sort_key": np_item.sort_key})
                self.budget_events.append(
                    {"seq": len(self.budget_events),
                     "node_id": node.node_id,
                     "code": E_NODE_BUDGET_EXHAUSTED, "macs": macs,
                     "remaining": self.node_left[node.node_id]})
                ordered.append(np_item)
                continue
            if macs > self.global_left:
                np_item.pending_code = E_GLOBAL_BUDGET_PENDING
                blocked.add(node.node_id)
                self.pending.append({"node_id": node.node_id,
                                     "code": E_GLOBAL_BUDGET_PENDING,
                                     "sort_key": np_item.sort_key})
                self.budget_events.append(
                    {"seq": len(self.budget_events),
                     "node_id": node.node_id,
                     "code": E_GLOBAL_BUDGET_PENDING, "macs": macs,
                     "remaining": self.global_left})
                ordered.append(np_item)
                continue
            if macs > self.batch_left:
                np_item.pending_code = E_RECOMPUTE_PENDING
                blocked.add(node.node_id)
                self.pending.append({"node_id": node.node_id,
                                     "code": E_RECOMPUTE_PENDING,
                                     "sort_key": np_item.sort_key})
                self.budget_events.append(
                    {"seq": len(self.budget_events),
                     "node_id": node.node_id,
                     "code": E_RECOMPUTE_PENDING, "macs": macs,
                     "remaining": self.batch_left})
                ordered.append(np_item)
                continue
            # budgets accepted: deduct in fixed order node -> global -> batch
            self.node_left[node.node_id] -= macs
            self.global_left -= macs
            self.batch_left -= macs
            np_item.recompute = True
            recomputed.append(node.node_id)
            self.budget_events.append(
                {"seq": len(self.budget_events),
                 "node_id": node.node_id, "code": "BUDGET_DEDUCT",
                 "macs": macs, "global_left": self.global_left,
                 "batch_left": self.batch_left,
                 "node_left": self.node_left[node.node_id]})
            np_item.reload_block = self._touch_residency(node, planned=True)
            ordered.append(np_item)
        self.pending.sort(key=lambda p: p["sort_key"])
        return Plan(ordered=ordered, budget_events=self.budget_events,
                    reused=reused, recomputed=recomputed,
                    pending=self.pending, evictions=self.evictions,
                    reloads=self.reloads,
                    remaining_global_budget=self.global_left,
                    remaining_batch_budget=self.batch_left,
                    remaining_node_budget=dict(self.node_left))

    def _input_id(self, node: Node) -> Optional[str]:
        return node.inputs[0] if node.inputs else None

    def _planned_block_bytes(self, node: Node) -> int:
        shape = self.tensor_shapes.get(node.node_id)
        if shape is None:
            return 0
        int_arr = np.zeros(shape, dtype=np.int64)
        dq_arr = np.zeros(shape, dtype=np.float64)
        return block_bytes(int_arr) + block_bytes(dq_arr)

    def _touch_residency(self, node: Node, planned: bool = False) -> bool:
        """Simulate LRU residency; returns True if block must be reloaded."""
        nid = node.node_id
        if nid in self.resident:
            self.resident.remove(nid)
            self.resident.append(nid)
            return False
        size = self._planned_block_bytes(node)
        reloaded = False
        if self.store.is_persisted(nid):
            reloaded = True
        evicted = []
        while self.resident_bytes + size > self.cfg.cache_resident_bytes \
                and self.resident:
            victim = self.resident.pop(0)
            victim_shape = self.tensor_shapes.get(victim)
            self.resident_bytes -= self._shape_bytes(victim_shape)
            evicted.append(victim)
        self.resident.append(nid)
        self.resident_bytes += size
        if evicted:
            self.evictions.append({"node_id": nid, "evicted": list(evicted),
                                   "round": self.round_no, "block_bytes": size})
        return reloaded

    def _shape_bytes(self, shape) -> int:
        if shape is None:
            return 0
        count = int(np.prod(shape))
        return count * 8 + count * 8


class Engine:
    def __init__(self, cfg: Config, snapshot_root: str,
                 injections=None, global_version: int = 1):
        self.cfg = cfg
        self.store = ProductStore(snapshot_root, cfg.cache_resident_bytes)
        self.snapshots = SnapshotManager(snapshot_root)
        self.injections = injections or []
        self.global_version = global_version
        self.faults: List[FaultRecord] = []
        self.hits: List[str] = []

    # ------------------------------------------------------------------
    def run(self, graph: Graph, pack_path: str, sidecar: Optional[dict] = None,
            previous_graph: Optional[Graph] = None,
            resume: bool = False, round_no: int = 0) -> dict:
        validate(graph)
        graph_hash = graph_content_hash(graph)
        cfg_hash = self.cfg.hash()
        self.removed_products = self.store.retain_only(
            [n.node_id for n in graph.nodes])
        manifest = None
        if resume:
            manifest = self.snapshots.check_resume(
                self.global_version, graph_hash, cfg_hash)
        blocks = self._load_pack_with_injection(graph, pack_path)
        sidecar_hash = hash_sidecar(sidecar or {"version": "none",
                                                "nodes": []})
        # stage 1: load + validate (already validated; record stage output)
        self.snapshots.write_stage(
            STAGES[0],
            {"graph_hash": graph_hash, "nodes": [n.node_id
                                                 for n in graph.topo_order()],
             "source": graph.source, "sidecar_hash": sidecar_hash})
        shapes = self._infer_shapes(graph, blocks)
        tensor_fp, invalid_ids = self._invalidate(graph, previous_graph,
                                                  shapes)
        plan = Planner(graph, self.cfg, self.store, shapes,
                       invalid_ids, round_no=round_no).plan()
        self._apply_planned_evictions(plan)
        compute_plans = [p for p in plan.ordered
                         if p.action == "compute" and not p.pending_code]
        products = self._execute(graph, blocks, tensor_fp, compute_plans,
                                 shapes, sidecar)
        # commit products in topological order
        committed_so_far = []
        for plan_item in plan.ordered:
            if plan_item.node_id in products:
                self.store.persist(products[plan_item.node_id])
                committed_so_far.append(plan_item.node_id)
                interval = self.cfg.snapshot_commit_interval
                if interval > 0 and len(committed_so_far) % interval == 0:
                    self.snapshots.commit({
                        "global_version": self.global_version,
                        "graph_hash": graph_hash,
                        "config_hash": cfg_hash,
                        "version_source": graph.source,
                        "sidecar_hash": sidecar_hash,
                        "completed_nodes": list(committed_so_far),
                        "product_versions": {
                            nid: products[nid].cache_key
                            for nid in committed_so_far},
                        "stage_outputs": list(STAGES),
                        "partial": True})
        dequant_meta = {nid: p.meta for nid, p in products.items()}
        self.snapshots.write_stage(STAGES[1], {
            "nodes": [p.node_id for p in plan.ordered
                      if p.node_id in products]})
        self.snapshots.write_stage(STAGES[2], {
            "nodes": sorted(dequant_meta),
            "boundary": {nid: m["boundary_samples"]
                         for nid, m in dequant_meta.items()}})
        report = self._build_report(graph, plan, products, graph_hash,
                                    cfg_hash, sidecar_hash, manifest, shapes,
                                    round_no)
        self.snapshots.write_stage(STAGES[3], report)
        completed = [p.node_id for p in plan.ordered
                     if p.pending_code is None]
        new_manifest = {
            "global_version": self.global_version,
            "graph_hash": graph_hash,
            "config_hash": cfg_hash,
            "version_source": graph.source,
            "sidecar_hash": sidecar_hash,
            "completed_nodes": completed,
            "product_versions": {
                nid: products[nid].cache_key for nid in products},
            "stage_outputs": list(STAGES),
        }
        self.snapshots.commit(new_manifest)
        return report

    # ------------------------------------------------------------------
    def _load_pack_with_injection(self, graph: Graph, pack_path: str) -> dict:
        for node in graph.topo_order():
            for inj in matching(self.injections, node.node_id, STAGES[0]):
                if inj.fault == FAULT_PACK_TRUNCATION:
                    self._record_fault(inj)
                    raise inj.trigger_error()
                if inj.fault == FAULT_SIDECAR_TAMPER:
                    self._record_fault(inj)
                    raise inj.trigger_error()
                if inj.fault == FAULT_BITWIDTH_OVERFLOW:
                    self._record_fault(inj)
                    raise inj.trigger_error()
        return read_pack(pack_path)

    def _infer_shapes(self, graph: Graph, blocks: dict) -> Dict[str, tuple]:
        shapes: Dict[str, tuple] = {}
        by_id = graph.by_id()
        for node in graph.topo_order():
            if node.kind == NODE_INPUT:
                key = "act:" + node.node_id
                if key not in blocks:
                    raise KeyError("missing activation block %s" % key)
                shapes[node.node_id] = blocks[key].shape
            else:
                src = node.inputs[0]
                if src not in shapes:
                    from .errors import QCGError, E_GRAPH_INVALID
                    raise QCGError(E_GRAPH_INVALID,
                                   "node %s references missing producer %s"
                                   % (node.node_id, src),
                                   node_id=node.node_id, field="inputs")
                in_h, in_w = shapes[src]
                from .kernel import conv_output_shape
                shapes[node.node_id] = conv_output_shape(in_h, in_w, node)
        return shapes

    def _invalidate(self, graph: Graph, previous_graph: Optional[Graph],
                    shapes) -> (Dict[str, str], List[str]):
        """Content-fingerprint invalidation.

        Two entry modes:
        * ``previous_graph`` supplied -> diff node parameter fingerprints
          between graph versions and close seeds over downstream nodes;
        * standalone continuation -> read persisted product metadata and
          mark a node invalid when its parameter fingerprint changed, it is
          new, or any upstream node is invalid. This makes incremental CLI
          runs precise without the caller retaining the previous graph.
        """
        persisted = self.store.persisted_keys()
        if previous_graph is not None:
            prev_by_id = previous_graph.by_id()
            seeds: List[str] = []
            for node in graph.topo_order():
                old_node = prev_by_id.get(node.node_id)
                if old_node is None:
                    seeds.append(node.node_id)
                    continue
                from .fingerprint import node_param_fingerprint
                if node_param_fingerprint(old_node) != \
                        node_param_fingerprint(node):
                    seeds.append(node.node_id)
            affected = downstream_closure(graph, seeds) if seeds else set()
            invalid = [n.node_id for n in graph.topo_order()
                       if n.node_id in affected
                       or n.node_id not in persisted]
            return {}, invalid
        if not persisted:
            return {}, [n.node_id for n in graph.topo_order()]
        from .fingerprint import node_param_fingerprint
        invalid: List[str] = []
        invalid_set: set = set()
        for node in graph.topo_order():
            is_invalid = False
            if node.node_id not in persisted:
                is_invalid = True
            else:
                meta = self.store.load_meta(node.node_id)
                if meta.get("param_fp") != node_param_fingerprint(node):
                    is_invalid = True
            if any(p in invalid_set for p in node.inputs):
                is_invalid = True
            if is_invalid:
                invalid.append(node.node_id)
                invalid_set.add(node.node_id)
        return {}, invalid

    def _apply_planned_evictions(self, plan: Plan) -> None:
        for event in plan.evictions:
            self.store.eviction_events.append(event)
        for event in plan.reloads:
            self.store.recompute_events.append(event)

    def _execute(self, graph: Graph, blocks: dict,
                 tensor_fp: Dict[str, str], compute_plans: List[NodePlan],
                 shapes: Dict[str, tuple], sidecar: Optional[dict]) -> dict:
        """Topology-streaming execution through an optional thread pool.

        The planner already decided every action and budget deduction in a
        single deterministic pass. Tasks only become eligible once their
        upstream products are available, so data dependencies are respected;
        each task is side-effect free, and results are committed in stable
        topological order, which makes concurrent and serial runs identical.
        """
        by_id = graph.by_id()
        chosen = {p.node_id for p in compute_plans
                  if p.action == "compute" and not p.pending_code}
        plan_by_id = {p.node_id: p for p in compute_plans}
        products: Dict[str, Product] = {}
        remaining = set(chosen)
        failed: set = set()
        while remaining:
            ready = [n for n in graph.topo_order()
                     if n.node_id in remaining
                     and all(p not in remaining for p in n.inputs)]
            wave = []
            for node in ready:
                if any(p in failed for p in node.inputs):
                    failed.add(node.node_id)
                    remaining.discard(node.node_id)
                else:
                    wave.append(node)
            if not wave:
                continue

            def run_node(node: Node):
                return self._compute_node(
                    node, graph, blocks, shapes, plan_by_id[node.node_id],
                    products)

            if self.cfg.max_workers <= 1 or len(wave) == 1:
                for node in wave:
                    product = run_node(node)
                    if product is not None:
                        products[product.node_id] = product
                    else:
                        failed.add(node.node_id)
            else:
                with ThreadPoolExecutor(
                        max_workers=min(self.cfg.max_workers, len(wave))
                        ) as pool:
                    futures = [pool.submit(run_node, n) for n in wave]
                    wave_products = [f.result() for f in futures]
                wave_ids = {n.node_id for n in wave}
                for product in sorted(
                        (p for p in wave_products if p is not None),
                        key=lambda p: plan_by_id[p.node_id].sort_key):
                    products[product.node_id] = product
                failed.update(nid for nid in wave_ids
                              if nid not in products)
            for node in wave:
                remaining.discard(node.node_id)
        return products

    def _compute_node(self, node: Node, graph: Graph, blocks: dict,
                      shapes: Dict[str, tuple], plan_item: NodePlan,
                      live_products: Dict[str, "Product"]):
        stage = STAGES[1]
        for inj in matching(self.injections, node.node_id, stage):
            if inj.fault in (FAULT_NONFINITE_SCALE, FAULT_BITWIDTH_OVERFLOW):
                self._record_fault(inj)
                self.faults.append(FaultRecord(
                    code=inj.trigger_error().code,
                    message=inj.trigger_error().message,
                    node_id=node.node_id, field=inj.trigger_error().field,
                    stage=stage, injection_id=inj.injection_id))
                return None
        if node.kind == NODE_INPUT:
            key = "act:" + node.node_id
            integer_output = blocks[key].astype(np.int64)
            macs = 0
            overflows = []
        else:
            src = node.inputs[0]
            src_node = graph.by_id()[src]
            if src_node.kind == NODE_INPUT:
                activation = blocks["act:" + src]
            elif src in live_products:
                activation = live_products[src].integer_output
            else:
                activation = self._load_integer(src)
            weight = blocks["w:" + node.node_id]
            conv = integer_conv(activation, weight, node, self.cfg)
            integer_output, macs, overflows = (conv.output, conv.macs,
                                                conv.overflows)
        act_scale = self._activation_scale(graph, node)
        dq = dequantize(integer_output, node, act_scale, self.cfg)
        node.boundary_samples = [list(p) for p in dq.boundary_positions]
        t_fp = tensor_fingerprint(integer_output)
        upstream_fp = {}
        for parent in node.inputs:
            if parent in live_products:
                upstream_fp[parent] = live_products[parent].tensor_fp
            elif self.store.is_persisted(parent):
                upstream_fp[parent] = self.store.load_meta(parent)[
                    "tensor_fp"]
        cache_key = compute_cache_key(node, upstream_fp)
        from .fingerprint import node_param_fingerprint
        meta = {
            "node_id": node.node_id,
            "cache_key": cache_key,
            "param_fp": node_param_fingerprint(node),
            "tensor_fp": t_fp,
            "acc_bits": self.cfg.accumulator_bits,
            "traversal": "row_major:oh,ow,kh,kw",
            "inplace": node.inplace,
            "rounding": node.rounding,
            "frac_bits": dq.frac_bits,
            "scale": dq.scale,
            "boundary_samples": dq.boundary_positions,
            "macs": macs,
            "overflows": overflows,
            "sort_key": plan_item.sort_key,
        }
        return Product(node_id=node.node_id, cache_key=cache_key,
                       integer_output=integer_output,
                       dequant_values=dq.values, meta=meta, tensor_fp=t_fp)

    def _load_integer(self, node_id: str) -> np.ndarray:
        product, reloaded = self.store.ensure_resident(node_id)
        if reloaded:
            self.store.mark_recompute(node_id, "evicted_block_reload", 0)
        return product.integer_output

    def _activation_scale(self, graph: Graph, node: Node) -> float:
        if not node.inputs:
            return 1.0
        src = graph.by_id()[node.inputs[0]]
        return src.scale_float if src.scale_float > 0 \
            else src.scale_permille / 1000.0

    def _record_fault(self, inj) -> None:
        self.hits.append(inj.injection_id)

    def _build_report(self, graph: Graph, plan: Plan, products: dict,
                      graph_hash: str, cfg_hash: str, sidecar_hash: str,
                      manifest: Optional[dict], shapes,
                      round_no: int = 0) -> dict:
        return {
            "global_version": self.global_version,
            "graph_hash": graph_hash,
            "config_hash": cfg_hash,
            "sidecar_hash": sidecar_hash,
            "version_source": graph.source,
            "resumed": manifest is not None,
            "round_no": round_no,
            "topo_order": [n.node_id for n in graph.topo_order()],
            "sort_keys": [p.sort_key for p in plan.ordered],
            "reused": plan.reused,
            "recomputed": plan.recomputed,
            "pending": plan.pending,
            "budget_events": plan.budget_events,
            "evictions": plan.evictions,
            "reloads": plan.reloads,
            "faults": [f.to_dict() for f in self.faults],
            "injection_hits": list(self.hits),
            "products": {
                nid: {"shape": list(products[nid].integer_output.shape),
                      "cache_key": products[nid].cache_key,
                      "boundary_samples": products[nid].meta[
                          "boundary_samples"],
                      "acc_bits": products[nid].meta["acc_bits"]}
                for nid in sorted(products)},
            "remaining_budget": {
                "global": plan.remaining_global_budget,
                "batch": plan.remaining_batch_budget,
                "node": plan.remaining_node_budget},
        }
