"""算子链加载、拓扑校验、形状推导与融合候选识别。

校验失败只产生 Failure 业务码并标记节点，不抛异常中断其余节点。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dmath
from .errors import (
    E_ACTIVATION_PARAM, E_EDGE_ENDPOINT, E_GRAPH_CYCLE, E_INPLACE_CONFLICT,
    E_MULTIPLE_PRODUCER, E_NODE_DUPLICATE, E_PERM_INVALID, E_UNKNOWN_OP,
    E_INPUT_UNBOUND, Failure,
)

ACTIVATION_OPS = set(dmath.ACTIVATIONS)
TRANSPOSE_OPS = {"transpose"}


@dataclass
class NodeState:
    spec: dict
    node_id: str
    op: str
    topo_index: int = -1
    shape: tuple | None = None
    producer: str | None = None          # 上游节点 id（无则外部输入）
    consumers: list = field(default_factory=list)
    input_ref: str | None = None         # 外部输入张量名
    valid: bool = True
    perm: list | None = None

    def to_public(self) -> dict:
        return {
            "node_id": self.node_id, "op": self.op,
            "topo_index": self.topo_index, "shape": list(self.shape)
            if self.shape is not None else None,
            "producer": self.producer, "consumers": list(self.consumers),
            "input_ref": self.input_ref, "valid": self.valid,
            "inplace": bool(self.spec.get("inplace", False)),
            "params": self.spec.get("params", {}),
        }


@dataclass
class Graph:
    nodes: dict
    order: list                       # 拓扑序 node id
    failures: list
    input_shapes: dict
    fusion_candidates: list
    edges: list

    def state(self, node_id: str) -> NodeState:
        return self.nodes[node_id]


def _is_complete_permutation(perm, rank):
    if not isinstance(perm, list) or len(perm) != rank:
        return False
    if any(not isinstance(p, int) or isinstance(p, bool) for p in perm):
        return False
    return sorted(perm) == list(range(rank))


def _validate_activation_params(op, params):
    params = params or {}
    if op == "relu":
        s = params.get("slope", 0.0)
        if not isinstance(s, (int, float)) or not np.isfinite(s) or not (0.0 <= s < 1.0):
            return "relu slope must be finite within [0, 1)"
    if op == "sigmoid":
        s = params.get("scale", 1.0)
        if not isinstance(s, (int, float)) or not np.isfinite(s) or s <= 0:
            return "sigmoid scale must be finite positive"
    if op == "tanh":
        g = params.get("gain", 1.0)
        if not isinstance(g, (int, float)) or not np.isfinite(g) or g <= 0:
            return "tanh gain must be finite positive"
    if op == "gelu":
        for k in params:
            if k not in ():
                return "gelu takes no parameters"
    return None


def build_graph(chain: dict, input_shapes: dict, injections=None) -> Graph:
    """chain: 算子链 JSON；input_shapes: 外部输入名 -> shape。

    injections: 校验期生效的注入（duplicate_perm / activation_param）。
    """
    injections = injections or {}
    failures: list[Failure] = []
    raw_nodes = chain.get("nodes", [])
    edges = [tuple(e) for e in chain.get("edges", [])]
    nodes: dict[str, NodeState] = {}

    # 节点 ID 唯一
    seen = set()
    for spec in raw_nodes:
        nid = spec.get("id")
        if nid in seen or nid in nodes:
            failures.append(Failure(-1, str(nid), "validate",
                                    E_NODE_DUPLICATE, "duplicate node id"))
            continue
        seen.add(nid)
        op = spec.get("op")
        if op not in ACTIVATION_OPS and op not in TRANSPOSE_OPS:
            failures.append(Failure(-1, str(nid), "validate", E_UNKNOWN_OP,
                                    f"unknown op {op!r}"))
        nodes[nid] = NodeState(spec=spec, node_id=nid, op=op,
                               input_ref=spec.get("input_ref"),
                               perm=spec.get("perm"))

    # 边端点存在
    clean_edges = []
    indeg = {nid: 0 for nid in nodes}
    for src, dst in edges:
        if src not in nodes or dst not in nodes:
            failures.append(Failure(-1, f"{src}->{dst}", "validate",
                                    E_EDGE_ENDPOINT, "edge endpoint missing"))
            continue
        clean_edges.append((src, dst))
        indeg[dst] += 1

    # 单生产者：每个节点至多一条入边
    multi = [n for n, d in indeg.items() if d > 1]
    for nid in multi:
        st = nodes[nid]
        st.valid = False
        failures.append(Failure(-1, nid, "validate", E_MULTIPLE_PRODUCER,
                                "input tensor has more than one producer"))

    # 邻接
    succ = {nid: [] for nid in nodes}
    pred = {nid: None for nid in nodes}
    for src, dst in clean_edges:
        succ[src].append(dst)
        if pred[dst] is None:
            pred[dst] = src

    # 稳定拓扑排序（Kahn，候选按 node_id 升序）
    available = sorted(n for n in nodes if indeg[n] == 0)
    order: list[str] = []
    indeg_work = dict(indeg)
    while available:
        nid = available.pop(0)
        order.append(nid)
        for d in sorted(succ[nid]):
            indeg_work[d] -= 1
            if indeg_work[d] == 0:
                # 保持有序插入
                available.append(d)
                available.sort()
    cyclic = [n for n in nodes if n not in order]
    for nid in cyclic:
        st = nodes[nid]
        st.valid = False
        failures.append(Failure(-1, nid, "validate", E_GRAPH_CYCLE,
                                "node participates in a directed cycle"))
    # 环上节点无法定位位次，给保守位次
    for idx, nid in enumerate(order):
        nodes[nid].topo_index = idx
    for j, nid in enumerate(sorted(cyclic)):
        nodes[nid].topo_index = len(order) + j

    # 绑定 producer / consumers / 外部输入
    for nid in order:
        st = nodes[nid]
        st.producer = pred[nid]
        st.consumers = sorted(succ[nid])
        if st.producer is None and not st.input_ref:
            st.valid = False
            failures.append(Failure(st.topo_index, nid, "validate",
                                    E_INPUT_UNBOUND,
                                    "no incoming edge and no input_ref"))
        if st.input_ref and st.input_ref not in input_shapes:
            st.valid = False
            failures.append(Failure(st.topo_index, nid, "validate",
                                    E_INPUT_UNBOUND,
                                    f"external input {st.input_ref!r} not found"))

    # 形状推导（按拓扑序，跳过已知非法/上游缺失）
    shapes: dict[str, tuple] = {}
    for nid in order:
        st = nodes[nid]
        in_shape = None
        if st.producer is not None:
            in_shape = shapes.get(st.producer)
        elif st.input_ref:
            in_shape = tuple(input_shapes.get(st.input_ref, ()))
        if in_shape is None:
            continue
        if st.op in ACTIVATION_OPS:
            shapes[nid] = in_shape
        elif st.op in TRANSPOSE_OPS:
            # 先按声明 perm 推形状；perm 合法性随后校验
            perm = st.perm
            if isinstance(perm, list) and len(perm) == len(in_shape) and \
                    all(isinstance(p, int) for p in perm) and \
                    len(set(perm)) == len(perm) and \
                    all(0 <= p < len(in_shape) for p in perm):
                shapes[nid] = tuple(in_shape[p] for p in perm)
            else:
                shapes[nid] = None
        st.shape = shapes.get(nid)

    # 节点级参数 / 置换校验（含校验期注入）
    for nid in order:
        st = nodes[nid]
        idx = st.topo_index
        if st.op in ACTIVATION_OPS:
            params = dict(st.spec.get("params", {}))
            inj = _injection_for(injections, nid, "validate",
                                 "activation_param")
            if inj is not None:
                params["slope"] = 2.0 if st.op == "relu" else params.get(
                    "slope", 2.0)
                if st.op != "relu":
                    params = {"scale": -1.0, "gain": -1.0}
                st.valid = False
                failures.append(Failure(idx, nid, "validate",
                                        E_ACTIVATION_PARAM,
                                        "injected out-of-range activation "
                                        "parameter",
                                        injection_id=inj["id"],
                                        detail={"injection": inj}))
                continue
            msg = _validate_activation_params(st.op, params)
            if msg:
                st.valid = False
                failures.append(Failure(idx, nid, "validate",
                                        E_ACTIVATION_PARAM, msg))
        elif st.op in TRANSPOSE_OPS:
            rank = len(shapes[st.producer]) if st.producer in shapes else None
            if rank is None and st.input_ref:
                rank = len(input_shapes.get(st.input_ref, ()))
            perm = list(st.perm) if isinstance(st.perm, list) else st.perm
            inj = _injection_for(injections, nid, "validate",
                                 "duplicate_perm")
            if inj is not None and isinstance(perm, list) and perm:
                perm = list(perm)
                perm[-1] = perm[0]  # 制造重复维度
            if rank is None:
                st.valid = False
                failures.append(Failure(idx, nid, "validate", E_PERM_INVALID,
                                        "cannot infer tensor rank for perm"))
            elif not _is_complete_permutation(perm, rank):
                st.valid = False
                code = E_PERM_INVALID
                failures.append(Failure(
                    idx, nid, "validate", code,
                    "perm must be a 0-based complete permutation matching rank",
                    injection_id=inj["id"] if inj else None,
                    detail={"perm": perm, "rank": rank,
                            "injection": inj} if inj else
                    {"perm": perm, "rank": rank}))
            else:
                st.perm = perm

    # 原地写回冲突：仅当写回会破坏其他节点仍在引用的输入张量，或形状改变。
    # 外部输入在执行期使用私有副本（copy），原地写回不污染原始输入，故允许。
    for nid in order:
        st = nodes[nid]
        if not st.spec.get("inplace", False):
            continue
        idx = st.topo_index
        if st.producer is not None and len(nodes[st.producer].consumers) > 1:
            st.valid = False
            failures.append(Failure(idx, nid, "validate", E_INPLACE_CONFLICT,
                                    "write-back would break another node's "
                                    "input reference (fan-out)"))
        elif st.op in TRANSPOSE_OPS:
            st.valid = False
            failures.append(Failure(idx, nid, "validate", E_INPLACE_CONFLICT,
                                    "transpose changes shape, cannot write "
                                    "back in place"))

    # 融合候选：激活 A -> 转置 T（T 的唯一入边来自 A）
    fusion_candidates = []
    for nid in order:
        st = nodes[nid]
        if st.op in TRANSPOSE_OPS and st.producer is not None:
            a = nodes[st.producer]
            if a.op in ACTIVATION_OPS and a.valid and st.valid:
                fusion_candidates.append({
                    "activation": a.node_id,
                    "transpose": st.node_id,
                    "topo_index": st.topo_index,
                })

    failures.sort(key=lambda f: (f.topo_index, f.node_id, f.code))
    return Graph(nodes=nodes, order=order, failures=failures,
                 input_shapes=dict(input_shapes),
                 fusion_candidates=fusion_candidates, edges=clean_edges)


def _injection_for(injections, node_id, stage, fault_type):
    for inj in injections:
        if inj.get("node_id") == node_id and inj.get("stage") == stage and \
                inj.get("fault") == fault_type:
            return inj
    return None
