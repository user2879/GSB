"""恢复后的一致性重放校验（E_REPLAY）。

依据声明与变更集序列重新推导每个已提交批次的失效集合、误差界与
逐节点期望回执，同账本逐项比较；首个不一致以 E_REPLAY 定位到
节点、配置键与字段。
"""

from .errors import ReplayError
from . import sample
from .compiler import compile_all
from .converge import expected_view
from .graph import compute_invalidation, node_subscription_map, topo_order
from .quantize import aggregation_part, chain_error_bounds
from .spec import apply_changeset, initial_state


def compute_error_bounds(state, decl, topo, scale_exp, groups):
    from .quantize import quantize
    derived_from = {k: it.derived_from for k, it in state.items.items()}
    coeff_q = {k: quantize(it.derived_coeff, scale_exp)
               for k, it in state.items.items() if it.derived_from}
    chain = chain_error_bounds(topo, derived_from, coeff_q, scale_exp)
    node_by_id = {nd.node_id: nd for nd in decl.nodes}
    bounds = {}
    for key in topo:
        sizes = [node_by_id[nid].sample_size for nid in state.subs.get(key, [])]
        agg = aggregation_part(sizes, groups) if sizes else 0
        tr, de, chain_total = chain[key]
        bounds[key] = (tr, de, agg, chain_total + agg)
    return bounds


def replay_verify(ledger, decl, changesets):
    """对账本中全部已提交批次执行重放校验；差异抛出 ReplayError。"""
    scale_exp = decl.session.scale_exp
    groups = decl.session.groups
    info = ledger.recovery_scan()
    committed = {b["batch_id"]: b for b in info.batches}
    if not committed:
        return

    state = initial_state(decl)
    topo = topo_order(list(state.items), {k: it.derived_from
                                          for k, it in state.items.items()})
    batches_by_seq = sorted(info.batches, key=lambda b: b["seq"])
    carried = {}
    for batch in batches_by_seq:
        batch_id = batch["batch_id"]
        seq = batch["seq"]
        if seq == 0:
            # 初始全量批次
            artifacts = compile_all(state, scale_exp, topo)
            node_subs = node_subscription_map(state)
            expected_steps = []
        else:
            cs = changesets[seq - 1]
            inv = compute_invalidation(state, cs, node_subscription_map(
                apply_changeset(state, cs)))
            state = apply_changeset(state, cs)
            artifacts = compile_all(state, scale_exp, topo)
            node_subs = node_subscription_map(state)
            expected_steps = [
                {"step": s.order, "kind": s.kind, "name": s.name,
                 "reason": s.reason}
                for s in inv.steps
            ]
        # 1) 失效传播轨迹
        got_steps = ledger.invalidation_of(batch_id)
        if got_steps != expected_steps:
            first = next((i for i, (a, b) in enumerate(
                zip(got_steps, expected_steps)) if a != b),
                min(len(got_steps), len(expected_steps)))
            name = (got_steps[first]["name"] if first < len(got_steps)
                    else expected_steps[first]["name"])
            raise ReplayError("invalidation trace mismatch at step %d" % first,
                              node=batch_id, key=name, field="invalidation")
        # 2) 误差界
        exp_bounds = compute_error_bounds(state, decl, topo, scale_exp, groups)
        got_bounds = ledger.error_bounds_of(batch_id)
        if got_bounds != exp_bounds:
            for key in sorted(set(got_bounds) | set(exp_bounds)):
                if got_bounds.get(key) != exp_bounds.get(key):
                    raise ReplayError("error bound mismatch",
                                      node=batch_id, key=key,
                                      field="error_bounds")
        # 3) 逐节点回执
        got_receipts = {r["node_id"]: r for r in ledger.receipts_of(batch_id)}
        for node in sorted(decl.nodes, key=lambda n: n.node_id):
            nid = node.node_id
            got = got_receipts.get(nid)
            if got is None:
                raise ReplayError("missing receipt record", node=nid,
                                  key="*", field="receipt")
            view = expected_view(state.items, node_subs.get(nid, []),
                                 artifacts, scale_exp)
            exp_digest = sample.view_digest(view, node.seed, node.sample_size,
                                            scale_exp, groups)
            exp_version = sample.config_version_of(view)
            if got["digest"] != exp_digest:
                raise ReplayError("receipt digest mismatch", node=nid,
                                  key="*", field="digest")
            if got["config_version"] != exp_version:
                raise ReplayError("receipt config version mismatch", node=nid,
                                  key="*", field="config_version")
            if got["source"] == "carried":
                basis = carried.get(nid)
                if basis is None or got["basis_version"] != basis["config_version"]:
                    raise ReplayError("carried basis version mismatch", node=nid,
                                      key="*", field="basis_version")
            carried[nid] = {"digest": got["digest"],
                            "config_version": got["config_version"]}
