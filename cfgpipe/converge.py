"""增量回执收敛判定（E_CONVERGE）。

- 仅对存在回执变化的（受影响）节点触发逐字节比对：
  回执摘要、逐键聚合摘要、所用口径版本（scale_exp）与配置版本。
- 未受影响节点标记为沿用上一轮回执（source="carried"），保留依据版本；
  并用当前世界状态重算其期望摘要，验证沿用仍然成立。
即视图未漂移。
- 偏差报告稳定排序：节点、配置键、字段均按字典序，首个偏差即抛出。
"""

from .errors import ConvergeError
from . import sample


def expected_view(state_items, node_keys, artifacts, scale_exp):
    """节点期望视图：{key: item payload dict}（来自当前产物）。"""
    import json
    view = {}
    for key in sorted(node_keys):
        blob, _ = artifacts[key]
        view[key] = json.loads(blob.decode("utf-8"))
    return view


def expected_digest(view, node, scale_exp, groups):
    return sample.view_digest(view, node.seed, node.sample_size, scale_exp, groups)


def expected_key_digests(view, node, scale_exp, groups):
    return sample.key_digests(view, node.seed, node.sample_size, scale_exp, groups)


def check_convergence(nodes, affected_nodes, receipts, expected_views,
                      carried_basis, scale_exp, groups):
    """返回 {node_id: receipt记录}（按节点字典序构造）。

    nodes: NodeSpec 列表；affected_nodes: set；
    receipts: {node_id: 回执 dict}；expected_views: {node_id: view}；
    carried_basis: {node_id: {"digest":..,"config_version":..}}（账本最近回执）。
    """
    out = {}
    for node in sorted(nodes, key=lambda n: n.node_id):
        nid = node.node_id
        view = expected_views[nid]
        exp_digest = sample.view_digest(view, node.seed, node.sample_size,
                                        scale_exp, groups)
        exp_version = sample.config_version_of(view)
        if nid in affected_nodes:
            receipt = receipts.get(nid)
            if receipt is None:
                raise ConvergeError("node did not return a receipt",
                                    node=nid, key="*", field="receipt")
            if receipt.get("scale_exp") != scale_exp:
                raise ConvergeError("quantization scale version mismatch",
                                    node=nid, key="*", field="scale_exp")
            if receipt.get("config_version") != exp_version:
                raise ConvergeError("config version mismatch",
                                    node=nid, key="*", field="config_version")
            if receipt.get("digest") != exp_digest:
                exp_kd = sample.key_digests(view, node.seed, node.sample_size,
                                            scale_exp, groups)
                got_kd = receipt.get("key_digests", {})
                first_key = "*"
                for key in sorted(set(exp_kd) | set(got_kd)):
                    if exp_kd.get(key) != got_kd.get(key):
                        first_key = key
                        break
                raise ConvergeError("aggregate digest mismatch",
                                    node=nid, key=first_key, field="digest")
            out[nid] = {
                "node_id": nid, "source": "new", "digest": receipt["digest"],
                "config_version": receipt["config_version"], "basis_version": None,
                "elapsed_ms": receipt.get("elapsed_ms", 0),
                "attempts": receipt.get("_attempts", 1),
                "wait_ms": receipt.get("_wait_ms", 0),
            }
        else:
            basis = carried_basis.get(nid)
            if basis is None:
                raise ConvergeError("no carried-over receipt basis for node",
                                    node=nid, key="*", field="basis_version")
            if basis["digest"] != exp_digest:
                raise ConvergeError(
                    "carried-over view drifted from expected digest",
                    node=nid, key="*", field="carried_digest")
            out[nid] = {
                "node_id": nid, "source": "carried", "digest": basis["digest"],
                "config_version": basis["config_version"],
                "basis_version": basis["config_version"],
                "elapsed_ms": 0, "attempts": 0, "wait_ms": 0,
            }
    return out
