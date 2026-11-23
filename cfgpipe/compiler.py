"""配置切片编译：生成逐字节稳定的下发产物。

产物为规范化 JSON（键排序、无空白），产物版本 = 内容 SHA-256，
因此未受影响项的产物与版本天然逐字节不变。
"""

import hashlib
import json

from .quantize import derived_value, quantize


def canonical_bytes(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def effective_values(state, scale_exp, topo):
    """按拓扑序计算每项有效定点值：根项取量化声明值，派生项取
    sat(trunc(coeff_q * parent_q / scale))。返回 {key: value_q}。"""
    values = {}
    for key in topo:
        it = state.items[key]
        if it.derived_from:
            coeff_q = quantize(it.derived_coeff, scale_exp)
            values[key] = derived_value(coeff_q, values[it.derived_from], scale_exp)
        else:
            values[key] = quantize(it.value, scale_exp)
    return values


def compile_item(state, key, value_q, scale_exp):
    """编译单个配置项的下发产物，返回规范化字节。"""
    it = state.items[key]
    coeff_q = quantize(it.derived_coeff, scale_exp) if it.derived_from else 0
    doc = {
        "key": key,
        "version": it.version,
        "value_q": value_q,
        "expire_ms": it.expire_ms,
        "derived_from": it.derived_from,
        "coeff_q": coeff_q,
        "scale_exp": scale_exp,
    }
    return canonical_bytes(doc)


def compile_all(state, scale_exp, topo):
    """编译全部配置项，返回 {key: (artifact_bytes, artifact_version)}。"""
    values = effective_values(state, scale_exp, topo)
    out = {}
    for key in topo:
        blob = compile_item(state, key, values[key], scale_exp)
        out[key] = (blob, sha256_hex(blob))
    return out


def dispatch_payload(batch_id, scale_exp, groups, upsert_blobs, remove_keys):
    """构造单次下发的线路字节（重试时逐字节不变）。

    upsert_blobs: 受影响项的产物字节列表（按 key 排序由调用方保证）。
    remove_keys: 需要节点从视图中移除的键（退订）。
    """
    upsert = [json.loads(b.decode("utf-8")) for b in upsert_blobs]
    doc = {
        "type": "dispatch",
        "batch_id": batch_id,
        "scale_exp": scale_exp,
        "groups": groups,
        "upsert": upsert,
        "remove": sorted(remove_keys),
    }
    return canonical_bytes(doc)
