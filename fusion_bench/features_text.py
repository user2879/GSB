"""文本断言编码：实体/属性断言映射到固定文本特征维度。

同一实体多次断言按 (entity, attribute, value) 字典序合并并记录合并来源；
断言缺失或字段为空时按配置指纹派生的默认向量补齐并标注补齐位置。
"""

import hashlib

import numpy as np

from .fixedpoint import div_round, quantize, scale_of

_DEFAULT_CACHE = {}


def default_vector(cfg):
    """默认补齐向量：由配置指纹确定性派生，禁止运行时随机状态。"""
    key = (cfg.text_dim, cfg.fingerprint)
    if key in _DEFAULT_CACHE:
        return _DEFAULT_CACHE[key]
    seed_material = f"{cfg.default_fill_seed_tag}|{cfg.fingerprint}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vec = rng.integers(0, cfg.scale, size=cfg.text_dim, dtype=np.int64)
    vec = quantize(vec, cfg.frac_bits, cfg.quant_levels, cfg.quant_range)
    _DEFAULT_CACHE[key] = vec
    return vec


def _slot(token, cfg):
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "big") % cfg.text_dim
    mag = 1 + int.from_bytes(digest[8:12], "big") % cfg.quant_levels
    mag_fixed = div_round(mag * scale_of(cfg.frac_bits), cfg.quant_levels)
    return idx, mag_fixed


def encode_text(assertions, cfg):
    """返回 (特征向量, 编码信息字典)。"""
    info = {
        "used_default": False,
        "filled_positions": [],
        "merge_sources": {},
        "conflicts": [],
        "n_assertions": 0,
    }
    if not assertions:
        info["used_default"] = True
        info["filled_positions"] = list(range(cfg.text_dim))
        return default_vector(cfg).copy(), info

    # 固定合并顺序：(entity, attribute, value) 字典序
    order = sorted(
        range(len(assertions)),
        key=lambda i: (str(assertions[i].get("entity", "")),
                       str(assertions[i].get("attribute", "")),
                       str(assertions[i].get("value", ""))),
    )
    vec = np.zeros(cfg.text_dim, dtype=np.int64)
    entity_slots = {}
    entity_attr_values = {}
    filled = set()
    n_effective = 0
    for rank, i in enumerate(order):
        item = assertions[i]
        entity = str(item.get("entity", ""))
        attribute = str(item.get("attribute", ""))
        value = item.get("value", "")
        entity_slots.setdefault(entity, []).append(i)  # 合并来源（原始序号）
        entity_attr_values.setdefault(entity, {}).setdefault(
            attribute, set()).add(str(value))
        if value is None or str(value) == "":
            # 字段为空：该断言槽位用默认向量补齐并标注位置
            idx, _ = _slot(f"{entity}|{attribute}", cfg)
            vec[idx] += default_vector(cfg)[idx]
            filled.add(idx)
        else:
            idx, mag = _slot(f"{entity}|{attribute}|{value}", cfg)
            vec[idx] += mag
        n_effective += 1

    # 冲突断言：同一实体同一属性出现多个不同取值 -> 固定折减
    conflict_entities = []
    for entity in sorted(entity_attr_values):
        for attribute in sorted(entity_attr_values[entity]):
            if len(entity_attr_values[entity][attribute]) > 1:
                conflict_entities.append(entity)
                break
    if conflict_entities:
        # 折减作用于冲突实体贡献过的槽位：重算非冲突部分 + 折减冲突部分
        reduced = np.zeros(cfg.text_dim, dtype=np.int64)
        for rank, i in enumerate(order):
            item = assertions[i]
            entity = str(item.get("entity", ""))
            attribute = str(item.get("attribute", ""))
            value = item.get("value", "")
            if value is None or str(value) == "":
                idx, _ = _slot(f"{entity}|{attribute}", cfg)
                mag = default_vector(cfg)[idx]
            else:
                idx, mag = _slot(f"{entity}|{attribute}|{value}", cfg)
            if entity in conflict_entities:
                mag = div_round(mag * cfg.conflict_num, cfg.conflict_den)
            reduced[idx] += mag
        vec = reduced
        info["conflicts"] = [
            {"entity": e, "reduction": [cfg.conflict_num, cfg.conflict_den]}
            for e in conflict_entities
        ]

    vec = div_round(vec, max(1, n_effective))  # 归一化
    vec = quantize(vec, cfg.frac_bits, cfg.quant_levels, cfg.quant_range)
    info["n_assertions"] = n_effective
    info["filled_positions"] = sorted(filled)
    info["merge_sources"] = {
        e: entity_slots[e] for e in sorted(entity_slots)
    }
    return vec, info
