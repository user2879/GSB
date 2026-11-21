"""分层策略清单、对象清单与种子参数的加载与强校验。

校验原则：
- bool 不被视为 int；浮点、负数、缺失类型全部拒绝
- 层标识/对象键唯一，阈值为 0..1000 的千分位整数
- 内容摘要必须与按固定整数种子确定性生成的内容一致
- 任一不合法：E_SPEC 给出首个违规字段与原因，整体拒绝，不做部分装载
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .content import expected_digest
from .errors import ErrorCollector


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonneg_int(value: Any) -> bool:
    return _is_int(value) and value >= 0


def _load_json(path: str | Path, collector: ErrorCollector,
               field_path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        collector.add(field_path, f"文件不存在: {p}")
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        collector.add(field_path, f"JSON 解析失败: {exc.msg} (line {exc.lineno})")
        return None
    if not isinstance(data, dict):
        collector.add(field_path, "顶层结构必须为 JSON 对象")
        return None
    return data


@dataclass(frozen=True)
class TierSpec:
    tier_id: str
    role: str  # hot | cold
    capacity_bytes: int
    unit_cost: int
    rw_weight: int
    eviction_priority: int


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    size_bytes: int
    last_access_ms: int
    access_count: int
    content_digest: str


@dataclass(frozen=True)
class Rules:
    hot_watermark_permille: int
    cold_reclaim_permille: int
    batch_limit: int
    heat_weight_recency: int
    heat_weight_frequency: int
    on_capacity: str  # reject | rewrite

    @property
    def hot_watermark(self) -> int:
        return self.hot_watermark_permille

    @property
    def cold_reclaim(self) -> int:
        return self.cold_reclaim_permille


@dataclass(frozen=True)
class Manifest:
    spec_version: int
    seed: int
    backend: str
    tiers: tuple[TierSpec, ...]
    objects: tuple[ObjectSpec, ...]
    rules: Rules
    tier_digest: str
    object_digest: str
    rules_digest: str
    spec_digest: str

    def tier(self, tier_id: str) -> TierSpec:
        for tier in self.tiers:
            if tier.tier_id == tier_id:
                return tier
        raise KeyError(tier_id)

    @property
    def hot_tier(self) -> TierSpec:
        return next(t for t in self.tiers if t.role == "hot")

    @property
    def cold_tier(self) -> TierSpec:
        return next(t for t in self.tiers if t.role == "cold")


def _check_permille(collector: ErrorCollector, field_path: str, value: Any) -> None:
    if not _nonneg_int(value) or value > 1000:
        collector.add(field_path, "必须为 0..1000 的整数（千分位阈值）")


def load_manifest(tiers_path: str | Path, objects_path: str | Path,
                  seed_path: str | Path) -> Manifest:
    collector = ErrorCollector()
    tiers_doc = _load_json(tiers_path, collector, "tiers_file")
    objects_doc = _load_json(objects_path, collector, "objects_file")
    seed_doc = _load_json(seed_path, collector, "seed_file")
    # JSON 本身无法解析时后续无从校验，直接整体拒绝
    collector.check()

    seed = _validate_seed(seed_doc, collector)
    backend = _validate_backend(seed_doc, collector)
    tiers = _validate_tiers(tiers_doc, collector)
    rules = _validate_rules(tiers_doc, collector)
    objects = _validate_objects(objects_doc, collector, seed, backend)

    # 容量前置约束：初始热层放置必须可行（计划推导本身另做运行期容量检查）
    if tiers and objects:
        hot = next((t for t in tiers if t.role == "hot"), None)
        if hot is not None:
            total = sum(o.size_bytes for o in objects)
            if total > hot.capacity_bytes:
                collector.add(
                    "tiers[hot].capacity_bytes",
                    f"对象总字节 {total} 超过热层容量 {hot.capacity_bytes}，"
                    "初始放置不可行",
                )
    collector.check()
    assert tiers is not None and rules is not None and objects is not None

    tier_digest, rules_digest, object_digest, spec_digest = _manifest_digests(
        tiers_doc, objects_doc, seed_doc)
    spec_version = tiers_doc.get("spec_version", 1)
    if not _is_int(spec_version) or spec_version < 1:
        collector.add("tiers_file.spec_version", "必须为 >=1 的整数")
        collector.check()

    return Manifest(
        spec_version=int(spec_version),
        seed=seed,
        backend=backend,
        tiers=tuple(tiers),
        objects=tuple(objects),
        rules=rules,
        tier_digest=tier_digest,
        object_digest=object_digest,
        rules_digest=rules_digest,
        spec_digest=spec_digest,
    )


def _validate_seed(doc: dict[str, Any], collector: ErrorCollector) -> int:
    seed = doc.get("seed")
    if not _nonneg_int(seed):
        collector.add("seed_file.seed", "必须为非负整数")
        return 0
    return int(seed)


def _validate_backend(doc: dict[str, Any], collector: ErrorCollector) -> str:
    backend = doc.get("backend", "python")
    if not isinstance(backend, str) or backend not in ("python", "numpy"):
        collector.add("seed_file.backend", "必须为 python 或 numpy")
        return "python"
    return backend


def _validate_tiers(doc: dict[str, Any],
                    collector: ErrorCollector) -> list[TierSpec] | None:
    raw = doc.get("tiers")
    if not isinstance(raw, list) or not raw:
        collector.add("tiers_file.tiers", "必须为非空数组")
        return None
    seen: set[str] = set()
    roles: set[str] = set()
    result: list[TierSpec] = []
    for index, item in enumerate(raw):
        base = f"tiers_file.tiers[{index}]"
        if not isinstance(item, dict):
            collector.add(base, "层定义必须为对象")
            continue
        tier_id = item.get("tier_id")
        if not isinstance(tier_id, str) or not tier_id:
            collector.add(f"{base}.tier_id", "必须为非空字符串")
        elif tier_id in seen:
            collector.add(f"{base}.tier_id", f"层标识重复: {tier_id}")
        else:
            seen.add(tier_id)
        role = item.get("role")
        if role not in ("hot", "cold"):
            collector.add(f"{base}.role", "必须为 hot 或 cold")
        else:
            if role in roles:
                collector.add(f"{base}.role", f"角色 {role} 的层只能有一个")
            roles.add(role)
        if not _nonneg_int(item.get("capacity_bytes")):
            collector.add(f"{base}.capacity_bytes", "必须为非负整数")
        if not _nonneg_int(item.get("unit_cost")):
            collector.add(f"{base}.unit_cost", "必须为非负整数")
        if not _nonneg_int(item.get("rw_weight")):
            collector.add(f"{base}.rw_weight", "必须为非负整数")
        if not _nonneg_int(item.get("eviction_priority")):
            collector.add(f"{base}.eviction_priority", "必须为非负整数")
        if all(k in item for k in ("tier_id", "role", "capacity_bytes",
                                   "unit_cost", "rw_weight", "eviction_priority")) \
                and isinstance(tier_id, str) and tier_id and role in ("hot", "cold"):
            result.append(TierSpec(
                tier_id=tier_id, role=role,
                capacity_bytes=int(item["capacity_bytes"]),
                unit_cost=int(item["unit_cost"]),
                rw_weight=int(item["rw_weight"]),
                eviction_priority=int(item["eviction_priority"]),
            ))
    if roles != {"hot", "cold"}:
        collector.add("tiers_file.tiers", "必须恰好包含一个 hot 层与一个 cold 层")
    return result


def _validate_rules(doc: dict[str, Any],
                    collector: ErrorCollector) -> Rules | None:
    raw = doc.get("rules")
    if not isinstance(raw, dict):
        collector.add("tiers_file.rules", "必须为对象")
        return None
    _check_permille(collector, "rules.hot_watermark_permille",
                    raw.get("hot_watermark_permille"))
    _check_permille(collector, "rules.cold_reclaim_permille",
                    raw.get("cold_reclaim_permille"))
    batch = raw.get("batch_limit")
    if not _is_int(batch) or batch <= 0:
        collector.add("rules.batch_limit", "必须为正整数")
    wr = raw.get("heat_weight_recency", 1)
    wf = raw.get("heat_weight_frequency", 1)
    if not _nonneg_int(wr):
        collector.add("rules.heat_weight_recency", "必须为非负整数")
    if not _nonneg_int(wf):
        collector.add("rules.heat_weight_frequency", "必须为非负整数")
    on_capacity = raw.get("on_capacity", "reject")
    if on_capacity not in ("reject", "rewrite"):
        collector.add("rules.on_capacity", "必须为 reject 或 rewrite")
    if collector.violations:
        return None
    return Rules(
        hot_watermark_permille=int(raw["hot_watermark_permille"]),
        cold_reclaim_permille=int(raw["cold_reclaim_permille"]),
        batch_limit=int(batch),
        heat_weight_recency=int(wr),
        heat_weight_frequency=int(wf),
        on_capacity=str(on_capacity),
    )


def _validate_objects(doc: dict[str, Any], collector: ErrorCollector,
                      seed: int, backend: str) -> list[ObjectSpec] | None:
    raw = doc.get("objects")
    if not isinstance(raw, list) or not raw:
        collector.add("objects_file.objects", "必须为非空数组")
        return None
    seen: set[str] = set()
    result: list[ObjectSpec] = []
    for index, item in enumerate(raw):
        base = f"objects_file.objects[{index}]"
        if not isinstance(item, dict):
            collector.add(base, "对象定义必须为对象")
            continue
        key = item.get("key")
        if not isinstance(key, str) or not key:
            collector.add(f"{base}.key", "必须为非空字符串")
        elif key in seen:
            collector.add(f"{base}.key", f"对象键重复: {key}")
        else:
            seen.add(key)
        size = item.get("size_bytes")
        if not _is_int(size) or size <= 0:
            collector.add(f"{base}.size_bytes", "必须为正整数")
        ts = item.get("last_access_ms")
        if not _nonneg_int(ts):
            collector.add(f"{base}.last_access_ms", "必须为非负整数")
        count = item.get("access_count")
        if not _nonneg_int(count):
            collector.add(f"{base}.access_count", "必须为非负整数")
        digest = item.get("content_digest")
        if not isinstance(digest, str) or not digest:
            collector.add(f"{base}.content_digest", "必须为非空字符串摘要")
        elif _is_int(size) and size > 0 and isinstance(key, str) and key:
            expected = expected_digest(seed, key, int(size), backend)
            if digest != expected:
                collector.add(
                    f"{base}.content_digest",
                    f"摘要与按种子 {seed} 生成的内容不一致 "
                    f"(期望 {expected})",
                )
        if all(isinstance(key, str) and key and _is_int(size) and size > 0
               and _nonneg_int(ts) and _nonneg_int(count)
               and isinstance(digest, str) and digest for _ in (0,)):
            result.append(ObjectSpec(
                key=key, size_bytes=int(size),
                last_access_ms=int(ts), access_count=int(count),
                content_digest=digest,
            ))
    return result


def canonical_json(data: Any) -> bytes:
    """规范序列化：sort_keys + 固定分隔符，保证逐字节一致。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _manifest_digests(tiers_doc: dict[str, Any], objects_doc: dict[str, Any],
                      seed_doc: dict[str, Any]):
    import hashlib

    def digest(data: Any) -> str:
        return hashlib.sha256(canonical_json(data)).hexdigest()

    tier_digest = digest(tiers_doc.get("tiers", []))
    rules_digest = digest(tiers_doc.get("rules", {}))
    object_digest = digest(objects_doc.get("objects", []))
    spec_digest = digest({
        "seed_doc": seed_doc,
        "tiers_doc": tiers_doc,
        "objects_doc": objects_doc,
    })
    return tier_digest, rules_digest, object_digest, spec_digest


def manifest_from_validated(manifest: Manifest) -> dict[str, Any]:
    """把已校验清单转回规范结构，供账本与重放使用。"""
    return {
        "spec_version": manifest.spec_version,
        "seed": manifest.seed,
        "backend": manifest.backend,
        "tiers": [t.__dict__ for t in manifest.tiers],
        "rules": manifest.rules.__dict__,
        "objects": [o.__dict__ for o in manifest.objects],
        "spec_digest": manifest.spec_digest,
    }
