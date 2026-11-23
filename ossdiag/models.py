from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError


KiB = 1024
MiB = 1024 * 1024

# The size set is intentionally finite and documented.  It keeps integer batch
# boundaries, fixed leaf sizes, and the three aggregation paths predictable.
SUPPORTED_SHARD_SIZES = tuple(size * KiB for size in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536))
SUPPORTED_BATCH_SIZES = tuple(size * KiB for size in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(
    value: Any,
    *,
    object_key: str,
    field: str,
    positive: bool = True,
    minimum: int | None = None,
) -> int:
    if not _is_int(value):
        raise SpecError(
            f"字段 {field} 必须是整数",
            object_key=object_key,
            field=field,
            detail={"value_type": type(value).__name__},
        )
    if positive and value <= 0:
        raise SpecError(
            f"字段 {field} 必须是正整数",
            object_key=object_key,
            field=field,
            detail={"value": value},
        )
    if minimum is not None and value < minimum:
        raise SpecError(
            f"字段 {field} 不得小于 {minimum}",
            object_key=object_key,
            field=field,
            detail={"value": value, "minimum": minimum},
        )
    return value


def _require_str(value: Any, *, object_key: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(
            f"字段 {field} 必须是非空字符串",
            object_key=object_key,
            field=field,
            detail={"value_type": type(value).__name__},
        )
    return value


@dataclass(frozen=True)
class LeasePolicy:
    lease_ms: int
    renew_interval_ms: int
    max_concurrent_leases: int

    def as_dict(self) -> dict[str, int]:
        return {
            "lease_ms": self.lease_ms,
            "renew_interval_ms": self.renew_interval_ms,
            "max_concurrent_leases": self.max_concurrent_leases,
        }


@dataclass(frozen=True)
class ObjectSpec:
    object_key: str
    total_bytes: int
    shard_size: int
    shard_seeds: tuple[int, ...]
    concurrency_limit: int
    lease_policy: LeasePolicy
    batch_size: int

    @property
    def shard_count(self) -> int:
        return math.ceil(self.total_bytes / self.shard_size)

    @property
    def effective_concurrency(self) -> int:
        return min(self.concurrency_limit, self.lease_policy.max_concurrent_leases)

    def shard_bounds(self, shard_index: int) -> tuple[int, int]:
        if shard_index < 0 or shard_index >= self.shard_count:
            raise IndexError(shard_index)
        start = shard_index * self.shard_size
        return start, min(start + self.shard_size, self.total_bytes)

    def shard_length(self, shard_index: int) -> int:
        start, end = self.shard_bounds(shard_index)
        return end - start

    def seed_for(self, shard_index: int) -> int:
        return self.shard_seeds[shard_index]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "object_key": self.object_key,
            "total_bytes": self.total_bytes,
            "shard_size": self.shard_size,
            "shard_seeds": list(self.shard_seeds),
            "concurrency_limit": self.concurrency_limit,
            "batch_size": self.batch_size,
            "lease_policy": self.lease_policy.as_dict(),
        }


@dataclass(frozen=True)
class SessionSpec:
    objects: tuple[ObjectSpec, ...]
    global_lease_quota: int
    throughput_budget_bytes: int
    scale_exponent: int
    tie_threshold_work: int
    worker_threads: int
    source_path: str | None = None

    @property
    def object_map(self) -> dict[str, ObjectSpec]:
        return {item.object_key: item for item in self.objects}

    @property
    def total_bytes(self) -> int:
        return sum(item.total_bytes for item in self.objects)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "global_lease_quota": self.global_lease_quota,
            "throughput_budget_bytes": self.throughput_budget_bytes,
            "scale_exponent": self.scale_exponent,
            "tie_threshold_work": self.tie_threshold_work,
            "worker_threads": self.worker_threads,
            "objects": [item.canonical_dict() for item in self.objects],
        }


def _session_block(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("session", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SpecError(
            "session 必须是对象",
            object_key="$session",
            field="session",
            detail={"value_type": type(value).__name__},
        )
    return value


def _session_value(document: dict[str, Any], block: dict[str, Any], name: str, default: Any) -> Any:
    if name in block:
        return block[name]
    if name in document:
        return document[name]
    return default


def _parse_lease_policy(raw: Any, *, object_key: str) -> LeasePolicy:
    if not isinstance(raw, dict):
        raise SpecError(
            "lease_policy 必须是对象",
            object_key=object_key,
            field="lease_policy",
        )
    lease_ms = _require_int(raw.get("lease_ms"), object_key=object_key, field="lease_policy.lease_ms")
    renew_interval_ms = _require_int(
        raw.get("renew_interval_ms"),
        object_key=object_key,
        field="lease_policy.renew_interval_ms",
    )
    max_leases = _require_int(
        raw.get("max_concurrent_leases"),
        object_key=object_key,
        field="lease_policy.max_concurrent_leases",
    )
    if renew_interval_ms >= lease_ms:
        raise SpecError(
            "lease_policy.renew_interval_ms 必须小于 lease_ms",
            object_key=object_key,
            field="lease_policy.renew_interval_ms",
            detail={"renew_interval_ms": renew_interval_ms, "lease_ms": lease_ms},
        )
    return LeasePolicy(
        lease_ms=lease_ms,
        renew_interval_ms=renew_interval_ms,
        max_concurrent_leases=max_leases,
    )


def _parse_object(raw: Any, *, index: int, session_batch_size: int) -> ObjectSpec:
    fallback_key = f"$objects[{index}]"
    if not isinstance(raw, dict):
        raise SpecError(
            "对象声明必须是 JSON 对象",
            object_key=fallback_key,
            field="$",
        )
    object_key_value = raw.get("object_key")
    object_key = object_key_value if isinstance(object_key_value, str) and object_key_value else fallback_key
    object_key = _require_str(raw.get("object_key"), object_key=object_key, field="object_key")

    total_bytes = _require_int(raw.get("total_bytes"), object_key=object_key, field="total_bytes")
    shard_size = _require_int(raw.get("shard_size"), object_key=object_key, field="shard_size")
    if shard_size not in SUPPORTED_SHARD_SIZES:
        raise SpecError(
            "shard_size 不在受支持集合内",
            object_key=object_key,
            field="shard_size",
            detail={"value": shard_size, "supported": list(SUPPORTED_SHARD_SIZES)},
        )

    expected_shards = math.ceil(total_bytes / shard_size)
    if "shard_count" in raw:
        declared = _require_int(raw.get("shard_count"), object_key=object_key, field="shard_count")
        if declared != expected_shards:
            raise SpecError(
                "shard_count 与 total_bytes/shard_size 不自洽",
                object_key=object_key,
                field="shard_count",
                detail={"declared": declared, "expected": expected_shards},
            )

    seeds_raw = raw.get("shard_seeds")
    if seeds_raw is None:
        seeds_raw = raw.get("content_seeds")
    if not isinstance(seeds_raw, list):
        raise SpecError(
            "shard_seeds 必须是整数数组",
            object_key=object_key,
            field="shard_seeds",
        )
    if len(seeds_raw) != expected_shards:
        raise SpecError(
            "shard_seeds 数量必须与分片数一致",
            object_key=object_key,
            field="shard_seeds",
            detail={"actual": len(seeds_raw), "expected": expected_shards},
        )
    seeds: list[int] = []
    for seed_index, seed in enumerate(seeds_raw):
        if not _is_int(seed):
            raise SpecError(
                f"分片种子必须是整数",
                object_key=object_key,
                field=f"shard_seeds[{seed_index}]",
                detail={"value_type": type(seed).__name__},
            )
        seeds.append(seed)

    concurrency_limit = _require_int(
        raw.get("concurrency_limit"),
        object_key=object_key,
        field="concurrency_limit",
    )
    lease_policy = _parse_lease_policy(raw.get("lease_policy"), object_key=object_key)

    batch_size = raw.get("batch_size", session_batch_size)
    batch_size = _require_int(batch_size, object_key=object_key, field="batch_size")
    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise SpecError(
            "batch_size 不在受支持集合内",
            object_key=object_key,
            field="batch_size",
            detail={"value": batch_size, "supported": list(SUPPORTED_BATCH_SIZES)},
        )

    return ObjectSpec(
        object_key=object_key,
        total_bytes=total_bytes,
        shard_size=shard_size,
        shard_seeds=tuple(seeds),
        concurrency_limit=concurrency_limit,
        lease_policy=lease_policy,
        batch_size=batch_size,
    )


def load_spec_data(document: Any) -> SessionSpec:
    if not isinstance(document, dict):
        raise SpecError(
            "规格文件顶层必须是 JSON 对象",
            object_key="$document",
            field="$",
        )
    block = _session_block(document)
    global_lease_quota = _require_int(
        _session_value(document, block, "global_lease_quota", 4),
        object_key="$session",
        field="global_lease_quota",
    )
    throughput_budget_bytes = _require_int(
        _session_value(document, block, "throughput_budget_bytes", 1024 * MiB),
        object_key="$session",
        field="throughput_budget_bytes",
    )
    scale_exponent = _require_int(
        _session_value(document, block, "scale_exponent", 6),
        object_key="$session",
        field="scale_exponent",
    )
    if scale_exponent > 9:
        raise SpecError(
            "scale_exponent 不得超过 9",
            object_key="$session",
            field="scale_exponent",
            detail={"value": scale_exponent},
        )
    tie_threshold_work = _require_int(
        _session_value(document, block, "tie_threshold_work", 32),
        object_key="$session",
        field="tie_threshold_work",
    )
    worker_threads = _require_int(
        _session_value(document, block, "worker_threads", 4),
        object_key="$session",
        field="worker_threads",
    )
    if worker_threads > 64:
        raise SpecError(
            "worker_threads 不得超过 64",
            object_key="$session",
            field="worker_threads",
            detail={"value": worker_threads},
        )
    session_batch_size = _session_value(document, block, "batch_size", 1024 * KiB)
    session_batch_size = _require_int(
        session_batch_size,
        object_key="$session",
        field="batch_size",
    )
    if session_batch_size not in SUPPORTED_BATCH_SIZES:
        raise SpecError(
            "batch_size 不在受支持集合内",
            object_key="$session",
            field="batch_size",
            detail={"value": session_batch_size, "supported": list(SUPPORTED_BATCH_SIZES)},
        )

    objects_raw = document.get("objects")
    if not isinstance(objects_raw, list) or not objects_raw:
        raise SpecError(
            "objects 必须是非空数组",
            object_key="$document",
            field="objects",
        )

    seen: set[str] = set()
    objects: list[ObjectSpec] = []
    for index, raw in enumerate(objects_raw):
        spec = _parse_object(raw, index=index, session_batch_size=session_batch_size)
        if spec.object_key in seen:
            raise SpecError(
                "对象键必须唯一",
                object_key=spec.object_key,
                field="object_key",
                detail={"duplicate_index": index},
            )
        seen.add(spec.object_key)
        objects.append(spec)

    return SessionSpec(
        objects=tuple(objects),
        global_lease_quota=global_lease_quota,
        throughput_budget_bytes=throughput_budget_bytes,
        scale_exponent=scale_exponent,
        tie_threshold_work=tie_threshold_work,
        worker_threads=worker_threads,
    )


def load_spec(path: str | Path) -> SessionSpec:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(
            "规格文件不存在",
            object_key="$document",
            field="path",
            detail={"path": str(source)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            "规格文件不是合法 JSON",
            object_key="$document",
            field="$",
            detail={"line": exc.lineno, "column": exc.colno, "message": exc.msg},
        ) from exc
    loaded = load_spec_data(document)
    return SessionSpec(
        objects=loaded.objects,
        global_lease_quota=loaded.global_lease_quota,
        throughput_budget_bytes=loaded.throughput_budget_bytes,
        scale_exponent=loaded.scale_exponent,
        tie_threshold_work=loaded.tie_threshold_work,
        worker_threads=loaded.worker_threads,
        source_path=str(source.resolve()),
    )
