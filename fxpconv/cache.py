"""基于内容指纹的输出位置级缓存与精确失效。

每个位置的三个阶段（整数累加 / 饱和截断 / 反量化）分别派生缓存键。
键中只纳入“真实影响该阶段取值”的策略指纹：未发生溢出的位置
不携带饱和策略标签，舍入结果稳定的位置不携带舍入模式标签。
因此替换输入张量、切换饱和策略或舍入模式时，只有取值真的会
变化的位置及其下游进入重算集合，其余位置直接复用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fingerprint import combine_fingerprints, content_fingerprint
from .models import OpDesc


def scale_fingerprint(desc: OpDesc) -> str:
    payload = {
        "act_scale": desc.act_scale,
        "weight_scales": desc.weight_scales,
    }
    return content_fingerprint(payload)


def _buffer_policy_tag(raw_sum: int, acc_bits: int, saturation: str) -> str:
    lo = -(1 << (acc_bits - 1))
    hi = (1 << (acc_bits - 1)) - 1
    if lo <= raw_sum <= hi:
        return "buf:inrange"
    return f"buf:{saturation}"


def _trunc_policy_tag(buffered: int, qbits: int, saturation: str) -> str:
    lo = -(1 << (qbits - 1))
    hi = (1 << (qbits - 1)) - 1
    if buffered == lo or buffered == hi:
        return "trunc:boundary"
    if lo < buffered < hi:
        return "trunc:inrange"
    return f"trunc:{saturation}"


def _rounding_tag(scaled: float, step: float, rounding: str) -> str:
    index_float = scaled / step
    if float(int(index_float)) == index_float and math.isfinite(index_float):
        return "r:exact"
    trunc_zero = int(math.copysign(math.floor(abs(index_float)), index_float))
    floor_index = math.floor(index_float)
    nearest = int(round(index_float))
    if trunc_zero == floor_index == nearest:
        return "r:stable"
    return f"r:{rounding}"


def build_accum_key(
    desc: OpDesc,
    window_fingerprint: str,
    act_row_ids: tuple[str, ...],
    weight_row_ids: tuple[str, ...],
    raw_sum: int,
    *,
    acc_bits: int | None = None,
    saturation: str | None = None,
) -> str:
    acc_bits = acc_bits or desc.acc_bits
    saturation = saturation or desc.saturation
    return combine_fingerprints(
        "accum",
        window_fingerprint,
        combine_fingerprints(*sorted(act_row_ids)),
        combine_fingerprints(*sorted(weight_row_ids)),
        f"acc_bits={acc_bits}",
        _buffer_policy_tag(raw_sum, acc_bits, saturation),
    )


def build_trunc_key_with_desc(
    desc: OpDesc,
    accum_key: str,
    buffered: int,
    *,
    qbits: int | None = None,
    saturation: str | None = None,
) -> str:
    qbits = qbits or desc.qbits
    saturation = saturation or desc.saturation
    return combine_fingerprints(
        "trunc",
        accum_key,
        f"qbits={qbits}",
        _trunc_policy_tag(buffered, qbits, saturation),
    )


def build_deq_key(
    trunc_key: str,
    scaled: float,
    step: float,
    rounding: str,
    frac_bits: int,
    scale_fp: str,
) -> str:
    return combine_fingerprints(
        "deq",
        trunc_key,
        f"frac={frac_bits}",
        scale_fp,
        _rounding_tag(scaled, step, rounding),
    )


@dataclass
class CacheEntry:
    position: int
    accum_key: str
    trunc_key: str
    deq_key: str
    payload: dict

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "accum_key": self.accum_key,
            "trunc_key": self.trunc_key,
            "deq_key": self.deq_key,
            "payload": self.payload,
        }


@dataclass
class PositionCache:
    entries: dict[int, CacheEntry] = field(default_factory=dict)
    reused: list[int] = field(default_factory=list)
    recomputed: list[int] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)

    def lookup(
        self,
        position: int,
        accum_key: str,
        trunc_key: str,
        deq_key: str,
    ) -> CacheEntry | None:
        entry = self.entries.get(position)
        if entry is None:
            return None
        if entry.accum_key == accum_key and entry.trunc_key == trunc_key and entry.deq_key == deq_key:
            return entry
        return None

    def record_reuse(self, position: int) -> None:
        self.reused.append(position)

    def record_recompute(self, position: int, reason: str) -> None:
        self.recomputed.append(position)
        self.reasons[position] = reason

    def put(self, entry: CacheEntry) -> None:
        self.entries[entry.position] = entry

    def invalidate_positions(self, positions: set[int]) -> None:
        for position in positions:
            self.entries.pop(position, None)

    def snapshot(self) -> dict:
        return {
            "entries": [entry.to_dict() for entry in sorted(self.entries.values(), key=lambda e: e.position)],
            "reused": sorted(self.reused),
            "recomputed": sorted(self.recomputed),
            "reasons": {str(key): self.reasons[key] for key in sorted(self.reasons)},
        }

    @classmethod
    def restore(cls, data: dict) -> "PositionCache":
        cache = cls()
        for item in data.get("entries", []):
            cache.entries[item["position"]] = CacheEntry(
                position=item["position"],
                accum_key=item["accum_key"],
                trunc_key=item["trunc_key"],
                deq_key=item["deq_key"],
                payload=item["payload"],
            )
        cache.reused = list(data.get("reused", []))
        cache.recomputed = list(data.get("recomputed", []))
        cache.reasons = {int(key): value for key, value in data.get("reasons", {}).items()}
        return cache
