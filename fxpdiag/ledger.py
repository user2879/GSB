"""行级缓存条目与复用账本（含复用/重算依据）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fingerprints import cache_key, conversion_cache_key


@dataclass
class RowCacheEntry:
    layer: str
    row_index: int
    version: str
    upstream_rows: list[tuple[str, int]]
    activation_fp: str
    param_fp: str
    config_fp: str
    keys: dict[str, str]
    result: dict
    cross_layer: dict

    def matches(
        self,
        activation_fingerprint: str,
        parameter_fingerprint: str,
        configuration_fingerprint: str,
        upstream_fingerprints: dict[tuple[str, int], str],
        conversion_key: str,
    ) -> bool:
        if self.activation_fp != activation_fingerprint:
            return False
        if self.param_fp != parameter_fingerprint:
            return False
        if self.config_fp != configuration_fingerprint:
            return False
        for upstream in self.upstream_rows:
            current = upstream_fingerprints.get(upstream)
            stored = self.keys.get(f"up:{upstream[0]}:{upstream[1]}")
            if current is None or current != stored:
                return False
        if self.keys.get("version_conversion") != conversion_key:
            return False
        return True


@dataclass
class LedgerDecision:
    layer: str
    row_index: int
    version: str
    action: str               # REUSE / RECOMPUTE / PENDING
    reason: str
    cache_key: str
    upstream_rows: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "row_index": self.row_index,
            "version": self.version,
            "action": self.action,
            "reason": self.reason,
            "cache_key": self.cache_key,
            "upstream_rows": [list(item) for item in self.upstream_rows],
        }


class ReuseLedger:
    """按稳定顺序记录每行的复用/重算/挂起决策。"""

    def __init__(self) -> None:
        self.decisions: list[LedgerDecision] = []
        self.reused = 0
        self.recomputed = 0
        self.pending = 0

    def record(self, decision: LedgerDecision) -> None:
        self.decisions.append(decision)
        if decision.action == "REUSE":
            self.reused += 1
        elif decision.action == "RECOMPUTE":
            self.recomputed += 1
        else:
            self.pending += 1

    def to_dicts(self) -> list[dict]:
        return [decision.to_dict() for decision in self.decisions]

    def summary(self) -> dict:
        return {
            "reused": self.reused,
            "recomputed": self.recomputed,
            "pending": self.pending,
            "total": len(self.decisions),
        }


def build_stage_keys(
    layer: str,
    row_index: int,
    version: str,
    activation_fingerprint: str,
    parameter_fingerprint: str,
    quant_key: str,
    upstream_fingerprints: dict[tuple[str, int], str],
    weight_fingerprint: str,
    sidecar_fingerprint: str | None,
) -> dict[str, str]:
    common_upstream = [activation_fingerprint, *upstream_fingerprints.values()]
    keys = {
        "softmax": cache_key(
            "softmax", [activation_fingerprint, *upstream_fingerprints.values()],
            parameter_fingerprint, quant_key, version
        ),
        "ln_stats": cache_key(
            "ln_stats", common_upstream, parameter_fingerprint, quant_key, version
        ),
        "ln_output": cache_key(
            "ln_output", common_upstream, parameter_fingerprint, quant_key, version
        ),
        "version_conversion": conversion_cache_key(
            layer, row_index, weight_fingerprint, version, sidecar_fingerprint
        ),
    }
    for upstream, fingerprint in upstream_fingerprints.items():
        keys[f"up:{upstream[0]}:{upstream[1]}"] = fingerprint
    return keys
