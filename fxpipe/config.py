"""Typed configuration / data model and schema validation layer.

All numeric kernel work happens in fixed-point integers. The pipeline config
declares the fractional bit width (``frac_bits``) and the quantization gear
(``gear``). Gears are small integer numerator/denominator scale shifts so that
switching gear genuinely changes every intermediate integer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .errors import Code, IssueLog


class FracBits(int, Enum):
    FB_8 = 8
    FB_16 = 16
    FB_24 = 24


class Gear(str, Enum):
    G0 = "g0"
    G1 = "g1"
    G2 = "g2"
    G3 = "g3"


# Ordered integer scale shift per gear. A gear scales the q-format unit by
# numerator/denominator (integer math). g0 == identity.
GEAR_SHIFTS: dict[str, tuple[int, int]] = {
    Gear.G0.value: (1, 1),
    Gear.G1.value: (5, 4),
    Gear.G2.value: (3, 2),
    Gear.G3.value: (7, 4),
}

# Allowed fixed-point integer container widths (e1 field) and their name map.
ALLOWED_WIDTHS = (8, 16, 32)
WIDTH_TO_SCHEME = {8: "qint8", 16: "qint16", 32: "qint32"}
SCHEME_TO_WIDTH = {v: k for k, v in WIDTH_TO_SCHEME.items()}

# Named quantization schemes accepted on the e2 side.
ALLOWED_SCHEMES = tuple(WIDTH_TO_SCHEME.values())

# Permille field ranges for the fixed-width e1 record.
PERMILLE_MIN, PERMILLE_MAX = 0, 9_999_999


class OpType(str, Enum):
    EMBED = "embed"
    POOL = "segment_pool"
    ATTEND = "multihead_attn"
    NORM = "layer_norm"
    FINAL = "final"


@dataclass(frozen=True)
class QuantSpec:
    """Quantization descriptor shared by both exchange versions.

    ``permille`` is the canonical integer scaling factor (per-mille). ``width``
    is the fixed-point integer container width. Missing fields are represented
    explicitly with ``None`` plus a completeness tag instead of silent zeros.
    """

    permille: int | None
    width: int | None
    scheme: str | None
    precision_label: str
    completeness: str  # "complete" | "missing:<field>"
    residual: float = 0.0
    missing: tuple[str, ...] = field(default_factory=tuple)

    def equivalent(self, other: "QuantSpec") -> bool:
        """Canonical quant semantics: permille + width decide equivalence."""
        return self.permille == other.permille and self.width == other.width


@dataclass
class OpSpec:
    chain_id: int
    seq: int
    op_type: str
    inputs: tuple[int, ...]
    params: dict
    quant: QuantSpec
    source_version: str  # "e1" | "e2" | "merged"
    missing_fields: tuple[str, ...] = field(default_factory=tuple)

    def node_id(self) -> str:
        return f"c{self.chain_id}op{self.seq}"


@dataclass
class Chain:
    chain_id: int
    ops: list[OpSpec]

    def ordered_ops(self) -> list[OpSpec]:
        return sorted(self.ops, key=lambda op: op.seq)


@dataclass
class PipelineConfig:
    heads: int
    dim: int
    frac_bits: int
    gear: str
    point_budget: int
    operator_quota: int
    max_workers: int
    vocab_size: int
    batch_id: str = "batch-0"

    def validate(self, log: IssueLog, where: str = "config") -> bool:
        ok = True
        if not isinstance(self.heads, int) or self.heads <= 0:
            log.add(Code.E_SCHEMA, where, f"heads must be positive integer, got {self.heads!r}")
            ok = False
        if not isinstance(self.dim, int) or self.dim <= 0:
            log.add(Code.E_SCHEMA, where, f"dim must be positive integer, got {self.dim!r}")
            ok = False
        if ok and self.dim % self.heads != 0:
            log.add(Code.E_SCHEMA, where, f"dim {self.dim} not divisible by heads {self.heads}")
            ok = False
        if self.frac_bits not in tuple(int(v) for v in FracBits):
            log.add(Code.E_SCHEMA, where, f"frac_bits {self.frac_bits} not in {[int(v) for v in FracBits]}")
            ok = False
        if self.gear not in GEAR_SHIFTS:
            log.add(Code.E_SCHEMA, where, f"gear {self.gear!r} not in {list(GEAR_SHIFTS)}")
            ok = False
        for name in ("point_budget", "operator_quota", "max_workers", "vocab_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                log.add(Code.E_SCHEMA, where, f"{name} must be positive integer, got {value!r}")
                ok = False
        return ok

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        return cls(
            heads=int(data["heads"]),
            dim=int(data["dim"]),
            frac_bits=int(data["frac_bits"]),
            gear=str(data["gear"]),
            point_budget=int(data["point_budget"]),
            operator_quota=int(data["operator_quota"]),
            max_workers=int(data["max_workers"]),
            vocab_size=int(data["vocab_size"]),
            batch_id=str(data.get("batch_id", "batch-0")),
        )


@dataclass
class PipelineModel:
    config: PipelineConfig
    tokens: list[int]
    segments: list[tuple[int, int]]
    chains: dict[int, Chain]
    log: IssueLog

    def all_ops(self) -> list[OpSpec]:
        ops: list[OpSpec] = []
        for chain_id in sorted(self.chains):
            ops.extend(self.chains[chain_id].ordered_ops())
        return ops


def validate_tokens_segments(
    tokens: list[int],
    segments: list[tuple[int, int]],
    vocab_size: int,
    log: IssueLog,
    where: str = "tokens",
) -> bool:
    """Covers every token exactly once; boundaries strictly increasing."""
    ok = True
    if not tokens or any(not isinstance(tok, int) or tok < 0 or tok >= vocab_size for tok in tokens):
        log.add(Code.E_SCHEMA, where, f"tokens must be ints within [0,{vocab_size})")
        ok = False
    if not segments:
        log.add(Code.E_SCHEMA, where, "segments must not be empty")
        return False
    expected_start = 0
    for index, (start, end) in enumerate(segments):
        if not (isinstance(start, int) and isinstance(end, int)) or start >= end:
            log.add(Code.E_SCHEMA, f"{where}.segments[{index}]", "segment needs start<end")
            ok = False
            continue
        if start != expected_start:
            log.add(
                Code.E_SCHEMA,
                f"{where}.segments[{index}]",
                f"segment boundary {start} not strictly increasing after {expected_start}",
            )
            ok = False
        expected_start = end
    if ok and segments[-1][1] != len(tokens):
        log.add(Code.E_SCHEMA, where, f"segments end {segments[-1][1]} != token count {len(tokens)}")
        ok = False
    return ok


def validate_chain(chain_id: int, ops: list[OpSpec], log: IssueLog) -> bool:
    """Unique chain, contiguous op seqs, existing refs, acyclic."""
    ok = True
    where = f"chain[{chain_id}]"
    seqs = [op.seq for op in ops]
    if len(set(seqs)) != len(seqs):
        log.add(Code.E_DUPLICATE, where, f"duplicate operator seqs {seqs}")
        ok = False
    if seqs and sorted(seqs) != list(range(min(seqs), max(seqs) + 1)):
        log.add(Code.E_SCHEMA, where, f"operator seqs not contiguous: {sorted(seqs)}")
        ok = False
    node_ids = {op.node_id() for op in ops}
    # Ref integrity within the same chain.
    for op in ops:
        for ref in op.inputs:
            if f"c{chain_id}op{ref}" not in node_ids:
                log.add(Code.E_REF, f"{where}.op{op.seq}", f"input op {ref} does not exist")
                ok = False
    # Cycle detection (Kahn) on seq-based edges.
    indegree = {op.seq: 0 for op in ops}
    adjacency: dict[int, list[int]] = {op.seq: [] for op in ops}
    for op in ops:
        for ref in op.inputs:
            if ref in indegree:
                adjacency[ref].append(op.seq)
                indegree[op.seq] += 1
    queue = sorted(seq for seq, deg in indegree.items() if deg == 0)
    seen = 0
    while queue:
        seq = queue.pop(0)
        seen += 1
        for nxt in adjacency[seq]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
        queue.sort()
    if seen != len(ops):
        log.add(Code.E_CYCLE, where, "operator graph contains a cycle")
        ok = False
    return ok


def load_config(path: str | Path, log: IssueLog) -> PipelineConfig | None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(data)
    return config if config.validate(log) else None
