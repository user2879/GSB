"""平台核心数据模型（不可变 dataclass，便于逐字节校验与持久化）。"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 受支持的数值类型：int / float（bool 是 int 的子类，需显式排除）
SUPPORTED_NUM_TYPES = (int, float)


@dataclass(frozen=True)
class Derivation:
    """线性派生关系：item == sum(coef[k] * upstream[k])。"""

    refs: Tuple[Tuple[str, float], ...]

    @property
    def ref_keys(self) -> Tuple[str, ...]:
        return tuple(k for k, _ in self.refs)

    def to_json(self) -> dict:
        return {"refs": [[k, c] for k, c in self.refs]}

    @staticmethod
    def from_json(data: dict) -> "Derivation":
        refs = tuple((str(k), float(c)) for k, c in data["refs"])
        return Derivation(refs=refs)


@dataclass(frozen=True)
class DispatchStrategy:
    """分发策略：字段完整性在加载阶段强校验。"""

    batch_size: int
    timeout_ms: int
    max_retries: int

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict) -> "DispatchStrategy":
        return DispatchStrategy(
            batch_size=int(data["batch_size"]),
            timeout_ms=int(data["timeout_ms"]),
            max_retries=int(data["max_retries"]),
        )


@dataclass(frozen=True)
class ConfigItem:
    """单个配置项（某一清单版本内的取值与派生声明）。"""

    key: str
    value: float
    targets: Tuple[str, ...]
    derivation: Optional[Derivation] = None

    @property
    def is_derived(self) -> bool:
        return self.derivation is not None and len(self.derivation.refs) > 0

    def to_json(self) -> dict:
        payload: dict = {
            "key": self.key,
            "value": self.value,
            "targets": list(self.targets),
        }
        if self.derivation is not None:
            payload["derivation"] = self.derivation.to_json()
        return payload

    @staticmethod
    def from_json(data: dict) -> "ConfigItem":
        derivation = (
            Derivation.from_json(data["derivation"]) if data.get("derivation") else None
        )
        return ConfigItem(
            key=str(data["key"]),
            value=float(data["value"]),
            targets=tuple(str(t) for t in data["targets"]),
            derivation=derivation,
        )


@dataclass(frozen=True)
class ManifestVersion:
    """一个清单版本：版本号、配置项集合、分发策略。"""

    version: int
    items: Tuple[ConfigItem, ...]
    strategy: DispatchStrategy

    def item_map(self) -> Dict[str, ConfigItem]:
        return {item.key: item for item in self.items}

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "items": [item.to_json() for item in self.items],
            "strategy": self.strategy.to_json(),
        }


@dataclass(frozen=True)
class NodeSpec:
    """节点注册表里的一个节点。"""

    node_id: str
    host: str
    port: int

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict) -> "NodeSpec":
        return NodeSpec(
            node_id=str(data["node_id"]),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data["port"]),
        )


@dataclass(frozen=True)
class CompiledItem:
    """配置项的编译产物。

    字段全部为基础类型/嵌套元组，:meth:`canonical_bytes` 给出确定性的
    逐字节表示，未变化项跨版本必须保持该字节串不变。
    """

    key: str
    source_value: float
    quantized: int
    is_derived: bool
    coef_quantized: Tuple[Tuple[str, int], ...]
    targets: Tuple[str, ...]
    error_bound: int
    contributions: Tuple[Tuple[str, int], ...]
    propagation_path: Tuple[str, ...]

    def canonical_bytes(self) -> bytes:
        """确定性 JSON + 固定分隔符，跨进程稳定。"""
        import json

        payload = dataclasses.asdict(self)
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            .encode("ascii")
        )

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict) -> "CompiledItem":
        return CompiledItem(
            key=data["key"],
            source_value=float(data["source_value"]),
            quantized=int(data["quantized"]),
            is_derived=bool(data["is_derived"]),
            coef_quantized=tuple(
                (str(k), int(v)) for k, v in data["coef_quantized"]
            ),
            targets=tuple(data["targets"]),
            error_bound=int(data["error_bound"]),
            contributions=tuple(
                (str(k), int(v)) for k, v in data["contributions"]
            ),
            propagation_path=tuple(data["propagation_path"]),
        )


@dataclass
class Receipt:
    """节点回执（含重试序号，首次与重试分开记录）。"""

    node_id: str
    version: int
    digest: str
    seq: int
    attempt: int
    accepted: bool

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict) -> "Receipt":
        return Receipt(
            node_id=str(data["node_id"]),
            version=int(data["version"]),
            digest=str(data["digest"]),
            seq=int(data["seq"]),
            attempt=int(data["attempt"]),
            accepted=bool(data["accepted"]),
        )


@dataclass
class DispatchRecord:
    """单次下发尝试的结构化记录（首次/重试各一条，绝不合并）。"""

    node_id: str
    version: int
    attempt: int
    queued_monotonic_ms: int
    started_monotonic_ms: int
    ended_monotonic_ms: int
    outcome: str  # ok | timeout | error
    seq: Optional[int] = None
    detail: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class StateTransition:
    """状态机迁移记录。"""

    seq: int
    from_state: str
    to_state: str
    reason: str
    monotonic_ms: int
    at_utc: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class DispatchPlan:
    """按节点组织的分发计划。"""

    version: int
    node_batches: Dict[str, List[List[str]]] = field(default_factory=dict)
    strategy: Optional[DispatchStrategy] = None

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "node_batches": {
                node: [list(batch) for batch in batches]
                for node, batches in self.node_batches.items()
            },
            "strategy": self.strategy.to_json() if self.strategy else None,
        }

    @staticmethod
    def from_json(data: dict) -> "DispatchPlan":
        return DispatchPlan(
            version=int(data["version"]),
            node_batches={
                node: [list(batch) for batch in batches]
                for node, batches in data["node_batches"].items()
            },
            strategy=DispatchStrategy.from_json(data["strategy"])
            if data.get("strategy")
            else None,
        )
