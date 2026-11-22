"""服务拓扑、依赖边与调用事件的领域数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

LOOPBACK = "127.0.0.1"


class DegradeStrategy(str, Enum):
    DROP = "discard"          # 丢弃
    CACHE = "cache_copy"      # 返回缓存副本
    DEFAULT = "default_value"  # 折叠为默认值

    @staticmethod
    def values() -> set[str]:
        return {m.value for m in DegradeStrategy}

    @property
    def cn_label(self) -> str:
        return {
            DegradeStrategy.DROP: "丢弃",
            DegradeStrategy.CACHE: "返回缓存副本",
            DegradeStrategy.DEFAULT: "折叠为默认值",
        }[self]


class PropagationMode(str, Enum):
    FORCED = "forced"       # 强制传播
    THRESHOLD = "threshold"  # 按阈值传播


class ServiceState(str, Enum):
    AVAILABLE = "available"
    DEGRADED_LIMITED = "limited"
    UNAVAILABLE = "unavailable"


class Outcome(str, Enum):
    SUCCESS = "success"
    REFUSED = "connection_refused"
    TIMEOUT = "timeout"
    SLOW_SUCCESS = "slow_success"


@dataclass(frozen=True)
class ServiceDef:
    service_id: str
    domain_id: str
    host: str
    port: int
    strategy: str
    capacity_limit: int
    seed: int
    source_line: int


@dataclass(frozen=True)
class EdgeDef:
    upstream: str
    downstream: str
    mode: str
    threshold: int | None  # 定点阈值（threshold 模式必填）
    source_line: int


@dataclass(frozen=True)
class CallEvent:
    seq: int
    arrival_ms: int
    upstream: str
    downstream: str
    client_key: str
    source_line: int


@dataclass(frozen=True)
class DomainQuota:
    domain_id: str
    in_flight_quota: int


@dataclass
class SessionParams:
    scale_exponent: int = 2
    global_concurrency: int = 4
    window_size: int = 5
    limited_threshold: int = 80   # 定点：成功率 >= 80/F 为可用
    unavailable_threshold: int = 40  # < 40/F 为不可用，之间为受限
    socket_timeout_ms: int = 300
    max_retries: int = 1

    @property
    def factor(self) -> int:
        return 10**self.scale_exponent


@dataclass
class Topology:
    topology_version: str
    services: dict[str, ServiceDef]
    domains: dict[str, DomainQuota]
    edges: list[EdgeDef]
    params: SessionParams
    order_index: list[str] = field(default_factory=list)
    adjacency: dict[str, list[str]] = field(default_factory=dict)   # upstream -> [downstream]
    reverse: dict[str, list[str]] = field(default_factory=dict)     # downstream -> [upstream]

    def outgoing(self, service_id: str) -> list[str]:
        return self.adjacency.get(service_id, [])

    def incoming_upstreams(self, service_id: str) -> list[str]:
        return self.reverse.get(service_id, [])
