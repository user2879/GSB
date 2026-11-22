"""工作台门面：装配区、缓存账本、上游客户端、快照与结构化日志。"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from .cache import MonotonicClock
from .errors import VersionError, WorkbenchError
from .ledger import CacheLedger
from .records import CacheEntry, load_cache_records
from .server import DnsDiagnosticEngine
from .snapshot import HitEvent, SnapshotStore
from .structured_log import configure_logger, log_error
from .upstream import TcpUpstreamClient
from .zone import AuthorityResolver, Zone, load_zone


class DeterministicClock:
    """可注入的确定性单调时钟（毫秒），用于可重复演示。"""

    def __init__(self, start_ms: int = 1_000_000) -> None:
        self._now = start_ms

    def now_ms(self) -> int:
        return self._now

    def advance_ms(self, delta_ms: int) -> int:
        self._now += max(0, delta_ms)
        return self._now


@dataclass
class LoadedSamples:
    zone: Zone
    entries: list[CacheEntry]


class DnsWorkbench:
    def __init__(
        self,
        *,
        sample_dir: str | Path,
        data_dir: str | Path,
        capacity: int = 256,
        negative_ttl_cap: int = 300,
        deterministic: bool = True,
        upstream_port: int = 5354,
        forward: bool = True,
        upstream_retries: int = 1,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.clock = DeterministicClock() if deterministic else MonotonicClock()
        self.logger = configure_logger(self.data_dir / "workbench.jsonl")
        self.ledger = CacheLedger(
            capacity=capacity,
            negative_ttl_cap=negative_ttl_cap,
            clock=self.clock,
        )
        self.zone: Zone | None = None
        self.resolver: AuthorityResolver | None = None
        self.engine: DnsDiagnosticEngine | None = None
        self.snapshot_store = SnapshotStore(self.data_dir / "ledger.sqlite3")
        self.upstream_client = TcpUpstreamClient(port=upstream_port, retries=upstream_retries)
        self.forward = forward
        self._hits: list[HitEvent] = []

    def load_zone_file(self, path: str | Path) -> Zone:
        self.zone = load_zone(path)
        self.resolver = AuthorityResolver(self.zone)
        return self.zone

    def load_cache_files(self, paths: list[str | Path]) -> list[CacheEntry]:
        payloads = []
        for path in paths:
            payloads.extend(json.loads(Path(path).read_text(encoding="utf-8")))
        return load_cache_records(payloads)

    def bootstrap(self, *, zone_path: str | Path, cache_paths: list[str | Path]) -> None:
        zone = self.load_zone_file(zone_path)
        entries = self.load_cache_files(cache_paths)
        self.ledger.initialize(zone_version=zone.zone_version, entries=entries)
        self._build_engine()

    def _build_engine(self) -> None:
        assert self.resolver is not None
        self.engine = DnsDiagnosticEngine(
            ledger=self.ledger,
            resolver=self.resolver,
            upstream_client=self.upstream_client,
            logger=self.logger,
            forward=self.forward,
        )

    def reload_zone(self, path: str | Path) -> None:
        """区版本变化后整体替换缓存视图。"""
        zone = load_zone(path)
        if self.zone is not None and zone.zone_version < self.zone.zone_version:
            raise VersionError(
                "区版本不得倒退",
                fields=["zone_version"],
                context={"given": zone.zone_version, "current": self.zone.zone_version},
            )
        entries = [stored.entry for stored in self.ledger.cache().view()]
        self.zone = zone
        self.resolver = AuthorityResolver(zone)
        diff = self.ledger.replace(zone_version=zone.zone_version, entries=entries)
        self._build_engine()
        self.logger.info("zone reloaded; wholesale replacement", extra={"data": diff.to_dict()})

    def save_snapshot(self) -> int:
        snapshot_id = self.snapshot_store.save(
            self.ledger,
            created_at_ms=self.clock.now_ms(),
            hits=list(self._hits),
        )
        self.logger.info("snapshot saved", extra={"data": {"snapshot_id": snapshot_id}})
        return snapshot_id

    def verify_snapshot(self, snapshot_id: int):
        return self.snapshot_store.verify_against(snapshot_id, self.ledger)

    def snapshots(self) -> list[dict]:
        return self.snapshot_store.list_snapshots()

    def record_hit_event(self, name: str, qtype: int, remaining_ttl_s: int) -> None:
        self._hits.append(HitEvent(self.clock.now_ms(), name, qtype, remaining_ttl_s))

    def safe(self, func, *args, **kwargs):
        """执行注入场景，捕获工作台异常并写结构化日志，返回 (结果, 异常)。"""
        try:
            return func(*args, **kwargs), None
        except WorkbenchError as exc:
            log_error(self.logger, exc)
            return None, exc

    def close(self) -> None:
        self.snapshot_store.close()
