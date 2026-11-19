"""Deterministic three-scenario loopback demonstration.

Only stable result lines are written to stdout. Observation timing is written to
stderr, so two runs can be compared byte-for-byte on stdout.
"""

from __future__ import annotations

import socket
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Counter

from .clock import FakeClock
from .errors import BusinessError
from .netutil import query_tcp, query_udp
from .service import QueryRecord, ResolverService, ServiceConfig
from .upstream import UpstreamSimulator
from .wire import ResourceRecord, make_query, parse_message


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _free_dual_port() -> int:
    for _ in range(100):
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.bind(("127.0.0.1", 0))
        port = tcp.getsockname()[1]
        tcp.close()
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(("127.0.0.1", port))
        except OSError:
            udp.close()
            continue
        udp.close()
        return port
    raise BusinessError("E_PORT", "could not reserve a loopback UDP/TCP port")


def _free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@dataclass
class ScenarioStats:
    name: str
    queries: int = 0
    codes: Counter[str] = field(default_factory=Counter)
    cache: Counter[str] = field(default_factory=Counter)

    def add(self, record: QueryRecord) -> None:
        self.queries += 1
        self.codes[record.business_code] += 1
        self.cache[record.cache_status] += 1

    def summary(self) -> str:
        codes = ",".join(f"{key}:{value}" for key, value in sorted(self.codes.items()))
        cache = ",".join(
            f"{key}:{self.cache.get(key, 0)}" for key in ("HIT", "MISS", "ERROR")
        )
        return f"SUMMARY scenario={self.name} queries={self.queries} codes={codes} cache={cache}"


def _rr_text(record: ResourceRecord) -> str:
    return f"{record.name}|{record.type_name}|{record.ttl}|{record.rdata_text()}"


def _basis(attribution: tuple[str, ...]) -> str:
    joined = " ".join(attribution)
    if "arbitration=local-authority-wins" in joined:
        return "local-authority-wins"
    if "merged=local-cname+upstream" in joined:
        return "local-cname+upstream"
    wildcard = next((item for item in attribution if item.startswith("match=wildcard")), None)
    if wildcard:
        return "wildcard"
    hops = next((item for item in attribution if item.startswith("cname_hops=")), None)
    if hops and not hops.endswith("=0"):
        return hops
    if "negative=nxdomain" in joined:
        return "negative-nxdomain"
    if "negative=nodata" in joined:
        return "negative-nodata"
    return "exact"


class Demo:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.service_port = _free_dual_port()
        self.upstream_port = _free_tcp_port()
        self.config = ServiceConfig(
            udp_port=self.service_port,
            tcp_port=self.service_port,
            upstream_port=self.upstream_port,
            upstream_probe_name="upstream.test.",
            upstream_timeout=1.0,
            cache_capacity=64,
            negative_max_ttl=20,
        )
        self._next_id = 0x4100
        self.scenarios: list[ScenarioStats] = []
        self.snapshots_saved = 0
        self.snapshots_restored = 0

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _find_record(self, transaction_id: int, transport: str) -> QueryRecord:
        for record in reversed(self.service.history()):
            if record.transaction_id == transaction_id and record.transport == transport:
                return record
        raise BusinessError("E_STATE", "demo query history record is missing")

    def query(
        self,
        scenario: ScenarioStats,
        transport: str,
        name: str,
        qtype: str,
    ) -> QueryRecord:
        transaction_id = self._id()
        raw = make_query(name, qtype, transaction_id=transaction_id)
        if transport == "udp":
            payload = query_udp(
                self.config.host,
                self.config.udp_port,
                raw,
                timeout=self.config.upstream_timeout + 1,
            )
        elif transport == "tcp":
            payload = query_tcp(
                self.config.host,
                self.config.tcp_port,
                raw,
                timeout=self.config.upstream_timeout + 1,
            )
        else:
            raise ValueError("transport must be udp or tcp")
        message = parse_message(payload)
        record = self._find_record(transaction_id, transport)
        answers = ";".join(_rr_text(rr) for rr in message.answers) or "-"
        authority = ";".join(_rr_text(rr) for rr in message.authority) or "-"
        print(
            f"QUERY scenario={scenario.name} id=0x{transaction_id:04x} transport={transport} "
            f"name={record.qname} type={record.qtype} rcode={record.rcode} code={record.business_code} "
            f"cache={record.cache_status} source={record.source} serial={record.source_serial} "
            f"ttl={record.remaining_ttl} basis={_basis(record.attribution)} truncated={str(record.truncated).lower()} "
            f"answers={answers} authority={authority}",
            flush=True,
        )
        scenario.add(record)
        return record

    def _scenario(self, name: str) -> ScenarioStats:
        stats = ScenarioStats(name)
        self.scenarios.append(stats)
        return stats

    def run(self) -> None:
        zone_v1 = PROJECT_ROOT / "data" / "zones" / "lab.v1.json"
        zone_v2 = PROJECT_ROOT / "data" / "zones" / "lab.v2.json"
        upstream_v1 = PROJECT_ROOT / "data" / "upstream" / "v1.json"
        upstream_v2 = PROJECT_ROOT / "data" / "upstream" / "v2.json"
        with tempfile.TemporaryDirectory(prefix="dns-local-demo-") as temp:
            snapshot_root = Path(temp) / "snapshots"
            upstream1 = UpstreamSimulator(upstream_v1, port=self.upstream_port)
            upstream2: UpstreamSimulator | None = None
            service = ResolverService(
                zone_v1,
                config=self.config,
                clock=self.clock,
                snapshot_root=snapshot_root,
            )
            service.start()
            self.service = service
            try:
                upstream1.start()
                print(
                    f"START zone_version={service.zone.version} listeners=loopback clock={self.clock.source}",
                    flush=True,
                )
                self._scenario_one(zone_v1)
                self._scenario_two(upstream1, upstream_v1, upstream_v2)
                upstream2 = self._upstream2
                self._scenario_three(zone_v1, zone_v2, snapshot_root)
                final_queries = sum(item.queries for item in self.scenarios)
                codes: Counter[str] = Counter()
                cache: Counter[str] = Counter()
                for item in self.scenarios:
                    codes.update(item.codes)
                    cache.update(item.cache)
                code_summary = ",".join(f"{key}:{value}" for key, value in sorted(codes.items()))
                cache_summary = ",".join(f"{key}:{cache.get(key, 0)}" for key in ("HIT", "MISS", "ERROR"))
                print(
                    f"FINAL scenarios={len(self.scenarios)} queries={final_queries} codes={code_summary} "
                    f"cache={cache_summary} snapshots_saved={self.snapshots_saved} "
                    f"snapshots_restored={self.snapshots_restored}",
                    flush=True,
                )
            finally:
                service.stop()
                upstream1.stop()
                if upstream2 is not None:
                    upstream2.stop()
                self._upstream2 = None

    def _scenario_one(self, zone_v1: Path) -> None:
        stats = self._scenario("cold-start-local")
        print("SCENARIO 1 cold-start-local", flush=True)
        self.query(stats, "udp", "www.lab.test.", "A")
        self.query(stats, "udp", "WWW.LAB.TEST.", "A")
        self.query(stats, "tcp", "alias.lab.test.", "A")
        self.query(stats, "udp", "anything.wild.lab.test.", "A")
        self.query(stats, "udp", "missing.lab.test.", "A")
        self.query(stats, "udp", "nodata.lab.test.", "A")
        print(stats.summary(), flush=True)

    def _scenario_two(self, upstream1: UpstreamSimulator, upstream_v1: Path, upstream_v2: Path) -> None:
        stats = self._scenario("upstream-version-change")
        print("SCENARIO 2 upstream-version-change", flush=True)
        self.query(stats, "udp", "api.upstream.test.", "A")
        self.query(stats, "udp", "api.upstream.test.", "A")
        upstream1.stop()
        second = UpstreamSimulator(upstream_v2, port=self.upstream_port)
        second.start()
        self._upstream2 = second
        report = self.service.refresh_upstream_version()
        print(
            f"UPSTREAM_SWITCH previous={report.previous} current={report.current} invalidated={report.invalidated}",
            flush=True,
        )
        self.query(stats, "udp", "api.upstream.test.", "A")
        self.query(stats, "udp", "conflict.upstream.test.", "A")
        self.query(stats, "tcp", "multi.upstream.test.", "A")
        print(stats.summary(), flush=True)

    def _scenario_three(self, zone_v1: Path, zone_v2: Path, snapshot_root: Path) -> None:
        stats = self._scenario("snapshot-save-restore")
        print("SCENARIO 3 snapshot-save-restore", flush=True)
        self.clock.advance(3)
        self.query(stats, "udp", "www.lab.test.", "A")
        snapshot1 = self.service.save_snapshot("demo", created_utc="2026-01-01T00:00:00+00:00")
        self.snapshots_saved += 1
        print(
            f"SNAPSHOT_SAVE id={snapshot1.snapshot_id} revision={snapshot1.revision} "
            f"zone_version={snapshot1.zone_version} entries={snapshot1.cache_entries} readonly=true",
            flush=True,
        )

        reloaded = self.service.reload_zone(zone_v2)
        print(
            f"ZONE_RELOAD version={reloaded.image.version} changed={str(reloaded.changed).lower()} cache=cleared",
            flush=True,
        )
        self.query(stats, "udp", "www.lab.test.", "A")
        self.query(stats, "udp", "v6.lab.test.", "AAAA")
        snapshot2 = self.service.save_snapshot("demo", created_utc="2026-01-01T00:01:00+00:00")
        self.snapshots_saved += 1
        print(
            f"SNAPSHOT_SAVE id={snapshot2.snapshot_id} revision={snapshot2.revision} "
            f"zone_version={snapshot2.zone_version} entries={snapshot2.cache_entries} readonly=true",
            flush=True,
        )
        revisions = ",".join(
            f"{item.snapshot_id}:rev{item.revision}:zone{item.zone_version}"
            for item in self.service.snapshots.list()
        )
        print(f"SNAPSHOT_LIST count={len(self.service.snapshots.list())} items={revisions}", flush=True)

        self.service.stop()
        replacement = ResolverService(
            zone_v1,
            config=self.config,
            clock=self.clock,
            snapshot_root=snapshot_root,
        )
        replacement.start()
        replacement.reload_zone(zone_v2)
        self.service = replacement
        try:
            self.query(stats, "udp", "v6.lab.test.", "AAAA")
            compared, consistent, differences = self.service.compare_snapshot(snapshot1.snapshot_id)
            print(
                f"SNAPSHOT_COMPARE id={compared.snapshot_id} status={'CONSISTENT' if consistent else 'MISMATCH'} "
                f"snapshot_zone={compared.zone_version} current_zone={self.service.zone.version} "
                f"snapshot_upstream={compared.upstream_version} current_upstream={self.service.upstream_version} "
                f"differences={len(differences)}",
                flush=True,
            )
            restored = self.service.restore_snapshot(snapshot1.snapshot_id)
            self.snapshots_restored += 1
            print(
                f"SNAPSHOT_RESTORE id={restored.info.snapshot_id} "
                f"status={'CONSISTENT' if restored.consistent else 'MISMATCH'} mode=whole "
                f"entries={restored.restored_entries} zone_version={self.service.zone.version}",
                flush=True,
            )
            keys = sorted(entry.key.display for entry in self.service.cache.entries())
            print(
                f"CACHE_AFTER_RESTORE entries={len(keys)} contains_v2_v6={str(any('v6.lab.test.' in key for key in keys)).lower()} "
                f"keys={','.join(keys)}",
                flush=True,
            )
            self.query(stats, "udp", "www.lab.test.", "A")
            self.query(stats, "udp", "v6.lab.test.", "AAAA")
            self.query(stats, "udp", "api.upstream.test.", "A")
        finally:
            replacement.stop()
        print(stats.summary(), flush=True)


def main() -> int:
    started = time.perf_counter()
    try:
        Demo().run()
    except BusinessError as exc:
        print(f"ERROR {exc.render()}", file=sys.stderr)
        return 2
    observed_ms = (time.perf_counter() - started) * 1000
    print(f"OBSERVED_ELAPSED_MS={observed_ms:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
