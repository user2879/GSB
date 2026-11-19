from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from dns_local.clock import FakeClock
from dns_local.errors import BusinessError
from dns_local.netutil import query_tcp, query_udp
from dns_local.service import ResolverService, ServiceConfig
from dns_local.upstream import UpstreamClient, UpstreamSimulator
from dns_local.wire import RCODE_FORMERR, RCODE_NXDOMAIN, make_query, parse_message

from .conftest import free_dual_port, free_tcp_port


def make_service(
    tmp_path: Path,
    zone_path: Path,
    *,
    upstream_port: int,
    clock: FakeClock | None = None,
    negative_max_ttl: int = 20,
    upstream_client=None,
) -> ResolverService:
    port = free_dual_port()
    config = ServiceConfig(
        udp_port=port,
        tcp_port=port,
        upstream_port=upstream_port,
        upstream_probe_name="upstream.test.",
        upstream_timeout=0.5,
        cache_capacity=32,
        negative_max_ttl=negative_max_ttl,
    )
    return ResolverService(
        zone_path,
        config=config,
        clock=clock or FakeClock(),
        snapshot_root=tmp_path / "snapshots",
        upstream_client=upstream_client,
    )


def test_udp_tcp_local_cache_ttl_cname_and_negative(tmp_path: Path, zone_v1: Path) -> None:
    upstream_port = free_tcp_port()
    clock = FakeClock()
    service = make_service(tmp_path, zone_v1, upstream_port=upstream_port, clock=clock)
    service.start()
    try:
        udp_query = make_query("www.lab.test.", "A", transaction_id=0x3001)
        udp_response = parse_message(query_udp("127.0.0.1", service.config.udp_port, udp_query))
        assert udp_response.header.rcode == 0
        assert udp_response.answers[0].rdata == "192.0.2.10"
        assert udp_response.answers[0].ttl == 120
        first = service.history()[-1]
        assert first.cache_status == "MISS"
        assert first.source == "ZONE"
        assert first.source_serial == 1

        clock.advance(5)
        tcp_response = parse_message(
            query_tcp("127.0.0.1", service.config.tcp_port, make_query("WWW.LAB.TEST.", "A", transaction_id=0x3002))
        )
        assert tcp_response.answers[0].ttl == 115
        second = service.history()[-1]
        assert second.cache_status == "HIT"
        assert second.remaining_ttl == 115

        cname = service.query("alias.lab.test.", "A")
        cname_message = parse_message(cname.wire)
        assert [record.type_name for record in cname_message.answers] == ["CNAME", "A"]
        assert cname.record.business_code == "OK"
        assert "cname_hops=1" in cname.record.attribution

        negative = service.query("missing.lab.test.", "A")
        negative_message = parse_message(negative.wire)
        assert negative_message.header.rcode == RCODE_NXDOMAIN
        assert negative.record.source == "NEGATIVE"
        assert negative.record.source_serial == 1
        assert negative.record.remaining_ttl == 20
        cached_negative = service.query("missing.lab.test.", "A")
        assert cached_negative.record.cache_status == "HIT"
        assert cached_negative.record.remaining_ttl == 20
    finally:
        service.stop()


def test_udp_truncates_and_tcp_carries_full_response(tmp_path: Path, zone_v1: Path) -> None:
    service = make_service(tmp_path, zone_v1, upstream_port=free_tcp_port())
    service.start()
    try:
        raw_query = make_query("bulk.lab.test.", "A", transaction_id=0x3100)
        udp = parse_message(query_udp("127.0.0.1", service.config.udp_port, raw_query))
        assert udp.header.flags & 0x0200
        assert udp.header.ancount == 0
        assert service.history()[-1].truncated is True

        tcp = parse_message(query_tcp("127.0.0.1", service.config.tcp_port, raw_query))
        assert not (tcp.header.flags & 0x0200)
        assert len(tcp.answers) == 45
        assert len(tcp.answers) * 16 + 30 > 512
    finally:
        service.stop()


class CountingClient(UpstreamClient):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.fetch_count = 0
        self.fetch_lock = threading.Lock()

    def fetch(self, question, transaction_id):  # type: ignore[no-untyped-def]
        with self.fetch_lock:
            self.fetch_count += 1
        time.sleep(0.08)
        return super().fetch(question, transaction_id)


def test_concurrent_same_key_uses_one_upstream_fill(tmp_path: Path, zone_v1: Path, upstream_v1: Path) -> None:
    upstream_port = free_tcp_port()
    with UpstreamSimulator(upstream_v1, port=upstream_port):
        client = CountingClient("127.0.0.1", upstream_port, timeout=1)
        service = make_service(tmp_path, zone_v1, upstream_port=upstream_port, upstream_client=client)
        barrier = threading.Barrier(8)
        outcomes = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            outcome = service.query("api.upstream.test.", "A")
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert client.fetch_count == 1
        assert all(outcome.record.business_code == "OK" for outcome in outcomes)
        assert {outcome.record.source_serial for outcome in outcomes} == {1}
        assert sum(1 for outcome in outcomes if outcome.record.cache_status == "MISS") == 1
        assert len(service.cache.entries()) == 1


class ReloadBlockingClient(UpstreamClient):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.started = threading.Event()
        self.release = threading.Event()
        self.fetch_count = 0

    def fetch(self, question, transaction_id):  # type: ignore[no-untyped-def]
        self.fetch_count += 1
        self.started.set()
        assert self.release.wait(timeout=3)
        return super().fetch(question, transaction_id)


def test_zone_reload_during_upstream_fill_never_commits_stale_entry(
    tmp_path: Path, zone_v1: Path, zone_v2: Path, upstream_v1: Path
) -> None:
    upstream_port = free_tcp_port()
    with UpstreamSimulator(upstream_v1, port=upstream_port):
        client = ReloadBlockingClient("127.0.0.1", upstream_port, timeout=1)
        service = make_service(tmp_path, zone_v1, upstream_port=upstream_port, upstream_client=client)
        result = []

        def worker() -> None:
            result.append(service.query("api.upstream.test.", "A"))

        thread = threading.Thread(target=worker)
        thread.start()
        assert client.started.wait(timeout=2)
        reloaded = service.reload_zone(zone_v2)
        assert reloaded.changed is True
        client.release.set()
        thread.join(timeout=5)
        assert result and result[0].record.business_code == "OK"
        assert client.fetch_count >= 2
        entries = service.cache.entries()
        assert len(entries) == 1
        assert entries[0].source == "UPSTREAM"


def test_non_loopback_configuration_is_rejected_before_any_connection(
    tmp_path: Path, zone_v1: Path
) -> None:
    config = ServiceConfig(host="203.0.113.1", udp_port=15353, tcp_port=15353)
    with pytest.raises(BusinessError) as caught:
        ResolverService(zone_v1, config=config, clock=FakeClock(), snapshot_root=tmp_path / "s")
    assert caught.value.code == "E_ARGUMENT"
    with pytest.raises(BusinessError) as upstream_error:
        UpstreamClient("203.0.113.1", 53, timeout=0.1).fetch(
            __import__("dns_local.wire", fromlist=["Question"]).Question("x.test.", 1),
            1,
        )
    assert upstream_error.value.code == "E_ARGUMENT"


def test_upstream_versions_replace_entries_and_conflicts_use_local_authority(
    tmp_path: Path, zone_v1: Path, upstream_v1: Path, upstream_v2: Path
) -> None:
    upstream_port = free_tcp_port()
    first = UpstreamSimulator(upstream_v1, port=upstream_port)
    second: UpstreamSimulator | None = None
    first.start()
    service = make_service(tmp_path, zone_v1, upstream_port=upstream_port)
    try:
        one = service.query("api.upstream.test.", "A")
        assert one.record.source == "UPSTREAM"
        assert one.record.source_serial == 1
        assert parse_message(one.wire).answers[-1].rdata == "198.51.100.11"
        hit = service.query("api.upstream.test.", "A")
        assert hit.record.cache_status == "HIT"

        first.stop()
        second = UpstreamSimulator(upstream_v2, port=upstream_port)
        second.start()
        report = service.refresh_upstream_version()
        assert (report.previous, report.current) == (1, 2)
        assert report.invalidated == 1
        assert service.cache.entries() == ()

        two = service.query("api.upstream.test.", "A")
        assert two.record.cache_status == "MISS"
        assert two.record.source_serial == 2
        assert parse_message(two.wire).answers[-1].rdata == "198.51.100.22"

        conflict = service.query("conflict.upstream.test.", "A")
        conflict_message = parse_message(conflict.wire)
        assert conflict_message.answers[-1].name == "www.lab.test."
        assert conflict_message.answers[-1].rdata == "192.0.2.10"
        assert any("arbitration=local-authority-wins" in item for item in conflict.record.attribution)

        external = service.query("external.lab.test.", "A")
        external_message = parse_message(external.wire)
        assert [record.type_name for record in external_message.answers] == ["CNAME", "CNAME", "A"]
        assert external_message.answers[-1].rdata == "198.51.100.22"
        assert any("merged=local-cname+upstream" in item for item in external.record.attribution)
    finally:
        service.stop()
        second.stop()
        first.stop()


def test_upstream_and_chain_failures_leave_cache_unchanged(
    tmp_path: Path, zone_v1: Path, upstream_v1: Path
) -> None:
    service = make_service(tmp_path, zone_v1, upstream_port=free_tcp_port())
    before = len(service.cache.entries())
    failed = service.query("direct.upstream.test.", "A")
    assert failed.record.business_code == "E_UPSTREAM"
    assert len(service.cache.entries()) == before

    bad_port = free_tcp_port()
    with UpstreamSimulator(upstream_v1, port=bad_port, mode="bad-chain"):
        config = ServiceConfig(
            udp_port=free_dual_port(),
            tcp_port=free_dual_port(),
            upstream_port=bad_port,
            upstream_timeout=0.5,
        )
        # Use one shared port for the service itself.
        dual = free_dual_port()
        config = ServiceConfig(udp_port=dual, tcp_port=dual, upstream_port=bad_port, upstream_timeout=0.5)
        bad_service = ResolverService(
            zone_v1,
            config=config,
            clock=FakeClock(),
            snapshot_root=tmp_path / "snapshots2",
        )
        outcome = bad_service.query("api.upstream.test.", "A")
        assert outcome.record.business_code == "E_CHAIN"
        assert len(bad_service.cache.entries()) == 0


def test_wire_cname_and_port_errors_are_business_codes(tmp_path: Path, zone_v1: Path, root: Path) -> None:
    service = make_service(tmp_path, zone_v1, upstream_port=free_tcp_port())
    malformed = service.process_wire(
        (root / "data" / "wire" / "pointer-loop.dns").read_bytes(),
        transport="udp",
    )
    assert malformed.record.business_code == "E_WIRE"
    assert malformed.record.error_offset == 14
    assert parse_message(malformed.wire).header.rcode == RCODE_FORMERR

    cname_error = service.query("loop1.lab.test.", "A")
    assert cname_error.record.business_code == "E_CNAME"
    assert len(service.cache.entries()) == 0

    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blocker.bind(("127.0.0.1", 0))
    taken_port = blocker.getsockname()[1]
    config = ServiceConfig(udp_port=taken_port, tcp_port=taken_port, upstream_port=free_tcp_port())
    conflicting = ResolverService(zone_v1, config=config, clock=FakeClock(), snapshot_root=tmp_path / "s3")
    try:
        with pytest.raises(BusinessError) as caught:
            conflicting.start()
        assert caught.value.code == "E_PORT"
    finally:
        blocker.close()


def test_snapshot_restore_is_whole_state_and_keeps_old_revisions(
    tmp_path: Path, zone_v1: Path, zone_v2: Path
) -> None:
    clock = FakeClock()
    service = make_service(tmp_path, zone_v1, upstream_port=free_tcp_port(), clock=clock)
    service.query("www.lab.test.", "A")
    service.query("alias.lab.test.", "A")
    service.query("missing.lab.test.", "A")
    info1 = service.save_snapshot("demo", created_utc="2026-01-01T00:00:00+00:00")
    assert info1.revision == 1
    assert info1.cache_entries == 3
    assert info1.path.stat().st_mode & 0o222 == 0
    snapshot_keys = {entry.key.display for entry in service.cache.entries()}
    snapshot_payload = service.snapshots.load(info1.snapshot_id)
    for cached in snapshot_payload["cache_entries"]:
        assert cached["first_write_basis"] == "CLOCK.monotonic_ns"
        assert cached["stored_age_seconds"] >= 0
        assert "source_serial" in cached

    clock.advance(5)
    service.query("www.lab.test.", "A")
    service.query("v6.lab.test.", "AAAA")
    assert any(entry.key.name == "v6.lab.test." for entry in service.cache.entries())

    reloaded = service.reload_zone(zone_v2)
    assert reloaded.changed is True
    assert service.zone.version == 2
    assert service.cache.entries() == ()
    info2 = service.save_snapshot("demo", created_utc="2026-01-01T00:01:00+00:00")
    assert info2.revision == 2
    assert [item.revision for item in service.snapshots.list()] == [1, 2]

    compared, consistent, differences = service.compare_snapshot(info1.snapshot_id)
    assert compared.snapshot_id == info1.snapshot_id
    assert consistent is False
    assert any("zone-version" in item for item in differences)

    restored = service.restore_snapshot(info1.snapshot_id)
    assert restored.consistent is False
    assert any("zone-version" in item for item in restored.differences)
    assert restored.restored_entries == 3
    assert service.zone.version == 1
    assert {entry.key.display for entry in service.cache.entries()} == snapshot_keys
    assert not any(entry.key.name == "v6.lab.test." for entry in service.cache.entries())
    hit = service.query("www.lab.test.", "A")
    assert hit.record.cache_status == "HIT"
    assert hit.record.remaining_ttl == 120
    assert len(service.history()) >= 4

    after_restore, consistent_after, no_diff = service.compare_snapshot(info1.snapshot_id)
    assert after_restore.snapshot_id == info1.snapshot_id
    assert consistent_after is True
    assert no_diff == ()
    assert service.snapshots.load(info2.snapshot_id)["revision"] == 2


def test_snapshot_checksum_tampering_is_rejected_without_state_change(
    tmp_path: Path, zone_v1: Path
) -> None:
    service = make_service(tmp_path, zone_v1, upstream_port=free_tcp_port())
    service.query("www.lab.test.", "A")
    info = service.save_snapshot("tamper-check", created_utc="2026-01-01T00:00:00+00:00")
    before_keys = [entry.key.display for entry in service.cache.entries()]
    info.path.chmod(0o600)
    payload = json.loads(info.path.read_text(encoding="utf-8"))
    payload["name"] = "changed-after-write"
    info.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BusinessError) as caught:
        service.restore_snapshot(info.snapshot_id)
    assert caught.value.code == "E_SNAPSHOT"
    assert caught.value.field == "checksum"
    assert [entry.key.display for entry in service.cache.entries()] == before_keys
