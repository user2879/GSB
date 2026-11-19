"""DNS service orchestration: authority, cache, upstream validation and snapshots."""

from __future__ import annotations

import socketserver
import struct
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .cache import Cache, CacheEntry, CacheKey, CacheSource, FillDiscarded
from .clock import Clock, SystemMonotonicClock
from .errors import BusinessError
from .netutil import check_loopback_ports, require_loopback_host
from .resolver import Resolution, ZoneResolver
from .snapshot import (
    SnapshotInfo,
    SnapshotStore,
    cache_entry_from_json,
    compare_state,
    zone_from_json,
)
from .upstream import UpstreamClient, UpstreamResult
from .wire import (
    CLASS_IN,
    RCODE_FORMERR,
    RCODE_NAMES,
    RCODE_NOERROR,
    RCODE_NXDOMAIN,
    RCODE_SERVFAIL,
    RCODE_NOTIMP,
    SOAData,
    TYPE_BY_NAME,
    DNSMessage,
    Question,
    ResourceRecord,
    build_error_response,
    build_response,
    make_query,
    maybe_truncate_udp,
    parse_message,
    peek_transaction_id,
)
from .zone import DEFAULT_ZONE_MAX_TTL, ZoneImage, ZoneLoadResult, ZoneManager


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "127.0.0.1"
    udp_port: int = 15353
    tcp_port: int = 15353
    upstream_host: str = "127.0.0.1"
    upstream_port: int = 15354
    upstream_probe_name: str = "upstream.test."
    upstream_timeout: float = 2.0
    cache_capacity: int = 128
    negative_max_ttl: int = 300
    zone_max_ttl: int = DEFAULT_ZONE_MAX_TTL
    cname_max_hops: int = 8
    udp_payload_limit: int = 512
    history_limit: int = 1000


@dataclass(frozen=True)
class QueryRecord:
    sequence: int
    transport: str
    client: str
    transaction_id: int
    qname: str
    qtype: str
    rcode: str
    business_code: str
    cache_status: str
    source: str
    source_serial: int
    remaining_ttl: int
    attribution: tuple[str, ...]
    answers: tuple[str, ...]
    error: str | None = None
    error_offset: int | None = None
    truncated: bool = False
    observed_elapsed_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "transport": self.transport,
            "client": self.client,
            "transaction_id": self.transaction_id,
            "qname": self.qname,
            "qtype": self.qtype,
            "rcode": self.rcode,
            "business_code": self.business_code,
            "cache_status": self.cache_status,
            "source": self.source,
            "source_serial": self.source_serial,
            "remaining_ttl": self.remaining_ttl,
            "attribution": list(self.attribution),
            "answers": list(self.answers),
            "error": self.error,
            "error_offset": self.error_offset,
            "truncated": self.truncated,
            "observed_elapsed_ns": self.observed_elapsed_ns,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryRecord":
        try:
            return cls(
                sequence=int(value["sequence"]),
                transport=str(value["transport"]),
                client=str(value["client"]),
                transaction_id=int(value["transaction_id"]),
                qname=str(value["qname"]),
                qtype=str(value["qtype"]),
                rcode=str(value["rcode"]),
                business_code=str(value["business_code"]),
                cache_status=str(value["cache_status"]),
                source=str(value["source"]),
                source_serial=int(value["source_serial"]),
                remaining_ttl=int(value["remaining_ttl"]),
                attribution=tuple(str(item) for item in value.get("attribution", [])),
                answers=tuple(str(item) for item in value.get("answers", [])),
                error=None if value.get("error") is None else str(value["error"]),
                error_offset=None if value.get("error_offset") is None else int(value["error_offset"]),
                truncated=bool(value.get("truncated", False)),
                observed_elapsed_ns=int(value.get("observed_elapsed_ns", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BusinessError("E_SNAPSHOT", f"invalid history record: {exc}", field="history") from exc


@dataclass(frozen=True)
class QueryOutcome:
    wire: bytes
    record: QueryRecord

    @property
    def business_code(self) -> str:
        return self.record.business_code


@dataclass(frozen=True)
class UpstreamVersionReport:
    previous: int
    current: int
    invalidated: int


@dataclass(frozen=True)
class RestoreResult:
    info: SnapshotInfo
    consistent: bool
    differences: tuple[str, ...]
    restored_entries: int


@dataclass(frozen=True)
class _ArbitratedUpstream:
    answers: tuple[ResourceRecord, ...]
    authority: tuple[ResourceRecord, ...]
    rcode: int
    negative: bool
    attribution: tuple[str, ...]
    conflict: tuple[str, ...]


class _UDPServer(socketserver.ThreadingUDPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: "ResolverService") -> None:
        self.service = service
        super().__init__(address, _UDPHandler)


class _UDPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data, sock = self.request
        service: ResolverService = self.server.service  # type: ignore[attr-defined]
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        outcome = service.process_wire(data, transport="udp", client=peer)
        try:
            sock.sendto(outcome.wire, self.client_address)
        except OSError:
            pass


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: "ResolverService") -> None:
        self.service = service
        super().__init__(address, _TCPHandler)


class _TCPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        service: ResolverService = self.server.service  # type: ignore[attr-defined]
        self.request.settimeout(service.config.upstream_timeout + 1.0)
        try:
            prefix = self._read_exact(2)
            length = struct.unpack("!H", prefix)[0]
            data = self._read_exact(length)
        except (OSError, ConnectionError, struct.error):
            return
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        outcome = service.process_wire(data, transport="tcp", client=peer)
        try:
            self.request.sendall(struct.pack("!H", len(outcome.wire)) + outcome.wire)
        except OSError:
            pass

    def _read_exact(self, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = self.request.recv(remaining)
            if not chunk:
                raise ConnectionError("TCP peer closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class ResolverService:
    """A local resolver whose data plane never touches non-loopback networks."""

    def __init__(
        self,
        zone_path: str | Path,
        *,
        config: ServiceConfig | None = None,
        clock: Clock | None = None,
        snapshot_root: str | Path = "snapshots",
        upstream_client: UpstreamClient | None = None,
    ) -> None:
        self.config = config or ServiceConfig()
        require_loopback_host(self.config.host)
        require_loopback_host(self.config.upstream_host)
        self.clock = clock or SystemMonotonicClock()
        self.zone_manager = ZoneManager(max_ttl=self.config.zone_max_ttl)
        load_result = self.zone_manager.load_file(zone_path)
        self._zone: ZoneImage = load_result.image
        self._state_epoch = 1
        self._state_lock = threading.RLock()
        self._upstream_lock = threading.RLock()
        self._history_lock = threading.RLock()
        self._query_id_lock = threading.Lock()
        self._next_transaction_id = 0x2000
        self._negative_max_ttl = max(0, int(self.config.negative_max_ttl))
        self._upstream_version = 0
        self.resolver = ZoneResolver(max_cname_hops=self.config.cname_max_hops)
        self.upstream = upstream_client or UpstreamClient(
            self.config.upstream_host,
            self.config.upstream_port,
            timeout=self.config.upstream_timeout,
            max_cname_hops=self.config.cname_max_hops,
        )
        self.cache = Cache(self.clock, capacity=self.config.cache_capacity)
        self.snapshots = SnapshotStore(snapshot_root)
        self._history: list[QueryRecord] = []
        self._next_history_sequence = 1
        self._udp_server: _UDPServer | None = None
        self._tcp_server: _TCPServer | None = None
        self._udp_thread: threading.Thread | None = None
        self._tcp_thread: threading.Thread | None = None
        self._running = False

    @property
    def zone(self) -> ZoneImage:
        with self._state_lock:
            return self._zone

    @property
    def state_epoch(self) -> int:
        with self._state_lock:
            return self._state_epoch

    @property
    def upstream_version(self) -> int:
        with self._upstream_lock:
            return self._upstream_version

    @property
    def negative_max_ttl(self) -> int:
        with self._state_lock:
            return self._negative_max_ttl

    @property
    def running(self) -> bool:
        return self._running

    def startup_report(self) -> dict[str, Any]:
        zone = self.zone
        return {
            "host": self.config.host,
            "udp_port": self.config.udp_port,
            "tcp_port": self.config.tcp_port,
            "upstream": f"{self.config.upstream_host}:{self.config.upstream_port}",
            "clock_source": self.clock.source,
            "zone_origin": zone.origin,
            "zone_version": zone.version,
            "zone_digest": zone.digest,
            "cache_capacity": self.config.cache_capacity,
            "negative_max_ttl": self.negative_max_ttl,
            "udp_payload_limit": self.config.udp_payload_limit,
            "upstream_version": self.upstream_version,
        }

    def start(self) -> dict[str, Any]:
        if self._running:
            return self.startup_report()
        check_loopback_ports(
            self.config.host,
            [("udp", self.config.udp_port), ("tcp", self.config.tcp_port)],
        )
        try:
            udp_server = _UDPServer((self.config.host, self.config.udp_port), self)
        except OSError as exc:
            raise BusinessError(
                "E_PORT",
                f"cannot bind UDP service port: {exc}",
                details={"host": self.config.host, "port": self.config.udp_port},
            ) from exc
        try:
            tcp_server = _TCPServer((self.config.host, self.config.tcp_port), self)
        except OSError as exc:
            udp_server.server_close()
            raise BusinessError(
                "E_PORT",
                f"cannot bind TCP service port: {exc}",
                details={"host": self.config.host, "port": self.config.tcp_port},
            ) from exc
        self._udp_server = udp_server
        self._tcp_server = tcp_server
        self._udp_thread = threading.Thread(target=udp_server.serve_forever, name="dns-udp", daemon=True)
        self._tcp_thread = threading.Thread(target=tcp_server.serve_forever, name="dns-tcp", daemon=True)
        self._udp_thread.start()
        self._tcp_thread.start()
        self._running = True
        return self.startup_report()

    def stop(self) -> None:
        if not self._running:
            return
        if self._udp_server is not None:
            self._udp_server.shutdown()
            self._udp_server.server_close()
        if self._tcp_server is not None:
            self._tcp_server.shutdown()
            self._tcp_server.server_close()
        if self._udp_thread is not None:
            self._udp_thread.join(timeout=3)
        if self._tcp_thread is not None:
            self._tcp_thread.join(timeout=3)
        self._udp_server = None
        self._tcp_server = None
        self._udp_thread = None
        self._tcp_thread = None
        self._running = False

    def __enter__(self) -> "ResolverService":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.stop()

    def _next_id(self) -> int:
        with self._query_id_lock:
            value = self._next_transaction_id
            self._next_transaction_id = 0x2000 + ((value + 1 - 0x2000) % 0x9FFF)
            return value

    def make_query(self, name: str, qtype: str | int, transaction_id: int | None = None) -> bytes:
        return make_query(name, qtype, transaction_id=self._next_id() if transaction_id is None else transaction_id)

    def query(
        self,
        name: str,
        qtype: str | int,
        *,
        transport: str = "internal",
        client: str = "internal",
        transaction_id: int | None = None,
    ) -> QueryOutcome:
        payload = self.make_query(name, qtype, transaction_id)
        return self.process_wire(payload, transport=transport, client=client)

    def _capture_state(self) -> tuple[ZoneImage, int]:
        with self._state_lock:
            return self._zone, self._state_epoch

    def _validate_query(self, message: DNSMessage) -> Question:
        header = message.header
        if header.is_response:
            raise BusinessError("E_WIRE", "received a DNS response on the query port", offset=2)
        if ((header.flags >> 11) & 0xF) != 0:
            raise BusinessError("E_WIRE", "only standard QUERY opcode is supported", offset=2)
        if header.qdcount != 1 or len(message.questions) != 1:
            raise BusinessError("E_WIRE", "query must contain exactly one question", offset=4)
        question = message.questions[0]
        if question.qclass != CLASS_IN:
            raise BusinessError(
                "E_ARGUMENT",
                "only the IN class is supported",
                details={"qclass": question.qclass},
            )
        if question.qtype not in TYPE_BY_NAME.values():
            raise BusinessError(
                "E_ARGUMENT",
                "unsupported query type",
                details={"qtype": question.type_name},
            )
        return question

    def _record(
        self,
        *,
        started_ns: int,
        transport: str,
        client: str,
        transaction_id: int,
        qname: str,
        qtype: str,
        rcode: str,
        business_code: str,
        cache_status: str,
        source: str = "-",
        source_serial: int = 0,
        remaining_ttl: int = 0,
        attribution: tuple[str, ...] = (),
        answers: tuple[str, ...] = (),
        error: str | None = None,
        error_offset: int | None = None,
        truncated: bool = False,
    ) -> QueryRecord:
        with self._history_lock:
            sequence = self._next_history_sequence
            self._next_history_sequence += 1
            record = QueryRecord(
                sequence=sequence,
                transport=transport,
                client=client,
                transaction_id=transaction_id,
                qname=qname,
                qtype=qtype,
                rcode=rcode,
                business_code=business_code,
                cache_status=cache_status,
                source=source,
                source_serial=source_serial,
                remaining_ttl=remaining_ttl,
                attribution=attribution,
                answers=answers,
                error=error,
                error_offset=error_offset,
                truncated=truncated,
                observed_elapsed_ns=max(0, self.clock.monotonic_ns() - started_ns),
            )
            self._history.append(record)
            if len(self._history) > self.config.history_limit:
                del self._history[: len(self._history) - self.config.history_limit]
            return record

    @staticmethod
    def _error_rcode(code: str) -> int:
        if code == "E_WIRE":
            return RCODE_FORMERR
        if code == "E_ARGUMENT":
            return RCODE_NOTIMP
        return RCODE_SERVFAIL

    def process_wire(self, data: bytes, *, transport: str, client: str = "unknown") -> QueryOutcome:
        started_ns = self.clock.monotonic_ns()
        transaction_id = peek_transaction_id(data) or 0
        message: DNSMessage | None = None
        question: Question | None = None
        try:
            message = parse_message(data)
            question = self._validate_query(message)
            served = self._serve_question(question, message.header.transaction_id)
            now = self.clock.monotonic_ns()
            answers, authority = served[0].aged_records(now)
            entry = served[0]
            remaining_ttl = served[1]
            authoritative = "authority=local" in entry.metadata
            wire = build_response(
                message,
                answers=answers,
                authority=authority,
                rcode=entry.rcode,
                authoritative=authoritative,
            )
            truncated = False
            if transport == "udp":
                wire, truncated = maybe_truncate_udp(wire, self.config.udp_payload_limit)
            record = self._record(
                started_ns=started_ns,
                transport=transport,
                client=client,
                transaction_id=message.header.transaction_id,
                qname=question.name,
                qtype=question.type_name,
                rcode=RCODE_NAMES.get(entry.rcode, f"RCODE{entry.rcode}"),
                business_code="OK",
                cache_status=served[2],
                source=entry.source,
                source_serial=entry.source_serial,
                remaining_ttl=remaining_ttl,
                attribution=entry.attribution,
                answers=tuple(record.canonical() for record in answers),
                truncated=truncated,
            )
            return QueryOutcome(wire, record)
        except BusinessError as exc:
            rcode = self._error_rcode(exc.code)
            try:
                wire = build_error_response(
                    data,
                    rcode,
                    question=question,
                    flags=message.header.flags if message is not None else 0,
                )
            except Exception:
                wire = build_error_response(data, RCODE_SERVFAIL, question=None)
            record = self._record(
                started_ns=started_ns,
                transport=transport,
                client=client,
                transaction_id=transaction_id,
                qname=question.name if question is not None else "<unparsed>",
                qtype=question.type_name if question is not None else "-",
                rcode=RCODE_NAMES.get(rcode, f"RCODE{rcode}"),
                business_code=exc.code,
                cache_status="ERROR",
                error=exc.render(),
                error_offset=exc.offset,
            )
            return QueryOutcome(wire, record)
        except Exception as exc:  # defensive boundary for a long-running service
            wire = build_error_response(data, RCODE_SERVFAIL, question=question)
            record = self._record(
                started_ns=started_ns,
                transport=transport,
                client=client,
                transaction_id=transaction_id,
                qname=question.name if question is not None else "<unparsed>",
                qtype=question.type_name if question is not None else "-",
                rcode="SERVFAIL",
                business_code="E_STATE",
                cache_status="ERROR",
                error=f"E_STATE | unexpected service error: {type(exc).__name__}: {exc}",
            )
            return QueryOutcome(wire, record)

    def _serve_question(self, question: Question, transaction_id: int) -> tuple[CacheEntry, int, str]:
        for attempt in range(2):
            zone, epoch = self._capture_state()
            key = CacheKey(question.name, question.qtype, question.qclass)
            holder: dict[str, CacheEntry] = {}

            def loader() -> CacheEntry:
                if zone.is_covered(question.name):
                    entry = self._load_local_entry(zone, question, transaction_id)
                else:
                    entry = self._load_upstream_entry(zone, question, transaction_id)
                holder["entry"] = entry
                return entry

            def guard() -> bool:
                # These integer reads intentionally avoid taking state/upstream
                # locks while the cache condition lock is held.
                if self._state_epoch != epoch:
                    return False
                entry = holder.get("entry")
                if entry is not None and self._entry_is_upstream(entry):
                    current_version = self._upstream_version
                    if current_version and entry.source_serial != current_version:
                        return False
                return True

            try:
                hit = self.cache.get_or_fill(key, loader, commit_guard=guard)
            except FillDiscarded:
                if attempt == 0:
                    continue
                raise BusinessError("E_STATE", "cache fill was discarded after a state change")
            status = "MISS" if hit.filled else "HIT"
            return hit.entry, hit.remaining_ttl, status
        raise BusinessError("E_STATE", "unable to obtain a coherent cache value")

    @staticmethod
    def _entry_is_upstream(entry: CacheEntry) -> bool:
        return entry.source == "UPSTREAM" or "origin=UPSTREAM" in entry.metadata

    def _load_local_entry(self, zone: ZoneImage, question: Question, transaction_id: int) -> CacheEntry:
        resolution = self.resolver.resolve(zone, question.name, question.qtype)
        if resolution.incomplete_target is not None:
            return self._load_local_cname_plus_upstream(zone, question, resolution, transaction_id)
        if resolution.negative:
            ttl = min(resolution.ttl, self.negative_max_ttl)
            authority = tuple(
                replace(record, ttl=min(record.ttl, ttl)) if record.rtype == TYPE_BY_NAME["SOA"] else record
                for record in resolution.authority
            )
            source: CacheSource = "NEGATIVE"
            metadata = ("authority=local", "origin=ZONE")
        else:
            ttl = resolution.ttl
            authority = resolution.authority
            source = "ZONE"
            metadata = ("authority=local", "origin=ZONE")
        return CacheEntry(
            key=CacheKey(question.name, question.qtype, question.qclass),
            rcode=resolution.rcode,
            answers=resolution.answers,
            authority=authority,
            ttl=ttl,
            stored_ns=self.clock.monotonic_ns(),
            source=source,
            source_serial=zone.version,
            attribution=resolution.attribution,
            negative=resolution.negative,
            metadata=metadata,
        )

    def _load_local_cname_plus_upstream(
        self,
        zone: ZoneImage,
        question: Question,
        resolution: Resolution,
        transaction_id: int,
    ) -> CacheEntry:
        target = resolution.incomplete_target
        assert target is not None
        upstream = self.upstream.fetch(
            Question(target, question.qtype, question.qclass),
            transaction_id,
        )
        self._observe_upstream_version(upstream.upstream_version)
        arbitrated = self._arbitrate_upstream(zone, target, question.qtype, upstream)
        answers = resolution.answers + arbitrated.answers
        combined = _combine_resolution_and_upstream(resolution, arbitrated)
        if arbitrated.negative:
            ttl = min(
                answers[0].ttl if answers else 0,
                self._negative_ttl(arbitrated.authority, answers),
                self.negative_max_ttl,
            )
        else:
            ttl = min(record.ttl for record in answers)
        serial = upstream.upstream_version or self.upstream_version
        return CacheEntry(
            key=CacheKey(question.name, question.qtype, question.qclass),
            rcode=combined[0],
            answers=answers,
            authority=combined[1],
            ttl=ttl,
            stored_ns=self.clock.monotonic_ns(),
            source="NEGATIVE" if arbitrated.negative else "UPSTREAM",
            source_serial=serial,
            attribution=resolution.attribution + upstream.attribution + arbitrated.attribution
            + ("merged=local-cname+upstream",),
            negative=arbitrated.negative,
            metadata=("origin=UPSTREAM", f"upstream_version={serial}", "chain=local-prefix"),
        )

    def _load_upstream_entry(self, zone: ZoneImage, question: Question, transaction_id: int) -> CacheEntry:
        upstream = self.upstream.fetch(question, transaction_id)
        serial = upstream.upstream_version or self.upstream_version
        self._observe_upstream_version(upstream.upstream_version)
        arbitrated = self._arbitrate_upstream(zone, question.name, question.qtype, upstream)
        if arbitrated.negative:
            ttl = min(self._negative_ttl(arbitrated.authority, arbitrated.answers), self.negative_max_ttl)
            source: CacheSource = "NEGATIVE"
        else:
            ttl = min((record.ttl for record in arbitrated.answers), default=0)
            source = "UPSTREAM"
        return CacheEntry(
            key=CacheKey(question.name, question.qtype, question.qclass),
            rcode=arbitrated.rcode,
            answers=arbitrated.answers,
            authority=arbitrated.authority,
            ttl=ttl,
            stored_ns=self.clock.monotonic_ns(),
            source=source,
            source_serial=serial,
            attribution=upstream.attribution + arbitrated.attribution,
            negative=arbitrated.negative,
            metadata=("origin=UPSTREAM", f"upstream_version={serial}"),
        )

    def _arbitrate_upstream(
        self,
        zone: ZoneImage,
        query_name: str,
        query_type: int,
        upstream: UpstreamResult,
    ) -> _ArbitratedUpstream:
        answers = list(upstream.answers)
        conflict_index = next(
            (index for index, record in enumerate(answers) if zone.is_covered(record.name)),
            None,
        )
        arbitration: list[str] = []
        result_answers: tuple[ResourceRecord, ...]
        result_authority: tuple[ResourceRecord, ...]
        rcode = upstream.rcode
        negative = upstream.rcode == RCODE_NXDOMAIN or (
            upstream.rcode == RCODE_NOERROR and not answers
        )
        conflicts: list[str] = []
        if conflict_index is None:
            result_answers = upstream.answers
            result_authority = tuple(
                record for record in upstream.authority if not zone.is_covered(record.name)
            )
            if len(result_authority) != len(upstream.authority):
                arbitration.append("arbitration=local-authority-wins scope=authority")
                conflicts.append("upstream-authority-covered-local-zone")
        else:
            prefix = answers[:conflict_index]
            owner = answers[conflict_index].name
            conflicts.append(f"upstream={answers[conflict_index].canonical()} owner={owner}")
            try:
                local = self.resolver.resolve(zone, owner, query_type)
            except BusinessError as exc:
                raise BusinessError(
                    "E_CHAIN",
                    "local authoritative arbitration failed",
                    details={"cause": exc.render(), "owner": owner},
                ) from exc
            result_answers = tuple(prefix) + local.answers
            result_authority = local.authority
            rcode = local.rcode
            negative = local.negative
            local_summary = ",".join(record.canonical() for record in local.answers) or "NEGATIVE"
            arbitration.append(
                "arbitration=local-authority-wins "
                f"owner={owner} upstream_record=discarded local_result={local_summary}"
            )
        if negative:
            arbitration.append("upstream-negative-cache-basis=SOA")
        return _ArbitratedUpstream(
            answers=result_answers,
            authority=result_authority,
            rcode=rcode,
            negative=negative,
            attribution=tuple(arbitration),
            conflict=tuple(conflicts),
        )

    @staticmethod
    def _negative_ttl(authority: Iterable[ResourceRecord], answers: Iterable[ResourceRecord] = ()) -> int:
        soa_ttls = [
            min(record.ttl, record.rdata.minimum)
            for record in authority
            if record.rtype == TYPE_BY_NAME["SOA"] and isinstance(record.rdata, SOAData)
        ]
        ttl = min(soa_ttls, default=0)
        answer_ttls = [record.ttl for record in answers]
        if answer_ttls:
            ttl = min(ttl, min(answer_ttls)) if ttl else min(answer_ttls)
        return max(0, ttl)

    def _observe_upstream_version(self, version: int, *, allow_rollback: bool = False) -> UpstreamVersionReport:
        if version <= 0:
            with self._upstream_lock:
                return UpstreamVersionReport(self._upstream_version, self._upstream_version, 0)
        with self._upstream_lock:
            previous = self._upstream_version
            if previous and version < previous and not allow_rollback:
                return UpstreamVersionReport(previous, previous, 0)
            if version == previous:
                return UpstreamVersionReport(previous, previous, 0)
            self._upstream_version = version
            stale_keys = [
                entry.key
                for entry in self.cache.entries()
                if self._entry_is_upstream(entry)
            ]
            removed = self.cache.invalidate_keys(stale_keys)
            return UpstreamVersionReport(previous, version, len(removed))

    def refresh_upstream_version(self) -> UpstreamVersionReport:
        version = self.upstream.probe_version(self.config.upstream_probe_name)
        if version <= 0:
            raise BusinessError("E_CHAIN", "upstream version probe returned no SOA serial")
        return self._observe_upstream_version(version, allow_rollback=True)

    def reload_zone(self, path: str | Path | None = None) -> ZoneLoadResult:
        if path is None and not self.zone.source_path:
            raise BusinessError("E_ZONE", "zone has no source path", field="file")
        source = Path(path) if path is not None else Path(self.zone.source_path)
        result = self.zone_manager.load_file(source)
        if result.changed:
            with self._state_lock:
                self._zone = result.image
                self._state_epoch += 1
                self.cache.clear()
        return result

    def set_negative_max_ttl(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BusinessError("E_ARGUMENT", "negative TTL cap must be a non-negative integer")
        with self._state_lock:
            self._negative_max_ttl = value
            self._state_epoch += 1
            self.cache.invalidate_sources(("NEGATIVE",))
            return value

    def history(self, limit: int | None = None) -> list[QueryRecord]:
        with self._history_lock:
            values = list(self._history)
        return values[-limit:] if limit is not None else values

    def history_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.history()]

    def cache_rows(self) -> list[dict[str, Any]]:
        now = self.clock.monotonic_ns()
        rows = []
        for entry in self.cache.entries():
            rows.append(
                {
                    "key": entry.key.display,
                    "source": entry.source,
                    "serial": entry.source_serial,
                    "ttl": entry.remaining_ttl(now),
                    "age_seconds": entry.age_seconds(now),
                    "first_write_monotonic_ns": entry.stored_ns,
                    "rcode": RCODE_NAMES.get(entry.rcode, f"RCODE{entry.rcode}"),
                    "negative": entry.negative,
                    "attribution": "; ".join(entry.attribution + entry.metadata),
                }
            )
        return rows

    def snapshot_rows(self) -> list[dict[str, Any]]:
        rows = []
        current_entries = self.cache.entries()
        for info in self.snapshots.list():
            try:
                payload = self.snapshots.load(info.path.name)
                consistent, differences = compare_state(
                    payload,
                    current_zone=self.zone,
                    current_cache_entries=current_entries,
                    current_negative_max_ttl=self.negative_max_ttl,
                    current_upstream_version=self.upstream_version,
                )
            except BusinessError as exc:
                consistent, differences = False, [exc.render()]
            rows.append(
                {
                    "snapshot": info.snapshot_id,
                    "name": info.name,
                    "revision": info.revision,
                    "zone_version": info.zone_version,
                    "entries": info.cache_entries,
                    "upstream_version": info.upstream_version,
                    "consistent": consistent,
                    "differences": differences,
                }
            )
        return rows

    def save_snapshot(self, name: str, *, created_utc: str | None = None) -> SnapshotInfo:
        with self._state_lock:
            zone = self._zone
            epoch = self._state_epoch
            upstream_version = self.upstream_version
            negative_policy = {
                "max_ttl": self._negative_max_ttl,
                "basis": "min(SOA TTL, SOA MINIMUM) capped by negative_max_ttl",
            }
        cache_entries = self.cache.snapshot_view()
        info, _ = self.snapshots.save(
            name=name,
            zone=zone,
            cache_entries=cache_entries,
            history=self.history_dicts(),
            negative_policy=negative_policy,
            upstream_version=upstream_version,
            cache_generation=self.cache.generation,
            state_epoch=epoch,
            clock=self.clock,
            created_utc=created_utc,
        )
        return info

    def compare_snapshot(self, locator: str) -> tuple[SnapshotInfo, bool, tuple[str, ...]]:
        payload = self.snapshots.load(locator)
        info = self.snapshots.payload_info(payload)
        consistent, differences = compare_state(
            payload,
            current_zone=self.zone,
            current_cache_entries=self.cache.entries(),
            current_negative_max_ttl=self.negative_max_ttl,
            current_upstream_version=self.upstream_version,
        )
        return info, consistent, tuple(differences)

    def restore_snapshot(self, locator: str) -> RestoreResult:
        payload = self.snapshots.load(locator)
        info = self.snapshots.payload_info(payload)
        with self._state_lock:
            consistent, differences = compare_state(
                payload,
                current_zone=self._zone,
                current_cache_entries=self.cache.entries(),
                current_negative_max_ttl=self._negative_max_ttl,
                current_upstream_version=self.upstream_version,
            )
            restored_zone = zone_from_json(payload.get("zone"), max_ttl=self.config.zone_max_ttl)
            policy = payload.get("negative_policy", {})
            if not isinstance(policy, dict):
                raise BusinessError("E_SNAPSHOT", "snapshot negative policy is invalid", field="negative_policy")
            restored_negative_max = int(policy.get("max_ttl", self._negative_max_ttl))
            if restored_negative_max < 0:
                raise BusinessError("E_SNAPSHOT", "snapshot negative TTL cap is negative", field="negative_policy")
            raw_entries = payload.get("cache_entries", [])
            if not isinstance(raw_entries, list):
                raise BusinessError("E_SNAPSHOT", "snapshot cache entries are invalid", field="cache_entries")
            now = self.clock.monotonic_ns()
            restored_entries = [cache_entry_from_json(raw, now_ns=now) for raw in raw_entries]
            if len(restored_entries) > self.cache.capacity:
                raise BusinessError(
                    "E_SNAPSHOT",
                    "snapshot has more entries than the configured cache capacity",
                    field="cache_entries",
                )
            restored_history = [
                QueryRecord.from_dict(raw)
                for raw in payload.get("history", [])
                if isinstance(raw, dict)
            ]
            restored_upstream = int(payload.get("upstream_version", 0))

            self.zone_manager.install_image(restored_zone)
            self._zone = restored_zone
            self._negative_max_ttl = restored_negative_max
            with self._upstream_lock:
                self._upstream_version = restored_upstream
            self._state_epoch += 1
            count = self.cache.replace_all(restored_entries)
            with self._history_lock:
                self._history = restored_history
                self._next_history_sequence = max(
                    (record.sequence for record in restored_history),
                    default=0,
                ) + 1
            return RestoreResult(info, consistent, tuple(differences), count)


def _combine_resolution_and_upstream(
    local: Resolution,
    upstream: _ArbitratedUpstream,
) -> tuple[int, tuple[ResourceRecord, ...]]:
    authority = local.authority + upstream.authority
    return upstream.rcode, authority
