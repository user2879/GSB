"""Interactive professional command-line application for the local DNS service."""

from __future__ import annotations

import argparse
import cmd
import shlex
import sys
from pathlib import Path
from typing import Iterable

from .clock import SystemMonotonicClock
from .errors import BusinessError
from .netutil import query_tcp, query_udp
from .service import ResolverService, ServiceConfig
from .wire import ResourceRecord, make_query, parse_message


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONE = PROJECT_ROOT / "data" / "zones" / "lab.v1.json"


def format_rr(record: ResourceRecord) -> str:
    return f"{record.name}|{record.type_name}|{record.ttl}|{record.rdata_text()}"


def print_table(headers: tuple[str, ...], rows: Iterable[Iterable[object]]) -> None:
    materialized = [[str(cell) for cell in row] for row in rows]
    if not materialized:
        print("(empty)")
        return
    widths = [len(header) for header in headers]
    for row in materialized:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in materialized:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


class DNSConsole(cmd.Cmd):
    intro = (
        "Loopback DNS workbench. Type 'help' for commands. "
        "No external network is used; the upstream is a loopback TCP simulator."
    )
    prompt = "dns-local> "

    def __init__(self, service: ResolverService) -> None:
        super().__init__()
        self.service = service
        self._next_id = 0x6000

    def emptyline(self) -> None:
        return

    def do_start(self, arg: str) -> None:
        """start — bind the configured loopback UDP and TCP service ports."""
        try:
            report = self.service.start()
            self._print_report(report, running=True)
        except BusinessError as exc:
            self._error(exc)

    def do_stop(self, arg: str) -> None:
        """stop — stop the UDP/TCP listeners (the session/cache stays in memory)."""
        self.service.stop()
        print("STATE running=false cache_retained=true")

    def do_status(self, arg: str) -> None:
        """status — show ports, clock source, zone version and cache size."""
        self._print_report(self.service.startup_report(), running=self.service.running)

    def _print_report(self, report: dict[str, object], *, running: bool) -> None:
        fields = [
            ("running", str(running).lower()),
            ("host", report["host"]),
            ("udp_port", report["udp_port"]),
            ("tcp_port", report["tcp_port"]),
            ("upstream", report["upstream"]),
            ("clock_source", report["clock_source"]),
            ("zone", report["zone_origin"]),
            ("zone_version", report["zone_version"]),
            ("zone_digest", str(report["zone_digest"])[:16]),
            ("cache_capacity", report["cache_capacity"]),
            ("negative_max_ttl", report["negative_max_ttl"]),
            ("upstream_version", report["upstream_version"]),
            ("cache_entries", len(self.service.cache.entries())),
        ]
        print("STATUS " + " ".join(f"{key}={value}" for key, value in fields))

    def do_query(self, arg: str) -> None:
        """query NAME [TYPE] [udp|tcp|internal] — resolve one local deterministic query."""
        parts = shlex.split(arg)
        if not parts:
            print("usage: query NAME [TYPE] [udp|tcp|internal]")
            return
        name = parts[0]
        qtype = parts[1].upper() if len(parts) > 1 else "A"
        transport = parts[2].lower() if len(parts) > 2 else ("udp" if self.service.running else "internal")
        if transport not in ("udp", "tcp", "internal"):
            print("ERROR code=E_ARGUMENT message=transport must be udp, tcp or internal")
            return
        transaction_id = self._next_id
        self._next_id = 0x6000 + ((self._next_id + 1 - 0x6000) % 0x8FFF)
        raw = make_query(name, qtype, transaction_id=transaction_id)
        try:
            if transport == "udp":
                if not self.service.running:
                    raise BusinessError("E_STATE", "service is not running; use 'start' or transport=internal")
                payload = query_udp(
                    self.service.config.host,
                    self.service.config.udp_port,
                    raw,
                    timeout=self.service.config.upstream_timeout + 1,
                )
                record = self.service.history()[-1]
            elif transport == "tcp":
                if not self.service.running:
                    raise BusinessError("E_STATE", "service is not running; use 'start' or transport=internal")
                payload = query_tcp(
                    self.service.config.host,
                    self.service.config.tcp_port,
                    raw,
                    timeout=self.service.config.upstream_timeout + 1,
                )
                record = self.service.history()[-1]
            else:
                outcome = self.service.process_wire(raw, transport="internal", client="cli")
                payload = outcome.wire
                record = outcome.record
            response = parse_message(payload)
        except BusinessError as exc:
            self._error(exc)
            return
        print(
            f"QUERY seq={record.sequence} id=0x{record.transaction_id:04x} transport={record.transport} "
            f"name={record.qname} type={record.qtype} rcode={record.rcode} code={record.business_code} "
            f"cache={record.cache_status} source={record.source} serial={record.source_serial} ttl={record.remaining_ttl}"
        )
        if response.answers:
            print_table(("NAME", "TYPE", "TTL", "RDATA"), (
                (rr.name, rr.type_name, rr.ttl, rr.rdata_text()) for rr in response.answers
            ))
        if response.authority:
            print_table(("AUTHORITY", "TYPE", "TTL", "RDATA"), (
                (rr.name, rr.type_name, rr.ttl, rr.rdata_text()) for rr in response.authority
            ))
        if record.attribution:
            print("ATTRIBUTION " + "; ".join(record.attribution))

    def do_zone(self, arg: str) -> None:
        """zone load PATH | zone show — reload JSON authority or show its state."""
        parts = shlex.split(arg)
        if not parts:
            print("usage: zone load PATH | zone show")
            return
        if parts[0] == "show":
            zone = self.service.zone
            print(
                f"ZONE origin={zone.origin} version={zone.version} records={len(zone.records)} "
                f"digest={zone.digest} source={zone.source_path or '-'}"
            )
            print_table(("NAME", "TYPE", "TTL", "RDATA"), (
                (rr.name, rr.type_name, rr.ttl, rr.rdata_text()) for rr in zone.records
            ))
            return
        if parts[0] == "load" and len(parts) == 2:
            try:
                result = self.service.reload_zone(parts[1])
            except BusinessError as exc:
                self._error(exc)
                return
            print(
                f"ZONE_LOAD version={result.image.version} changed={str(result.changed).lower()} "
                f"digest={result.image.digest} cache=cleared"
            )
            return
        print("usage: zone load PATH | zone show")

    def do_cache(self, arg: str) -> None:
        """cache [list|clear] — inspect or atomically clear the TTL cache."""
        action = arg.strip().lower() or "list"
        if action == "clear":
            removed = self.service.cache.clear()
            print(f"CACHE_CLEAR removed={removed}")
            return
        if action != "list":
            print("usage: cache [list|clear]")
            return
        rows = self.service.cache_rows()
        print_table(
            ("KEY", "SOURCE", "SERIAL", "TTL", "AGE", "RCODE", "NEGATIVE", "ATTRIBUTION"),
            (
                (row["key"], row["source"], row["serial"], row["ttl"], row["age_seconds"], row["rcode"],
                 str(row["negative"]).lower(), row["attribution"])
                for row in rows
            ),
        )

    def do_history(self, arg: str) -> None:
        """history [N] — show stored query history (newest last)."""
        limit = 20
        if arg.strip():
            try:
                limit = int(arg.strip())
            except ValueError:
                print("ERROR code=E_ARGUMENT message=history limit must be an integer")
                return
        rows = self.service.history(limit)
        print_table(
            ("SEQ", "TRANSPORT", "NAME", "TYPE", "RCODE", "CODE", "CACHE", "SOURCE", "SERIAL", "TTL"),
            (
                (record.sequence, record.transport, record.qname, record.qtype, record.rcode,
                 record.business_code, record.cache_status, record.source, record.source_serial,
                 record.remaining_ttl)
                for record in rows
            ),
        )

    def do_snapshot(self, arg: str) -> None:
        """snapshot save NAME | list | diff ID | open ID — immutable session snapshots."""
        parts = shlex.split(arg)
        if not parts:
            print("usage: snapshot save NAME | list | diff ID | open ID")
            return
        action = parts[0].lower()
        try:
            if action == "save" and len(parts) >= 2:
                info = self.service.save_snapshot(" ".join(parts[1:]))
                print(
                    f"SNAPSHOT_SAVE id={info.snapshot_id} name={info.name} revision={info.revision} "
                    f"zone_version={info.zone_version} entries={info.cache_entries} readonly=true"
                )
                return
            if action == "list":
                print_table(
                    ("ID", "NAME", "REV", "ZONE", "ENTRIES", "UPSTREAM", "CONSISTENT"),
                    (
                        (row["snapshot"], row["name"], row["revision"], row["zone_version"],
                         row["entries"], row["upstream_version"], str(row["consistent"]).lower())
                        for row in self.service.snapshot_rows()
                    ),
                )
                return
            if action == "diff" and len(parts) == 2:
                info, consistent, differences = self.service.compare_snapshot(parts[1])
                print(
                    f"SNAPSHOT_DIFF id={info.snapshot_id} consistent={str(consistent).lower()} "
                    f"differences={len(differences)}"
                )
                for item in differences:
                    print(f"DIFF {item}")
                return
            if action == "open" and len(parts) == 2:
                result = self.service.restore_snapshot(parts[1])
                print(
                    f"SNAPSHOT_OPEN id={result.info.snapshot_id} consistent={str(result.consistent).lower()} "
                    f"mode=whole entries={result.restored_entries} differences={len(result.differences)}"
                )
                for item in result.differences:
                    print(f"DIFF {item}")
                return
        except BusinessError as exc:
            self._error(exc)
            return
        print("usage: snapshot save NAME | list | diff ID | open ID")

    def do_upstream(self, arg: str) -> None:
        """upstream version | upstream refresh — inspect or probe the loopback simulator."""
        action = arg.strip().lower() or "version"
        if action == "version":
            print(f"UPSTREAM version={self.service.upstream_version} endpoint={self.service.config.upstream_host}:{self.service.config.upstream_port}")
            return
        if action == "refresh":
            try:
                report = self.service.refresh_upstream_version()
            except BusinessError as exc:
                self._error(exc)
                return
            print(
                f"UPSTREAM_REFRESH previous={report.previous} current={report.current} "
                f"invalidated={report.invalidated}"
            )
            return
        print("usage: upstream version | upstream refresh")

    def do_negative(self, arg: str) -> None:
        """negative TTL — set the negative-cache cap and invalidate negative entries."""
        try:
            value = int(arg.strip())
            applied = self.service.set_negative_max_ttl(value)
        except (ValueError, BusinessError) as exc:
            if isinstance(exc, BusinessError):
                self._error(exc)
            else:
                print("ERROR code=E_ARGUMENT message=negative TTL must be an integer")
            return
        print(f"NEGATIVE_POLICY max_ttl={applied} entries_invalidated=true")

    @staticmethod
    def _error(exc: BusinessError) -> None:
        print(f"ERROR {exc.render()}")

    def do_quit(self, arg: str) -> bool:
        """quit — stop listeners and leave the process."""
        self.service.stop()
        print("BYE")
        return True

    do_exit = do_quit
    do_EOF = do_quit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive loopback DNS and cache workbench")
    parser.add_argument("--zone", default=str(DEFAULT_ZONE))
    parser.add_argument("--snapshot-dir", default=str(PROJECT_ROOT / "snapshots"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15353)
    parser.add_argument("--tcp-port", type=int, default=15353)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=15354)
    parser.add_argument("--upstream-probe-name", default="upstream.test.")
    parser.add_argument("--cache-capacity", type=int, default=128)
    parser.add_argument("--negative-max-ttl", type=int, default=300)
    parser.add_argument("--cname-max-hops", type=int, default=8)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--run", help="run one console command and exit")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = ServiceConfig(
        host=args.host,
        udp_port=args.udp_port,
        tcp_port=args.tcp_port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
        upstream_probe_name=args.upstream_probe_name,
        cache_capacity=args.cache_capacity,
        negative_max_ttl=args.negative_max_ttl,
        cname_max_hops=args.cname_max_hops,
    )
    try:
        service = ResolverService(
            args.zone,
            config=config,
            clock=SystemMonotonicClock(),
            snapshot_root=args.snapshot_dir,
        )
    except BusinessError as exc:
        print(f"ERROR {exc.render()}", file=sys.stderr)
        return 2
    if args.auto_start:
        try:
            service.start()
        except BusinessError as exc:
            print(f"ERROR {exc.render()}", file=sys.stderr)
            return 2
    console = DNSConsole(service)
    try:
        if args.run:
            console.onecmd(args.run)
            service.stop()
            return 0
        console.cmdloop()
    except KeyboardInterrupt:
        print()
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
