"""Tkinter workbench for Windows/local-desktop operation.

The module imports cleanly without Tk installed; run_gui() reports a business
error explaining that Python's optional Tk component is missing.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Callable, Iterable

from .clock import SystemMonotonicClock
from .errors import BusinessError
from .netutil import query_tcp, query_udp
from .service import ResolverService, ServiceConfig
from .wire import make_query, parse_message

try:  # pragma: no cover - availability is environment-specific
    import tkinter as tk
    from tkinter import messagebox, ttk
except ModuleNotFoundError:  # pragma: no cover
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONE = PROJECT_ROOT / "data" / "zones" / "lab.v1.json"


class DNSWorkbench:
    def __init__(self, root, service: ResolverService) -> None:  # type: ignore[no-untyped-def]
        self.root = root
        self.service = service
        self._next_id = 0x7000
        self._busy = False
        root.title("Loopback DNS Workbench — authority, cache, upstream, snapshots")
        root.geometry("1240x800")
        root.minsize(980, 640)
        self._build_styles()
        self._build_header()
        self._build_body()
        self._build_status()
        self.refresh_all()
        root.after(1000, self._periodic)
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Subtle.TLabel", foreground="#4a5568")
        style.configure("Treeview", rowheight=23)
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=(14, 12, 14, 6))
        header.pack(fill="x")
        left = ttk.Frame(header)
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="LOOPBACK DNS WORKBENCH", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Strict wire codec · JSON authority · monotonic TTL cache · validated TCP upstream · immutable snapshots",
            style="Subtle.TLabel",
        ).pack(anchor="w")
        buttons = ttk.Frame(header)
        buttons.pack(side="right")
        ttk.Button(buttons, text="Start listeners", style="Accent.TButton", command=self.start_service).pack(side="left", padx=3)
        ttk.Button(buttons, text="Stop", command=self.stop_service).pack(side="left", padx=3)
        ttk.Button(buttons, text="Probe upstream version", command=self.refresh_upstream).pack(side="left", padx=3)

    def _build_body(self) -> None:
        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=6)
        query_panel = ttk.Frame(body, padding=8)
        tabs = ttk.Notebook(body)
        body.add(query_panel, weight=2)
        body.add(tabs, weight=3)
        self._build_query_panel(query_panel)
        self._build_cache_tab(tabs)
        self._build_history_tab(tabs)
        self._build_snapshot_tab(tabs)
        self._build_zone_tab(tabs)

    def _build_query_panel(self, parent) -> None:  # type: ignore[no-untyped-def]
        ttk.Label(parent, text="Query console", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        form = ttk.Frame(parent)
        form.pack(fill="x", pady=(6, 4))
        self.qname = tk.StringVar(value="www.lab.test.")
        self.qtype = tk.StringVar(value="A")
        self.transport = tk.StringVar(value="UDP")
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.qname).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(6, 0))
        ttk.Label(form, text="Type").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Combobox(form, textvariable=self.qtype, values=("A", "AAAA", "CNAME", "NS", "SOA"), width=10, state="readonly").grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(5, 0))
        ttk.Combobox(form, textvariable=self.transport, values=("UDP", "TCP", "Internal"), width=10, state="readonly").grid(row=1, column=2, sticky="e", pady=(5, 0))
        form.columnconfigure(1, weight=1)
        ttk.Button(parent, text="Resolve query", style="Accent.TButton", command=self.run_query).pack(fill="x", pady=(4, 8))
        self.query_output = tk.Text(parent, wrap="word", font=("Consolas", 9), bg="#fbfcfe", relief="solid", borderwidth=1)
        self.query_output.pack(fill="both", expand=True)
        self.query_output.configure(state="disabled")

    def _tree(self, parent, columns: tuple[str, ...], widths: dict[str, int]):  # type: ignore[no-untyped-def]
        frame = ttk.Frame(parent, padding=8)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=widths.get(column, 90), stretch=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return frame, tree

    def _build_cache_tab(self, tabs) -> None:  # type: ignore[no-untyped-def]
        tab = ttk.Frame(tabs)
        tabs.add(tab, text="Cache")
        frame, self.cache_tree = self._tree(
            tab,
            ("KEY", "SOURCE", "SERIAL", "TTL", "AGE", "RCODE", "NEG", "ATTRIBUTION"),
            {"KEY": 255, "ATTRIBUTION": 320, "SOURCE": 90, "SERIAL": 70, "TTL": 60, "AGE": 55, "RCODE": 80, "NEG": 50},
        )
        frame.pack(fill="both", expand=True)
        actions = ttk.Frame(tab, padding=(8, 0, 8, 8))
        actions.pack(fill="x")
        ttk.Button(actions, text="Refresh", command=self.refresh_cache).pack(side="left")
        ttk.Button(actions, text="Clear cache", command=self.clear_cache).pack(side="left", padx=6)

    def _build_history_tab(self, tabs) -> None:  # type: ignore[no-untyped-def]
        tab = ttk.Frame(tabs)
        tabs.add(tab, text="History")
        frame, self.history_tree = self._tree(
            tab,
            ("SEQ", "TRANSPORT", "NAME", "TYPE", "RCODE", "CODE", "CACHE", "SOURCE", "SERIAL", "TTL"),
            {"SEQ": 50, "TRANSPORT": 75, "NAME": 190, "TYPE": 60, "RCODE": 80, "CODE": 90, "CACHE": 65, "SOURCE": 85, "SERIAL": 60, "TTL": 55},
        )
        frame.pack(fill="both", expand=True)
        ttk.Button(tab, text="Refresh history", command=self.refresh_history).pack(anchor="w", padx=8, pady=(0, 8))

    def _build_snapshot_tab(self, tabs) -> None:  # type: ignore[no-untyped-def]
        tab = ttk.Frame(tabs)
        tabs.add(tab, text="Snapshots")
        top = ttk.Frame(tab, padding=8)
        top.pack(fill="x")
        self.snapshot_name = tk.StringVar(value="desktop-session")
        ttk.Label(top, text="Name").pack(side="left")
        ttk.Entry(top, textvariable=self.snapshot_name, width=24).pack(side="left", padx=6)
        ttk.Button(top, text="Save immutable snapshot", command=self.save_snapshot).pack(side="left")
        ttk.Button(top, text="Show diff", command=self.diff_snapshot).pack(side="left", padx=6)
        ttk.Button(top, text="Open (whole replace)", style="Accent.TButton", command=self.open_snapshot).pack(side="left")
        frame, self.snapshot_tree = self._tree(
            tab,
            ("ID", "NAME", "REV", "ZONE", "ENTRIES", "UPSTREAM", "CONSISTENT"),
            {"ID": 125, "NAME": 180, "REV": 50, "ZONE": 55, "ENTRIES": 65, "UPSTREAM": 80, "CONSISTENT": 90},
        )
        frame.pack(fill="both", expand=False, padx=8)
        self.snapshot_diff = tk.Text(tab, height=12, wrap="word", font=("Consolas", 9), bg="#fbfcfe", relief="solid", borderwidth=1)
        self.snapshot_diff.pack(fill="both", expand=True, padx=8, pady=8)
        self.snapshot_diff.configure(state="disabled")

    def _build_zone_tab(self, tabs) -> None:  # type: ignore[no-untyped-def]
        tab = ttk.Frame(tabs)
        tabs.add(tab, text="Authority zone")
        top = ttk.Frame(tab, padding=8)
        top.pack(fill="x")
        self.zone_path = tk.StringVar(value=str(self.service.zone.source_path or DEFAULT_ZONE))
        ttk.Entry(top, textvariable=self.zone_path).pack(side="left", fill="x", expand=True)
        ttk.Button(top, text="Load JSON zone", command=self.load_zone).pack(side="left", padx=6)
        self.zone_text = tk.Text(tab, wrap="none", font=("Consolas", 9), bg="#fbfcfe", relief="solid", borderwidth=1)
        self.zone_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.zone_text.configure(state="disabled")

    def _build_status(self) -> None:
        frame = ttk.Frame(self.root, padding=(14, 4, 14, 8))
        frame.pack(fill="x")
        self.status = tk.StringVar()
        ttk.Label(frame, textvariable=self.status, style="Subtle.TLabel").pack(side="left")
        negative = ttk.Frame(frame)
        negative.pack(side="right")
        ttk.Label(negative, text="Negative TTL cap").pack(side="left")
        self.negative_ttl = tk.StringVar(value=str(self.service.negative_max_ttl))
        ttk.Spinbox(negative, from_=0, to=86400, textvariable=self.negative_ttl, width=7).pack(side="left", padx=5)
        ttk.Button(negative, text="Apply", command=self.apply_negative).pack(side="left")

    def _async(self, task: Callable[[], object], success: Callable[[object], None]) -> None:
        if self._busy:
            messagebox.showinfo("Busy", "Another operation is still running.")
            return
        self._busy = True
        self._set_status("Operation in progress…")

        def runner() -> None:
            try:
                result = task()
            except Exception as exc:  # delivered on the Tk main thread
                self.root.after(0, lambda error=exc: self._failed(error))
                return
            self.root.after(0, lambda value=result: self._succeeded(value, success))

        threading.Thread(target=runner, daemon=True).start()

    def _succeeded(self, value: object, success: Callable[[object], None]) -> None:
        self._busy = False
        try:
            success(value)
        finally:
            self.refresh_all()

    def _failed(self, exc: Exception) -> None:
        self._busy = False
        if isinstance(exc, BusinessError):
            messagebox.showerror("Business error", exc.render())
            self._set_status(f"{exc.code}: {exc.message}")
        else:
            messagebox.showerror("Unexpected error", f"{type(exc).__name__}: {exc}")
            self._set_status("E_STATE: unexpected UI worker failure")

    def _set_status(self, prefix: str) -> None:
        running = "running" if self.service.running else "stopped"
        self.status.set(
            f"{prefix}  |  {running}  |  zone v{self.service.zone.version}  |  "
            f"cache {len(self.service.cache.entries())}  |  clock {self.service.clock.source}"
        )

    def start_service(self) -> None:
        self._async(self.service.start, lambda report: self._set_status("UDP/TCP listeners bound on 127.0.0.1"))

    def stop_service(self) -> None:
        self.service.stop()
        self._set_status("Listeners stopped; session retained")

    def refresh_upstream(self) -> None:
        def done(report) -> None:  # type: ignore[no-untyped-def]
            self._set_status(
                f"Upstream {report.previous}→{report.current}; invalidated {report.invalidated} cache entries"
            )
        self._async(self.service.refresh_upstream_version, done)

    def run_query(self) -> None:
        name = self.qname.get().strip()
        qtype = self.qtype.get().strip().upper()
        transport = self.transport.get().strip().lower()
        transaction_id = self._next_id
        self._next_id = 0x7000 + ((self._next_id + 1 - 0x7000) % 0x8FFF)
        raw = make_query(name, qtype, transaction_id=transaction_id)

        def task():  # type: ignore[no-untyped-def]
            if transport == "udp":
                if not self.service.running:
                    raise BusinessError("E_STATE", "start the UDP/TCP listeners first")
                payload = query_udp(
                    self.service.config.host,
                    self.service.config.udp_port,
                    raw,
                    timeout=self.service.config.upstream_timeout + 1,
                )
                record = self._find_record(transaction_id, "udp")
            elif transport == "tcp":
                if not self.service.running:
                    raise BusinessError("E_STATE", "start the UDP/TCP listeners first")
                payload = query_tcp(
                    self.service.config.host,
                    self.service.config.tcp_port,
                    raw,
                    timeout=self.service.config.upstream_timeout + 1,
                )
                record = self._find_record(transaction_id, "tcp")
            else:
                outcome = self.service.process_wire(raw, transport="internal", client="gui")
                payload = outcome.wire
                record = outcome.record
            return payload, record

        self._async(task, lambda value: self._show_query(value))  # type: ignore[arg-type]

    def _find_record(self, transaction_id: int, transport: str):  # type: ignore[no-untyped-def]
        for record in reversed(self.service.history()):
            if record.transaction_id == transaction_id and record.transport == transport:
                return record
        raise BusinessError("E_STATE", "query completed but no history record was produced")

    def _show_query(self, value) -> None:  # type: ignore[no-untyped-def]
        payload, record = value
        response = parse_message(payload)
        lines = [
            f"QUERY seq={record.sequence} id=0x{record.transaction_id:04x} transport={record.transport}",
            f"name={record.qname} type={record.qtype} rcode={record.rcode} code={record.business_code}",
            f"cache={record.cache_status} source={record.source} serial={record.source_serial} ttl={record.remaining_ttl} truncated={str(record.truncated).lower()}",
            "",
        ]
        if response.answers:
            lines.append("ANSWER")
            for rr in response.answers:
                lines.append(f"  {rr.name}  {rr.type_name}  TTL={rr.ttl}  {rr.rdata_text()}")
        if response.authority:
            lines.append("AUTHORITY")
            for rr in response.authority:
                lines.append(f"  {rr.name}  {rr.type_name}  TTL={rr.ttl}  {rr.rdata_text()}")
        if record.attribution:
            lines.extend(["", "ATTRIBUTION"])
            lines.extend(f"  {item}" for item in record.attribution)
        if record.error:
            lines.extend(["", f"ERROR {record.error}"])
        self.query_output.configure(state="normal")
        self.query_output.delete("1.0", "end")
        self.query_output.insert("end", "\n".join(lines))
        self.query_output.configure(state="disabled")
        self._set_status(f"Query {record.business_code} / {record.cache_status}")

    def refresh_cache(self) -> None:
        for item in self.cache_tree.get_children():
            self.cache_tree.delete(item)
        for row in self.service.cache_rows():
            self.cache_tree.insert(
                "", "end",
                values=(row["key"], row["source"], row["serial"], row["ttl"], row["age_seconds"], row["rcode"],
                        "yes" if row["negative"] else "no", row["attribution"]),
            )

    def refresh_history(self) -> None:
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for record in self.service.history(250):
            self.history_tree.insert(
                "", "end",
                values=(record.sequence, record.transport, record.qname, record.qtype, record.rcode,
                        record.business_code, record.cache_status, record.source, record.source_serial,
                        record.remaining_ttl),
            )

    def refresh_snapshots(self) -> None:
        for item in self.snapshot_tree.get_children():
            self.snapshot_tree.delete(item)
        for row in self.service.snapshot_rows():
            self.snapshot_tree.insert(
                "", "end", iid=row["snapshot"],
                values=(row["snapshot"], row["name"], row["revision"], row["zone_version"],
                        row["entries"], row["upstream_version"], "yes" if row["consistent"] else "no"),
            )

    def load_zone(self) -> None:
        path = self.zone_path.get().strip()

        def done(result) -> None:  # type: ignore[no-untyped-def]
            self._set_status(
                f"Zone v{result.image.version}; changed={str(result.changed).lower()}; cache atomically cleared"
            )
            self._render_zone()
        self._async(lambda: self.service.reload_zone(path), done)

    def clear_cache(self) -> None:
        if messagebox.askyesno("Clear cache", "Atomically remove all cache entries?"):
            removed = self.service.cache.clear()
            self._set_status(f"Cache cleared: {removed} entries")
            self.refresh_cache()

    def save_snapshot(self) -> None:
        name = self.snapshot_name.get().strip()

        def done(info) -> None:  # type: ignore[no-untyped-def]
            self._set_status(
                f"Saved snapshot {info.snapshot_id}: zone v{info.zone_version}, {info.cache_entries} entries, read-only"
            )
        self._async(lambda: self.service.save_snapshot(name), done)

    def _selected_snapshot(self) -> str | None:
        selection = self.snapshot_tree.selection()
        if not selection:
            messagebox.showinfo("Snapshot", "Select a snapshot first.")
            return None
        return str(selection[0])

    def diff_snapshot(self) -> None:
        snapshot_id = self._selected_snapshot()
        if snapshot_id is None:
            return

        def done(result) -> None:  # type: ignore[no-untyped-def]
            info, consistent, differences = result
            text = [
                f"Snapshot: {info.snapshot_id} ({info.name}, revision {info.revision})",
                f"Snapshot zone: v{info.zone_version}; entries: {info.cache_entries}",
                f"Current zone: v{self.service.zone.version}; entries: {len(self.service.cache.entries())}",
                f"Status: {'CONSISTENT' if consistent else 'MISMATCH'}",
                "",
            ]
            text.extend(differences if differences else ["No version/source differences."])
            self.snapshot_diff.configure(state="normal")
            self.snapshot_diff.delete("1.0", "end")
            self.snapshot_diff.insert("end", "\n".join(text))
            self.snapshot_diff.configure(state="disabled")
        self._async(lambda: self.service.compare_snapshot(snapshot_id), done)

    def open_snapshot(self) -> None:
        snapshot_id = self._selected_snapshot()
        if snapshot_id is None:
            return
        if not messagebox.askyesno(
            "Open snapshot",
            "Restore the complete snapshot state?\n\nThis atomically replaces the active zone, cache, negative policy and history. Old snapshots remain read-only.",
        ):
            return

        def done(result) -> None:  # type: ignore[no-untyped-def]
            status = "CONSISTENT" if result.consistent else "MISMATCH"
            self._set_status(
                f"Snapshot {result.info.snapshot_id} restored as whole state ({status}); {result.restored_entries} entries"
            )
            self.query_output.configure(state="normal")
            self.query_output.delete("1.0", "end")
            self.query_output.insert(
                "end",
                f"Restored {result.info.snapshot_id} as a whole state.\n"
                f"Pre-restore comparison: {status}\n"
                f"Differences: {len(result.differences)}\n"
                + "\n".join(result.differences),
            )
            self.query_output.configure(state="disabled")
        self._async(lambda: self.service.restore_snapshot(snapshot_id), done)

    def apply_negative(self) -> None:
        try:
            value = int(self.negative_ttl.get())
        except ValueError:
            messagebox.showerror("Negative policy", "TTL cap must be an integer.")
            return
        try:
            applied = self.service.set_negative_max_ttl(value)
        except BusinessError as exc:
            messagebox.showerror("Negative policy", exc.render())
            return
        self._set_status(f"Negative TTL cap={applied}; negative cache entries invalidated")
        self.refresh_all()

    def _render_zone(self) -> None:
        zone = self.service.zone
        lines = [
            f"origin={zone.origin}",
            f"version={zone.version}",
            f"file_digest={zone.digest}",
            f"content_digest={zone.content_digest}",
            f"records={len(zone.records)}",
            f"source={zone.source_path or '<snapshot>'}",
            "",
        ]
        for rr in zone.records:
            lines.append(f"{rr.name:<32} {rr.type_name:<6} {rr.ttl:>8}  {rr.rdata_text()}")
        self.zone_text.configure(state="normal")
        self.zone_text.delete("1.0", "end")
        self.zone_text.insert("end", "\n".join(lines))
        self.zone_text.configure(state="disabled")
        if zone.source_path:
            self.zone_path.set(zone.source_path)

    def refresh_all(self) -> None:
        self.refresh_cache()
        self.refresh_history()
        self.refresh_snapshots()
        self._render_zone()
        self._set_status("Ready")

    def _periodic(self) -> None:
        if not self._busy:
            self.refresh_cache()
            self.refresh_history()
            self._set_status("Ready")
        self.root.after(1500, self._periodic)

    def _close(self) -> None:
        self.service.stop()
        self.root.destroy()


def run_gui(service: ResolverService) -> None:
    if tk is None:
        raise BusinessError(
            "E_STATE",
            "Tkinter is unavailable in this Python installation; install the official Windows Python build or the OS python3-tk package",
        )
    try:
        root = tk.Tk()
    except Exception as exc:  # TclError when no display server is available
        raise BusinessError("E_STATE", f"Tk display could not be opened: {exc}") from exc
    DNSWorkbench(root, service)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tkinter loopback DNS workbench")
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
    )
    try:
        service = ResolverService(
            args.zone,
            config=config,
            clock=SystemMonotonicClock(),
            snapshot_root=args.snapshot_dir,
        )
        run_gui(service)
    except BusinessError as exc:
        print(f"ERROR {exc.render()}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
