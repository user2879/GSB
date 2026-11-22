from __future__ import annotations

import threading
import time
import queue
from pathlib import Path
from typing import Any

from .engine import MeshRuntime
from .errors import MeshError
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .model import load_requests, load_spec
from .snapshot import SnapshotStore
from .viewmodel import build_view_model


def _ratio_text(fixed: int, scale: int) -> str:
    return f"{fixed} / {scale} = {fixed / scale:.6f}"


def launch_ui(
    *,
    spec_path: str,
    requests_path: str,
    ledger_path: str,
    snapshot_dir: str,
    revision: int | None,
    logger: StructuredLogger,
) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # Tk can be absent on minimal CI hosts
        raise MeshError(
            "E_HOST",
            f"Tkinter is unavailable in this Python environment: {exc}",
            field="ui.tkinter",
            location=Path(spec_path).parent.as_posix(),
        ) from exc

    class DiagnosticsApp:
        def __init__(self, root: "tk.Tk") -> None:
            self.root = root
            self.root.title("Meshwork · Local Service-Mesh Diagnostics")
            self.root.geometry("1320x820")
            self.root.minsize(980, 640)
            self.spec = load_spec(spec_path)
            self.revision = self.spec.active_revision(revision)
            self.requests = load_requests(requests_path, self.spec.services.keys())
            self.logger = logger
            self.snapshot_store = SnapshotStore(snapshot_dir, logger)
            self.ledger_path = ledger_path
            self._io_lock = threading.RLock()
            self._busy = False
            self._closed = False
            self._refresh_job: str | None = None
            self._ui_events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
            self._last_report: dict[str, Any] | None = None
            self._last_view: dict[str, Any] | None = None
            self.ledger = Ledger(self.ledger_path, logger=logger)
            self.runtime = MeshRuntime(
                self.spec,
                self.revision,
                self.ledger,
                self.snapshot_store,
                logger,
                f"ui-{int(time.time())}",
            )
            self.runtime.validate_or_replace_snapshot()
            self._build_style(ttk)
            self._build_widgets(tk, ttk)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self._refresh()

        def _build_style(self, ttk: Any) -> None:
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
            style.configure("Mono.TLabel", font=("Consolas", 9))
            style.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
            style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
            style.configure("Status.TLabel", font=("Consolas", 9))

        def _build_widgets(self, tk: Any, ttk: Any) -> None:
            header = ttk.Frame(self.root, padding=(10, 8, 10, 4))
            header.pack(fill="x")
            ttk.Label(header, text="Meshwork", style="Header.TLabel").pack(side="left")
            ttk.Label(
                header,
                text="deterministic loopback-only traffic governance diagnostics",
            ).pack(side="left", padx=(12, 0))
            self.clock_label = ttk.Label(header, text="", style="Mono.TLabel")
            self.clock_label.pack(side="right")
            self.refresh_digest_label = ttk.Label(header, text="ledger refresh: -", style="Mono.TLabel")
            self.refresh_digest_label.pack(side="right", padx=(0, 18))

            toolbar = ttk.Frame(self.root, padding=(10, 2, 10, 6))
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="Refresh", command=self._refresh).pack(side="left")
            ttk.Button(toolbar, text="Reset + Run sample", command=self._run_sample).pack(
                side="left", padx=(6, 0)
            )
            ttk.Button(toolbar, text="Write snapshot", command=self._write_snapshot).pack(
                side="left", padx=(6, 0)
            )
            self.status_label = ttk.Label(toolbar, text="ready", style="Status.TLabel")
            self.status_label.pack(side="right")

            main = ttk.Panedwindow(self.root, orient="horizontal")
            main.pack(fill="both", expand=True, padx=10, pady=(0, 8))

            left = ttk.Frame(main, padding=(0, 0, 8, 0))
            main.add(left, weight=2)
            ttk.Label(left, text="Services / instances / weighted selection", style="Header.TLabel").pack(
                anchor="w"
            )
            service_columns = (
                "service",
                "instance",
                "quota",
                "base_w",
                "effective_w",
                "selected",
                "breaker",
                "cooldown_ms",
            )
            self.service_tree = ttk.Treeview(
                left, columns=service_columns, show="headings", height=12
            )
            headings = {
                "service": "service",
                "instance": "instance",
                "quota": "quota",
                "base_w": "base w",
                "effective_w": "rule w",
                "selected": "selected",
                "breaker": "breaker",
                "cooldown_ms": "cooldown ms",
            }
            widths = {
                "service": 92,
                "instance": 116,
                "quota": 52,
                "base_w": 58,
                "effective_w": 60,
                "selected": 66,
                "breaker": 76,
                "cooldown_ms": 88,
            }
            for column in service_columns:
                self.service_tree.heading(column, text=headings[column])
                self.service_tree.column(column, width=widths[column], anchor="w", stretch=True)
            self.service_tree.pack(fill="both", expand=True, pady=(6, 8))

            ttk.Label(left, text="Recent deterministic selections", style="Header.TLabel").pack(
                anchor="w"
            )
            selection_columns = (
                "request",
                "service",
                "attempt",
                "instance",
                "position",
                "range",
                "outcome",
            )
            self.selection_tree = ttk.Treeview(
                left, columns=selection_columns, show="headings", height=9
            )
            selection_headings = {
                "request": "request",
                "service": "service",
                "attempt": "try",
                "instance": "instance",
                "position": "weight pos",
                "range": "cumulative range",
                "outcome": "outcome",
            }
            for column in selection_columns:
                self.selection_tree.heading(column, text=selection_headings[column])
                self.selection_tree.column(column, width=94, anchor="w", stretch=True)
            self.selection_tree.pack(fill="both", expand=True, pady=(6, 0))

            right = ttk.Notebook(main)
            main.add(right, weight=5)
            overview = ttk.Frame(right, padding=8)
            timeline_tab = ttk.Frame(right, padding=8)
            stats_tab = ttk.Frame(right, padding=8)
            versions_tab = ttk.Frame(right, padding=8)
            right.add(overview, text="Breakers / budget")
            right.add(timeline_tab, text="Concurrency timeline")
            right.add(stats_tab, text="Fixed-point SLO")
            right.add(versions_tab, text="Versions / snapshots")

            breaker_columns = (
                "service",
                "instance",
                "state",
                "failures",
                "threshold",
                "cooldown",
                "probes",
            )
            self.breaker_tree = ttk.Treeview(
                overview, columns=breaker_columns, show="headings", height=8
            )
            for column in breaker_columns:
                self.breaker_tree.heading(column, text=column)
                self.breaker_tree.column(column, width=112, anchor="w", stretch=True)
            self.breaker_tree.pack(fill="x")
            self.overview_text = tk.Text(
                overview, height=18, wrap="word", font=("Consolas", 9), state="disabled"
            )
            self.overview_text.pack(fill="both", expand=True, pady=(8, 0))

            timeline_header = ttk.Frame(timeline_tab)
            timeline_header.pack(fill="x")
            ttk.Label(
                timeline_header,
                text="Gray = budget wait · colored = real socket attempt · label shows g/s in-flight",
            ).pack(side="left")
            timeline_frame = ttk.Frame(timeline_tab)
            timeline_frame.pack(fill="both", expand=True, pady=(6, 0))
            self.timeline_canvas = tk.Canvas(
                timeline_frame, background="#101820", highlightthickness=0
            )
            y_scroll = ttk.Scrollbar(
                timeline_frame, orient="vertical", command=self.timeline_canvas.yview
            )
            self.timeline_canvas.configure(yscrollcommand=y_scroll.set)
            self.timeline_canvas.pack(side="left", fill="both", expand=True)
            y_scroll.pack(side="right", fill="y")
            self.timeline_canvas.bind("<Configure>", lambda _event: self._render_timeline())

            stat_columns = (
                "service",
                "requests",
                "success",
                "success_fixed",
                "p95_ms",
                "p95_fixed",
                "throughput_fixed",
            )
            self.stats_tree = ttk.Treeview(
                stats_tab, columns=stat_columns, show="headings", height=9
            )
            for column in stat_columns:
                self.stats_tree.heading(column, text=column)
                self.stats_tree.column(column, width=140, anchor="w", stretch=True)
            self.stats_tree.pack(fill="x")
            self.stats_text = tk.Text(
                stats_tab, height=20, wrap="word", font=("Consolas", 9), state="disabled"
            )
            self.stats_text.pack(fill="both", expand=True, pady=(8, 0))

            self.version_text = tk.Text(
                versions_tab, height=9, wrap="word", font=("Consolas", 9), state="disabled"
            )
            self.version_text.pack(fill="x")
            snapshot_columns = ("path", "valid", "cursor", "revision", "digest")
            self.snapshot_tree = ttk.Treeview(
                versions_tab, columns=snapshot_columns, show="headings", height=14
            )
            for column in snapshot_columns:
                self.snapshot_tree.heading(column, text=column)
                self.snapshot_tree.column(
                    column,
                    width=180 if column in {"path", "digest"} else 100,
                    anchor="w",
                    stretch=True,
                )
            self.snapshot_tree.pack(fill="both", expand=True, pady=(8, 0))

        @staticmethod
        def _set_text(widget: Any, text: str) -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")

        def _current_view(self) -> dict[str, Any]:
            runtime_state = self.runtime.runtime_state()
            return build_view_model(
                self.spec,
                self.revision,
                self.ledger,
                self.snapshot_store,
                runtime_state,
                last_report=self._last_report,
            )

        def _refresh(self) -> None:
            if self._closed:
                return
            if self._refresh_job is not None:
                try:
                    self.root.after_cancel(self._refresh_job)
                except Exception:
                    pass
                self._refresh_job = None
            self._drain_ui_events()
            try:
                with self._io_lock:
                    view = self._current_view()
                    self._last_view = view
                self._render_view(view)
            except Exception as exc:
                self.status_label.configure(text=f"refresh error: {exc}")
                self.logger.error("ui_refresh_error", code="E_INTERNAL", message=str(exc))
            finally:
                if not self._closed:
                    self._refresh_job = self.root.after(1000, self._refresh)

        def _drain_ui_events(self) -> None:
            while True:
                try:
                    kind, payload = self._ui_events.get_nowait()
                except queue.Empty:
                    break
                if kind == "status":
                    self.status_label.configure(text=str(payload))
                elif kind == "report":
                    self._last_report = payload

        def _render_view(self, view: dict[str, Any]) -> None:
            session = view["session"]
            self.clock_label.configure(text=time.strftime("%Y-%m-%d %H:%M:%S"))
            unchanged = view["ledger_refresh"]["unchanged"]
            self.refresh_digest_label.configure(
                text=(
                    f"refresh digest {'stable' if unchanged else 'CHANGED'} · "
                    f"{view['versions']['ledger_content_digest'][:12]}"
                )
            )
            self.status_label.configure(
                text=(f"revision {session['route_revision']} · scale {session['scale']} · busy={self._busy}")
            )
            for row in self.service_tree.get_children():
                self.service_tree.delete(row)
            for service in view["services"]:
                for instance in service["instances"]:
                    breaker = instance["breaker"]
                    self.service_tree.insert(
                        "",
                        "end",
                        values=(
                            service["id"],
                            instance["id"],
                            service["quota"],
                            instance["base_weight"],
                            instance["effective_weight"],
                            instance["selected_count"],
                            breaker["state"],
                            breaker["cooldown_remaining_ms"],
                        ),
                    )
            for row in self.selection_tree.get_children():
                self.selection_tree.delete(row)
            for item in view["recent_selections"]:
                range_item = next(
                    (
                        entry
                        for entry in item["weight_ranges"]
                        if entry["instance_id"] == item["instance_id"]
                    ),
                    None,
                )
                range_text = "-" if range_item is None else f"[{range_item['start']},{range_item['end']})/{item['total_weight']}"
                self.selection_tree.insert(
                    "",
                    "end",
                    values=(
                        item["request_id"],
                        item["service"],
                        item["attempt_no"],
                        item["instance_id"],
                        item["weight_position"],
                        range_text,
                        item["outcome"],
                    ),
                )
            for row in self.breaker_tree.get_children():
                self.breaker_tree.delete(row)
            for breaker in view["breakers"]:
                self.breaker_tree.insert(
                    "",
                    "end",
                    values=(
                        breaker["service"],
                        breaker["instance_id"],
                        breaker["state"],
                        breaker["consecutive_failures"],
                        breaker.get("failure_threshold", "-"),
                        breaker["cooldown_remaining_ms"],
                        f"{breaker['half_open_dispatched']}/{breaker.get('half_open_probes', '-')}",
                    ),
                )
            overall = view["stats"]["overall"]
            overview_lines = [
                f"global concurrency limit : {view['budget']['global_limit']}",
                f"committed budget grants  : {view['budget']['global_used']}",
                "service budget usage     : " + ", ".join(
                    f"{key}={value}" for key, value in view["budget"]["by_service"].items()
                ),
                f"requests / successes     : {overall['requests']} / {overall['successes']}",
                "final outcome distribution: "
                + json_dumps(overall["final_distribution"]),
                "attempt distribution     : "
                + json_dumps(overall["attempt_distribution"]),
                "error distribution       : "
                + json_dumps(overall["error_distribution"]),
            ]
            runtime = view["budget"].get("runtime")
            if runtime:
                overview_lines.extend(
                    [
                        "",
                        f"live global inflight     : {runtime.get('global_inflight')}",
                        f"peak global inflight     : {runtime.get('peak_global')}",
                        "peak service inflight    : " + json_dumps(runtime.get("peak_service", {})),
                        "waiting for budget       : " + str(runtime.get("waiting", 0)),
                    ]
                )
            recovery = view["versions"].get("recovery")
            if recovery:
                overview_lines.extend(
                    [
                        "",
                        "recovery state           : restored from committed ledger only",
                        f"unknown attempts         : {recovery.get('unknown_attempts')}",
                        f"exit report available    : {recovery.get('exit_report_available')}",
                    ]
                )
            self._set_text(self.overview_text, "\n".join(overview_lines))
            self._render_timeline()
            for row in self.stats_tree.get_children():
                self.stats_tree.delete(row)
            for item in list(view["stats"]["services"]) + [overall]:
                self.stats_tree.insert(
                    "",
                    "end",
                    values=(
                        item["service"],
                        item["requests"],
                        item["successes"],
                        _ratio_text(item["success_ratio_fixed"], view["stats"]["scale"]),
                        item["p95_latency_ms"],
                        item["p95_latency_ms_fixed"],
                        item["throughput_rps_fixed"],
                    ),
                )
            method = view["stats"]["method"]
            stats_lines = [
                f"rounding                 : {method['rounding']}",
                f"ratio formula            : {method['ratio_formula']}",
                f"p95 formula              : {method['p95_formula']}",
                f"throughput formula       : {method['throughput_formula']}",
                f"single-value error       : {json_dumps(method['single_value_quantization_error'])}",
                f"comparison error bound   : {json_dumps(method['comparison_error_bound'])}",
                f"stable tie break         : {method['stable_tie_break']}",
                "",
                "success-ratio ranking:",
            ]
            for rank in view["stats"]["rankings"]["success_ratio"]:
                stats_lines.append(
                    f"  {rank['rank']}. {rank['service']} value={rank['value']} "
                    f"error_affected={rank['within_error_bound']} tie={rank['tie_break']}"
                )
            stats_lines.append("")
            stats_lines.append("error-affected pair comparisons:")
            affected = [
                item
                for group in view["stats"]["pairwise_comparisons"].values()
                for item in group
                if item["within_error_bound"]
            ]
            if affected:
                for item in affected:
                    stats_lines.append(
                        f"  {item['metric']}: {item['left']} vs {item['right']} "
                        f"diff={item['difference']} <= bound={item['comparison_error_bound']} "
                        f"stable_winner={item['winner_by_stable_rule']}"
                    )
            else:
                stats_lines.append("  none")
            stats_lines.append("")
            stats_lines.append("service final/attempt/error distributions:")
            for item in view["stats"]["services"]:
                stats_lines.append(
                    f"  {item['service']}: final={json_dumps(item['final_distribution'])} "
                    f"attempts={json_dumps(item['attempt_distribution'])} "
                    f"errors={json_dumps(item['error_distribution'])}"
                )
            self._set_text(self.stats_text, "\n".join(stats_lines))
            versions = view["versions"]
            self._set_text(
                self.version_text,
                "\n".join(
                    [
                        f"schema version           : {versions['schema_version']}",
                        f"route revision           : {versions['route_revision']}",
                        f"instance-set version     : {versions['instance_version']}",
                        f"config digest            : {versions['config_digest']}",
                        f"ledger format version    : {versions['ledger_format_version']}",
                        f"ledger content digest    : {versions['ledger_content_digest']}",
                        f"request-sequence digest  : {versions['request_sequence_digest']}",
                        f"snapshot repair          : {json_dumps(versions.get('snapshot_repair'))}",
                    ]
                ),
            )
            for row in self.snapshot_tree.get_children():
                self.snapshot_tree.delete(row)
            for snapshot in view["snapshots"]:
                self.snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        snapshot.get("path"),
                        snapshot.get("valid"),
                        snapshot.get("completed_cursor", "-"),
                        snapshot.get("route_revision", "-"),
                        str(snapshot.get("digest", snapshot.get("code", "")))[:16],
                    ),
                )

        def _render_timeline(self) -> None:
            if self._last_view is None:
                return
            canvas = self.timeline_canvas
            canvas.delete("all")
            rows = self._last_view["timeline"]
            width = max(760, canvas.winfo_width())
            if not rows:
                canvas.create_text(18, 18, anchor="nw", fill="#9fb3c8", text="No committed attempts yet.")
                canvas.configure(scrollregion=(0, 0, width, 80))
                return
            min_t = min(int(row["wait_start_mono_ms"]) for row in rows)
            max_t = max(int(row["finished_mono_ms"]) for row in rows)
            span = max(1, max_t - min_t)
            label_width = 260
            plot_width = max(200, width - label_width - 28)
            height = 32 + len(rows) * 28 + 20
            colors = {
                "success": "#39d98a",
                "error": "#ff6b6b",
                "timeout": "#ff9f43",
                "reply_error": "#c56cf0",
                "host_error": "#ff5252",
                "circuit_open": "#8d99ae",
            }
            canvas.create_text(
                12,
                10,
                anchor="nw",
                fill="#d9e2ec",
                text=f"relative timeline · {len(rows)} attempts · span {span} ms",
                font=("Consolas", 9, "bold"),
            )
            for index, row in enumerate(rows):
                y = 34 + index * 28
                wait_start = int(row["wait_start_mono_ms"])
                wait_end = int(row["wait_end_mono_ms"])
                started = int(row["started_mono_ms"])
                finished = int(row["finished_mono_ms"])
                x_wait_start = label_width + (wait_start - min_t) / span * plot_width
                x_wait_end = label_width + (wait_end - min_t) / span * plot_width
                x_start = label_width + (started - min_t) / span * plot_width
                x_finish = label_width + (finished - min_t) / span * plot_width
                label = f"{row['service']}/{row['instance_id']} {row['request_id']}#{row['attempt_no']}"
                canvas.create_text(10, y + 7, anchor="w", fill="#d9e2ec", text=label[:46], font=("Consolas", 8))
                if row["budget_used"] and x_wait_end > x_wait_start:
                    canvas.create_rectangle(
                        x_wait_start,
                        y,
                        x_wait_end,
                        y + 14,
                        fill="#536878",
                        outline="",
                    )
                color = colors.get(str(row["outcome"]), "#adb5bd")
                canvas.create_rectangle(
                    max(x_start, x_finish - 2),
                    y,
                    max(x_start + 2, x_finish),
                    y + 14,
                    fill=color,
                    outline="",
                )
                canvas.create_text(
                    min(width - 8, x_finish + 6),
                    y + 7,
                    anchor="w",
                    fill="#9fb3c8",
                    text=f"g{row['global_inflight']}/{row['global_limit']} s{row['service_inflight']}/{row['service_quota']} wait{row['budget_wait_ms']}ms {row['outcome']}",
                    font=("Consolas", 8),
                )
            canvas.configure(scrollregion=(0, 0, width, height))

        def _run_sample(self) -> None:
            if self._busy:
                return
            self._busy = True
            self.status_label.configure(text="resetting ledger and running sample…")
            self.requests = load_requests(requests_path, self.spec.services.keys())
            thread = threading.Thread(target=self._run_sample_worker, name="ui-run", daemon=True)
            thread.start()

        def _run_sample_worker(self) -> None:
            try:
                with self._io_lock:
                    try:
                        self.runtime.stop()
                    except Exception:
                        pass
                    self.ledger.close()
                    self.ledger = Ledger(self.ledger_path, reset=True, logger=self.logger)
                    self.runtime = MeshRuntime(
                        self.spec,
                        self.revision,
                        self.ledger,
                        self.snapshot_store,
                        self.logger,
                        f"ui-{int(time.time())}",
                    )
                self.runtime.start()
                report = self.runtime.run_requests(self.requests)
                snapshot = self.runtime.write_snapshot("ui_run_complete")
                report["snapshot"] = {"path": snapshot.path, "digest": snapshot.digest}
                self._ui_events.put(("report", report))
                self._ui_events.put(("status", "sample run complete"))
            except Exception as exc:
                self.logger.error("ui_run_error", code="E_INTERNAL", message=str(exc))
                self._ui_events.put(("status", f"run error: {exc}"))
            finally:
                self._busy = False

        def _write_snapshot(self) -> None:
            if self._busy:
                self.status_label.configure(text="snapshot deferred: sample run is active")
                return
            try:
                with self._io_lock:
                    envelope = self.runtime.write_snapshot("ui_manual")
                self.status_label.configure(text=f"snapshot written: {Path(envelope.path).name}")
                self._refresh()
            except Exception as exc:
                self.status_label.configure(text=f"snapshot error: {exc}")

        def _on_close(self) -> None:
            if self._busy:
                self.status_label.configure(text="waiting for the active sample run before closing…")
                self.root.after(250, self._on_close)
                return
            self._closed = True
            try:
                self.runtime.stop()
            finally:
                self.ledger.close()
                self.root.destroy()

    try:
        root = tk.Tk()
    except Exception as exc:
        raise MeshError(
            "E_HOST",
            f"Tk display could not be opened: {exc}",
            field="ui.display",
            location=Path(spec_path).as_posix(),
        ) from exc
    app = DiagnosticsApp(root)
    logger.info("ui_started", ledger=ledger_path, revision=revision)
    root.mainloop()
    logger.info("ui_stopped", ledger=ledger_path)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
