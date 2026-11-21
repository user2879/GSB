"""Tkinter diagnostic workbench for Windows local use."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .echo import run_echo_replay
from .errors import DiagnosticError
from .pipeline import DiagnosisSession, build_session
from .replay import verify_ledger
from .store import WorkbenchStore, restore_snapshot, restored_session_view
from .util import pretty_json


def _message(error: DiagnosticError) -> str:
    return pretty_json(error.to_dict())


class WorkbenchApp:
    def __init__(self, root: Any, *, db_path: str) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk
        self.root = root
        self.root.title("丢包观测归并与诊断工作台")
        self.root.geometry("1400x860")
        self.db_path = db_path
        self.session: DiagnosisSession | None = None
        self._build_ui()
        self._load_defaults()

    def _build_ui(self) -> None:
        ttk = self.ttk
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        form = ttk.LabelFrame(outer, text="输入与运行参数", padding=8)
        form.pack(fill="x", pady=(0, 8))
        self.wire_var = self.tk.StringVar()
        self.manifest_var = self.tk.StringVar()
        self.events_var = self.tk.StringVar()
        self.window_var = self.tk.StringVar(value="100")
        self.db_var = self.tk.StringVar(value=self.db_path)
        self.port_var = self.tk.StringVar(value="47731")

        entries = [
            ("逐包观测", self.wire_var, self._choose_wire),
            ("观测清单", self.manifest_var, self._choose_manifest),
            ("事件 JSON", self.events_var, self._choose_events),
            ("窗口 ms", self.window_var, None),
            ("SQLite", self.db_var, self._choose_db),
            ("回环端口", self.port_var, None),
        ]
        for row, (label, variable, command) in enumerate(entries):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(form, textvariable=variable, width=110).grid(
                row=row, column=1, columnspan=3, sticky="we", pady=2
            )
            if command:
                ttk.Button(form, text="浏览", command=command).grid(row=row, column=4, padx=4)
        form.columnconfigure(1, weight=1)

        buttons = ttk.Frame(form)
        buttons.grid(row=0, column=5, rowspan=6, padx=(12, 0), sticky="ns")
        for label, command in (
            ("运行诊断", self.run_diagnosis),
            ("保存快照", self.save_snapshot),
            ("保存账本", self.save_ledger),
            ("回环校验", self.run_echo),
        ):
            ttk.Button(buttons, text=label, command=command, width=14).pack(fill="x", pady=2)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.summary_text = self._add_text_tab("总览")
        self.frames_tree = self._add_tree_tab(
            "逐帧摘要",
            ("frame", "flow", "seq", "send_ms", "flags", "arrival", "payload", "offset"),
            (70, 300, 70, 100, 150, 80, 80, 90),
        )
        self.events_tree = self._add_tree_tab(
            "事件归一",
            ("record", "version", "flow", "event_seq", "count", "link", "window", "retrans", "depth", "padded"),
            (60, 70, 300, 80, 70, 90, 160, 80, 70, 230),
        )
        self.alignment_tree = self._add_tree_tab(
            "基准对齐",
            ("flow", "frame_clock", "event_clock", "offset", "aligned", "sequence_only", "missing"),
            (320, 170, 190, 90, 80, 100, 300),
        )
        self.gaps_tree = self._add_tree_tab(
            "缺口与归因",
            ("flow", "start", "end", "classification", "count", "time", "source", "evidence"),
            (300, 70, 70, 120, 70, 170, 140, 520),
        )
        self.rates_tree = self._add_tree_tab(
            "窗口丢包率",
            ("flow", "window", "observed", "lost", "expected", "ppm", "exact", "unwindowed"),
            (310, 170, 80, 70, 80, 90, 90, 100),
        )
        self.store_text = self._add_text_tab("快照与账本")

        store_frame = ttk.LabelFrame(outer, text="快照 / 账本操作", padding=8)
        store_frame.pack(fill="x", pady=(8, 0))
        self.snapshot_var = self.tk.StringVar()
        self.ledger_var = self.tk.StringVar()
        ttk.Label(store_frame, text="快照").grid(row=0, column=0, sticky="w")
        self.snapshot_combo = ttk.Combobox(store_frame, textvariable=self.snapshot_var, width=52, state="readonly")
        self.snapshot_combo.grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(store_frame, text="刷新", command=self.refresh_snapshots).grid(row=0, column=2, padx=2)
        ttk.Button(store_frame, text="整体替换恢复", command=self.restore_snapshot).grid(row=0, column=3, padx=2)

        ttk.Label(store_frame, text="账本").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.ledger_combo = ttk.Combobox(store_frame, textvariable=self.ledger_var, width=52, state="readonly")
        self.ledger_combo.grid(row=1, column=1, sticky="we", padx=4, pady=(4, 0))
        ttk.Button(store_frame, text="刷新", command=self.refresh_ledgers).grid(row=1, column=2, padx=2, pady=(4, 0))
        ttk.Button(store_frame, text="重放校验", command=self.verify_ledger).grid(row=1, column=3, padx=2, pady=(4, 0))
        store_frame.columnconfigure(1, weight=1)
        self.refresh_snapshots()
        self.refresh_ledgers()

    def _add_text_tab(self, title: str) -> Any:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        text = self.tk.Text(frame, wrap="none", font=("Consolas", 10))
        y_scroll = self.ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        x_scroll = self.ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return text

    def _add_tree_tab(self, title: str, columns: tuple[str, ...], widths: tuple[int, ...]) -> Any:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings")
        for column, width in zip(columns, widths):
            tree.heading(column, text=column)
            tree.column(column, width=width, stretch=False)
        y_scroll = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=y_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _load_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1] / "samples"
        if (root / "observations.bin").exists():
            self.wire_var.set(str(root / "observations.bin"))
        if (root / "wire_manifest.json").exists():
            self.manifest_var.set(str(root / "wire_manifest.json"))
        if (root / "events_v3.json").exists():
            self.events_var.set(str(root / "events_v3.json"))

    def _choose_wire(self) -> None:
        path = self.filedialog.askopenfilename(filetypes=(("Binary", "*.bin"), ("All", "*.*")))
        if path:
            self.wire_var.set(path)

    def _choose_manifest(self) -> None:
        path = self.filedialog.askopenfilename(filetypes=(("JSON", "*.json"), ("All", "*.*")))
        if path:
            self.manifest_var.set(path)

    def _choose_events(self) -> None:
        paths = self.filedialog.askopenfilenames(filetypes=(("JSON", "*.json"), ("All", "*.*")))
        if paths:
            self.events_var.set(" ".join(paths))

    def _choose_db(self) -> None:
        path = self.filedialog.asksaveasfilename(filetypes=(("SQLite", "*.db"), ("All", "*.*")))
        if path:
            self.db_var.set(path)

    def _event_paths(self) -> list[str]:
        return [part for part in self.events_var.get().split() if part]

    def _set_text(self, widget: Any, value: str) -> None:
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def _clear_tree(self, tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _show_error(self, error: Exception) -> None:
        if isinstance(error, DiagnosticError):
            text = _message(error)
        else:
            text = f"{type(error).__name__}: {error}"
        self.messagebox.showerror("诊断失败", text)

    def run_diagnosis(self) -> None:
        try:
            window_ms = int(self.window_var.get())
            self.session = build_session(
                wire_path=self.wire_var.get(),
                wire_manifest_path=self.manifest_var.get(),
                event_paths=self._event_paths(),
                window_ms=window_ms,
            )
            self._render_session(self.session)
            self.refresh_snapshots()
            self.refresh_ledgers()
        except Exception as exc:
            self._show_error(exc)

    def _render_session(self, session: Any) -> None:
        report = session.report
        summary = {
            "report_digest": session.report_digest,
            "input_digest": session.input_digest,
            "version_combo": session.version_combo,
            "unaligned_flow_ids": report["alignment"]["unaligned_flow_ids"],
            "ranking": report["rates"]["ranking"],
            "rate_error_policy": {
                "rounding": report["rates"]["rounding"],
                "quantization_error_upper_ppm": report["rates"]["quantization_error_upper_ppm"],
                "combined_error_bound_ppm": report["rates"]["combined_error_bound_ppm"],
            },
        }
        self._set_text(self.summary_text, pretty_json(summary))

        self._clear_tree(self.frames_tree)
        for frame in report["wire"]["frames"]:
            self.frames_tree.insert(
                "",
                "end",
                values=(
                    frame["frame_index"],
                    frame["flow_id"],
                    frame["send_seq"],
                    frame["send_monotonic_ms"],
                    ",".join(frame["flag_names"]),
                    frame["arrival_ordinal"],
                    frame["payload_len"],
                    frame["byte_offset"],
                ),
            )

        self._clear_tree(self.events_tree)
        for bundle in report["events"]["bundles"]:
            for record in bundle["records"]:
                window = record["time_window_ms"]
                window_text = f"{window['start_ms']}..{window['end_ms']}" if window else "null"
                self.events_tree.insert(
                    "",
                    "end",
                    values=(
                        record["record_number"],
                        record["source_version"],
                        record["flow_id"],
                        record["event_seq"],
                        record["loss_count"],
                        record["link_id"] or "null",
                        window_text,
                        record["retransmit"],
                        record["reorder_depth"],
                        ",".join(record["padded_fields"]),
                    ),
                )

        self._clear_tree(self.alignment_tree)
        for record in report["alignment"]["records"]:
            self.alignment_tree.insert(
                "",
                "end",
                values=(
                    record["flow_id"],
                    record["frame_clock"],
                    record["event_clock"],
                    record["offset_to_frame_ms"],
                    record["aligned"],
                    record["sequence_only"],
                    ",".join(record["missing_basis"]),
                ),
            )

        self._clear_tree(self.gaps_tree)
        for flow in report["diagnosis"]["flows"]:
            for gap in flow["gaps"]:
                window = gap["time_window_ms"]
                window_text = f"{window['start_ms']}..{window['end_ms']}" if window else "null"
                self.gaps_tree.insert(
                    "",
                    "end",
                    values=(
                        flow["flow_id"],
                        gap["gap_interval"]["start_seq"],
                        gap["gap_interval"]["end_seq"],
                        gap["classification"],
                        gap["count"],
                        window_text,
                        gap["time_source"],
                        ",".join(gap["evidence"]),
                    ),
                )

        self._clear_tree(self.rates_tree)
        for flow in report["rates"]["flows"]:
            if not flow["windows"]:
                self.rates_tree.insert(
                    "",
                    "end",
                    values=(flow["flow_id"], "unwindowed", 0, flow["true_loss"], flow["expected"], flow["rate_ppm"], flow["exact_fraction"], flow["unwindowed_true_loss"]),
                )
            for window in flow["windows"]:
                self.rates_tree.insert(
                    "",
                    "end",
                    values=(
                        flow["flow_id"],
                        f"{window['window_start_ms']}..{window['window_end_ms']}",
                        window["observed_unique"],
                        window["true_loss"],
                        window["expected"],
                        window["rate_ppm"],
                        window["exact_fraction"],
                        flow["unwindowed_true_loss"],
                    ),
                )
        self._append_store_log("diagnosis", summary)

    def _append_store_log(self, event: str, payload: Any) -> None:
        current = self.store_text.get("1.0", "end").rstrip()
        entry = pretty_json({"event": event, "payload": payload})
        self._set_text(self.store_text, (current + "\n\n" + entry).lstrip())

    def save_snapshot(self) -> None:
        if self.session is None:
            self.messagebox.showinfo("提示", "请先运行诊断。")
            return
        try:
            with WorkbenchStore(self.db_var.get()) as store:
                uid = store.save_snapshot(self.session)
            self._append_store_log("snapshot_saved", {"snapshot_uid": uid, "db": self.db_var.get()})
            self.refresh_snapshots()
        except Exception as exc:
            self._show_error(exc)

    def save_ledger(self) -> None:
        if self.session is None:
            self.messagebox.showinfo("提示", "请先运行诊断。")
            return
        try:
            with WorkbenchStore(self.db_var.get()) as store:
                uid = store.save_ledger(self.session)
            self._append_store_log("ledger_saved", {"ledger_uid": uid, "db": self.db_var.get()})
            self.refresh_ledgers()
        except Exception as exc:
            self._show_error(exc)

    def refresh_snapshots(self) -> None:
        try:
            with WorkbenchStore(self.db_var.get()) as store:
                rows = store.list_snapshots()
            values = [row["snapshot_uid"] for row in rows]
            self.snapshot_combo["values"] = values
            if values and self.snapshot_var.get() not in values:
                self.snapshot_var.set(values[-1])
        except Exception:
            self.snapshot_combo["values"] = []

    def refresh_ledgers(self) -> None:
        try:
            with WorkbenchStore(self.db_var.get()) as store:
                rows = store.list_ledger()
            values = [row["ledger_uid"] for row in rows]
            self.ledger_combo["values"] = values
            if values and self.ledger_var.get() not in values:
                self.ledger_var.set(values[-1])
        except Exception:
            self.ledger_combo["values"] = []

    def restore_snapshot(self) -> None:
        snapshot_uid = self.snapshot_var.get().strip()
        if not snapshot_uid:
            self.messagebox.showinfo("提示", "请选择快照。")
            return
        try:
            with WorkbenchStore(self.db_var.get()) as store:
                result = restore_snapshot(
                    store,
                    snapshot_uid,
                    current_session=self.session,
                )
            self.session = restored_session_view(result)
            self._render_session(self.session)
            self._append_store_log("snapshot_restored", result["comparison"])
            self.messagebox.showinfo(
                "快照恢复",
                f"整体替换: {result['comparison']['replaced']}\n"
                f"模式: {result['comparison']['replacement_mode']}\n"
                f"差异项: {len(result['comparison']['differences'])}",
            )
        except Exception as exc:
            self._show_error(exc)

    def verify_ledger(self) -> None:
        ledger_uid = self.ledger_var.get().strip()
        if not ledger_uid:
            self.messagebox.showinfo("提示", "请选择账本记录。")
            return
        try:
            with WorkbenchStore(self.db_var.get()) as store:
                result = verify_ledger(store, ledger_uid)
            self._append_store_log("ledger_verified", result)
            self.messagebox.showinfo("账本重放", f"逐项一致: {result['matches']}")
        except Exception as exc:
            self._show_error(exc)

    def run_echo(self) -> None:
        if self.session is None:
            self.messagebox.showinfo("提示", "请先运行诊断。")
            return
        if not isinstance(self.session, DiagnosisSession):
            self.messagebox.showinfo("提示", "快照恢复视图是只读的；请重新加载输入并运行诊断后再做回环校验。")
            return
        try:
            port = int(self.port_var.get())
            result = run_echo_replay(
                self.session,
                host="127.0.0.1",
                port=port,
                request_id=f"gui-{self.session.report_digest[:16]}",
                send_ack=True,
            )
            self._append_store_log("echo_replay", result)
            self.messagebox.showinfo("回环校验", "对端缺口形态与诊断结论一致。")
        except Exception as exc:
            self._show_error(exc)


def main(*, db_path: str = "artifacts/workbench.db") -> int:
    try:
        import tkinter as tk
    except ImportError as exc:
        sys.stderr.write(
            "Tkinter is unavailable in this Python build. "
            "On Windows, use the official Python installer (tkinter included).\n"
        )
        raise SystemExit(1) from exc
    root = tk.Tk()
    WorkbenchApp(root, db_path=db_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
