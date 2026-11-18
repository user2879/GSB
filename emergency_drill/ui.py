"""Tkinter operator console for the local emergency drill workbench."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .workbench import Workbench

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError:  # pragma: no cover - exercised on the Linux verification host
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    simpledialog = None


if tk is not None:

    class DrillApp(tk.Tk):
        def __init__(
            self,
            workbench: Workbench,
            session_id: str,
            *,
            plan_path: Path,
        ) -> None:
            super().__init__()
            self.workbench = workbench
            self.session_id = session_id
            self.plan_path = plan_path
            self.title("应急预案演练工作台")
            self.geometry("1240x860")
            self.minsize(1040, 720)
            self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
            self.engine = None
            self.abort_event = threading.Event()
            self._worker: threading.Thread | None = None
            self._build_layout()
            self._refresh()
            self.after(100, self._poll_worker_events)

        def _build_layout(self) -> None:
            outer = ttk.Frame(self, padding=8)
            outer.pack(fill="both", expand=True)
            header = ttk.Frame(outer)
            header.pack(fill="x", pady=(0, 6))
            self.header_var = tk.StringVar(value="加载中…")
            ttk.Label(
                header,
                textvariable=self.header_var,
                font=("TkDefaultFont", 11, "bold"),
            ).pack(anchor="w")
            self.unknown_var = tk.StringVar(value="")
            ttk.Label(header, textvariable=self.unknown_var, foreground="#9a3412").pack(
                anchor="w", pady=(2, 0)
            )

            controls = ttk.Frame(outer)
            controls.pack(fill="x", pady=(0, 8))
            ttk.Button(controls, text="加载本地预案", command=self._load_plan_file).pack(side="left", padx=(0, 4))
            ttk.Button(controls, text="编辑预案", command=self._edit_plan).pack(side="left", padx=4)
            ttk.Button(controls, text="注入事件", command=self._inject_event).pack(side="left", padx=4)
            ttk.Button(controls, text="开始演练", command=self._start_run).pack(side="left", padx=4)
            ttk.Button(controls, text="中止演练", command=self._abort_run).pack(side="left", padx=4)
            ttk.Button(controls, text="保存快照", command=self._save_snapshot).pack(side="left", padx=4)
            ttk.Button(controls, text="恢复所选", command=self._restore_selected).pack(side="left", padx=4)
            ttk.Button(controls, text="崩溃恢复", command=self._recover).pack(side="left", padx=4)
            ttk.Button(controls, text="重放校验", command=self._replay_selected).pack(side="left", padx=4)
            ttk.Button(controls, text="刷新", command=self._refresh).pack(side="left", padx=4)

            paned = ttk.Panedwindow(outer, orient="vertical")
            paned.pack(fill="both", expand=True)

            stage_frame = ttk.LabelFrame(paned, text="处置阶段与依赖 / 状态机", padding=4)
            stage_columns = (
                "seq",
                "name",
                "deps",
                "status",
                "skip_reason",
                "business_code",
                "skip_from",
                "logical_ms",
                "unknown",
            )
            self.stage_tree = ttk.Treeview(
                stage_frame, columns=stage_columns, show="headings", height=8
            )
            headings = {
                "seq": "序号",
                "name": "阶段",
                "deps": "前置阶段",
                "status": "状态",
                "skip_reason": "跳过原因",
                "business_code": "业务码",
                "skip_from": "触发上游",
                "logical_ms": "逻辑毫秒",
                "unknown": "未知标记",
            }
            widths = {
                "seq": 54,
                "name": 150,
                "deps": 100,
                "status": 90,
                "skip_reason": 320,
                "business_code": 90,
                "skip_from": 82,
                "logical_ms": 92,
                "unknown": 82,
            }
            for column in stage_columns:
                self.stage_tree.heading(column, text=headings[column])
                self.stage_tree.column(column, width=widths[column], anchor="w")
            stage_scroll = ttk.Scrollbar(
                stage_frame, orient="vertical", command=self.stage_tree.yview
            )
            self.stage_tree.configure(yscrollcommand=stage_scroll.set)
            self.stage_tree.pack(side="left", fill="both", expand=True)
            stage_scroll.pack(side="right", fill="y")
            paned.add(stage_frame, weight=3)

            lower = ttk.Frame(paned)
            paned.add(lower, weight=2)

            event_frame = ttk.LabelFrame(lower, text="注入事件时间轴", padding=4)
            event_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
            event_columns = (
                "injection_seq",
                "offset_ms",
                "event_id",
                "type",
                "stage_seq",
                "status",
                "source",
            )
            self.event_tree = ttk.Treeview(
                event_frame, columns=event_columns, show="headings", height=7
            )
            event_headings = {
                "injection_seq": "注入序",
                "offset_ms": "逻辑偏移",
                "event_id": "事件 ID",
                "type": "类型",
                "stage_seq": "作用阶段",
                "status": "状态",
                "source": "来源",
            }
            for column in event_columns:
                self.event_tree.heading(column, text=event_headings[column])
                self.event_tree.column(column, width=108 if column != "event_id" else 180)
            self.event_tree.pack(fill="both", expand=True)

            snapshot_frame = ttk.LabelFrame(lower, text="快照列表", padding=4)
            snapshot_frame.pack(side="right", fill="both", expand=True, padx=(4, 0))
            snapshot_columns = ("snapshot_id", "plan_version", "mode", "clock", "digest")
            self.snapshot_tree = ttk.Treeview(
                snapshot_frame, columns=snapshot_columns, show="headings", height=7
            )
            snapshot_headings = {
                "snapshot_id": "快照",
                "plan_version": "预案版本",
                "mode": "模式",
                "clock": "逻辑毫秒",
                "digest": "内容摘要",
            }
            for column in snapshot_columns:
                self.snapshot_tree.heading(column, text=snapshot_headings[column])
                self.snapshot_tree.column(
                    column, width=92 if column != "digest" else 210
                )
            self.snapshot_tree.pack(fill="both", expand=True)

            log_frame = ttk.LabelFrame(outer, text="操作与结构化事件摘要", padding=4)
            log_frame.pack(fill="both", expand=False, pady=(8, 0))
            self.log_text = tk.Text(log_frame, height=7, wrap="word", state="disabled")
            self.log_text.pack(fill="both", expand=True)

        def _append_log(self, message: str) -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message.rstrip() + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        def _refresh(self) -> None:
            try:
                model = self.workbench.render_model(self.session_id)
            except BusinessError as exc:
                self._show_error(exc)
                return
            self.header_var.set(
                f"会话 {model['session_id']} | 预案 {model['plan_name']} v{model['plan_version']} "
                f"| 代 {model['generation']} | 状态 {model['status']} "
                f"| 逻辑时钟 {model['logical_clock_ms']} ms ({model['clock_source']})"
            )
            unknown = model["unknown_stage_seqs"]
            self.unknown_var.set(
                f"未知阶段（必须重跑）: {unknown}" if unknown else "未知阶段: 无"
            )

            self.stage_tree.delete(*self.stage_tree.get_children())
            for stage in sorted(model["stages"], key=lambda item: item["seq"]):
                self.stage_tree.insert(
                    "",
                    "end",
                    iid=f"stage-{stage['seq']:06d}",
                    values=(
                        stage["seq"],
                        stage["name"],
                        ",".join(str(item) for item in stage["depends_on"]) or "-",
                        stage["status"],
                        stage["skip_reason"] or "",
                        stage["business_code"] or "",
                        stage["skip_from_stage_seq"] or "",
                        stage["updated_ms"],
                        "UNKNOWN/需重跑" if stage["unknown"] else "",
                    ),
                )

            self.event_tree.delete(*self.event_tree.get_children())
            for event in model["events"]:
                self.event_tree.insert(
                    "",
                    "end",
                    iid=f"event-{event['injection_seq']:06d}",
                    values=(
                        event["injection_seq"],
                        event["offset_ms"],
                        event["event_id"],
                        event["type"],
                        event["stage_seq"],
                        event["status"],
                        event["source"],
                    ),
                )

            self.snapshot_tree.delete(*self.snapshot_tree.get_children())
            for snapshot in sorted(model["snapshots"], key=lambda item: item["snapshot_id"]):
                self.snapshot_tree.insert(
                    "",
                    "end",
                    iid=f"snapshot-{snapshot['snapshot_id']}",
                    values=(
                        snapshot["snapshot_id"],
                        snapshot["plan_version"],
                        snapshot["display_mode"],
                        snapshot["created_ms"],
                        snapshot["content_digest"][:24] + "…",
                    ),
                )

        def _show_error(self, error: BusinessError) -> None:
            text = (
                f"{error.code} | stage={error.stage_seq} | path={error.field_path} | "
                f"{error.message}"
            )
            self._append_log(text)
            self.workbench.log_error(error, event="ui_error", session_id=self.session_id)
            messagebox.showerror("操作被拒绝", text)

        def _run_action(self, callback):
            try:
                callback()
            except BusinessError as exc:
                self._show_error(exc)
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                self._append_log(f"UI_ERROR {exc!r}")
                messagebox.showerror("操作失败", str(exc))

        def _load_plan_file(self) -> None:
            path = filedialog.askopenfilename(
                title="选择预案 JSON",
                filetypes=[("JSON", "*.json"), ("All", "*.*")],
                initialdir=str(Path.cwd()),
            )
            if not path:
                return

            def action() -> None:
                report = self.workbench.publish_plan(self.session_id, path, auto_bump=True)
                self.plan_path = Path(path)
                self._append_log(
                    "预案已整体替换: "
                    + json.dumps(report["diff"], ensure_ascii=False, sort_keys=True)
                )
                self._refresh()

            self._run_action(action)

        def _edit_plan(self) -> None:
            try:
                current = self.workbench.repository.get_plan(self.session_id)
            except BusinessError as exc:
                self._show_error(exc)
                return
            window = tk.Toplevel(self)
            window.title("编辑预案（保存时版本号自动递增）")
            window.geometry("920x660")
            text = tk.Text(window, wrap="none", font=("TkFixedFont", 10))
            text.pack(fill="both", expand=True)
            text.insert(
                "1.0",
                json.dumps(current.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            )

            def save() -> None:
                try:
                    raw = json.loads(text.get("1.0", "end"))
                    report = self.workbench.publish_plan(
                        self.session_id, raw, auto_bump=True
                    )
                except (json.JSONDecodeError, BusinessError) as exc:
                    if isinstance(exc, BusinessError):
                        self._show_error(exc)
                    else:
                        messagebox.showerror("JSON 解析失败", str(exc))
                    return
                self._append_log(
                    f"预案编辑已发布 v{report['new_plan_version']}，旧版本 "
                    f"v{report['stale_plan_version']} 标记为陈旧"
                )
                window.destroy()
                self._refresh()

            button_bar = ttk.Frame(window)
            button_bar.pack(fill="x", pady=4)
            ttk.Button(button_bar, text="校验并整体发布", command=save).pack(side="right")
            ttk.Button(button_bar, text="取消", command=window.destroy).pack(side="right", padx=6)

        def _inject_event(self) -> None:
            window = tk.Toplevel(self)
            window.title("注入演练事件")
            window.geometry("440x250")
            fields = {
                "event_id": tk.StringVar(value="manual-event"),
                "type": tk.StringVar(value="switchover_failed"),
                "offset_ms": tk.StringVar(value="6000"),
                "stage_seq": tk.StringVar(value="3"),
            }
            labels = [
                ("事件 ID", "event_id"),
                ("事件类型", "type"),
                ("逻辑偏移毫秒", "offset_ms"),
                ("作用阶段序号", "stage_seq"),
            ]
            for row, (label, key) in enumerate(labels):
                ttk.Label(window, text=label).grid(
                    row=row, column=0, padx=8, pady=6, sticky="e"
                )
                if key == "type":
                    widget = ttk.Combobox(
                        window,
                        textvariable=fields[key],
                        values=["node_lost", "switchover_failed", "validation_timeout"],
                        state="readonly",
                    )
                else:
                    widget = ttk.Entry(window, textvariable=fields[key])
                widget.grid(row=row, column=1, padx=8, pady=6, sticky="ew")
            window.columnconfigure(1, weight=1)

            def save() -> None:
                try:
                    runtime = self.workbench.inject_event(
                        self.session_id,
                        {
                            "event_id": fields["event_id"].get().strip(),
                            "type": fields["type"].get(),
                            "offset_ms": int(fields["offset_ms"].get()),
                            "stage_seq": int(fields["stage_seq"].get()),
                        },
                    )
                except ValueError:
                    messagebox.showerror("输入错误", "偏移与阶段序号必须是整数")
                    return
                except BusinessError as exc:
                    self._show_error(exc)
                    return
                self._append_log(
                    f"已注入事件 {runtime.event_id}，作用阶段 {runtime.stage_seq}，"
                    f"偏移 {runtime.offset_ms} ms"
                )
                window.destroy()
                self._refresh()

            ttk.Button(window, text="注入", command=save).grid(
                row=4, column=1, padx=8, pady=8, sticky="e"
            )

        def _start_run(self) -> None:
            if self._worker is not None and self._worker.is_alive():
                messagebox.showinfo("演练进行中", "当前已有演练线程在运行")
                return
            self.abort_event.clear()
            self.engine = self.workbench.new_engine(self.session_id)
            self._append_log("开始演练：逻辑时钟推进，不使用墙钟做判定")

            def stage_callback(summary: dict[str, Any]) -> None:
                self.events.put(("progress", summary))
                time.sleep(0.12)

            def worker() -> None:
                try:
                    summary = self.workbench.run_session(
                        self.session_id,
                        abort_check=self.abort_event.is_set,
                        stage_callback=stage_callback,
                        engine=self.engine,
                    )
                    self.events.put(("done", summary))
                except BusinessError as exc:
                    self.events.put(("error", exc.to_dict()))
                except Exception as exc:  # pragma: no cover - thread boundary
                    self.events.put(("fatal", repr(exc)))

            self._worker = threading.Thread(target=worker, daemon=True)
            self._worker.start()

        def _abort_run(self) -> None:
            self.abort_event.set()
            if self.engine is not None:
                self.engine.request_abort()
            self._append_log("已请求中止；当前阶段结束后跳过剩余阶段")

        def _poll_worker_events(self) -> None:
            try:
                while True:
                    kind, payload = self.events.get_nowait()
                    if kind == "progress":
                        self._refresh()
                    elif kind == "done":
                        self._append_log(
                            f"演练结束: status={payload['status']} clock={payload['logical_clock_ms']} ms"
                        )
                        self._refresh()
                    elif kind == "error":
                        self._append_log(
                            f"{payload['code']} stage={payload['stage_seq']} path={payload['field_path']}"
                        )
                        self.workbench.log_error(
                            BusinessError(
                                payload["code"],
                                payload["message"],
                                stage_seq=payload.get("stage_seq"),
                                field_path=payload.get("field_path"),
                            ),
                            event="ui_worker_error",
                            session_id=self.session_id,
                        )
                        messagebox.showerror("演练失败", payload["message"])
                    elif kind == "fatal":
                        self._append_log(f"FATAL {payload}")
                        messagebox.showerror("演练异常", payload)
            except queue.Empty:
                pass
            self.after(100, self._poll_worker_events)

        def _save_snapshot(self) -> None:
            self._run_action(lambda: self._save_snapshot_action())

        def _save_snapshot_action(self) -> None:
            snapshot = self.workbench.save_snapshot(self.session_id)
            self._append_log(
                f"已保存快照 {snapshot['snapshot_id']} "
                f"(plan v{snapshot['plan_version']}, digest={snapshot['content_digest'][:12]}…)"
            )
            self._refresh()

        def _selected_snapshot(self) -> dict[str, Any] | None:
            selection = self.snapshot_tree.selection()
            snapshots = self.workbench.list_snapshots(self.session_id)
            if selection:
                snapshot_id = selection[0].removeprefix("snapshot-")
                return next(
                    (item for item in snapshots if item["snapshot_id"] == snapshot_id),
                    None,
                )
            return snapshots[-1] if snapshots else None

        def _restore_selected(self) -> None:
            snapshot = self._selected_snapshot()
            if snapshot is None:
                messagebox.showinfo("无快照", "请先保存快照")
                return

            def action() -> None:
                report = self.workbench.restore_snapshot(
                    self.session_id, snapshot["file_path"]
                )
                self._append_log(
                    f"已整体恢复 {report['snapshot_id']}：plan v{report['restored_plan_version']}，"
                    f"代 {report['generation']}，历史快照只读"
                )
                self._refresh()

            self._run_action(action)

        def _recover(self) -> None:
            def action() -> None:
                report = self.workbench.recover(self.session_id)
                self._append_log(
                    "恢复报告: "
                    + json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
                )
                self._refresh()

            self._run_action(action)

        def _replay_selected(self) -> None:
            snapshot = self._selected_snapshot()
            if snapshot is None:
                messagebox.showinfo("无快照", "请先保存快照")
                return

            def action() -> None:
                report = self.workbench.replay_snapshot(
                    self.session_id, snapshot["file_path"]
                )
                self._append_log(
                    f"重放通过: {report.compared_stage_count} 个阶段逐项一致 "
                    f"(snapshot={report.snapshot_id})"
                )
                self._refresh()

            self._run_action(action)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="应急预案演练工作台 Tkinter 界面")
    parser.add_argument("--db", default="data/session.db")
    parser.add_argument("--snapshots-dir", default="data/snapshots")
    parser.add_argument("--log", default="data/logs/drill.jsonl")
    parser.add_argument("--session", default="ui-session")
    parser.add_argument("--plan", default="samples/plan_valid.json")
    args = parser.parse_args(argv)
    if tk is None:
        print(
            "E_UI: 当前 Python 未安装 Tkinter。请使用 Windows 官方 Python 3.10+ "
            "（通常自带 Tkinter）运行 python run_ui.py。",
            file=sys.stderr,
        )
        return 2
    workbench = Workbench(args.db, args.snapshots_dir, args.log)
    if not workbench.repository.session_exists(args.session):
        workbench.create_session(args.session, args.plan)
    try:
        app = DrillApp(workbench, args.session, plan_path=Path(args.plan))
    except tk.TclError as exc:
        print(f"E_UI: 无法启动图形界面: {exc}", file=sys.stderr)
        return 2
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
