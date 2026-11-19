"""Tkinter diagnostic workbench."""

from __future__ import annotations

from tkinter import (
    END,
    LEFT,
    RIGHT,
    StringVar,
    Tk,
    messagebox,
    simpledialog,
    ttk,
)
from tkinter.scrolledtext import ScrolledText

from .errors import BusinessError
from .service import SessionService
from .ui_model import build_ui_model


class DiagnosticWorkbench:
    def __init__(self, root: Tk, service: SessionService) -> None:
        self.root = root
        self.service = service
        self.root.title("TCP 连接状态与拥塞诊断工作台")
        self.root.geometry("1380x900")
        self.root.minsize(1100, 700)

        self.status_var = StringVar(value="")
        self.state_var = StringVar(value="")
        self.cwnd_var = StringVar(value="")
        self.ssthresh_var = StringVar(value="")
        self.dup_var = StringVar(value="")
        self.unacked_var = StringVar(value="")
        self.rtt_var = StringVar(value="")
        self.version_var = StringVar(value="")
        self.recovery_var = StringVar(value="")

        self._build_layout()
        self.refresh()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=3)
        self.root.rowconfigure(3, weight=2)

        controls = ttk.Frame(self.root, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Button(controls, text="刷新", command=self.refresh).pack(side=LEFT, padx=3)
        ttk.Button(controls, text="保存快照", command=self.save_snapshot).pack(
            side=LEFT, padx=3
        )
        ttk.Button(
            controls, text="校验所选快照", command=self.verify_selected_snapshot
        ).pack(side=LEFT, padx=3)
        ttk.Button(
            controls, text="恢复所选快照", command=self.restore_selected_snapshot
        ).pack(side=LEFT, padx=3)
        ttk.Button(controls, text="全量重算比对", command=self.verify_recompute).pack(
            side=LEFT, padx=3
        )
        ttk.Label(controls, textvariable=self.status_var).pack(side=RIGHT, padx=8)

        metrics = ttk.LabelFrame(self.root, text="连接与拥塞指标", padding=8)
        metrics.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        for index in range(4):
            metrics.columnconfigure(index, weight=1)
        pairs = [
            ("状态", self.state_var),
            ("拥塞窗口", self.cwnd_var),
            ("慢启动阈值", self.ssthresh_var),
            ("重复确认", self.dup_var),
            ("已发送未确认", self.unacked_var),
            ("SRTT / RTO", self.rtt_var),
            ("当前版本", self.version_var),
            ("恢复状态", self.recovery_var),
        ]
        for index, (label, variable) in enumerate(pairs):
            row = index // 4
            column = (index % 4) * 1
            frame = ttk.Frame(metrics)
            frame.grid(row=row, column=column, sticky="ew", padx=6, pady=4)
            ttk.Label(frame, text=label + "：").pack(side=LEFT)
            ttk.Label(
                frame, textvariable=variable, font=("TkDefaultFont", 10, "bold")
            ).pack(side=LEFT)

        conclusions_frame = ttk.LabelFrame(
            self.root, text="逐项结论（稳定顺序、版本、依赖事件）", padding=6
        )
        conclusions_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        conclusions_frame.columnconfigure(0, weight=1)
        conclusions_frame.rowconfigure(0, weight=1)
        columns = (
            "label",
            "value",
            "revision",
            "basis",
            "affected",
            "version",
            "dependencies",
        )
        self.conclusions = ttk.Treeview(
            conclusions_frame, columns=columns, show="headings", height=12
        )
        headings = {
            "label": "结论项",
            "value": "值",
            "revision": "版本号",
            "basis": "依据事件",
            "affected": "本次受影响",
            "version": "配置/脚本版本",
            "dependencies": "依赖事件序号",
        }
        widths = {
            "label": 150,
            "value": 540,
            "revision": 70,
            "basis": 80,
            "affected": 90,
            "version": 120,
            "dependencies": 250,
        }
        for column in columns:
            self.conclusions.heading(column, text=headings[column])
            self.conclusions.column(
                column, width=widths[column], anchor="w", stretch=column == "value"
            )
        scrollbar = ttk.Scrollbar(
            conclusions_frame, orient="vertical", command=self.conclusions.yview
        )
        self.conclusions.configure(yscrollcommand=scrollbar.set)
        self.conclusions.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        bottom = ttk.Panedwindow(self.root, orient="horizontal")
        bottom.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 8))
        snapshot_frame = ttk.LabelFrame(bottom, text="快照列表", padding=6)
        snapshot_frame.columnconfigure(0, weight=1)
        snapshot_frame.rowconfigure(0, weight=1)
        snapshot_columns = ("id", "name", "version", "events", "digest")
        self.snapshots = ttk.Treeview(
            snapshot_frame, columns=snapshot_columns, show="headings", height=8
        )
        for column, title, width in (
            ("id", "ID", 50),
            ("name", "名称", 200),
            ("version", "版本", 100),
            ("events", "事件数", 70),
            ("digest", "内容摘要", 120),
        ):
            self.snapshots.heading(column, text=title)
            self.snapshots.column(column, width=width, anchor="w")
        self.snapshots.grid(row=0, column=0, sticky="nsew")
        bottom.add(snapshot_frame, weight=1)

        log_frame = ttk.LabelFrame(bottom, text="结构化日志（末尾）", padding=6)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.logs = ScrolledText(log_frame, height=8, wrap="none")
        self.logs.grid(row=0, column=0, sticky="nsew")
        self.logs.configure(state="disabled")
        bottom.add(log_frame, weight=2)

    def refresh(self) -> None:
        model = build_ui_model(self.service)
        header = model["header"]
        rtt = header["rtt"]
        self.state_var.set(str(header["connection_state"]))
        self.cwnd_var.set(str(header["cwnd"]))
        self.ssthresh_var.set(str(header["ssthresh"]))
        self.dup_var.set(str(header["dup_ack_count"]))
        self.unacked_var.set(
            f"{header['sent_unacked']['bytes']} B / {header['sent_unacked']['segments']} seg"
        )
        self.rtt_var.set(f"{rtt['srtt_ms']} ms / {rtt['rto_ms']} ms")
        self.version_var.set(str(header["version"]))
        self.recovery_var.set("需要重跑" if header["needs_rerun"] else "一致")
        self.status_var.set(f"事件 {header['event_count']} 条")

        for row in self.conclusions.get_children():
            self.conclusions.delete(row)
        for item in model["conclusion_rows"]:
            self.conclusions.insert(
                "",
                END,
                iid=item["item"],
                values=(
                    item["label"],
                    item["value"],
                    item["revision"],
                    item["basis_event_seq"],
                    item["affected"],
                    item["version"],
                    item["depends_on_events"],
                ),
            )

        for row in self.snapshots.get_children():
            self.snapshots.delete(row)
        for snapshot in model["snapshots"]:
            self.snapshots.insert(
                "",
                END,
                iid=str(snapshot["snapshot_id"]),
                values=(
                    snapshot["snapshot_id"],
                    snapshot["name"],
                    f"C{snapshot['config_version']}/S{snapshot['script_version']}",
                    snapshot["event_count"],
                    snapshot["content_digest"],
                ),
            )

        self.logs.configure(state="normal")
        self.logs.delete("1.0", END)
        for record in model["logs"]:
            self.logs.insert(END, str(record) + "\n")
        self.logs.configure(state="disabled")

    def _run_business(self, operation) -> None:
        try:
            operation()
            self.refresh()
        except BusinessError as exc:
            self.service.logger.business_error("ui_operation_failed", exc)
            messagebox.showerror(
                exc.code,
                exc.message + ("\n" + "\n".join(exc.fields) if exc.fields else ""),
            )
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            self.service.logger.error(
                "ui_unexpected_error", code="E_UI", message=str(exc)
            )
            messagebox.showerror("E_UI", str(exc))
            self.refresh()

    def save_snapshot(self) -> None:
        name = simpledialog.askstring(
            "保存快照", "快照名称", initialvalue="manual-snapshot"
        )
        if not name:
            return

        def operation() -> None:
            snapshot_id = self.service.save_snapshot(name)
            self.status_var.set(f"已保存快照 #{snapshot_id}")

        self._run_business(operation)

    def _selected_snapshot_id(self) -> int | None:
        selection = self.snapshots.selection()
        if not selection:
            messagebox.showinfo("快照", "请先选择一个快照")
            return None
        return int(selection[0])

    def verify_selected_snapshot(self) -> None:
        snapshot_id = self._selected_snapshot_id()
        if snapshot_id is None:
            return

        def operation() -> None:
            result = self.service.verify_snapshot(snapshot_id)
            self.status_var.set(
                f"快照 #{snapshot_id} 校验通过：{result['event_count']} 个事件"
            )

        self._run_business(operation)

    def restore_selected_snapshot(self) -> None:
        snapshot_id = self._selected_snapshot_id()
        if snapshot_id is None:
            return
        if not messagebox.askyesno(
            "恢复快照", "恢复会整体替换当前结论集，旧结论进入只读历史。继续？"
        ):
            return

        def operation() -> None:
            report = self.service.restore_snapshot(snapshot_id)
            self.status_var.set(
                f"快照 #{snapshot_id} 已整体替换：{len(report.affected_fields)} 个字段受影响"
            )

        self._run_business(operation)

    def verify_recompute(self) -> None:
        def operation() -> None:
            self.service.verify_incremental_equivalence()
            self.status_var.set("增量结论与全量重算逐项一致")

        self._run_business(operation)


def run_ui(service: SessionService) -> None:
    root = Tk()
    DiagnosticWorkbench(root, service)
    root.mainloop()
