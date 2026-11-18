"""Tkinter operator console for the connection-pool workbench.

The console shows live connection states, per-workload quota usage, queued
tasks, outcome distribution and the snapshot list, with buttons for running a
batch, saving a snapshot and restoring one.  Tkinter is imported lazily so the
rest of the workbench works on headless machines; the class also exposes the
same view-model data for tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .app import WorkbenchSession
from .sample_db import build_sample_db

try:  # Windows and desktop Linux/macOS ship Tkinter with the standard installer.
    import tkinter as tk
    from tkinter import messagebox, ttk
    _TK_AVAILABLE = True
except Exception:  # pragma: no cover - environment without Tk
    tk = None  # type: ignore
    ttk = None  # type: ignore
    messagebox = None  # type: ignore
    _TK_AVAILABLE = False


class PoolViewModel:
    """Headless view model used by both the GUI and the unit tests."""

    def __init__(self, session: WorkbenchSession) -> None:
        self.session = session

    def connection_rows(self) -> list[dict[str, Any]]:
        if self.session.pool is None:
            return []
        return [entry.snapshot() for entry in self.session.pool.entries()]

    def quota_rows(self) -> list[dict[str, Any]]:
        if self.session.pool is None or self.session.config is None:
            return []
        active = self.session.pool.workload_active()
        return [
            {
                "workload": workload,
                "quota": self.session.config.pool.workload_quotas[workload],
                "active": active.get(workload, 0),
            }
            for workload in sorted(self.session.config.pool.workload_quotas)
        ]

    def queued_task_ids(self) -> list[str]:
        if self.session.result is None:
            return []
        return [t.task_id for t in self.session.result.traces
                if t.outcome == "GET_TIMEOUT"]

    def outcome_distribution(self) -> dict[str, int]:
        if self.session.result is None:
            return {}
        return dict(self.session.result.outcome_counts)

    def snapshots(self) -> list[str]:
        return [str(path) for path in self.session.store.list_snapshots()]


class WorkbenchGUI:
    def __init__(self, workspace: str | Path, config_path: str | Path) -> None:
        if not _TK_AVAILABLE:
            raise RuntimeError(
                "tkinter is not available in this Python installation; "
                "use the CLI or the headless PoolViewModel instead"
            )
        self.workspace = Path(workspace)
        self.config_path = Path(config_path)
        self.db_path = self.workspace / "sample.sqlite"
        if not self.db_path.exists():
            build_sample_db(self.db_path)
        self.session = WorkbenchSession(self.workspace)
        self.session.load(self.config_path, self.db_path)
        self.view_model = PoolViewModel(self.session)
        self._build()

    def _build(self) -> None:
        self.root = tk.Tk()
        self.root.title("SQLite 连接池并行调度工作台")
        self.root.geometry("980x640")
        columns = ttk.Frame(self.root)
        columns.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(columns)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(columns)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="连接状态").pack(anchor="w")
        self.conn_tree = ttk.Treeview(
            left, columns=("seq", "type", "state", "task"),
            show="headings", height=8,
        )
        for col, title in (("seq", "序号"), ("type", "类型"),
                           ("state", "状态"), ("task", "任务")):
            self.conn_tree.heading(col, text=title)
        self.conn_tree.pack(fill=tk.X)

        ttk.Label(left, text="工作负载配额用量").pack(anchor="w", pady=(8, 0))
        self.quota_tree = ttk.Treeview(
            left, columns=("workload", "active", "quota"),
            show="headings", height=5,
        )
        for col, title in (("workload", "工作负载"), ("active", "活跃"),
                           ("quota", "配额")):
            self.quota_tree.heading(col, text=title)
        self.quota_tree.pack(fill=tk.X)

        ttk.Label(right, text="排队任务（获取超时）").pack(anchor="w")
        self.queue_list = tk.Listbox(right, height=6)
        self.queue_list.pack(fill=tk.X)

        ttk.Label(right, text="任务结局分布").pack(anchor="w", pady=(8, 0))
        self.outcome_text = tk.Text(right, height=6)
        self.outcome_text.pack(fill=tk.X)

        ttk.Label(right, text="快照列表").pack(anchor="w", pady=(8, 0))
        self.snapshot_list = tk.Listbox(right, height=8)
        self.snapshot_list.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(self.root)
        buttons.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(buttons, text="执行任务批次", command=self.run_batch).pack(side=tk.LEFT)
        ttk.Button(buttons, text="保存快照", command=self.save_snapshot).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="恢复选中快照", command=self.restore_snapshot).pack(side=tk.LEFT)
        ttk.Button(buttons, text="刷新", command=self.refresh).pack(side=tk.LEFT, padx=6)
        self.status = ttk.Label(self.root, text="就绪")
        self.status.pack(anchor="w", padx=8)
        self.refresh()

    def run_batch(self) -> None:
        try:
            self.session.execute()
            self.status.config(text="批次执行完成")
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("执行失败", str(exc))
        self.refresh()

    def save_snapshot(self) -> None:
        try:
            path = self.session.save_snapshot()
            self.status.config(text=f"已保存 {path}")
        except RuntimeError as exc:  # pragma: no cover
            messagebox.showwarning("提示", str(exc))
        self.refresh()

    def restore_snapshot(self) -> None:
        selection = self.snapshot_list.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选择一个快照")
            return
        path = self.snapshot_list.get(selection[0])
        restored = self.session.restore_snapshot(path)
        self.status.config(text=(
            f"已恢复快照（整体替换={restored['replaced']}）"
        ))
        self.refresh()

    def refresh(self) -> None:
        self.conn_tree.delete(*self.conn_tree.get_children())
        for row in self.view_model.connection_rows():
            self.conn_tree.insert("", tk.END, values=(
                row["seq"], row["conn_type"], row["state"], row["task_id"] or "-",
            ))
        self.quota_tree.delete(*self.quota_tree.get_children())
        for row in self.view_model.quota_rows():
            self.quota_tree.insert("", tk.END, values=(
                row["workload"], row["active"], row["quota"],
            ))
        self.queue_list.delete(0, tk.END)
        for task_id in self.view_model.queued_task_ids():
            self.queue_list.insert(tk.END, task_id)
        self.outcome_text.delete("1.0", tk.END)
        for outcome, count in sorted(self.view_model.outcome_distribution().items()):
            self.outcome_text.insert(tk.END, f"{outcome}: {count}\n")
        self.snapshot_list.delete(0, tk.END)
        for snapshot in self.view_model.snapshots():
            self.snapshot_list.insert(tk.END, snapshot)

    def mainloop(self) -> None:  # pragma: no cover - interactive only
        self.root.mainloop()


def launch(workspace: str | Path, config_path: str | Path) -> int:
    if not _TK_AVAILABLE:
        print(
            "tkinter 不可用：当前 Python 未包含 Tk 绑定。\n"
            "在 Windows 上使用 python.org 官方安装包即自带 tkinter；\n"
            "无头环境可使用 CLI（python -m pool_workbench.cli）或 PoolViewModel。"
        )
        return 1
    gui = WorkbenchGUI(workspace, config_path)
    gui.mainloop()
    return 0


def _main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="连接池工作台 Tkinter 界面")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    workspace = args.workspace or (Path(__file__).resolve().parent.parent / "workspace")
    config = args.config or (Path(__file__).resolve().parent / "samples" / "tasks_cold.json")
    return launch(workspace, config)


if __name__ == "__main__":
    raise SystemExit(_main())
