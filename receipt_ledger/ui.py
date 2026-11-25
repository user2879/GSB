from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .planner import build_plan
from .net import free_loopback_port
from .processes import WorkerPool
from .scheduler import LeaseScheduler
from .spec import load_bundle


def short_digest(value: Any, length: int = 16) -> str:
    if not value:
        return ""
    text = str(value)
    return text[:length] + "…" if len(text) > length else text


def read_only_state(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"missing": str(db_path), "meta": {}, "batches": [], "invalidations": [], "snapshots": [], "leases": []}
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM meta ORDER BY key")}
        batches = [dict(row) for row in conn.execute("SELECT * FROM batches ORDER BY worker_id,batch_index,batch_id")]
        invalidations = [dict(row) for row in conn.execute("SELECT * FROM invalidations ORDER BY worker_id,batch_id,request_id,id")]
        snapshots = [dict(row) for row in conn.execute("SELECT * FROM snapshots ORDER BY snapshot_id")]
        leases = [dict(row) for row in conn.execute("SELECT * FROM leases ORDER BY worker_id,batch_id,lease_id")]
        return {"meta": meta, "batches": batches, "invalidations": invalidations, "snapshots": snapshots, "leases": leases}
    finally:
        conn.close()


class DiagnosticsApp:
    def __init__(self, root: Any, sample_dir: Path, runtime_dir: Path) -> None:
        self.root = root
        self.sample_dir = sample_dir
        self.runtime_dir = runtime_dir
        verify_db = runtime_dir / "verify" / "ledger.sqlite3"
        ui_db = runtime_dir / "ui" / "ledger.sqlite3"
        self.db_path = verify_db if verify_db.exists() else ui_db
        self.bundle = load_bundle(sample_dir)
        self.plan = build_plan(self.bundle)
        self.demo_bundle = self.bundle
        self.scheduler: LeaseScheduler | None = None
        self.pool: WorkerPool | None = None
        self.demo_thread: threading.Thread | None = None
        root.title("Receipt Ledger Diagnostics — 127.0.0.1")
        root.geometry("1280x760")

        self.summary = self._label(root, "加载中…")
        self.summary.pack(anchor="w", padx=10, pady=6)
        buttons = self._frame(root)
        buttons.pack(fill="x", padx=10)
        self._button(buttons, "刷新只读账本", self.refresh)
        self._button(buttons, "启动实时租约演示", self.start_demo)
        self._button(buttons, "停止演示工作进程", self.stop_demo)

        import tkinter.ttk as ttk

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=8)
        self.trees: dict[str, Any] = {}
        self._create_tab(
            "workers",
            ["worker_id", "running", "quota", "water", "in_flight_leases", "expires_ms"],
            widths=[160, 90, 80, 100, 260, 220],
        )
        self._create_tab(
            "queue",
            ["batch_id", "worker_id", "batch_index", "wait_started_ms", "wait_ended_ms", "wait_ms/current"],
            widths=[260, 140, 90, 150, 150, 140],
        )
        self._create_tab(
            "batches",
            ["status", "unknown", "batch_id", "worker", "idx", "param_v", "param_digest", "output_digest", "frozen_digest", "wait_ms"],
            widths=[100, 80, 230, 130, 50, 70, 150, 150, 150, 80],
        )
        self._create_tab(
            "invalidations",
            ["worker", "batch_id", "request_id", "node", "level", "cause", "basis"],
            widths=[130, 220, 130, 150, 90, 260, 420],
        )
        self._create_tab(
            "snapshots",
            ["snapshot_id", "format", "ledger_v", "content_digest", "path"],
            widths=[220, 220, 80, 180, 420],
        )
        self.status = self._label(root, "")
        self.status.pack(anchor="w", padx=10, pady=4)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh()
        self.root.after(1000, self.tick)

    def _label(self, parent: Any, text: str) -> Any:
        import tkinter as tk

        return tk.Label(parent, text=text, justify="left", anchor="w")

    def _frame(self, parent: Any) -> Any:
        import tkinter as tk

        return tk.Frame(parent)

    def _button(self, parent: Any, text: str, command: Any) -> None:
        import tkinter as tk

        tk.Button(parent, text=text, command=command).pack(side="left", padx=4)

    def _create_tab(self, name: str, columns: list[str], widths: list[int]) -> None:
        import tkinter.ttk as ttk

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column, width in zip(columns, widths):
            tree.heading(column, text=column)
            tree.column(column, width=width, anchor="w")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self.trees[name] = tree

    def _set_rows(self, tab: str, rows: list[tuple[Any, ...]]) -> None:
        tree = self.trees[tab]
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", "end", values=row)

    def tick(self) -> None:
        self.refresh(live_only=True)
        self.root.after(1000, self.tick)

    def refresh(self, live_only: bool = False) -> None:
        state = read_only_state(self.db_path)
        meta = state.get("meta", {})
        unknown = [batch for batch in state.get("batches", []) if batch.get("status") == "unknown"]
        db_note = state.get("missing") or str(self.db_path)
        self.summary.config(
            text=(
                f"账本格式版本: {meta.get('ledger_format_version', 'N/A')}    "
                f"内容摘要: {short_digest(meta.get('content_digest', ''), 24)}    "
                f"未知批次: {len(unknown)}    快照: {len(state.get('snapshots', []))}    "
                f"数据库(只读): {db_note}"
            )
        )

        water_rows: list[tuple[Any, ...]] = []
        queue_rows: list[tuple[Any, ...]] = []
        if self.scheduler is not None:
            water = self.scheduler.water_snapshot()
            active_by_worker: dict[str, list[Any]] = {}
            expiry_by_worker: dict[str, list[int]] = {}
            for lease in water["active_leases"]:
                active_by_worker.setdefault(lease["worker_id"], []).append(lease["batch_id"][:24])
                expiry_by_worker.setdefault(lease["worker_id"], []).append(lease["expires_monotonic_ms"])
            for worker_id, item in sorted(water["workers"].items()):
                water_rows.append(
                    (
                        worker_id,
                        item["running"],
                        item["quota"],
                        item["water_level"],
                        ", ".join(active_by_worker.get(worker_id, [])),
                        ", ".join(str(v) for v in expiry_by_worker.get(worker_id, [])),
                    )
                )
            for entry in self.scheduler.queue_snapshot():
                queue_rows.append(
                    (
                        entry["batch_id"],
                        entry["worker_id"],
                        entry["batch_index"],
                        entry["wait_started_monotonic_ms"],
                        entry["wait_ended_monotonic_ms"],
                        entry["wait_ms"] if entry["wait_ms"] is not None else f"active {entry['current_wait_ms']}ms",
                    )
                )
        else:
            quotas = {worker.worker_id: worker.quota for worker in self.bundle.workers}
            running: dict[str, int] = {}
            leases_by_worker: dict[str, list[dict[str, Any]]] = {}
            for lease in state.get("leases", []):
                if lease.get("status") == "active":
                    running[lease["worker_id"]] = running.get(lease["worker_id"], 0) + 1
                    leases_by_worker.setdefault(lease["worker_id"], []).append(lease)
            for worker in self.bundle.workers:
                count = running.get(worker.worker_id, 0)
                leases = leases_by_worker.get(worker.worker_id, [])
                water_rows.append(
                    (
                        worker.worker_id,
                        count,
                        worker.quota,
                        f"{count}/{quotas[worker.worker_id]}",
                        ", ".join(lease["batch_id"][:24] for lease in leases),
                        ", ".join(str(lease["expires_monotonic_ms"]) for lease in leases),
                    )
                )
            for batch in state.get("batches", []):
                if batch.get("status") != "committed":
                    queue_rows.append(
                        (
                            batch["batch_id"],
                            batch["worker_id"],
                            batch["batch_index"],
                            batch.get("wait_started_ms"),
                            batch.get("wait_ended_ms"),
                            batch.get("wait_ms"),
                        )
                    )
        self._set_rows("workers", water_rows)
        self._set_rows("queue", queue_rows)

        batch_rows = []
        for batch in state.get("batches", []):
            batch_rows.append(
                (
                    batch.get("status"),
                    "YES" if batch.get("status") == "unknown" else "",
                    batch.get("batch_id"),
                    batch.get("worker_id"),
                    batch.get("batch_index"),
                    batch.get("param_version"),
                    short_digest(batch.get("param_digest")),
                    short_digest(batch.get("output_digest")),
                    short_digest(batch.get("frozen_output_digest")),
                    batch.get("wait_ms"),
                )
            )
        self._set_rows("batches", batch_rows)

        invalidation_rows = []
        for row in state.get("invalidations", []):
            basis = row.get("basis_json", "")
            invalidation_rows.append(
                (
                    row.get("worker_id"),
                    row.get("batch_id"),
                    row.get("request_id"),
                    row.get("node_type"),
                    row.get("dependency_level"),
                    row.get("cause"),
                    basis[:400],
                )
            )
        self._set_rows("invalidations", invalidation_rows)

        snapshot_rows = []
        for row in state.get("snapshots", []):
            snapshot_rows.append(
                (
                    row.get("snapshot_id"),
                    row.get("format_version"),
                    row.get("ledger_version"),
                    short_digest(row.get("content_digest"), 24),
                    row.get("path"),
                )
            )
        self._set_rows("snapshots", snapshot_rows)
        if not live_only:
            self.status.config(text=f"已按稳定顺序刷新；只读查询不写入账本。当前时间 {time.strftime('%H:%M:%S')}")

    def start_demo(self) -> None:
        if self.scheduler is not None:
            return
        try:
            demo_dir = self.runtime_dir / "ui-loopback-sample"
            shutil.copytree(self.sample_dir, demo_dir, dirs_exist_ok=True)
            declaration_path = demo_dir / "workers.json"
            declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
            for worker in declaration["workers"]:
                worker["port"] = free_loopback_port()
            declaration_path.write_text(json.dumps(declaration, ensure_ascii=False, indent=2), encoding="utf-8")
            self.demo_bundle = load_bundle(demo_dir)
            self.pool = WorkerPool(self.demo_bundle.workers, demo_dir).start_all()
        except Exception as exc:
            self.status.config(text=f"工作进程启动失败: {type(exc).__name__}: {exc}")
            self.pool = None
            return
        self.scheduler = LeaseScheduler(
            {worker.worker_id: worker.quota for worker in self.demo_bundle.workers},
            self.demo_bundle.rules.global_concurrency,
            lease_ttl_ms=self.demo_bundle.rules.lease_ttl_ms,
        )

        def process(batch: Any, lease: LeaseScheduler) -> dict[str, Any]:
            assert self.pool is not None
            result = self.pool.forward(batch.worker_id, batch)
            time.sleep(0.35)
            return result

        def run_demo() -> None:
            assert self.scheduler is not None
            demo_plan = build_plan(self.demo_bundle)
            self.scheduler.run(demo_plan.batches, process)

        self.demo_thread = threading.Thread(target=run_demo, daemon=True)
        self.demo_thread.start()
        self.status.config(text="实时演示已启动：全局 2、按工作进程配额放行；可切换到 queue/workers 页观察。")

    def stop_demo(self) -> None:
        if self.pool is not None:
            self.pool.stop_all()
        self.pool = None
        self.scheduler = None
        self.status.config(text="演示工作进程已强制回收；只读账本内容未改变。")

    def on_close(self) -> None:
        self.stop_demo()
        self.root.destroy()


def launch_ui(sample_dir: str | Path = "samples/local", runtime_dir: str | Path = ".runtime") -> int:
    try:
        import tkinter as tk
    except ImportError as exc:
        print("Tkinter 不可用。Windows 官方 Python 安装包默认包含 tcl/tk；Linux 可安装 python3-tk。")
        print(f"ImportError: {exc}")
        return 1
    root = tk.Tk()
    DiagnosticsApp(root, Path(sample_dir), Path(runtime_dir))
    root.mainloop()
    return 0
