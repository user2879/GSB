from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from .engine import RunOptions, run_engine
from .errors import LeaseError
from .verifier import verify_ledger


def main(
    spec_path: str = "samples/demo_spec.json",
    db_path: str = "reports/gui.sqlite",
    workspace: str = "reports/gui-workspace",
) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:  # pragma: no cover - platform dependent
        raise LeaseError(
            "E_TK",
            f"Tkinter is not available in this Python distribution: {exc}",
            field="tkinter",
            expected="Python built with Tk support",
            actual=repr(exc),
        ) from exc

    root = tk.Tk()
    root.title("LeaseCache · 跨节点制品租约仲裁")
    root.geometry("1180x760")
    root.minsize(900, 600)

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    busy = {"running": False}

    spec_var = tk.StringVar(value=str(Path(spec_path).resolve()))
    db_var = tk.StringVar(value=str(Path(db_path).resolve()))
    ws_var = tk.StringVar(value=str(Path(workspace).resolve()))
    status_var = tk.StringVar(value="就绪")
    lease_var = tk.StringVar(value="")
    latest_report: dict[str, Any] = {}

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)

    form = ttk.Frame(frame)
    form.pack(fill="x")
    for row, (label, variable, choose) in enumerate(
        (
            ("规格 JSON", spec_var, "file"),
            ("SQLite 账本", db_var, "save"),
            ("工作区", ws_var, "dir"),
        )
    ):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=6, pady=3
        )

        def pick(var=variable, kind=choose):
            if kind == "file":
                path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            elif kind == "dir":
                path = filedialog.askdirectory()
            else:
                path = filedialog.asksaveasfilename(defaultextension=".sqlite")
            if path:
                var.set(path)

        ttk.Button(form, text="浏览", command=pick).grid(row=row, column=2, pady=3)
    form.columnconfigure(1, weight=1)

    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=(8, 6))
    run_button = ttk.Button(controls, text="增量运行")
    run_button.pack(side="left")
    reuse_button = ttk.Button(controls, text="仅评估复用")
    reuse_button.pack(side="left", padx=6)
    verify_button = ttk.Button(controls, text="独立验证账本")
    verify_button.pack(side="left", padx=6)
    open_button = ttk.Button(controls, text="加载报告")
    open_button.pack(side="left", padx=6)
    ttk.Label(controls, text="期望租约 ms:").pack(side="left", padx=(20, 4))
    ttk.Entry(controls, textvariable=lease_var, width=8).pack(side="left")
    ttk.Label(controls, textvariable=status_var, foreground="#164e63").pack(
        side="right"
    )

    notebook = ttk.Notebook(frame)
    notebook.pack(fill="both", expand=True)

    nodes_tree = ttk.Treeview(
        notebook,
        columns=("node", "port", "cap", "limit", "used", "active", "free"),
        show="headings",
    )
    for column, title, width in (
        ("node", "节点", 160),
        ("port", "回环端口", 100),
        ("cap", "容量槽位", 100),
        ("limit", "租约上限", 100),
        ("used", "占用", 80),
        ("active", "活动租约", 100),
        ("free", "剩余", 80),
    ):
        nodes_tree.heading(column, text=title)
        nodes_tree.column(column, width=width, anchor="center")
    notebook.add(nodes_tree, text="节点")

    artifacts_tree = ttk.Treeview(
        notebook,
        columns=("key", "digest", "sources", "reusable", "selected"),
        show="headings",
    )
    for column, title, width in (
        ("key", "制品键", 190),
        ("digest", "规范摘要", 300),
        ("sources", "来源节点集合", 180),
        ("reusable", "可复用", 90),
        ("selected", "服务节点", 130),
    ):
        artifacts_tree.heading(column, text=title)
        artifacts_tree.column(column, width=width, anchor="w")
    notebook.add(artifacts_tree, text="制品与仲裁")

    queue_tree = ttk.Treeview(
        notebook,
        columns=("seq", "kind", "key", "node", "stage", "start", "end", "waited"),
        show="headings",
    )
    for column, title, width in (
        ("seq", "队列序号", 80),
        ("kind", "类型", 90),
        ("key", "制品键", 160),
        ("node", "节点", 100),
        ("stage", "阶段", 100),
        ("start", "等待开始 ms", 120),
        ("end", "等待结束 ms", 120),
        ("waited", "等待 ms", 90),
    ):
        queue_tree.heading(column, text=title)
        queue_tree.column(column, width=width, anchor="center")
    notebook.add(queue_tree, text="排队")

    events_tree = ttk.Treeview(
        notebook,
        columns=("seq", "kind", "node", "key", "lease", "rationale"),
        show="headings",
    )
    for column, title, width in (
        ("seq", "事件序号", 90),
        ("kind", "事件", 150),
        ("node", "节点", 110),
        ("key", "制品键", 170),
        ("lease", "租约", 210),
        ("rationale", "裁决依据", 400),
    ):
        events_tree.heading(column, text=title)
        events_tree.column(column, width=width, anchor="w")
    notebook.add(events_tree, text="归属事件")

    raw_text = tk.Text(notebook, wrap="none", font=("Consolas", 10))
    notebook.add(raw_text, text="原始报告 / 验证结果")

    def clear_trees() -> None:
        for tree in (nodes_tree, artifacts_tree, queue_tree, events_tree):
            for item in tree.get_children():
                tree.delete(item)

    def show_report(report: dict[str, Any]) -> None:
        clear_trees()
        for node in report.get("nodes", []):
            nodes_tree.insert(
                "",
                "end",
                values=(
                    node["node_id"],
                    node["port"],
                    node["capacity_slots"],
                    node["lease_limit"],
                    node["used_slots"],
                    node["active_leases"],
                    node["free_slots"],
                ),
            )
        for artifact in report.get("artifacts", []):
            artifacts_tree.insert(
                "",
                "end",
                values=(
                    artifact["artifact_key"],
                    artifact["canonical_digest"],
                    ", ".join(artifact["source_nodes"]),
                    "是" if artifact["reusable"] else "否",
                    artifact.get("selected_node") or "排队",
                ),
            )
        scheduling = report.get("scheduling") or {}
        for wait in scheduling.get("queue_waits", []):
            queue_tree.insert(
                "",
                "end",
                values=(
                    wait["queue_seq"],
                    wait["kind"],
                    wait["artifact_key"],
                    wait.get("node_id") or "",
                    wait.get("stage_id") or "",
                    wait["wait_started_mono_ms"],
                    wait["wait_ended_mono_ms"],
                    wait["waited_mono_ms"],
                ),
            )
        for event in report.get("ownership_events", []):
            events_tree.insert(
                "",
                "end",
                values=(
                    event["seq"],
                    event["kind"],
                    event.get("node_id") or "",
                    event.get("artifact_key") or "",
                    event.get("lease_id") or "",
                    event.get("rationale") or "",
                ),
            )
        raw_text.delete("1.0", "end")
        raw_text.insert(
            "end", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        )
        status_var.set(
            f"完成 · 复用 {len((report.get('incremental') or {}).get('reusable_keys', []))}"
            f" · 重算 {len((report.get('incremental') or {}).get('recomputed_keys', []))}"
        )

    def worker(action: str) -> None:
        try:
            spec = spec_var.get()
            db = db_var.get()
            workspace = ws_var.get()
            if action in {"run", "reuse"}:
                override = None
                if lease_var.get().strip():
                    override = int(lease_var.get().strip())
                report = run_engine(
                    spec,
                    db,
                    workspace,
                    options=RunOptions(
                        lease_ms=override,
                        wait_timeout_ms=20000,
                        settle=True,
                        reuse_only=(action == "reuse"),
                    ),
                )
                events.put(("report", report))
            elif action == "verify":
                result = verify_ledger(spec, db)
                events.put(("verify", result))
            elif action == "open":
                path = Path(db).with_suffix("").with_suffix(".report.json")
                report = json.loads(path.read_text(encoding="utf-8"))
                events.put(("report", report))
        except LeaseError as exc:
            events.put(("error", exc.to_dict()))
        except Exception as exc:
            events.put(("error", {"code": "E_INTERNAL", "message": repr(exc)}))
        finally:
            events.put(("done", None))

    def start(action: str) -> None:
        if busy["running"]:
            messagebox.showinfo("正在执行", "当前运行尚未完成。")
            return
        busy["running"] = True
        status_var.set("运行中… 正在启动真实回环子进程")
        thread = threading.Thread(target=worker, args=(action,), daemon=True)
        thread.start()

    run_button.configure(command=lambda: start("run"))
    reuse_button.configure(command=lambda: start("reuse"))
    verify_button.configure(command=lambda: start("verify"))
    open_button.configure(command=lambda: start("open"))

    def poll() -> None:
        try:
            while True:
                event, payload = events.get_nowait()
                if event == "report":
                    latest_report.clear()
                    latest_report.update(payload)
                    show_report(payload)
                elif event == "verify":
                    raw_text.delete("1.0", "end")
                    raw_text.insert(
                        "end", json.dumps(payload, ensure_ascii=False, indent=2)
                    )
                    status_var.set("独立验证通过")
                    notebook.select(raw_text)
                elif event == "error":
                    status_var.set(f"失败 {payload.get('code')}: {payload.get('message')}")
                    raw_text.delete("1.0", "end")
                    raw_text.insert(
                        "end", json.dumps(payload, ensure_ascii=False, indent=2)
                    )
                    notebook.select(raw_text)
                    messagebox.showerror(
                        payload.get("code", "错误"), payload.get("message", "")
                    )
                elif event == "done":
                    busy["running"] = False
        except queue.Empty:
            pass
        root.after(100, poll)

    def on_close() -> None:
        if busy["running"]:
            messagebox.showinfo("运行进行中", "请等待当前租约运行完成后再关闭窗口。")
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    poll()
    root.mainloop()
    return 0
