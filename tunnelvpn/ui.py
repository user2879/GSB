"""Tkinter diagnostics console for the loopback tunnel lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import TunnelConfig, load_config
from .engine import TunnelEngine
from .errors import TunnelError
from .peer import PeerServer
from .snapshot import SnapshotStore, restore_snapshot


def launch_ui(config: TunnelConfig, state_dir: str | Path) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError:
        print(
            "tkinter 不可用。Windows 官方 Python 安装包请勾选 Tcl/Tk；"
            "核心引擎与演示脚本无需 tkinter。",
            flush=True,
        )
        return 2

    state_path = Path(state_dir)
    store = SnapshotStore(state_path / "snapshots")
    root = tk.Tk()
    root.title("Loopback Tunnel Diagnostics")
    root.geometry("1380x900")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Treeview", rowheight=24)
    style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))

    app_state: dict[str, Any] = {
        "engine": None,
        "peers": [],
        "restore_report": None,
    }
    refresh_job: list[str | None] = [None]

    top = ttk.Frame(root, padding=8)
    top.pack(fill=tk.X)
    ttk.Label(top, text="本地隧道与 VPN 转发诊断", style="Title.TLabel").pack(side=tk.LEFT)
    status_var = tk.StringVar(value="未启动")
    ttk.Label(top, textvariable=status_var).pack(side=tk.LEFT, padx=16)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

    routes_frame = ttk.Frame(notebook)
    budget_frame = ttk.Frame(notebook)
    sessions_frame = ttk.Frame(notebook)
    stats_frame = ttk.Frame(notebook)
    snapshots_frame = ttk.Frame(notebook)
    report_frame = ttk.Frame(notebook)
    for frame, title in (
        (routes_frame, "隧道路由"),
        (budget_frame, "四重预算"),
        (sessions_frame, "会话状态"),
        (stats_frame, "吞吐与排队"),
        (snapshots_frame, "快照"),
        (report_frame, "恢复报告"),
    ):
        notebook.add(frame, text=title)

    route_tree = _tree(
        routes_frame,
        ("id", "listen", "target", "mtu", "cipher", "kbps", "max_sessions"),
        (110, 150, 150, 70, 180, 90, 100),
    )
    budget_tree = _tree(
        budget_frame,
        ("scope", "active", "max", "waiting", "token_bytes", "used_bytes", "rate_kbps"),
        (120, 70, 70, 70, 130, 140, 110),
    )
    session_tree = _tree(
        sessions_frame,
        (
            "session",
            "route",
            "sequence",
            "status",
            "code",
            "payload",
            "forwarded",
            "wire",
            "queues",
            "queue_ms",
            "fragments",
            "error",
        ),
        (130, 90, 70, 90, 100, 90, 90, 90, 70, 90, 80, 250),
    )
    stats_tree = _tree(
        stats_frame,
        ("scope", "sessions", "completed", "failed", "forwarded", "throughput", "avg_queue"),
        (120, 80, 90, 80, 120, 130, 130),
    )
    snapshot_tree = _tree(
        snapshots_frame,
        ("name", "complete", "route_version", "created", "sessions", "digest"),
        (370, 80, 180, 230, 80, 480),
    )
    report_text = tk.Text(report_frame, wrap=tk.WORD, font=("Consolas", 10))
    report_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    controls = ttk.Frame(root, padding=8)
    controls.pack(fill=tk.X)

    def fill_routes() -> None:
        route_tree.delete(*route_tree.get_children())
        for route in config.routes:
            route_tree.insert(
                "",
                tk.END,
                values=(
                    route.id,
                    f"{route.listen.host}:{route.listen.port}",
                    f"{route.target.host}:{route.target.port}",
                    route.mtu,
                    route.cipher,
                    route.bandwidth_kbps,
                    route.max_sessions,
                ),
            )

    def start_engine() -> None:
        if app_state["engine"] is not None:
            return
        peers: list[PeerServer] = []
        engine: TunnelEngine | None = None
        try:
            for peer in config.peers:
                if peer.behavior == "unreachable":
                    continue
                server = PeerServer(
                    config,
                    peer,
                    log_path=state_path / "logs" / f"peer-{peer.id}.jsonl",
                )
                server.start()
                peers.append(server)
            engine = TunnelEngine(
                config, log_path=state_path / "logs" / "engine-ui.jsonl"
            )
            engine.start()
        except Exception as exc:
            if engine is not None:
                engine.stop()
            for server in peers:
                server.stop()
            messagebox.showerror("启动失败", str(exc))
            return
        app_state["engine"] = engine
        app_state["peers"] = peers
        status_var.set("运行中")
        refresh()

    def stop_engine() -> None:
        engine = app_state.get("engine")
        if engine is not None:
            engine.stop()
        for server in app_state.get("peers", []):
            server.stop()
        app_state["engine"] = None
        app_state["peers"] = []
        status_var.set("已停止")
        refresh()

    def save_snapshot() -> None:
        engine = app_state.get("engine")
        if engine is None:
            messagebox.showinfo("快照", "请先启动引擎。")
            return
        try:
            snapshot = store.save_engine(engine)
            status_var.set(f"快照已保存: {snapshot.path.name}")
            refresh_snapshots()
        except Exception as exc:
            messagebox.showerror("快照失败", str(exc))

    def restore_selected() -> None:
        selection = snapshot_tree.selection()
        if not selection:
            messagebox.showinfo("恢复", "请选择一个快照。")
            return
        name = snapshot_tree.item(selection[0], "values")[0]
        path = state_path / "snapshots" / name
        try:
            restored = restore_snapshot(store.load(path), config)
            app_state["restore_report"] = restored.report.to_dict()
            report_text.delete("1.0", tk.END)
            report_text.insert(
                tk.END,
                json.dumps(
                    {
                        "restore_report": restored.report.to_dict(),
                        "ledger_digest": restored.ledger.digest(),
                        "unknown_sessions": [
                            item.session_id
                            for item in restored.sessions
                            if item.requires_rerun
                        ],
                        "statistics": restored.statistics,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ),
            )
            notebook.select(report_frame)
        except TunnelError as exc:
            messagebox.showerror("恢复失败", json.dumps(exc.to_dict(), ensure_ascii=False))

    def refresh_snapshots() -> None:
        snapshot_tree.delete(*snapshot_tree.get_children())
        for entry in store.list_snapshots():
            snapshot_tree.insert(
                "",
                tk.END,
                values=(
                    entry.path.name,
                    "yes" if entry.complete else "no",
                    entry.route_version or "",
                    entry.created_utc or "",
                    entry.session_count if entry.session_count is not None else "",
                    entry.ledger_digest or "",
                ),
            )

    def refresh() -> None:
        if refresh_job[0] is not None:
            try:
                root.after_cancel(refresh_job[0])
            except tk.TclError:
                pass
        engine: TunnelEngine | None = app_state.get("engine")
        budget_tree.delete(*budget_tree.get_children())
        session_tree.delete(*session_tree.get_children())
        stats_tree.delete(*stats_tree.get_children())
        if engine is not None:
            budgets = engine.budget.snapshot().to_dict()
            global_budget = budgets["global"]
            budget_tree.insert(
                "",
                tk.END,
                values=(
                    "GLOBAL",
                    global_budget["active_sessions"],
                    global_budget["max_sessions"],
                    global_budget["waiting"],
                    global_budget["available_bytes_floor"],
                    global_budget["used_bytes"],
                    global_budget["bandwidth_kbps"],
                ),
            )
            for route_id, route_budget in sorted(budgets["routes"].items()):
                budget_tree.insert(
                    "",
                    tk.END,
                    values=(
                        route_id,
                        route_budget["active_sessions"],
                        route_budget["max_sessions"],
                        route_budget["waiting"],
                        route_budget["available_bytes_floor"],
                        route_budget["used_bytes"],
                        route_budget["bandwidth_kbps"],
                    ),
                )
            for session in engine.sessions_snapshot()[-200:]:
                error = session.get("error") or {}
                session_tree.insert(
                    "",
                    tk.END,
                    values=(
                        session["session_id"],
                        session["route_id"],
                        session["sequence"],
                        session["status"],
                        session["business_code"],
                        session["payload_bytes"],
                        session["forwarded_bytes"],
                        session["wire_bytes"],
                        session["queue_count"],
                        session["total_queue_ms"],
                        session["fragment_count"],
                        error.get("message", ""),
                    ),
                )
            statistics = engine.ledger.statistics()
            for scope, data in [
                ("GLOBAL", statistics["global"]),
                *[(name, value) for name, value in statistics["routes"].items()],
            ]:
                stats_tree.insert(
                    "",
                    tk.END,
                    values=(
                        scope,
                        data["session_count"],
                        data["completed_count"],
                        data["failed_count"],
                        data["forwarded_bytes"],
                        data["throughput_kbps"],
                        data["average_queue_duration_ms"],
                    ),
                )
            status_var.set(
                f"运行中 | {len(engine.sessions_snapshot())} 会话 | "
                f"{len(engine.ledger.records())} 账本 | 业务码 {statistics['business_code_counts']}"
            )
        refresh_snapshots()
        refresh_job[0] = root.after(800, refresh)

    def on_close() -> None:
        stop_engine()
        root.destroy()

    ttk.Button(controls, text="启动引擎", command=start_engine).pack(side=tk.LEFT)
    ttk.Button(controls, text="停止", command=stop_engine).pack(side=tk.LEFT, padx=4)
    ttk.Button(controls, text="保存快照", command=save_snapshot).pack(side=tk.LEFT, padx=4)
    ttk.Button(controls, text="恢复选中快照", command=restore_selected).pack(
        side=tk.LEFT, padx=4
    )
    ttk.Button(controls, text="刷新", command=refresh).pack(side=tk.LEFT, padx=4)
    ttk.Label(
        controls,
        text="四重预算 = 全局/隧道并发 + 全局/隧道令牌桶；所有端点均为 127.0.0.1",
    ).pack(side=tk.RIGHT)

    root.protocol("WM_DELETE_WINDOW", on_close)
    fill_routes()
    refresh()
    root.mainloop()
    return 0


def _tree(parent: Any, columns: tuple[str, ...], widths: tuple[int, ...]) -> Any:
    import tkinter as tk
    from tkinter import ttk

    frame = ttk.Frame(parent)
    frame.pack(fill=tk.BOTH, expand=True)
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    for column, width in zip(columns, widths):
        tree.heading(column, text=column)
        tree.column(column, width=width, anchor=tk.W)
    y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tkinter tunnel diagnostics")
    parser.add_argument("--config", default="data/demo_routes.json")
    parser.add_argument("--state-dir", default="artifacts/ui-state")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except TunnelError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True))
        return 2
    return launch_ui(config, args.state_dir)


if __name__ == "__main__":
    raise SystemExit(main())
