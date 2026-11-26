from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .ledger import Ledger
from .spec import load_spec_bundle


def tkinter_status() -> dict[str, Any]:
    try:
        import tkinter  # noqa: F401
        from tkinter import ttk  # noqa: F401

        return {"tkinter_available": True}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"tkinter_available": False, "error": type(exc).__name__, "message": str(exc)}


def run_tk_ui(args: Any) -> int:
    status = tkinter_status()
    if getattr(args, "check", False):
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if not status["tkinter_available"]:
        print(json.dumps({"ok": False, **status, "hint": "Install python3-tk or run with the system Python that includes Tk."}, ensure_ascii=False, indent=2))
        return 2

    import tkinter as tk
    from tkinter import messagebox, ttk

    class DiagnosticUI:
        def __init__(self, root: tk.Tk, spec_path: str, db_path: str) -> None:
            self.root = root
            self.spec_path = spec_path
            self.db_path = db_path
            self.root.title("边缘推理分片调度诊断 / Edge Shard Diagnostics")
            self.root.geometry("1180x720")
            self._build_layout()
            self.refresh()

        def _build_layout(self) -> None:
            toolbar = ttk.Frame(self.root, padding=8)
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="校验声明", command=self.validate_spec).pack(side="left", padx=3)
            ttk.Button(toolbar, text="归一回执", command=self.normalize_receipts).pack(side="left", padx=3)
            ttk.Button(toolbar, text="运行调度", command=self.run_scheduler).pack(side="left", padx=3)
            ttk.Button(toolbar, text="强制终止验收", command=self.run_acceptance).pack(side="left", padx=3)
            ttk.Button(toolbar, text="刷新账本", command=self.refresh).pack(side="left", padx=3)
            self.status = ttk.Label(toolbar, text="ready")
            self.status.pack(side="right")

            main = ttk.Panedwindow(self.root, orient="horizontal")
            main.pack(fill="both", expand=True, padx=8, pady=4)

            left = ttk.Frame(main)
            columns = ("status", "capacity", "node", "lease", "summary", "marker")
            self.tree = ttk.Treeview(left, columns=columns, show="headings", height=18)
            headings = {
                "status": ("状态", 110),
                "capacity": ("容量版本", 80),
                "node": ("节点", 90),
                "lease": ("租约版本", 80),
                "summary": ("聚合摘要", 430),
                "marker": ("恢复标记", 130),
            }
            for key, (text, width) in headings.items():
                self.tree.heading(key, text=text)
                self.tree.column(key, width=width, anchor="w")
            self.tree.tag_configure("unknown", background="#fff1f0", foreground="#b42318")
            self.tree.tag_configure("stale", foreground="#98a2b3")
            self.tree.pack(fill="both", expand=True)
            main.add(left, weight=3)

            right = ttk.Frame(main)
            ttk.Label(right, text="快照版本列表", font=("Sans", 11, "bold")).pack(anchor="w")
            self.snapshots = tk.Listbox(right, height=12)
            self.snapshots.pack(fill="x", pady=4)
            ttk.Label(right, text="当前视图 / 事件", font=("Sans", 11, "bold")).pack(anchor="w", pady=(8, 0))
            self.details = tk.Text(right, wrap="word", height=24)
            self.details.pack(fill="both", expand=True)
            main.add(right, weight=2)

        def set_text(self, value: str) -> None:
            self.details.delete("1.0", "end")
            self.details.insert("1.0", value)

        def refresh(self) -> None:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.snapshots.delete(0, "end")
            if not Path(self.db_path).exists():
                self.status.config(text="账本尚未创建")
                return
            ledger = Ledger(self.db_path)
            try:
                outcomes = ledger.conn.execute(
                    "SELECT * FROM shard_outcomes ORDER BY capacity_version DESC, shard_id"
                ).fetchall()
                for row in outcomes:
                    marker = "⚠ UNKNOWN/需重跑" if row["unknown"] or row["status"] == "UNKNOWN" else ("STALE" if row["stale"] else "")
                    tags = ("unknown",) if marker else (("stale",) if row["stale"] else ())
                    self.tree.insert(
                        "",
                        "end",
                        values=(row["status"], row["capacity_version"], row["node_id"] or "", row["lease_version"] or "", (row["summary"] or "")[:64], marker),
                        tags=tags,
                    )
                for snap in ledger.list_snapshots():
                    digest = str(snap["content_digest"])[:10]
                    self.snapshots.insert(
                        "end",
                        f"#{snap['snapshot_id']} fmt={snap['format_version']} nodes=v{snap['node_set_version']} cap=v{snap['capacity_version']} fixed=r{snap['fixed_revision']} digest={digest}",
                    )
                events = ledger.list_events(limit=30)
                self.set_text(json.dumps(events, ensure_ascii=False, indent=2, default=str))
                self.status.config(text=f"{len(outcomes)} outcome rows / {len(ledger.list_snapshots())} snapshots")
            finally:
                ledger.close()

        def validate_spec(self) -> None:
            try:
                bundle = load_spec_bundle(self.spec_path)
                messagebox.showinfo("E_SPEC 校验通过", f"{len(bundle.nodes)} nodes, {len(bundle.shards)} shards\ndigest={bundle.content_digest[:16]}")
            except DiagnosticError as exc:
                messagebox.showerror(exc.code, str(exc.to_dict()))

        def normalize_receipts(self) -> None:
            from .acceptance import normalize_demo_receipts

            try:
                result = normalize_demo_receipts(Path(self.spec_path), Path("samples/receipts"))
                self.set_text(json.dumps({"counts": result["counts"], "adopted": result["adopted"]}, ensure_ascii=False, indent=2, default=str))
            except DiagnosticError as exc:
                messagebox.showerror(exc.code, str(exc))

        def _run_background(self, target: Any) -> None:
            self.status.config(text="运行中…")

            def worker() -> None:
                try:
                    message = target()
                    self.root.after(0, lambda: self.status.config(text=message))
                    self.root.after(0, self.refresh)
                except Exception as exc:  # pragma: no cover - UI callback
                    self.root.after(0, lambda: messagebox.showerror("运行失败", str(exc)))

            threading.Thread(target=worker, daemon=True).start()

        def run_scheduler(self) -> None:
            def work() -> str:
                from .cli import cmd_run

                args = argparse_namespace(spec=self.spec_path, db=self.db_path, logs="artifacts/runtime/ui-logs", ttl_ms=30000)
                cmd_run(args)
                return "调度完成"

            self._run_background(work)

        def run_acceptance(self) -> None:
            def work() -> str:
                from .acceptance import run_acceptance

                run_acceptance("artifacts/ui-acceptance")
                return "强制终止验收完成"

            self._run_background(work)

    root = tk.Tk()
    ui = DiagnosticUI(root, getattr(args, "spec", "config/demo"), getattr(args, "db", "artifacts/runtime/ledger.sqlite"))
    smoke_ms = __import__("os").environ.get("EDGE_UI_SMOKE_MS")
    if smoke_ms:
        root.after(int(smoke_ms), root.destroy)
    root.mainloop()
    return 0


def argparse_namespace(**kwargs: Any) -> Any:
    import argparse

    return argparse.Namespace(**kwargs)
