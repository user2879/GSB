from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from .engine import DiagnosticEngine
from .ledger import Ledger, load_or_replace_snapshot
from .runtime import NodeCluster
from .spec import load_spec_bundle


def launch(root: Path, db_name: str = "ledger/diagnostic.sqlite") -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        print(json.dumps({"ok": False, "code": "E_TK", "error": repr(exc)}, ensure_ascii=False))
        return 2

    root_path = Path(root)
    db_path = root_path / db_name
    try:
        window = tk.Tk()
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "code": "E_DISPLAY",
            "error": "无法打开 Tk 显示；在 WSL 中请配置 WSLg 或 X Server",
            "detail": repr(exc),
        }, ensure_ascii=False))
        return 2

    window.title("本地推理分片下发诊断")
    window.geometry("980x680")
    frame = ttk.Frame(window, padding=8)
    frame.pack(fill="both", expand=True)

    output = tk.Text(frame, wrap="none", font=("Consolas", 10))
    output.pack(fill="both", expand=True)
    status = ttk.Label(frame, text="就绪")
    status.pack(anchor="w", pady=(6, 0))

    def show(value: Any) -> None:
        output.delete("1.0", "end")
        output.insert("1.0", json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))

    def run_async(title: str, work: Callable[[], Any]) -> None:
        status.config(text=f"{title}运行中…")

        def worker() -> None:
            try:
                result = work()
                window.after(0, lambda: (show({"ok": True, **result}), status.config(text=f"{title}完成")))
            except Exception as exc:
                payload = getattr(exc, "to_dict", lambda: {"ok": False, "error": str(exc)})()
                window.after(0, lambda: (show(payload), status.config(text=f"{title}失败")))

        threading.Thread(target=worker, daemon=True).start()

    def validate() -> dict[str, Any]:
        bundle = load_spec_bundle(root_path)
        return {"nodes": len(bundle.nodes), "shards": len(bundle.shards), "quota_versions": [v["version"] for v in bundle.quota_versions]}

    def plan() -> dict[str, Any]:
        with Ledger(db_path) as ledger:
            engine = DiagnosticEngine(root_path, ledger)
            readiness = engine.evaluate_readiness()
            return {
                "topology": list(engine.topology.order),
                "readiness": readiness,
                "decisions": engine.candidate_decisions(readiness),
                "water": engine.current_water(),
            }

    def run_schedule() -> dict[str, Any]:
        bundle = load_spec_bundle(root_path)
        with Ledger(db_path) as ledger:
            recovery = ledger.recover_interrupted()
            engine = DiagnosticEngine(root_path, ledger, bundle)
            cluster = NodeCluster(root_path, bundle.nodes)
            cluster.start_all()
            try:
                report = engine.run_all(cluster, batch_size=4, max_seconds=60)
                snapshot, replaced = load_or_replace_snapshot(root_path / "ledger" / "session_snapshot.json", ledger)
                return {"recovery": recovery, "report": report, "snapshot_replaced": replaced, "snapshot_reason": snapshot.get("reason")}
            finally:
                cluster.stop_all()

    def replay() -> dict[str, Any]:
        with Ledger(db_path) as ledger:
            result = ledger.replay_and_verify()
            snapshot, replaced = load_or_replace_snapshot(root_path / "ledger" / "session_snapshot.json", ledger)
            return {"replay": result, "snapshot_replaced": replaced}

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(0, 8))
    ttk.Button(buttons, text="强校验声明", command=lambda: run_async("校验", validate)).pack(side="left", padx=4)
    ttk.Button(buttons, text="就绪/水位", command=lambda: run_async("规划", plan)).pack(side="left", padx=4)
    ttk.Button(buttons, text="启动并调度", command=lambda: run_async("调度", run_schedule)).pack(side="left", padx=4)
    ttk.Button(buttons, text="恢复重放", command=lambda: run_async("重放", replay)).pack(side="left", padx=4)
    show({"ok": True, "hint": "选择上方操作；所有节点均为 127.0.0.1 JSON-line 子进程。"})
    window.mainloop()
    return 0
