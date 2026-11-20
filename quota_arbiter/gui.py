"""tkinter 领域可视化与诊断工作台。

tkinter 采用延迟导入：在缺少 Tk 的环境（如最小化 Linux）下，
import 本模块不会失败，仅在真正启动 GUI 时给出可操作提示。
Windows 官方 Python 安装默认包含 tkinter。
"""
from __future__ import annotations

import json
import os
from .generator import generate_dataset
from .ledger import Ledger
from .loader import load_requests, load_spec
from .paths import run_path
from .runner import (
    choose_and_describe,
    persist_run,
    recover_and_rerun,
    run_all_paths,
    spec_digest,
)
from .snapshot import restore_snapshot, verify_replay
from .util import StructuredLogger, canonical_json


def launch():
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as exc:  # pragma: no cover - 环境相关
        raise SystemExit(
            "当前 Python 环境缺少 tkinter。Windows 请使用 python.org 官方安装包并勾选 "
            "tcl/tk；Debian/Ubuntu 需安装 python3-tk。核心仲裁与 CLI 不依赖 tkinter，"
            f"可继续使用 run_arbiter.py。原始错误: {exc}"
        )

    app = ArbiterApp(tk, ttk, messagebox)
    app.mainloop()


class ArbiterApp:
    def __init__(self, tk, ttk, messagebox):
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("数据库租户资源配额仲裁工作台")
        self.root.geometry("1080x720")
        self.workdir = tk.StringVar(value=os.path.abspath("data"))
        self._build()

    def mainloop(self):
        self.root.mainloop()

    def _build(self):
        top = self.ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        self.ttk.Label(top, text="工作目录").pack(side="left")
        self.ttk.Entry(top, textvariable=self.workdir, width=70).pack(side="left", padx=6)
        self.ttk.Button(top, text="生成样本", command=self.on_generate).pack(side="left", padx=4)
        self.ttk.Button(top, text="运行默认路径", command=self.on_run).pack(side="left", padx=4)
        self.ttk.Button(top, text="三路径一致性", command=self.on_consistency).pack(side="left", padx=4)
        self.ttk.Button(top, text="强制终止恢复", command=self.on_crash).pack(side="left", padx=4)
        self.ttk.Button(top, text="快照恢复重放", command=self.on_restore).pack(side="left", padx=4)

        self.notebook = self.ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.texts = {}
        for name in ("路径选择", "结局账本", "排队轨迹", "恢复诊断", "异常注入"):
            frame = self.ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            text = self.tk.Text(frame, wrap="none")
            yscroll = self.ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            xscroll = self.ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
            text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
            text.grid(row=0, column=0, sticky="nsew")
            yscroll.grid(row=0, column=1, sticky="ns")
            xscroll.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            self.texts[name] = text

        fault_bar = self.ttk.Frame(self.root, padding=6)
        fault_bar.pack(fill="x")
        for kind in ("order", "priority", "ledger"):
            self.ttk.Button(
                fault_bar, text=f"注入 {kind}", command=lambda k=kind: self.on_inject(k)
            ).pack(side="left", padx=4)
        self._write("路径选择", "就绪：先生成样本，再运行仲裁与诊断。\n")

    def _write(self, tab: str, content: str):
        text = self.texts[tab]
        text.delete("1.0", "end")
        text.insert("1.0", content)

    def _paths(self):
        workdir = self.workdir.get()
        return {
            "spec": os.path.join(workdir, "spec.json"),
            "requests": os.path.join(workdir, "requests.ndjson"),
            "ledger": os.path.join(workdir, "ledger.sqlite3"),
            "snapshot": os.path.join(workdir, "snapshot.json"),
            "log": os.path.join(workdir, "events.jsonl"),
        }

    def _load(self):
        paths = self._paths()
        tenants, pools = load_spec(paths["spec"])
        requests = load_requests(paths["requests"], tenants, pools)
        return paths, tenants, pools, requests

    def on_generate(self):
        info = generate_dataset(self.workdir.get())
        self._write("路径选择", json.dumps(info, ensure_ascii=False, indent=2))

    def on_run(self):
        try:
            paths, tenants, pools, requests = self._load()
            selection = choose_and_describe(tenants, pools, requests)
            digest = spec_digest(tenants, pools, requests)
            logger = StructuredLogger(paths["log"])
            if os.path.exists(paths["ledger"]):
                os.remove(paths["ledger"])
            ledger = Ledger(paths["ledger"])
            ledger.initialize_session(tenants, pools, selection["selected"], digest)
            summary = persist_run(tenants, pools, requests, selection["selected"], ledger, logger=logger)
            ledger.verify_digest_chain()
            ledger.close()
            result = run_path(selection["selected"], tenants, pools, requests)
            from .snapshot import write_snapshot

            write_snapshot(paths["snapshot"], tenants, pools, selection, result, result.last_rid)
            self._write(
                "路径选择",
                json.dumps({"selection": selection, "persistence": summary},
                           ensure_ascii=False, indent=2),
            )
            lines = [canonical_json(o.canonical_tuple()) for o in result.outcomes[:200]]
            self._write("结局账本", "\n".join(lines) + f"\n\n共 {len(result.outcomes)} 条结局")
            trace = [
                canonical_json([e.rid, e.kind, e.at_ms, e.wait_ms, e.basis, e.detail])
                for e in result.queue_events[:200]
            ]
            self._write("排队轨迹", "\n".join(trace) + f"\n\n共 {len(result.queue_events)} 条轨迹")
        except Exception as exc:  # pragma: no cover - GUI 兜底
            self.messagebox.showerror("运行失败", str(exc))

    def on_consistency(self):
        try:
            _, tenants, pools, requests = self._load()
            results, report = run_all_paths(tenants, pools, requests)
            self._write("路径选择", json.dumps(report, ensure_ascii=False, indent=2))
        except Exception as exc:  # pragma: no cover
            self.messagebox.showerror("一致性校验失败", str(exc))

    def on_crash(self):
        try:
            paths, tenants, pools, requests = self._load()
            selection = choose_and_describe(tenants, pools, requests)
            digest = spec_digest(tenants, pools, requests)
            logger = StructuredLogger(paths["log"])
            if os.path.exists(paths["ledger"]):
                os.remove(paths["ledger"])
            ledger = Ledger(paths["ledger"])
            ledger.initialize_session(tenants, pools, selection["selected"], digest)
            try:
                persist_run(
                    tenants, pools, requests, selection["selected"], ledger,
                    crash_after_batch=2, logger=logger,
                )
            except Exception:
                ledger.close()
            point = recover_and_rerun(
                tenants, pools, requests, Ledger(paths["ledger"]), digest,
                selection["selected"], logger=logger,
            )
            safe = {k: v for k, v in point.items() if k not in ("tenant_state", "pool_state")}
            self._write("恢复诊断", json.dumps(safe, ensure_ascii=False, indent=2))
        except Exception as exc:  # pragma: no cover
            self.messagebox.showerror("恢复失败", str(exc))

    def on_restore(self):
        try:
            paths, tenants, pools, requests = self._load()
            payload, mismatch, backup = restore_snapshot(paths["snapshot"], tenants, pools)
            replay = run_path(payload["default_path"], tenants, pools, requests)
            verify_replay(payload, replay, payload["processed_cursor"])
            self._write(
                "恢复诊断",
                json.dumps(
                    {
                        "mismatch_replaced": mismatch,
                        "readonly_backup": backup,
                        "default_path": payload["default_path"],
                        "replay_verified": True,
                        "cursor": payload["processed_cursor"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception as exc:  # pragma: no cover
            self.messagebox.showerror("快照恢复失败", str(exc))

    def on_inject(self, kind: str):
        from .generator import (
            inject_ledger_digest,
            inject_missing_priority_tenant,
            inject_order_regression,
        )

        paths = self._paths()
        try:
            if kind == "order":
                dst = paths["requests"] + ".fault"
                meta = inject_order_regression(paths["requests"], dst)
                tenants, pools = load_spec(paths["spec"])
                load_requests(dst, tenants, pools, check_global_order=True)
            elif kind == "priority":
                dst = paths["spec"] + ".fault"
                meta = inject_missing_priority_tenant(paths["spec"], dst)
                load_spec(dst)
            else:
                tenants, pools, requests = self._load()
                ledger = Ledger(paths["ledger"])
                from .paths import canonical_requests

                rid = canonical_requests(requests)[0].rid
                meta = inject_ledger_digest(ledger, rid)
                ledger.close()
                Ledger(paths["ledger"]).verify_digest_chain()
            self._write("异常注入", f"注入 {kind} 未触发异常：{meta}")
        except Exception as exc:
            if hasattr(exc, "to_dict"):
                payload = exc.to_dict()
            else:
                payload = {"code": "E_UNKNOWN", "message": str(exc)}
            StructuredLogger(paths["log"]).error("fault_injected", injection=kind, **payload)
            self._write("异常注入", json.dumps({"injection": kind, "error": payload},
                                               ensure_ascii=False, indent=2))
