"""Tkinter 界面入口：清单编辑、下发观察、权威重算与恢复。

单窗布局：左侧清单编辑器，右侧节点回执/进度配额/延迟统计/日志。
下发在后台线程执行，界面经 after() 轮询保持可交互。
运行：python -m configdist.gui --workdir demo_ws
"""

import argparse
import json
import queue
import sys
import threading


def main(argv=None):
    parser = argparse.ArgumentParser(prog="configdist.gui")
    parser.add_argument("--workdir", default="demo_ws")
    args = parser.parse_args(argv)
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter 不可用：请通过官方安装包勾选 Tcl/Tk 组件后重试。",
              file=sys.stderr)
        return 2

    from .errors import PlatformError
    from .platform import Platform

    platform = Platform(args.workdir)
    events = queue.Queue()

    root = tk.Tk()
    root.title("configdist 配置分发平台")
    root.geometry("1100x700")

    toolbar = ttk.Frame(root)
    toolbar.pack(side=tk.TOP, fill=tk.X)
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)

    left = ttk.Frame(paned)
    right = ttk.Frame(paned)
    paned.add(left, weight=1)
    paned.add(right, weight=2)

    ttk.Label(left, text="清单编辑（JSON，保存后版本单调递增）").pack(anchor=tk.W)
    editor = tk.Text(left, wrap=tk.NONE, width=52)
    editor.pack(fill=tk.BOTH, expand=True)

    notebook = ttk.Notebook(right)
    notebook.pack(fill=tk.BOTH, expand=True)

    receipts_frame = ttk.Frame(notebook)
    notebook.add(receipts_frame, text="节点回执")
    tree = ttk.Treeview(receipts_frame,
                        columns=("node", "key", "digest", "stale"),
                        show="headings")
    for col in ("node", "key", "digest", "stale"):
        tree.heading(col, text=col)
        tree.column(col, width=140)
    tree.pack(fill=tk.BOTH, expand=True)

    progress_frame = ttk.Frame(notebook)
    notebook.add(progress_frame, text="进度与配额")
    progress_var = tk.StringVar(value="尚未下发")
    ttk.Label(progress_frame, textvariable=progress_var,
              justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=8)
    quota_var = tk.StringVar(value="")
    ttk.Label(progress_frame, textvariable=quota_var,
              justify=tk.LEFT).pack(anchor=tk.W, padx=8)

    latency_frame = ttk.Frame(notebook)
    notebook.add(latency_frame, text="延迟统计")
    latency_var = tk.StringVar(value="暂无样本")
    ttk.Label(latency_frame, textvariable=latency_var,
              justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=8)

    log_frame = ttk.Frame(notebook)
    notebook.add(log_frame, text="日志")
    log_text = tk.Text(log_frame, height=12, state=tk.DISABLED)
    log_text.pack(fill=tk.BOTH, expand=True)

    def log(msg):
        log_text.configure(state=tk.NORMAL)
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
        log_text.configure(state=tk.DISABLED)

    def background(fn):
        def runner():
            try:
                result = fn()
                events.put(("ok", result))
            except PlatformError as exc:
                events.put(("error", exc.to_dict()))
            except Exception as exc:  # 界面兜底
                events.put(("error", {"code": "E_UNKNOWN",
                                      "message": str(exc)}))
        threading.Thread(target=runner, daemon=True).start()

    def do_gen_samples():
        background(lambda: ("样本已生成", platform.gen_samples()))

    def do_normalize():
        background(lambda: ("归一报告", platform.normalize()))

    def do_open():
        def work():
            recovery = platform.open()
            platform.start_nodes()
            return ("平台已打开", recovery)
        background(work)

    def do_load():
        name = "app_v3.json"
        manifest = platform.load_manifest(name)
        from .manifest import encode_manifest
        editor.delete("1.0", tk.END)
        editor.insert("1.0", encode_manifest(manifest))
        log(f"已加载 {name} (mv={manifest.manifest_version})")

    def do_dispatch():
        def work():
            text = editor.get("1.0", tk.END)
            platform.edit_manifest(platform.current_manifest_name
                                   or "app_v3.json", text)
            return ("下发报告", platform.dispatch(platform.current_manifest))
        background(work)

    def do_recompute():
        background(lambda: ("权威重算", platform.recompute_authoritative()))

    def do_recover():
        background(lambda: ("恢复报告", platform.recover()))

    for label, fn in (("生成样本", do_gen_samples), ("归一目录", do_normalize),
                      ("打开平台", do_open), ("加载清单", do_load),
                      ("保存并下发", do_dispatch), ("权威重算", do_recompute),
                      ("恢复", do_recover)):
        ttk.Button(toolbar, text=label, command=fn).pack(side=tk.LEFT, padx=2)

    def refresh_view():
        try:
            view = platform.ui_query()
        except Exception:
            view = None
        if view:
            tree.delete(*tree.get_children())
            for node, key, digest in view["receipts"]:
                tree.insert("", tk.END,
                            values=(node, key, digest[:16] + "…", False))
            prog = view["progress"]
            if prog:
                progress_var.set(
                    f"版本={view['version']} 进度={prog['done']}/"
                    f"{prog['total']} 运行中={prog['in_flight_global']} "
                    f"排队={prog['queued']}")
                quota_var.set(
                    "每节点运行中: "
                    + json.dumps(prog["in_flight_node"], sort_keys=True)
                    + "  峰值: "
                    + json.dumps(prog["node_peak"], sort_keys=True))
            report = platform.latency_report()
            lines = []
            for kind, stats in report["stats"].items():
                lines.append(
                    f"{kind}: n={stats['count']} "
                    f"mean={stats['mean_milli_ms'] / 1000:.3f}ms "
                    f"p95={stats['p95_ms']}ms max={stats['max_ms']}ms "
                    f"量化上界=±{stats['quant_bound_milli_ms'] / 1000:.3f}ms")
            for event in report["events"]:
                lines.append(f"E_LATENCY: {event}")
            latency_var.set("\n".join(lines) or "暂无样本")
        root.after(150, refresh_view)

    def drain_events():
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == "ok" and isinstance(payload, tuple):
                title, body = payload
                log(f"{title}: "
                    f"{json.dumps(body, ensure_ascii=False, default=str)[:800]}")
            else:
                log(f"错误: {json.dumps(payload, ensure_ascii=False)}")
        root.after(100, drain_events)

    def on_close():
        try:
            platform.close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(150, refresh_view)
    root.after(100, drain_events)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
