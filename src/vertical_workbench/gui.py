from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Any


def _bootstrap_tkinter():
    try:
        import tkinter as tk
        from tkinter import ttk

        return tk, ttk
    except ImportError:
        local_lib = Path.home() / ".local" / "tk-local" / "usr" / "lib" / "x86_64-linux-gnu"
        local_tcl = Path.home() / ".local" / "tk-local" / "usr" / "share" / "tcltk"
        if local_lib.exists():
            for library in ("libXss.so.1", "libXft.so.2", "libtcl8.6.so", "libtk8.6.so"):
                path = local_lib / library
                if path.exists():
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            os.environ["TCL_LIBRARY"] = str(local_tcl / "tcl8.6")
            os.environ["TK_LIBRARY"] = str(local_tcl / "tk8.6")
        import tkinter as tk
        from tkinter import ttk

        return tk, ttk


tk, ttk = _bootstrap_tkinter()

from .config import ALLOWED_FRAC_BITS, ALLOWED_QUANT_GEARS, WorkbenchConfig
from .injection import run_injection_suite
from .studio import Studio


class WorkbenchGUI:
    def __init__(self, root: "tk.Misc", workspace_root: str | Path | None = None):
        self.root = root
        self.workspace_root = Path(workspace_root or Path.cwd())
        if not (self.workspace_root / "configs" / "workbench.json").exists():
            Studio.create(self.workspace_root)
        self.studio = Studio(self.workspace_root)
        if self.studio.engine.last_result is None:
            self.studio.global_version = max(1, self.studio.global_version)
            self.studio.engine.run(1)
        self.busy = False
        self.photo: Any = None
        self.image_var = tk.StringVar(value="")
        self.window_size_var = tk.StringVar()
        self.stride_var = tk.StringVar()
        self.quant_var = tk.StringVar()
        self.frac_var = tk.StringVar()
        self.workers_var = tk.StringVar()
        self.show_windows_var = tk.BooleanVar(value=False)
        self.show_candidates_var = tk.BooleanVar(value=True)
        self.show_components_var = tk.BooleanVar(value=True)
        self.show_truth_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="ready")

        root.title("竖排古籍关键点定位诊断工作台")
        root.geometry("1440x920")
        self._build_controls()
        self._build_body()
        self.refresh_all()

    @property
    def result(self) -> dict[str, Any]:
        return self.studio.engine.last_result or {}

    def _build_controls(self) -> None:
        config = self.studio.current_config()
        self.window_size_var.set(str(config.window_size))
        self.stride_var.set(str(config.stride))
        self.quant_var.set(config.quant_gear)
        self.frac_var.set(str(config.fixed_frac_bits))
        self.workers_var.set(str(config.max_workers))

        frame = ttk.LabelFrame(self.root, text="编辑与增量重算")
        frame.pack(fill="x", padx=8, pady=6)
        specs = [
            ("窗尺寸", self.window_size_var, 6),
            ("步长", self.stride_var, 6),
            ("量化档", self.quant_var, 10),
            ("小数位", self.frac_var, 6),
            ("并发", self.workers_var, 5),
        ]
        for index, (label, variable, width) in enumerate(specs):
            ttk.Label(frame, text=label).grid(row=0, column=index * 2, padx=4, pady=6, sticky="e")
            if label == "量化档":
                widget = ttk.Combobox(frame, textvariable=variable, values=list(ALLOWED_QUANT_GEARS), width=width)
            elif label == "小数位":
                widget = ttk.Combobox(frame, textvariable=variable, values=list(ALLOWED_FRAC_BITS), width=width)
            else:
                widget = ttk.Entry(frame, textvariable=variable, width=width)
            widget.grid(row=0, column=index * 2 + 1, padx=2)

        ttk.Button(frame, text="应用参数并重算", command=self.apply_parameters).grid(row=0, column=10, padx=6)
        ttk.Button(frame, text="标注改名(摘要不变)", command=self.relabel_annotation).grid(row=0, column=11, padx=4)
        ttk.Button(frame, text="替换标注内容", command=self.replace_annotation_content).grid(row=0, column=12, padx=4)
        ttk.Button(frame, text="注入演练", command=self.run_injections).grid(row=0, column=13, padx=4)
        ttk.Button(frame, text="刷新", command=lambda: self.run_command({"op": "run"}, mutate=False)).grid(row=0, column=14, padx=4)

        ttk.Label(frame, text="图像").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        self.image_combo = ttk.Combobox(frame, textvariable=self.image_var, width=18)
        self.image_combo.grid(row=1, column=1, padx=4)
        self.image_combo.bind("<<ComboboxSelected>>", lambda _e: self.draw_canvas())
        for col, (text, variable) in enumerate(
            [
                ("全部窗", self.show_windows_var),
                ("候选窗", self.show_candidates_var),
                ("连通域", self.show_components_var),
                ("真值", self.show_truth_var),
            ],
            start=2,
        ):
            ttk.Checkbutton(frame, text=text, variable=variable, command=self.draw_canvas).grid(
                row=1, column=col, padx=5
            )
        ttk.Label(frame, textvariable=self.status_var, foreground="#555").grid(
            row=1, column=10, columnspan=5, sticky="w", padx=8
        )

    def _build_body(self) -> None:
        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        body.add(left, weight=3)
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.metrics_var, justify="left", font=("Noto Sans CJK SC", 11, "bold")).pack(
            anchor="w", pady=4
        )
        canvas_holder = ttk.Frame(left, relief="sunken")
        canvas_holder.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_holder, width=540, height=780, background="#202020")
        yscroll = ttk.Scrollbar(canvas_holder, orient="vertical", command=self.canvas.yview)
        xscroll = ttk.Scrollbar(canvas_holder, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        canvas_holder.rowconfigure(0, weight=1)
        canvas_holder.columnconfigure(0, weight=1)

        right = ttk.Notebook(body)
        body.add(right, weight=2)
        self.lease_tree = self._make_tree(right, "租约", ["target", "gen", "issued", "expires", "remain_ms", "points"])
        self.conflict_tree = self._make_tree(right, "冲突", ["id", "type", "pair", "truth_orders", "reason"])
        self.reuse_text = self._make_text_tab(right, "失效/复用")
        self.queue_tree = self._make_tree(right, "待办/业务码", ["kind", "target", "detail"])

    def _make_tree(self, notebook: ttk.Notebook, title: str, columns: list[str]) -> ttk.Treeview:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150 if col in ("target", "reason") else 90, anchor="w")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return tree

    def _make_text_tab(self, notebook: ttk.Notebook, title: str) -> tk.Text:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        text = tk.Text(frame, wrap="none", height=24)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        return text

    def run_command(self, command: dict[str, Any], mutate: bool = True) -> None:
        if self.busy:
            return
        self.busy = True
        self.status_var.set("计算中…")

        def worker() -> None:
            try:
                if command.get("op") == "run":
                    result = self.studio.engine.run(self.studio.global_version)
                else:
                    result = self.studio.apply(command)
                self.root.after(0, lambda: self._after_command(result))
            except Exception as exc:  # GUI must survive a bad edit.
                self.root.after(0, lambda: self._command_failed(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _after_command(self, result: dict[str, Any] | None) -> None:
        self.busy = False
        self.status_var.set("完成；快照已整体替换")
        self.refresh_all()

    def _command_failed(self, exc: Exception) -> None:
        self.busy = False
        self.status_var.set(f"命令失败: {exc}")

    def apply_parameters(self) -> None:
        try:
            patch = {
                "window_size": int(self.window_size_var.get()),
                "stride": int(self.stride_var.get()),
                "quant_gear": self.quant_var.get(),
                "fixed_frac_bits": int(self.frac_var.get()),
                "max_workers": max(1, min(16, int(self.workers_var.get()))),
            }
        except ValueError as exc:
            self.status_var.set(f"参数无效: {exc}")
            return
        self.run_command({"op": "update_config", "patch": patch})

    def relabel_annotation(self) -> None:
        version = f"kv-relabel-{self.studio.global_version + 1:04d}"
        self.run_command({"op": "replace_annotation_version", "version_id": version, "jitter": 0})

    def replace_annotation_content(self) -> None:
        image_id = self.image_var.get() or "sample_001"
        version = f"kv-content-{self.studio.global_version + 1:04d}"
        self.run_command(
            {
                "op": "replace_annotation_version",
                "version_id": version,
                "jitter_by_image": {image_id: 1},
            }
        )

    def run_injections(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.status_var.set("运行注入隔离演练…")

        def worker() -> None:
            report = run_injection_suite(self.workspace_root)
            self.root.after(0, lambda: self._injection_done(report))

        threading.Thread(target=worker, daemon=True).start()

    def _injection_done(self, report: dict[str, Any]) -> None:
        self.busy = False
        self.status_var.set(
            f"注入命中 {sum(x['hit'] for x in report['injection_results'])}/4；正常样本隔离={report['good_samples_isolated']}"
        )
        self.refresh_all()

    def refresh_all(self) -> None:
        result = self.result
        image_ids = [sample["image_id"] for sample in result.get("samples", [])]
        if not image_ids:
            image_ids = sorted(path.stem for path in (self.workspace_root / "samples").glob("*.png"))
        self.image_combo["values"] = image_ids
        if not self.image_var.get() and image_ids:
            self.image_var.set(image_ids[0])
        self._refresh_metrics()
        self._refresh_leases()
        self._refresh_conflicts()
        self._refresh_reuse()
        self._refresh_queue_codes()
        self.draw_canvas()

    def _refresh_metrics(self) -> None:
        result = self.result
        counts = result.get("business_code_counts", {})
        conflict_counts = {"跨栏": 0, "错序": 0, "缺号": 0}
        for conflict in result.get("conflicts", []):
            conflict_counts[conflict["type"]] = conflict_counts.get(conflict["type"], 0) + 1
        self.metrics_var.set(
            "\n".join(
                [
                    f"全局版本 {result.get('global_version')}  配置哈希 {str(result.get('config_hash'))[:12]}",
                    f"已完成图数 {len(result.get('samples', []))}  关键点数 {len(result.get('keypoints', []))}",
                    f"冲突分布  跨栏={conflict_counts['跨栏']}  错序={conflict_counts['错序']}  缺号={conflict_counts['缺号']}",
                    f"候选窗 {sum(1 for w in result.get('windows', []) if w.get('candidate'))} / 总窗 {len(result.get('windows', []))}  连通域 {len(result.get('components', []))}",
                    f"待办 {len(result.get('pending_queue', []))}  失效记录 {len(result.get('invalidations', []))}  复用命中 {len(result.get('reuse', []))}  重算 {len(result.get('recompute', []))}",
                    "业务码 " + "  ".join(f"{key}={value}" for key, value in sorted(counts.items()) if value),
                ]
            )
        )

    def _refresh_leases(self) -> None:
        self._clear_tree(self.lease_tree)
        now = self.studio.clock.now_ms()
        for lease in self.result.get("lease_snapshot", []):
            self.lease_tree.insert(
                "",
                "end",
                values=(
                    lease["target"],
                    lease["generation"],
                    lease["issued_ms"],
                    lease["expires_ms"],
                    max(0, lease["expires_ms"] - now),
                    lease["reserved_points"],
                ),
            )

    def _refresh_conflicts(self) -> None:
        self._clear_tree(self.conflict_tree)
        for conflict in self.result.get("conflicts", [])[:500]:
            self.conflict_tree.insert(
                "",
                "end",
                values=(
                    conflict["conflict_id"],
                    conflict["type"],
                    conflict.get("pair_index"),
                    ",".join(str(x) for x in conflict.get("truth_reading_orders", [])),
                    "; ".join(conflict.get("reasons", []))[:180],
                ),
            )

    def _refresh_reuse(self) -> None:
        text = self.reuse_text
        text.delete("1.0", "end")
        text.insert("end", "== 版本事件 ==\n")
        for event in self.result.get("version_events", []):
            text.insert("end", f"{event.get('kind')}: {event.get('image_id') or '*'} -> {event.get('decision')}\n")
        text.insert("end", "\n== 失效依据（前 200 条）==\n")
        for item in self.result.get("invalidations", [])[:200]:
            text.insert("end", f"{item}\n")
        text.insert("end", "\n== 复用命中（前 300 条）==\n")
        for item in self.result.get("reuse", [])[:300]:
            text.insert("end", f"REUSE {item['stage']} {item['target']} {item['cache_key'][:16]} because {item['reason']}\n")
        text.insert("end", "\n== 重算清单（前 200 条）==\n")
        for item in self.result.get("recompute", [])[:200]:
            text.insert("end", f"RECOMPUTE {item['stage']} {item['target']} {item['cache_key'][:16]}\n")

    def _refresh_queue_codes(self) -> None:
        tree = self.queue_tree
        self._clear_tree(tree)
        for item in self.result.get("pending_queue", []):
            tree.insert("", "end", values=("PENDING", item["unit_id"], item["reason"]))
        for code, count in sorted(self.result.get("business_code_counts", {}).items()):
            if count:
                tree.insert("", "end", values=("CODE", code, f"count={count}"))
        budget = self.result.get("budget", {})
        for key in ("consumed_points", "remaining_points", "comparisons_used_this_run", "reserved_lease_points"):
            if key in budget:
                tree.insert("", "end", values=("BUDGET", key, str(budget[key])))

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def draw_canvas(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        image_id = self.image_var.get()
        if not image_id:
            return
        png_path = self.workspace_root / "samples" / f"{image_id}.png"
        if png_path.exists():
            self.photo = tk.PhotoImage(file=str(png_path))
            canvas.create_image(0, 0, image=self.photo, anchor="nw")
            canvas.configure(scrollregion=(0, 0, 512, 768))

        result = self.result
        if self.show_windows_var.get() or self.show_candidates_var.get():
            for window in result.get("windows", []):
                if window["image_id"] != image_id:
                    continue
                if self.show_windows_var.get():
                    color = "#7aa2f7" if not window["candidate"] else "#ff9e64"
                    canvas.create_rectangle(
                        window["x"], window["y"], window["x"] + window["w"], window["y"] + window["h"],
                        outline=color, width=1,
                    )
                elif self.show_candidates_var.get() and window["candidate"]:
                    canvas.create_rectangle(
                        window["x"], window["y"], window["x"] + window["w"], window["y"] + window["h"],
                        outline="#ff9e64", width=1,
                    )

        if self.show_components_var.get():
            for component in result.get("components", []):
                if not component.get("domain_id", "").startswith(image_id + ":"):
                    continue
                x0, y0, x1, y1 = component["bbox"]
                canvas.create_rectangle(x0, y0, x1, y1, outline="#9ece6a", width=2)

        if self.show_truth_var.get():
            truth_path = self.workspace_root / "truth" / f"{image_id}.json"
            if truth_path.exists():
                import json

                truth = json.loads(truth_path.read_text(encoding="utf-8"))
                for point in truth["keypoints"]:
                    x, y = int(point["x"]), int(point["y"])
                    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#e0af68", width=2)

        for point in result.get("keypoints", []):
            if not point.get("keypoint_id", "").startswith(image_id + ":"):
                continue
            x, y = int(point["x"]), int(point["y"])
            canvas.create_line(x - 7, y, x + 7, y, fill="#f7768e", width=2)
            canvas.create_line(x, y - 7, x, y + 7, fill="#f7768e", width=2)
            canvas.create_text(x + 10, y - 10, text=str(point.get("reading_order", "?")), fill="#c0caf5")


def main() -> int:
    tk_module, _ttk = _bootstrap_tkinter()
    root = tk_module.Tk()
    WorkbenchGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
