from __future__ import annotations

import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tkinter as tk

from .acceptance import run_acceptance
from .errors import WorkbenchError
from .spec import dump_example_spec, load_spec


class WorkbenchGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("分片租约水位诊断工作台")
        self.root.geometry("980x680")
        self.spec_path = tk.StringVar(value=str(Path("examples") / "sample-spec.json"))
        self.workdir = tk.StringVar(value=str(Path("artifacts") / "gui-run"))

        top = ttk.Frame(root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="JSON 规范").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.spec_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )
        ttk.Button(top, text="选择", command=self.choose_spec).pack(side=tk.LEFT)
        ttk.Button(top, text="初始化示例", command=self.init_example).pack(side=tk.LEFT, padx=4)

        actions = ttk.Frame(root, padding=(8, 0))
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="强校验", command=self.validate_spec).pack(side=tk.LEFT)
        ttk.Button(actions, text="运行调度", command=self.run_spec).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="一键验收", command=self.run_acceptance).pack(side=tk.LEFT)
        ttk.Label(actions, text="工作目录").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(actions, textvariable=self.workdir, width=28).pack(side=tk.LEFT)

        self.status = ttk.Label(root, text="就绪：所有节点仅监听 127.0.0.1", padding=8)
        self.status.pack(fill=tk.X)
        self.output = tk.Text(root, wrap=tk.NONE, font=("Noto Sans Mono CJK SC", 10))
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def choose_spec(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.spec_path.set(path)

    def write(self, value: object) -> None:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        self.output.insert(tk.END, value + "\n")
        self.output.see(tk.END)

    def set_status(self, text: str) -> None:
        self.status.config(text=text)

    def init_example(self) -> None:
        path = Path(self.spec_path.get())
        dump_example_spec(path)
        self.write({"initialized": str(path)})
        self.validate_spec()

    def validate_spec(self) -> None:
        try:
            spec = load_spec(self.spec_path.get())
        except WorkbenchError as exc:
            self.write(exc.to_dict())
            messagebox.showerror(exc.code, f"{exc.field}: {exc.location}")
            return
        self.write(
            {
                "ok": True,
                "spec_digest": spec.digest,
                "topo_order": list(spec.topo_order),
            }
        )
        self.set_status("规范有效")

    def _run_in_thread(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def run_spec(self) -> None:
        def task() -> None:
            try:
                from .engine import WorkbenchEngine

                spec = load_spec(self.spec_path.get())
                engine = WorkbenchEngine(spec, self.workdir.get(), parallel=True, max_workers=3)
                result = engine.run_until_complete()
                engine.close()
                self.root.after(0, lambda: self.write({"ok": True, **result}))
                self.root.after(0, lambda: self.set_status("调度完成"))
            except Exception as exc:  # noqa: BLE001 - GUI diagnostic surface
                payload = getattr(exc, "to_dict", lambda: {"code": "E_GUI", "message": str(exc)})()
                self.root.after(0, lambda: self.write(payload))
                self.root.after(0, lambda: self.set_status(payload.get("code", "E_GUI")))

        self.set_status("调度运行中…")
        self._run_in_thread(task)

    def run_acceptance(self) -> None:
        def task() -> None:
            report = run_acceptance(Path("artifacts") / "gui-acceptance")
            self.root.after(0, lambda: self.write(report))
            self.root.after(
                0,
                lambda: self.set_status("验收通过" if report["ok"] else "验收失败，见报告"),
            )

        self.set_status("验收运行中：包含真实 SIGKILL 与子进程重启…")
        self._run_in_thread(task)


def main() -> int:
    root = tk.Tk()
    WorkbenchGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
