from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .cli import execute_demo
from .config_paths import ROOT
from .errors import ObserverError
from .spec import load_session


def launch() -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # Tk is an optional system-python dependency here.
        raise SystemExit(
            "Tkinter is not available in this interpreter: "
            f"{type(exc).__name__}: {exc}. Install/repair system tk and libtk8.6, "
            "or use the documented CLI commands."
        ) from exc

    root = tk.Tk()
    root.title("TCP Observer — deterministic local state ledger")
    root.geometry("1100x760")
    path_value = tk.StringVar(value=str(ROOT / "configs/session.json"))

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="会话配置").pack(side="left")
    ttk.Entry(top, textvariable=path_value, width=90).pack(side="left", padx=8)

    editor = ttk.PanedWindow(root, orient="horizontal")
    editor.pack(fill="both", expand=True, padx=8, pady=4)
    config_frame = ttk.Labelframe(editor, text="session JSON（缩放因子/封存长度/上限）")
    script_frame = ttk.Labelframe(editor, text="peer script JSON（条目）")
    editor.add(config_frame, weight=1)
    editor.add(script_frame, weight=1)

    config_text = tk.Text(config_frame, wrap="none", undo=True)
    script_text = tk.Text(script_frame, wrap="none", undo=True)
    config_text.pack(fill="both", expand=True)
    script_text.pack(fill="both", expand=True)

    output_frame = ttk.Labelframe(root, text="报告")
    output_frame.pack(fill="both", expand=False, padx=8, pady=4)
    output_text = tk.Text(output_frame, height=14, wrap="word")
    output_text.pack(fill="both", expand=True)

    def set_output(value: Any) -> None:
        output_text.delete("1.0", "end")
        output_text.insert("1.0", value if isinstance(value, str) else canonical_text(value))

    def current_path() -> Path:
        return Path(path_value.get()).expanduser().resolve()

    def load_files() -> None:
        path = current_path()
        config = json.loads(path.read_text(encoding="utf-8"))
        config_text.delete("1.0", "end")
        config_text.insert("1.0", json.dumps(config, ensure_ascii=False, indent=2))
        script_file = config.get("script_file")
        if script_file:
            script_path = (path.parent / script_file).resolve()
            script = json.loads(script_path.read_text(encoding="utf-8"))
            script_text.delete("1.0", "end")
            script_text.insert("1.0", json.dumps(script, ensure_ascii=False, indent=2))
        set_output(f"loaded {path}")

    def save_files() -> Path | None:
        path = current_path()
        config = json.loads(config_text.get("1.0", "end"))
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        script_file = config.get("script_file")
        if script_file:
            script_path = (path.parent / script_file).resolve()
            script = json.loads(script_text.get("1.0", "end"))
            script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        set_output(f"saved {path}")
        return path

    def validate() -> None:
        try:
            path = save_files()
            spec = load_session(path or current_path())
            set_output({"ok": True, "script_digest": spec.script_digest, "scale": spec.scale})
        except (ObserverError, json.JSONDecodeError, OSError) as exc:
            if isinstance(exc, ObserverError):
                set_output(exc.to_dict())
            else:
                set_output(f"{type(exc).__name__}: {exc}")
            messagebox.showerror("校验失败", str(exc))

    def run_demo() -> None:
        try:
            path = save_files()
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return

        def worker() -> None:
            try:
                report = execute_demo(ROOT / "artifacts/gui-run", path)
                root.after(0, lambda: set_output(report))
            except Exception as exc:
                payload = exc.to_dict() if isinstance(exc, ObserverError) else {"error": type(exc).__name__, "message": str(exc)}
                root.after(0, lambda: set_output(payload))

        threading.Thread(target=worker, daemon=True).start()
        set_output("running deterministic demo...")

    buttons = ttk.Frame(root, padding=8)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="重新加载", command=load_files).pack(side="left")
    ttk.Button(buttons, text="保存", command=save_files).pack(side="left", padx=6)
    ttk.Button(buttons, text="保存并强校验", command=validate).pack(side="left")
    ttk.Button(buttons, text="保存并运行全部验收", command=run_demo).pack(side="left", padx=6)
    ttk.Button(buttons, text="退出", command=root.destroy).pack(side="right")

    load_files()
    root.mainloop()
