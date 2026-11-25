from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def launch(root_dir: str | Path | None = None) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": {
                "code": "E_TKINTER_UNAVAILABLE",
                "message": f"{type(exc).__name__}: {exc}",
                "windows_hint": "Use the official Python 3.11+ installer with tcl/tk enabled, then run python -m tkinter.",
            },
        }, ensure_ascii=False, indent=2))
        return 2

    root = Path(root_dir or Path.cwd()).resolve()
    app = tk.Tk()
    app.title("分层扰动预算诊断工作台")
    app.geometry("980x680")

    controls = ttk.Frame(app, padding=8)
    controls.pack(fill=tk.X)
    output = ScrolledText(app, font=("Consolas", 10), wrap=tk.WORD)
    output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    status = tk.StringVar(value=f"root={root}")
    ttk.Label(controls, textvariable=status).pack(side=tk.LEFT, padx=(0, 12))

    def show(payload: Any) -> None:
        output.delete("1.0", tk.END)
        output.insert(tk.END, json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        output.see("1.0")

    def generate() -> None:
        from workbench.samples import generate_workspace
        paths = generate_workspace(root)
        show({"ok": True, "generated": len(paths), "root": str(root)})
        status.set("样本与样例配置已生成")

    def validate() -> None:
        try:
            from workbench.commands import validate_command_stream
            from workbench.workflows import load_bundle
            config, rectangles, pixels, fingerprint = load_bundle(root / "configs" / "workbench.json")
            commands = validate_command_stream(root / "configs" / "edit_stream.json", config, rectangles)
            show({
                "ok": True,
                "config_hash": config.config_hash,
                "image_fingerprint": fingerprint,
                "mask_rectangles": len(rectangles),
                "levels": config.levels,
                "commands": len(commands),
            })
            status.set("校验通过")
        except Exception as exc:
            show(_error_payload(exc))
            status.set("校验失败")

    def run_demo_action() -> None:
        status.set("正在运行确定性验收，请稍候……")
        app.update_idletasks()
        try:
            from workbench.demo import run_demo
            report = run_demo(root)
            show({
                "ok": report["ok"],
                "report": "artifacts/reports/verification-report.json",
                "consistency_equal": report["consistency"]["equal"],
                "injection_consistent": report["injection_diagnostic"]["consistent"],
                "all_injections_hit": report["injection_diagnostic"]["all_injections_hit"],
                "budget_story": report["budget_story"],
            })
            status.set("验收完成")
        except Exception as exc:
            show(_error_payload(exc))
            status.set("验收失败")

    def open_report() -> None:
        path = root / "artifacts" / "reports" / "verification-report.json"
        if not path.exists():
            messagebox.showinfo("报告不存在", "请先运行验收演示。")
            return
        show(json.loads(path.read_text(encoding="utf-8")))
        status.set(str(path))

    ttk.Button(controls, text="生成样本", command=generate).pack(side=tk.LEFT, padx=3)
    ttk.Button(controls, text="校验输入", command=validate).pack(side=tk.LEFT, padx=3)
    ttk.Button(controls, text="运行验收", command=run_demo_action).pack(side=tk.LEFT, padx=3)
    ttk.Button(controls, text="查看完整报告", command=open_report).pack(side=tk.LEFT, padx=3)
    ttk.Button(controls, text="退出", command=app.destroy).pack(side=tk.RIGHT, padx=3)

    report_path = root / "artifacts" / "reports" / "verification-report.json"
    if report_path.exists():
        try:
            show(json.loads(report_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    app.mainloop()
    return 0


def _error_payload(exc: Exception) -> dict[str, Any]:
    if hasattr(exc, "to_dict"):
        return exc.to_dict()
    return {"ok": False, "error": {"code": "E_GUI_FAILED", "message": f"{type(exc).__name__}: {exc}"}}


def main() -> int:
    return launch(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
