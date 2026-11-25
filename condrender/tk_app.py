from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _enable_bundled_tk(root: Path) -> None:
    vendor_lib = root / "vendor/tk/usr/lib/x86_64-linux-gnu"
    tcl_lib = root / "vendor/tk/usr/share/tcltk/tcl8.6"
    tk_lib = root / "vendor/tk/usr/share/tcltk/tk8.6"
    if vendor_lib.exists():
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(vendor_lib) + (os.pathsep + current if current else "")
    if tcl_lib.exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_lib))
    if tk_lib.exists():
        os.environ.setdefault("TK_LIBRARY", str(tk_lib))


def launch(root: Path) -> None:
    root = Path(root).resolve()
    _enable_bundled_tk(root)
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        print("Tkinter is unavailable. Install system dependencies on WSL/Ubuntu:")
        print("  sudo apt update && sudo apt install -y python3 python3-tk python3-numpy python3-yaml")
        print(f"import error: {type(exc).__name__}: {exc}")
        raise SystemExit(2)

    from .bundle import load_bundle
    from .session import InteractiveSession

    bundle, failures = load_bundle(root)
    if failures:
        for failure in failures:
            print(failure.code, failure.message)
        raise SystemExit(2)

    session = InteractiveSession(bundle, workers=int(bundle["config"]["max_workers"]))
    session.finish()

    window = tk.Tk()
    window.title("CondFusion 定点渲染诊断台")
    window.geometry("980x680")

    prompt_var = tk.StringVar(value=bundle["frame_prompts"][0])
    frame_var = tk.IntVar(value=0)
    steps_var = tk.IntVar(value=int(bundle["config"]["denoise_steps"]))
    tiles_var = tk.IntVar(value=int(bundle["config"]["tile_rows"]))
    status_var = tk.StringVar(value="baseline render complete")

    top = ttk.Frame(window, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="帧").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(top, from_=0, to=3, textvariable=frame_var, width=5).grid(row=0, column=1)
    ttk.Label(top, text="提示词").grid(row=0, column=2, sticky="w", padx=(12, 0))
    ttk.Entry(top, textvariable=prompt_var, width=58).grid(row=0, column=3, padx=4)

    controls = ttk.Frame(window, padding=(8, 0))
    controls.pack(fill="x")
    ttk.Label(controls, text="去噪步").pack(side="left")
    ttk.Spinbox(controls, from_=1, to=8, textvariable=steps_var, width=5).pack(side="left", padx=4)
    ttk.Label(controls, text="分块行").pack(side="left", padx=(12, 0))
    ttk.Spinbox(controls, from_=1, to=8, textvariable=tiles_var, width=5).pack(side="left", padx=4)

    text = tk.Text(window, height=24, wrap="none")
    text.pack(fill="both", expand=True, padx=8, pady=8)
    status = ttk.Label(window, textvariable=status_var, padding=6)
    status.pack(fill="x")

    def show_report(report: dict[str, Any]) -> None:
        summary = report.get("summary", {})
        lines = [
            f"status={report.get('status')} code={report.get('code', 'E_OK')}",
            f"global_version={summary.get('global_version')} fresh={summary.get('fresh')}",
            f"completed={summary.get('completed_frames')} range={summary.get('affected_frame_range')}",
            f"HIT={summary.get('reused_entries')} RECOMPUTE={summary.get('recomputed_entries')} "
            f"progress={summary.get('progress')}",
            f"config_hash={summary.get('config_hash')}",
            f"condition_fp={summary.get('condition_fingerprint')}",
            "",
            "tile rows:",
        ]
        for frame, tile_info in sorted(summary.get("tile_rows", {}).items()):
            lines.append(f"  frame {frame}: order={tile_info['order']} bounds={tile_info['bounds']}")
        if report.get("failures"):
            lines.extend(["", "failures:"])
            lines.extend(f"  {item}" for item in report["failures"])
        text.delete("1.0", "end")
        text.insert("1.0", "\n".join(lines))

    def apply_prompt() -> None:
        report = session.handle({
            "id": "GUI-PROMPT", "op": "replace_prompt",
            "frame": int(frame_var.get()), "text": prompt_var.get(),
        }, budget_ms=float(bundle["config"]["interaction_budget_ms"]))
        show_report(report)
        status_var.set("prompt command handled")

    def apply_render_params() -> None:
        step_report = session.handle({
            "id": "GUI-STEPS", "op": "set_denoise_steps", "steps": int(steps_var.get())
        })
        tile_report = session.handle({
            "id": "GUI-TILES", "op": "set_tile_rows", "tile_rows": int(tiles_var.get())
        })
        show_report(tile_report if tile_report else step_report)
        status_var.set("render parameters handled")

    button_bar = ttk.Frame(window, padding=8)
    button_bar.pack(fill="x")
    ttk.Button(button_bar, text="替换提示词", command=apply_prompt).pack(side="left")
    ttk.Button(button_bar, text="应用步数/分块", command=apply_render_params).pack(side="left", padx=8)
    ttk.Button(button_bar, text="退出", command=window.destroy).pack(side="right")

    show_report({"status": "complete", "code": "E_OK", "summary": session.last_summary or {}})
    window.mainloop()


if __name__ == "__main__":
    launch(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
