"""Optional read-only Tkinter viewer for deterministic conclusion files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="View lease conclusion JSON")
    parser.add_argument(
        "--path",
        default="artifacts/demo/order-forward/conclusion.json",
        help="conclusion.json produced by leasectl demo",
    )
    args = parser.parse_args(argv)

    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": "E_TK",
                    "message": "python3-tk is unavailable",
                    "fix": "sudo apt-get update && sudo apt-get install -y python3-tk",
                    "detail": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2

    path = Path(args.path)
    root = tk.Tk()
    root.title("Lease middleware conclusion")
    root.geometry("980x640")
    text = tk.Text(root, wrap="none")
    text.pack(fill="both", expand=True)

    def reload_file() -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            text.delete("1.0", "end")
            text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        except Exception as exc:
            messagebox.showerror("Cannot load conclusion", str(exc))

    toolbar = tk.Frame(root)
    toolbar.pack(fill="x")
    tk.Button(toolbar, text="Reload", command=reload_file).pack(side="left")
    tk.Label(toolbar, text=str(path)).pack(side="left", padx=8)
    reload_file()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
