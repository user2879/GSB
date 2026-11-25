from __future__ import annotations

import json
from pathlib import Path


def launch_viewer(root: str | Path) -> int:
    root = Path(root)
    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageTk
    except Exception as exc:
        print("Tk viewer unavailable. Install system package: sudo apt install python3-tk")
        print(f"underlying error: {exc}")
        return 2

    report_path = root / "reports" / "last_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"run pipeline first; missing {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    app = tk.Tk()
    app.title("Vertical OCR Deterministic Workspace")
    app.geometry("1180x820")
    left = tk.Frame(app)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
    right = tk.Frame(app)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

    listbox = tk.Listbox(left, width=28)
    listbox.pack(fill=tk.Y, expand=True)
    text = tk.Text(right, height=18, wrap=tk.NONE)
    text.pack(fill=tk.X)
    image_label = tk.Label(right)
    image_label.pack(fill=tk.BOTH, expand=True)
    photo = None

    images = report.get("images", [])

    def show(_event=None):
        nonlocal photo
        selection = listbox.curselection()
        if not selection:
            return
        item = images[selection[0]]
        image_name = item["image"]
        text.delete("1.0", tk.END)
        text.insert(tk.END, json.dumps(item, ensure_ascii=False, indent=2)[:12000])
        image_path = root / "input" / image_name
        if image_path.exists():
            pillow = Image.open(image_path)
            pillow.thumbnail((680, 680))
            photo = ImageTk.PhotoImage(pillow)
            image_label.configure(image=photo)
        else:
            image_label.configure(image="", text=f"missing {image_path}")

    for item in images:
        listbox.insert(tk.END, item["image"])
    listbox.bind("<<ListboxSelect>>", show)
    if images:
        listbox.selection_set(0)
        show()

    info = tk.Label(left, text=f"records: {report.get('receipt_counts', {}).get('generated_records')}\n"
                               f"elapsed: {report.get('elapsed_ms')} ms", justify=tk.LEFT)
    info.pack(pady=8)
    try:
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("viewer error", str(exc))
        return 1
    return 0
