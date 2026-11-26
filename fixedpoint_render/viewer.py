"""Optional Tkinter frame viewer.

The rendering pipeline never imports this module, so a headless environment
without a display is fully supported.  Launching the window requires an
available Tk display; otherwise a clear, actionable error is raised.
"""
from __future__ import annotations

import os

from PIL import Image, ImageTk


def view_output(output_dir: str, scale: int = 16) -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("tkinter is unavailable: install python3-tk") from exc
    paths = sorted(os.path.join(output_dir, name)
                   for name in os.listdir(output_dir) if name.endswith(".png"))
    if not paths:
        raise FileNotFoundError(f"no PNG outputs in {output_dir}")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - headless
        raise RuntimeError(
            "no Tk display available; the renderer itself is headless-safe. "
            "Run on a desktop or inspect output PNGs directly.") from exc
    root.title("fixedpoint-render output")
    images = []
    for col, path in enumerate(paths):
        image = Image.open(path)
        scaled = image.resize((image.width * scale, image.height * scale),
                              Image.NEAREST)
        photo = ImageTk.PhotoImage(scaled)
        images.append(photo)
        label = tk.Label(root, image=photo)
        label.grid(row=0, column=col, padx=4, pady=4)
        tk.Label(root, text=os.path.basename(path)).grid(row=1, column=col)
    root.mainloop()
