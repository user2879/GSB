"""Optional Tk layout inspector; the diagnostic pipeline never imports Tk."""

from __future__ import annotations

import json
from pathlib import Path


def launch_gui(snapshot_dir: str | Path = "diagnostics/current-run/current") -> None:
    import tkinter as tk
    from PIL import Image, ImageTk

    current = Path(snapshot_dir)
    if not current.is_dir():
        raise FileNotFoundError(f"snapshot not found: {current}")
    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    keypoints = json.loads((current / "keypoints.json").read_text(encoding="utf-8"))["keypoints"]
    root = tk.Tk()
    root.title("Layout Diagnostic Inspector")
    canvas = tk.Canvas(root, width=760, height=1020, bg="#dddddd")
    canvas.pack(side=tk.LEFT)
    text = tk.Text(root, width=54, height=60)
    text.pack(side=tk.LEFT, fill=tk.Y)
    text.insert("1.0", json.dumps(manifest, ensure_ascii=False, indent=2))
    image_path = Path("samples") / "images" / (manifest.get("completed_images") or [""])[0]
    if image_path.is_file():
        source = Image.open(image_path).convert("L").resize((640, 960))
        photo = ImageTk.PhotoImage(source)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
    for point in keypoints:
        if point["image"] != (manifest.get("completed_images") or [""])[0]:
            continue
        x, y = point["x_phys"], point["y_phys"]
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#d40000", width=2)
        canvas.create_text(x + 5, y - 5, text=str(point["keypoint_index"]), fill="#0055aa", anchor=tk.SW)
    root.mainloop()
