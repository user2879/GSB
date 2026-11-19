"""本地桌面诊断视图（业务要求 8）。

视图只读取 output 目录下已持久化、已校验的 timeline.json 与
diagnostic_report.json，不重算任何表达状态，也不修改状态块。
"""
from __future__ import annotations

import json
import os

from .atomicio import read_json
from .errors import WorkbenchError
from .report import REPORT_NAME, TIMELINE_NAME

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as exc:  # pragma: no cover - 无 tkinter 环境
    tk = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def launch_gui(output_dir: str) -> None:
    if tk is None:
        raise WorkbenchError("E_IO", f"tkinter 不可用: {_IMPORT_ERROR}")
    try:
        timeline = read_json(os.path.join(output_dir, TIMELINE_NAME))
        report = read_json(os.path.join(output_dir, REPORT_NAME))
    except WorkbenchError:
        raise
    root = tk.Tk()
    root.title(f"演奏表达诊断工作台 - {timeline.get('title', '')}")
    root.geometry("1180x760")

    main = ttk.Frame(root, padding=8)
    main.pack(fill=tk.BOTH, expand=True)

    info = ttk.Frame(main)
    info.pack(fill=tk.X)
    consistency = report.get("consistency", {})
    digest_status = report.get("digest_verification", {}).get("status", "n/a")
    ttk.Label(
        info,
        text=(
            f"版本 {report.get('version_id')} | 总 tick {report.get('total_ticks')} | "
            f"块数 {report.get('chunk_count')} | 编辑步数 {report.get('steps_applied')} | "
            f"一致性={consistency.get('consistent')} (时长差 {consistency.get('max_duration_diff')}, "
            f"力度差 {consistency.get('max_velocity_diff')}) | 摘要校验={digest_status}"
        ),
    ).pack(side=tk.LEFT)

    body = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
    body.pack(fill=tk.BOTH, expand=True, pady=6)

    canvas_frame = ttk.Frame(body)
    body.add(canvas_frame, weight=4)
    canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
    scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    canvas.configure(xscrollcommand=scroll.set)
    canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.BOTTOM, fill=tk.X)

    side = ttk.Frame(body)
    body.add(side, weight=1)
    ttk.Label(side, text="编辑影响范围（逐步）").pack(anchor=tk.W)
    impact_box = tk.Listbox(side, height=12)
    impact_box.pack(fill=tk.X)
    for item in report.get("edit_impact", []):
        impact_box.insert(
            tk.END,
            f"步骤{item['step']} {item['op']} -> 失效块 {item['invalidated_chunks']}",
        )

    ttk.Label(side, text="跨声部重叠").pack(anchor=tk.W, pady=(8, 0))
    overlap_box = tk.Listbox(side, height=6)
    overlap_box.pack(fill=tk.X)
    for item in report.get("cross_voice_overlaps", []):
        overlap_box.insert(
            tk.END,
            f"{item['voice_a']}/{item['voice_b']} tick {item['start_tick']}..{item['end_tick']}",
        )

    detail = ttk.Label(side, text="选择音符条带或 tick 块查看依赖", wraplength=300, justify=tk.LEFT)
    detail.pack(anchor=tk.W, pady=(8, 0))

    chunks = timeline["chunks"]
    total_ticks = timeline["total_ticks"]
    scale = max(1.0, 900.0 / max(total_ticks, 1))
    row_h = 18
    lane_top = 30
    colors = {"active": "#2e7d32", "silent": "#90a4ae"}
    note_items: list[tuple] = []

    # 速度与力度曲线轨道
    canvas.create_text(8, 8, anchor=tk.W, text="音符条带 / 速度·力度曲线 / 踏板轨道")
    max_tempo = max((c["tempo"][i] for c in chunks for i in range(len(c["tempo"]))), default=1.0)
    curve_base_y = lane_top + 5 * row_h + 10
    prev_x, prev_y = None, None
    tick_cursor = 0
    for chunk in chunks:
        for i, tempo in enumerate(chunk["tempo"]):
            x = (tick_cursor + i) * scale
            y = curve_base_y - (tempo / max_tempo) * 40 if max_tempo else curve_base_y
            if prev_x is not None:
                canvas.create_line(prev_x, prev_y, x, y, fill="#1565c0", width=1)
            prev_x, prev_y = x, y
        tick_cursor += chunk["declared_ticks"]
    canvas.create_text(8, curve_base_y + 2, anchor=tk.NW, text="速度", fill="#1565c0")

    # 逐 tick 实际力度曲线：取该 tick 上活跃音符的最大实际力度
    velocity_base_y = curve_base_y + 26
    prev_x, prev_y = None, None
    tick_cursor = 0
    for chunk in chunks:
        for i, notes in enumerate(chunk["per_tick_notes"]):
            active = [n["velocity"] for n in notes if n["state"] == "active"]
            velocity = max(active) if active else 0
            x = (tick_cursor + i) * scale
            y = velocity_base_y - (velocity / 127.0) * 40
            if prev_x is not None:
                canvas.create_line(prev_x, prev_y, x, y, fill="#c62828", width=1)
            prev_x, prev_y = x, y
        tick_cursor += chunk["declared_ticks"]
    canvas.create_text(8, velocity_base_y + 2, anchor=tk.NW, text="力度", fill="#c62828")

    pedal_y = velocity_base_y + 18
    tick_cursor = 0
    for chunk in chunks:
        for i, pedal in enumerate(chunk["pedal"]):
            if pedal:
                x = (tick_cursor + i) * scale
                canvas.create_rectangle(x, pedal_y, x + max(1, scale), pedal_y + 8,
                                        fill="#6a1b9a", outline="")
        tick_cursor += chunk["declared_ticks"]
    canvas.create_text(8, pedal_y - 4, anchor=tk.W, text="踏板", fill="#6a1b9a")

    # 音符条带（按声部排行为序）
    voices = sorted({seg["voice"] for c in chunks for seg in c["note_segments"]})
    voice_row = {voice: idx for idx, voice in enumerate(voices)}
    drawn: set[str] = set()
    for chunk in chunks:
        for seg in chunk["note_segments"]:
            key = seg["note_id"]
            if key in drawn:
                continue
            drawn.add(key)
            row = voice_row[seg["voice"]]
            x0 = seg["start_tick"] * scale
            x1 = seg["active_end_tick"] * scale
            x2 = seg["actual_end_tick"] * scale
            y = lane_top + row * row_h
            active_item = canvas.create_rectangle(x0, y, x1, y + row_h - 4,
                                                  fill=colors["active"], outline="")
            if x2 > x1:
                canvas.create_rectangle(x1, y, x2, y + row_h - 4, fill=colors["silent"], outline="")
            canvas.create_text(x0 + 2, y, anchor=tk.NW, text=key, fill="white", font=("TkDefaultFont", 7))
            note_items.append((active_item, seg, chunk))

    # 块边界与点击选择
    chunk_items = []
    for chunk in chunks:
        x0 = chunk["tick_lo"] * scale
        x1 = chunk["tick_hi"] * scale
        item = canvas.create_rectangle(x0, lane_top - 12, x1, pedal_y + 14,
                                       outline="#bdbdbd", dash=(2, 3), fill="")
        chunk_items.append((item, chunk))

    def show_chunk(chunk):
        detail.configure(
            text=(
                f"块 {chunk['chunk_index']} tick {chunk['tick_lo']}..{chunk['tick_hi']}\n"
                f"依赖键: {', '.join(chunk['dependency_keys'])}"
            )
        )

    def on_canvas_click(event):
        x = canvas.canvasx(event.x)
        for item, chunk in chunk_items:
            x0, _y0, x1, _y1 = canvas.coords(item)
            if x0 <= x <= x1:
                canvas.itemconfigure(item, outline="#e65100", width=2)
                show_chunk(chunk)
                return

    canvas.bind("<Button-1>", on_canvas_click)
    canvas.configure(scrollregion=canvas.bbox("all"))

    root.mainloop()
