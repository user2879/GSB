from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_view_data(output_dir: str | Path) -> dict[str, Any]:
    """Load the committed human-review mirror without mutating any state."""

    root = Path(output_dir)
    report_path = root / "diagnostic-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("diagnostic-report.json must contain an object")

    final: dict[str, Any] | None = None
    final_path = root / "final.json"
    if final_path.exists():
        value = json.loads(final_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            final = value

    state: dict[str, Any] | None = None
    pointer_path = root / "CURRENT.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if isinstance(pointer, dict) and isinstance(pointer.get("snapshot"), str):
            state_path = root / pointer["snapshot"] / "state.json"
            if state_path.exists():
                value = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    state = value

    receipts: dict[str, str] = {}
    for name in ("evidence.a1", "evidence.a2.json"):
        path = root / "receipts" / name
        if path.exists():
            receipts[name] = path.read_text(encoding="utf-8")

    return {
        "root": str(root),
        "report": report,
        "final": final,
        "state": state,
        "receipts": receipts,
    }


def interval_rows(report: Mapping[str, Any]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for item in report.get("interval_evidence") or []:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            (
                str(item.get("interval_id", "")),
                str(item.get("speaker_id", "")),
                str(item.get("endpoint_confidence_ppm", "")),
                str(item.get("command_code", "")),
                str(item.get("command_probability_ppm", "")),
                str(item.get("truth_command_code", "")),
                "yes" if item.get("truth_matches") else "no",
                str(item.get("command_arbitration", "")),
            )
        )
    return rows


def cache_rows(report: Mapping[str, Any], *, limit: int = 500) -> list[tuple[str, ...]]:
    cache = report.get("cache") or {}
    rows: list[tuple[str, ...]] = []
    for item in cache.get("reuse_entries") or []:
        if not isinstance(item, Mapping):
            continue
        rows.append(("reuse", str(item.get("stage", "")), str(item.get("cache_id", "")), ""))
    for item in cache.get("invalidation_entries") or []:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            (
                "invalidate",
                str(item.get("stage", "")),
                str(item.get("cache_id", "")),
                ",".join(str(reason) for reason in item.get("reasons", [])),
            )
        )
    return rows[: max(0, limit)]


def launch_viewer(output_dir: str | Path) -> None:
    """Open a read-only Tkinter review window.

    Tkinter is imported lazily so all parsing, scheduling, and recovery paths
    remain usable on headless servers where the optional display stack is not
    available.
    """

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # pragma: no cover - depends on host packaging
        raise RuntimeError("Tkinter is not installed; install the system python3-tk package") from exc

    data = load_view_data(output_dir)
    report = data["report"]
    root = tk.Tk()
    root.title("Voice diagnostic review")
    root.geometry("1180x760")

    header = ttk.Frame(root, padding=10)
    header.pack(fill=tk.X)
    summary = {
        "conversation": report.get("conversation_id"),
        "status": report.get("status"),
        "final_digest": report.get("final_digest"),
        "receipts a1/a2": f"{report.get('receipts', {}).get('generated_a1_count', 0)}/"
        f"{report.get('receipts', {}).get('generated_a2_count', 0)}",
        "overlap frames": report.get("overlap_frame_count"),
        "boundary frames": report.get("boundary_frame_count"),
        "cache reuse/invalidation": f"{report.get('cache', {}).get('reuse_count', 0)}/"
        f"{report.get('cache', {}).get('invalidation_count', 0)}",
    }
    for row, (label, value) in enumerate(summary.items()):
        ttk.Label(header, text=f"{label}:", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Label(header, text=str(value)).grid(row=row, column=1, sticky="w")

    notebook = ttk.Notebook(root, padding=(10, 0, 10, 10))
    notebook.pack(fill=tk.BOTH, expand=True)

    interval_frame = ttk.Frame(notebook)
    notebook.add(interval_frame, text="Interval evidence")
    interval_columns = (
        "interval",
        "speaker",
        "endpoint_ppm",
        "command",
        "probability_ppm",
        "truth",
        "match",
        "arbitration",
    )
    interval_tree = ttk.Treeview(interval_frame, columns=interval_columns, show="headings")
    for column in interval_columns:
        interval_tree.heading(column, text=column)
        interval_tree.column(column, width=125, anchor="w")
    for row in interval_rows(report):
        interval_tree.insert("", tk.END, values=row)
    interval_scroll = ttk.Scrollbar(interval_frame, orient=tk.VERTICAL, command=interval_tree.yview)
    interval_tree.configure(yscrollcommand=interval_scroll.set)
    interval_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    interval_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    overlap_frame = ttk.Frame(notebook)
    notebook.add(overlap_frame, text="Overlap / boundary")
    overlap_text = tk.Text(overlap_frame, wrap=tk.WORD)
    commands = report.get("command_conclusions") or {}
    overlap_lines = [
        f"overlap_frame_count: {report.get('overlap_frame_count', 0)}",
        f"contested_frame_count: {report.get('contested_frame_count', 0)}",
        f"boundary_frames: {report.get('boundary_frames', [])}",
        f"multi_source_window_count: {report.get('multi_source_window_count', 0)}",
        "",
        "accepted command segments:",
    ]
    overlap_lines.extend(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in commands.get("accepted", []))
    overlap_lines.append("")
    overlap_lines.append("suppressed command segments:")
    overlap_lines.extend(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in commands.get("suppressed", []))
    overlap_text.insert("1.0", "\n".join(overlap_lines))
    overlap_text.configure(state=tk.DISABLED)
    overlap_text.pack(fill=tk.BOTH, expand=True)

    cache_frame = ttk.Frame(notebook)
    notebook.add(cache_frame, text="Reuse / invalidation")
    cache_tree = ttk.Treeview(cache_frame, columns=("kind", "stage", "cache_id", "reason"), show="headings")
    for column in ("kind", "stage", "cache_id", "reason"):
        cache_tree.heading(column, text=column)
        cache_tree.column(column, width=180 if column != "reason" else 560, anchor="w")
    for row in cache_rows(report):
        cache_tree.insert("", tk.END, values=row)
    cache_scroll = ttk.Scrollbar(cache_frame, orient=tk.VERTICAL, command=cache_tree.yview)
    cache_tree.configure(yscrollcommand=cache_scroll.set)
    cache_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    cache_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    receipt_frame = ttk.Frame(notebook)
    notebook.add(receipt_frame, text="Receipt text")
    receipt_text = tk.Text(receipt_frame, wrap=tk.NONE)
    for name, text in sorted(data["receipts"].items()):
        receipt_text.insert(tk.END, f"=== {name} ===\n{text}\n")
    receipt_text.configure(state=tk.DISABLED)
    receipt_text.pack(fill=tk.BOTH, expand=True)

    report_frame = ttk.Frame(notebook)
    notebook.add(report_frame, text="Raw report")
    raw_text = tk.Text(report_frame, wrap=tk.NONE)
    raw_text.insert("1.0", json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    raw_text.configure(state=tk.DISABLED)
    raw_text.pack(fill=tk.BOTH, expand=True)

    root.mainloop()
