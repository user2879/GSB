from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ledger import open_readonly


def load_dashboard(
    ledger_path: str | Path,
    *,
    session_id: str | None = None,
    view_id: str | None = None,
) -> dict[str, Any]:
    connection = open_readonly(ledger_path)
    try:
        if session_id is None:
            row = connection.execute(
                "SELECT session_id FROM sessions WHERE session_id NOT LIKE 'inject-%' ORDER BY updated_at DESC, session_id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT session_id FROM sessions ORDER BY updated_at DESC, session_id DESC LIMIT 1"
                ).fetchone()
            if row is None:
                raise ValueError("ledger has no registered sessions")
            session_id = row["session_id"]
        runtime = connection.execute(
            "SELECT * FROM session_runtime WHERE session_id = ?", (session_id,)
        ).fetchone()
        if runtime is None:
            raise ValueError(f"unknown session {session_id!r}")
        if view_id is None:
            view_id = runtime["current_view_id"]
        manifest = connection.execute(
            "SELECT * FROM view_manifest WHERE session_id = ? AND view_id = ?",
            (session_id, view_id),
        ).fetchone()
        if manifest is None:
            raise ValueError(f"unknown view {view_id!r}")
        traces = connection.execute(
            "SELECT * FROM watermark_trace WHERE session_id = ? AND view_id = ? ORDER BY arrival_no, event_id",
            (session_id, view_id),
        ).fetchall()
        windows = connection.execute(
            "SELECT w.*, a.quantization_truncation, a.late_admission, a.window_boundary, a.sum_bound, a.mean_bound, a.single_value_quantization, a.dominant_segment, a.dominant_share_bp, a.threshold_bp, a.quantization_share_bp, a.late_share_bp, a.boundary_share_bp FROM window_state w LEFT JOIN error_attribution a ON a.session_id = w.session_id AND a.view_id = w.view_id AND a.rule_id = w.rule_id AND a.partition_key = w.partition_key AND a.window_start_ms = w.window_start_ms AND a.window_end_ms = w.window_end_ms WHERE w.session_id = ? AND w.view_id = ? AND w.stale = 0 AND w.status = 'COMMITTED' ORDER BY w.partition_key, w.window_start_ms, w.window_end_ms, w.rule_id",
            (session_id, view_id),
        ).fetchall()
        decisions = connection.execute(
            "SELECT * FROM late_decisions WHERE session_id = ? AND view_id = ? AND stale = 0 ORDER BY arrival_no, partition_key, rule_id, event_id",
            (session_id, view_id),
        ).fetchall()
        batches = connection.execute(
            "SELECT * FROM ledger_batches WHERE session_id = ? AND view_id = ? ORDER BY batch_no",
            (session_id, view_id),
        ).fetchall()
        ambiguities = connection.execute(
            "SELECT * FROM ambiguity_pairs WHERE session_id = ? AND view_id = ? AND stale = 0 ORDER BY pair_id",
            (session_id, view_id),
        ).fetchall()
        pending = connection.execute(
            "SELECT status, COUNT(*) AS count FROM window_state WHERE session_id = ? AND view_id = ? AND stale = 0 AND status != 'COMMITTED' GROUP BY status",
            (session_id, view_id),
        ).fetchall()
        stale_counts = connection.execute(
            "SELECT 'window_state' AS table_name, COUNT(*) AS count FROM window_state WHERE session_id = ? AND view_id = ? AND stale = 1 UNION ALL SELECT 'late_decisions', COUNT(*) FROM late_decisions WHERE session_id = ? AND view_id = ? AND stale = 1 UNION ALL SELECT 'error_attribution', COUNT(*) FROM error_attribution WHERE session_id = ? AND view_id = ? AND stale = 1",
            (session_id, view_id, session_id, view_id, session_id, view_id),
        ).fetchall()
        budget_by_window: dict[tuple[Any, ...], dict[str, Any]] = {}
        for decision in decisions:
            key = (
                decision["rule_id"],
                decision["partition_key"],
                decision["window_start_ms"],
                decision["window_end_ms"],
            )
            budget_by_window[key] = {
                "rule_id": decision["rule_id"],
                "partition_key": decision["partition_key"],
                "window_start_ms": decision["window_start_ms"],
                "window_end_ms": decision["window_end_ms"],
                "remaining_count": decision["budget_count_after"],
                "remaining_amount": decision["budget_amount_after"],
                "cumulative_late_abs_amount": decision["cumulative_late_abs_amount"],
                "last_arrival_no": decision["arrival_no"],
                "last_decision": decision["decision"],
                "last_reason": decision["reason"],
            }
        return {
            "session_id": session_id,
            "view_id": view_id,
            "runtime": dict(runtime),
            "manifest": dict(manifest),
            "sessions": [dict(row) for row in connection.execute("SELECT * FROM sessions ORDER BY created_at, session_id")],
            "traces": [dict(row) for row in traces],
            "windows": [dict(row) for row in windows],
            "decisions": [dict(row) for row in decisions],
            "batches": [dict(row) for row in batches],
            "ambiguities": [dict(row) for row in ambiguities],
            "pending_status": {row["status"]: row["count"] for row in pending},
            "stale_counts": {row["table_name"]: row["count"] for row in stale_counts},
            "budget_by_window": [budget_by_window[key] for key in sorted(budget_by_window)],
            "counts": {
                "traces": len(traces),
                "committed_windows": len(windows),
                "decisions": len(decisions),
                "drops": sum(1 for row in decisions if row["decision"] == "DROP"),
                "late_accepts": sum(
                    1
                    for row in decisions
                    if row["decision"] == "ACCEPT" and row["was_late"]
                ),
                "batches": len(batches),
                "ambiguities": len(ambiguities),
            },
        }
    finally:
        connection.close()


def run_diagnostics(
    ledger_path: str | Path,
    *,
    session_id: str | None = None,
    view_id: str | None = None,
) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Tkinter is unavailable. On Windows use the official CPython installer and run `python -m tkinter`."
        ) from exc

    root = tk.Tk()
    root.title("StreamAgg Fixed-Point Diagnostics")
    root.geometry("1220x820")
    root.minsize(980, 680)

    current_session = {"session_id": session_id, "view_id": view_id}
    header = tk.Frame(root, padx=10, pady=8)
    header.pack(fill="x")
    title_var = tk.StringVar(value="StreamAgg diagnostics")
    subtitle_var = tk.StringVar(value="")
    tk.Label(header, textvariable=title_var, font=("Segoe UI", 16, "bold"), anchor="w").pack(fill="x")
    tk.Label(header, textvariable=subtitle_var, font=("Consolas", 10), anchor="w", justify="left").pack(fill="x")
    controls = tk.Frame(root, padx=10)
    controls.pack(fill="x")
    session_var = tk.StringVar(value=session_id or "")
    tk.Label(controls, text="Session").pack(side="left")
    session_entry = tk.Entry(controls, textvariable=session_var, width=32)
    session_entry.pack(side="left", padx=(6, 10))
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=(6, 10))

    chart_frame = tk.Frame(notebook)
    windows_frame = tk.Frame(notebook)
    late_frame = tk.Frame(notebook)
    error_frame = tk.Frame(notebook)
    batch_frame = tk.Frame(notebook)
    notebook.add(chart_frame, text="水位曲线")
    notebook.add(windows_frame, text="窗口定稿")
    notebook.add(late_frame, text="迟到裁决与预算")
    notebook.add(error_frame, text="误差界与归因")
    notebook.add(batch_frame, text="规则版本与账本批次")

    canvas = tk.Canvas(chart_frame, background="#101820", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=6, pady=6)

    def make_tree(parent: tk.Widget, columns: tuple[tuple[str, str, int], ...]) -> ttk.Treeview:
        frame = tk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=[column[0] for column in columns], show="headings")
        for key, label, width in columns:
            tree.heading(key, text=label)
            tree.column(key, width=width, anchor="w", stretch=True)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    windows_tree = make_tree(
        windows_frame,
        (
            ("partition", "分区", 90),
            ("start", "窗口起", 150),
            ("end", "窗口止", 150),
            ("rule", "规则", 120),
            ("count", "计数", 70),
            ("sum", "定点求和", 150),
            ("mean", "定点均值", 150),
            ("sum_bound", "求和误差界", 130),
            ("mean_bound", "均值误差界", 130),
            ("dominant", "主导段", 130),
        ),
    )
    late_tree = make_tree(
        late_frame,
        (
            ("partition", "分区", 80),
            ("window", "窗口", 300),
            ("remaining_count", "剩余条数预算", 120),
            ("remaining_amount", "剩余量预算", 140),
            ("cumulative", "累计迟到绝对量", 150),
            ("decision", "最后裁决", 100),
            ("reason", "原因", 260),
        ),
    )
    error_tree = make_tree(
        error_frame,
        (
            ("partition", "分区", 80),
            ("window", "窗口", 280),
            ("quant", "量化截断", 120),
            ("late", "迟到接纳", 120),
            ("boundary", "边界归属", 120),
            ("sum_bound", "求和界", 120),
            ("mean_bound", "均值界", 120),
            ("dominant", "主导段", 130),
            ("share", "占比bp", 80),
        ),
    )
    batch_tree = make_tree(
        batch_frame,
        (
            ("field", "字段", 220),
            ("value", "值", 900),
        ),
    )

    def clear_tree(tree: ttk.Treeview) -> None:
        children = tree.get_children()
        if children:
            tree.delete(*children)

    def draw_chart(data: dict[str, Any]) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 360)
        left, right, top, bottom = 70, 20, 30, 50
        plot_width = width - left - right
        plot_height = height - top - bottom
        traces = data["traces"]
        canvas.create_rectangle(left, top, width - right, height - bottom, outline="#4A6072")
        if not traces:
            canvas.create_text(width / 2, height / 2, text="No watermark trace", fill="#C7D3DD")
            return
        min_arrival = min(row["arrival_no"] for row in traces)
        max_arrival = max(row["arrival_no"] for row in traces)
        min_value = min(row["watermark_after_ms"] for row in traces)
        max_value = max(row["watermark_after_ms"] for row in traces)
        arrival_span = max(1, max_arrival - min_arrival)
        value_span = max(1, max_value - min_value)
        partitions = sorted({row["partition_key"] for row in traces})
        palette = ["#57C7FF", "#F6BD60", "#84D9A0", "#E5989B", "#B8A1E3", "#90BE6D"]

        def point(row: dict[str, Any]) -> tuple[float, float]:
            x = left + (row["arrival_no"] - min_arrival) / arrival_span * plot_width
            y = top + (max_value - row["watermark_after_ms"]) / value_span * plot_height
            return x, y

        canvas.create_text(left, 12, text=f"watermark ms  [{min_value} .. {max_value}]", fill="#C7D3DD", anchor="w")
        for index, partition in enumerate(partitions):
            points = [point(row) for row in traces if row["partition_key"] == partition]
            color = palette[index % len(palette)]
            if len(points) >= 2:
                canvas.create_line(*[coordinate for pair in points for coordinate in pair], fill=color, width=2)
            for x, y in points[:: max(1, len(points) // 30)]:
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
            legend_y = top + 12 + index * 18
            canvas.create_line(left + 12, legend_y, left + 34, legend_y, fill=color, width=3)
            canvas.create_text(left + 40, legend_y, text=partition, fill="#E8EEF2", anchor="w")

    def refresh(*_: Any) -> None:
        requested_session = session_var.get().strip() or None
        data = load_dashboard(
            ledger_path,
            session_id=requested_session,
            view_id=current_session["view_id"] if requested_session == current_session["session_id"] else None,
        )
        current_session["session_id"] = data["session_id"]
        current_session["view_id"] = data["view_id"]
        session_var.set(data["session_id"])
        manifest = data["manifest"]
        runtime = data["runtime"]
        counts = data["counts"]
        title_var.set(
            f"Session {data['session_id']} | view {data['view_id']} | rules v{manifest['rules_version']} | scale {manifest['fixed_scale']}"
        )
        subtitle_var.set(
            f"ledger batch {runtime['last_batch_digest'][:16]} | cursor {runtime['event_cursor']} | "
            f"windows {counts['committed_windows']} | drops {counts['drops']} | late accepts {counts['late_accepts']} | "
            f"pending {data['pending_status']} | stale {data['stale_counts']}"
        )
        draw_chart(data)
        clear_tree(windows_tree)
        for row in data["windows"]:
            windows_tree.insert(
                "",
                "end",
                values=(
                    row["partition_key"],
                    row["window_start_ms"],
                    row["window_end_ms"],
                    row["rule_id"],
                    row["count"],
                    row["sum_fixed"],
                    row["mean_fixed"],
                    row["sum_bound"],
                    row["mean_bound"],
                    f"{row['dominant_segment']} {row['dominant_share_bp']}bp",
                ),
            )
        clear_tree(late_tree)
        for row in data["budget_by_window"]:
            late_tree.insert(
                "",
                "end",
                values=(
                    row["partition_key"],
                    f"[{row['window_start_ms']}, {row['window_end_ms']})",
                    row["remaining_count"],
                    row["remaining_amount"],
                    row["cumulative_late_abs_amount"],
                    row["last_decision"],
                    f"{row['last_reason']} @ {row['last_arrival_no']}",
                ),
            )
        clear_tree(error_tree)
        for row in data["windows"]:
            error_tree.insert(
                "",
                "end",
                values=(
                    row["partition_key"],
                    f"[{row['window_start_ms']}, {row['window_end_ms']})",
                    row["quantization_truncation"],
                    row["late_admission"],
                    row["window_boundary"],
                    row["sum_bound"],
                    row["mean_bound"],
                    row["dominant_segment"],
                    row["dominant_share_bp"],
                ),
            )
        clear_tree(batch_tree)
        manifest_rows = (
            ("session_id", data["session_id"]),
            ("view_id", data["view_id"]),
            ("rules_version", manifest["rules_version"]),
            ("fixed_scale", manifest["fixed_scale"]),
            ("view_status", manifest["status"]),
            ("view_content_digest", manifest["content_digest"]),
            ("spec_digest", manifest["spec_digest"]),
            ("runtime_batch_digest", runtime["last_batch_digest"]),
            ("event_cursor", runtime["event_cursor"]),
            ("pending_status", json.dumps(data["pending_status"], ensure_ascii=False, sort_keys=True)),
            ("stale_counts", json.dumps(data["stale_counts"], ensure_ascii=False, sort_keys=True)),
            ("ambiguity_pairs", counts["ambiguities"]),
        )
        for field, value in manifest_rows:
            batch_tree.insert("", "end", values=(field, value))
        for batch in data["batches"]:
            batch_tree.insert(
                "",
                "end",
                values=(f"batch_{batch['batch_no']}_digest", batch["content_digest"]),
            )

    tk.Button(controls, text="刷新（只读）", command=refresh).pack(side="left")
    tk.Label(controls, text="Refresh opens SQLite in mode=ro and query_only=ON", fg="#52616B").pack(side="left", padx=12)
    root.after(80, refresh)
    root.mainloop()
