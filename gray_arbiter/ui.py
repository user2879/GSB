"""Read-only Tkinter diagnostics.

Importing this module is safe on systems without Tkinter.  The toolkit is
loaded only when :func:`launch` is called.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_ledger(path: str | Path) -> dict[str, Any]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {"batches": [], "epochs": [], "batches_table": []}
    try:
        connection = sqlite3.connect(f"{ledger_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return {"batches_table": [], "epochs": [], "error": str(exc)}
    try:
        try:
            batches_table = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT b.batch_id, b.config_version, b.ledger_format_version,
                           b.start_cursor, b.end_cursor, b.content_digest,
                           COALESCE(e.status, 'unknown') AS epoch_status
                    FROM batches b
                    LEFT JOIN version_epochs e ON e.version = b.config_version
                    ORDER BY b.start_cursor, b.batch_id
                    """
                ).fetchall()
            ]
            epochs = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM version_epochs ORDER BY version"
                ).fetchall()
            ]
            return {"batches_table": batches_table, "epochs": epochs}
        except sqlite3.Error as exc:
            return {"batches_table": [], "epochs": [], "error": str(exc)}
    finally:
        connection.close()


def _batch_file_key(path: Path) -> tuple[int, int, int]:
    try:
        parts = path.stem.split("-")
        return (int(parts[1][1:]), int(parts[2]), int(parts[3]))
    except (IndexError, ValueError):
        return (0, 0, 0)


def _load_diagnostics(ledger_path: str | Path, snapshot_dir: str | Path) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_dir)
    batch_files = (
        sorted(snapshot_dir.glob("batch-v*.json"), key=_batch_file_key)
        if snapshot_dir.exists()
        else []
    )
    replacement_files = (
        sorted(snapshot_dir.glob("replacement-v*.json")) if snapshot_dir.exists() else []
    )
    latest_result = _read_json(snapshot_dir / "latest.json") if snapshot_dir.exists() else None
    latest_archived_batch = _read_json(batch_files[-1]) if batch_files else None
    latest_batch = latest_archived_batch
    if latest_result and latest_result.get("batches"):
        result_batch = latest_result["batches"][-1]
        if latest_batch is None or (
            int(result_batch.get("config_version", 0)),
            int(result_batch.get("end_cursor", 0)),
        ) >= (
            int(latest_batch.get("config_version", 0)),
            int(latest_batch.get("end_cursor", 0)),
        ):
            latest_batch = result_batch
    current_view = (latest_result or {}).get("current_view")
    replacements = [
        report
        for report in (_read_json(path) for path in replacement_files)
        if isinstance(report, dict)
    ]
    latest_replacement = max(
        replacements,
        key=lambda report: (int(report.get("new_version", 0)), str(report.get("new_config_digest", ""))),
        default=None,
    )
    if latest_replacement is not None:
        batch_version = int((latest_batch or {}).get("config_version", 0))
        if int(latest_replacement.get("new_version", 0)) > batch_version:
            current_view = latest_replacement.get("current_view", current_view)
            # Old-version batch conclusions are archival only after a whole
            # replacement and must not appear as the current report.
            latest_batch = None
    ledger = _read_ledger(ledger_path)
    return {
        "latest_result": latest_result,
        "latest_batch": latest_batch,
        "latest_archived_batch": latest_archived_batch,
        "current_view": current_view,
        "latest_replacement": latest_replacement,
        "ledger": ledger,
        "snapshot_files": [path.name for path in batch_files],
    }


def launch(ledger_path: str | Path, snapshot_dir: str | Path) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        print(f"Tkinter is not available: {exc}")
        return 2

    class DiagnosticsApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Gray Arbiter - Connection Watermark Diagnostics")
            self.geometry("1180x760")
            self.ledger_path = ledger_path
            self.snapshot_dir = snapshot_dir
            self.data: dict[str, Any] = {}
            self._build()
            self.refresh()

        def _build(self) -> None:
            toolbar = ttk.Frame(self, padding=(8, 6))
            toolbar.pack(side=tk.TOP, fill=tk.X)
            ttk.Button(toolbar, text="Refresh (read-only)", command=self.refresh).pack(side=tk.LEFT)
            self.status = ttk.Label(toolbar, text="")
            self.status.pack(side=tk.LEFT, padx=12)
            self.notebook = ttk.Notebook(self)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

            self.watermark_frame = ttk.Frame(self.notebook)
            self.requests_frame = ttk.Frame(self.notebook)
            self.borrow_frame = ttk.Frame(self.notebook)
            self.heat_frame = ttk.Frame(self.notebook)
            self.config_frame = ttk.Frame(self.notebook)
            self.ledger_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.watermark_frame, text="Pool Watermarks")
            self.notebook.add(self.requests_frame, text="Requests")
            self.notebook.add(self.borrow_frame, text="Borrow / Return")
            self.notebook.add(self.heat_frame, text="Error-bound Heat")
            self.notebook.add(self.config_frame, text="Rules & Versions")
            self.notebook.add(self.ledger_frame, text="Ledger")

            self.watermark_canvas = tk.Canvas(self.watermark_frame, background="white")
            self.watermark_canvas.pack(fill=tk.BOTH, expand=True)

            request_columns = (
                "ordinal",
                "request_id",
                "destination",
                "pool",
                "connection",
                "reuse",
                "borrow",
                "status",
                "mirror",
            )
            self.request_tree = ttk.Treeview(
                self.requests_frame, columns=request_columns, show="headings"
            )
            for column in request_columns:
                self.request_tree.heading(column, text=column)
                self.request_tree.column(column, width=115, anchor=tk.W)
            self._scrollable_tree(self.requests_frame, self.request_tree)

            borrow_columns = (
                "borrow_id",
                "lender",
                "borrower",
                "amount",
                "deadline",
                "returned",
                "state",
                "overdue",
            )
            self.borrow_tree = ttk.Treeview(
                self.borrow_frame, columns=borrow_columns, show="headings"
            )
            for column in borrow_columns:
                self.borrow_tree.heading(column, text=column)
                self.borrow_tree.column(column, width=130, anchor=tk.W)
            self._scrollable_tree(self.borrow_frame, self.borrow_tree)

            self.heat_canvas = tk.Canvas(self.heat_frame, background="white")
            self.heat_canvas.pack(fill=tk.BOTH, expand=True)

            self.config_text = tk.Text(self.config_frame, wrap=tk.NONE)
            self.config_text.pack(fill=tk.BOTH, expand=True)

            ledger_columns = (
                "batch_id",
                "version",
                "format",
                "start",
                "end",
                "digest",
                "status",
            )
            self.ledger_tree = ttk.Treeview(
                self.ledger_frame, columns=ledger_columns, show="headings"
            )
            for column in ledger_columns:
                self.ledger_tree.heading(column, text=column)
                self.ledger_tree.column(column, width=150, anchor=tk.W)
            self._scrollable_tree(self.ledger_frame, self.ledger_tree)

        @staticmethod
        def _scrollable_tree(parent: Any, tree: Any) -> None:
            scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def refresh(self) -> None:
            # Every load is read-only.  No ledger row or snapshot file is
            # rewritten when the window refreshes.
            self.data = _load_diagnostics(self.ledger_path, self.snapshot_dir)
            self._draw_watermarks()
            self._fill_requests()
            self._fill_borrows()
            self._draw_heat()
            self._fill_config()
            self._fill_ledger()
            batch = self.data.get("latest_batch") or {}
            current = self.data.get("current_view") or {}
            version = batch.get("config_version", (current.get("config") or {}).get("version", "-"))
            batch_id = batch.get("batch_id", "no completed batch for current version")
            self.status.configure(text=f"version={version}  batch={batch_id}  read-only refresh")

        def _draw_watermarks(self) -> None:
            canvas = self.watermark_canvas
            canvas.delete("all")
            batch = self.data.get("latest_batch") or {}
            series = batch.get("pool_watermark_series") or {}
            if not series:
                canvas.create_text(24, 24, anchor=tk.NW, text="No watermark samples")
                return
            width = max(640, canvas.winfo_width() or 900)
            height = max(420, canvas.winfo_height() or 520)
            left, top, right, bottom = 58, 34, width - 24, height - 58
            max_points = max(len(values) for values in series.values())
            max_value = max(
                [1]
                + [
                    max(
                        [1]
                        + [
                            max(item.get("active_connections", 0), item.get("lent_out", 0))
                            for item in values
                        ]
                    )
                    for values in series.values()
                ]
            )
            canvas.create_line(left, top, left, bottom, fill="#555")
            canvas.create_line(left, bottom, right, bottom, fill="#555")
            for tick in range(0, max_value + 1):
                y = bottom - (tick / max_value) * (bottom - top)
                canvas.create_line(left - 4, y, left, y, fill="#777")
                canvas.create_text(left - 8, y, anchor=tk.E, text=str(tick))
            canvas.create_text((left + right) / 2, 12, text="Active (solid) / Lent-out (dashed) by event sample")
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
            legend_y = bottom + 18
            for pool_index, pool_id in enumerate(sorted(series)):
                values = series[pool_id]
                color = palette[pool_index % len(palette)]
                for field, dash in (("active_connections", None), ("lent_out", (5, 3))):
                    points: list[float] = []
                    for index, item in enumerate(values):
                        x = left + ((index + 0.5) / max(1, max_points)) * (right - left)
                        y = bottom - (item.get(field, 0) / max_value) * (bottom - top)
                        points.extend((x, y))
                    if len(points) >= 4:
                        canvas.create_line(*points, fill=color, width=2, dash=dash)
                canvas.create_text(
                    left + (pool_index % 4) * 210,
                    legend_y + (pool_index // 4) * 18,
                    anchor=tk.W,
                    text=f"{pool_id}: active/lent",
                    fill=color,
                )

        def _fill_requests(self) -> None:
            self.request_tree.delete(*self.request_tree.get_children())
            batch = self.data.get("latest_batch") or {}
            for request in sorted(batch.get("requests", []), key=lambda item: item["ordinal"]):
                mirror = request.get("mirror_status", "not_sampled")
                if request.get("mirror_error_code"):
                    mirror = f"{mirror}:{request['mirror_error_code']}"
                self.request_tree.insert(
                    "",
                    tk.END,
                    values=(
                        request.get("ordinal"),
                        request.get("request_id"),
                        request.get("main_destination"),
                        request.get("main_pool_id"),
                        request.get("connection_id"),
                        request.get("reuse_sequence"),
                        request.get("borrow_id"),
                        request.get("main_status"),
                        mirror,
                    ),
                )

        def _fill_borrows(self) -> None:
            self.borrow_tree.delete(*self.borrow_tree.get_children())
            batch = self.data.get("latest_batch") or {}
            for borrow in sorted(batch.get("borrows", []), key=lambda item: item["borrow_id"]):
                self.borrow_tree.insert(
                    "",
                    tk.END,
                    values=(
                        borrow.get("borrow_id"),
                        borrow.get("lender_pool"),
                        borrow.get("borrower_pool"),
                        borrow.get("amount"),
                        borrow.get("deadline_mono_ms"),
                        borrow.get("returned_mono_ms"),
                        borrow.get("state"),
                        borrow.get("overdue"),
                    ),
                )

        def _draw_heat(self) -> None:
            canvas = self.heat_canvas
            canvas.delete("all")
            batch = self.data.get("latest_batch") or {}
            report = batch.get("error_report") or {}
            pools = report.get("pool_errors") or {}
            mirrors = report.get("mirror_pool_errors") or {}
            rows = [(pool_id, pools[pool_id], "main") for pool_id in sorted(pools)]
            rows.extend((pool_id, mirrors[pool_id], "mirror") for pool_id in sorted(mirrors))
            if not rows:
                canvas.create_text(24, 24, anchor=tk.NW, text="No error-bound data")
                return
            max_bound = max(row[1].get("aggregate_error_upper_bound_scaled", 1) for row in rows) or 1
            y = 28
            canvas.create_text(24, 12, anchor=tk.W, text="Pool error upper bound (scaled x1000)")
            for pool_id, entry, kind in rows:
                bound = entry.get("aggregate_error_upper_bound_scaled", 0)
                ratio = min(1.0, bound / max_bound)
                red = int(230 * ratio + 20)
                green = int(230 * (1.0 - ratio) + 20)
                blue = int(210 * (1.0 - ratio) + 30)
                color = f"#{red:02x}{green:02x}{blue:02x}"
                canvas.create_rectangle(24, y, 360, y + 28, fill=color, outline="#777")
                canvas.create_text(32, y + 14, anchor=tk.W, text=f"{pool_id} [{kind}]")
                canvas.create_text(372, y + 14, anchor=tk.W, text=f"bound={bound}")
                contributions = entry.get("contributions", [])
                label = "  ".join(
                    f"{item['rule_id']}:{item['component']}={item['share_bp']}bp"
                    for item in contributions
                )
                canvas.create_text(520, y + 14, anchor=tk.W, text=label)
                y += 38
            comparisons = report.get("watermark_comparisons", [])
            y += 10
            canvas.create_text(24, y, anchor=tk.W, text="Stable tie-break comparisons")
            y += 24
            for comparison in comparisons:
                text = (
                    f"{comparison['rule_id']}: stable={comparison['stable_watermark_scaled']} "
                    f"gray={comparison['gray_watermark_scaled']} "
                    f"delta={comparison['delta_scaled']} "
                    f"bound={comparison['combined_error_upper_bound_scaled']} "
                    f"winner={comparison['winner_by_stable_tiebreak']} "
                    f"error_affected={comparison['within_error_bound']}"
                )
                canvas.create_text(32, y, anchor=tk.W, text=text)
                y += 22

        def _fill_config(self) -> None:
            self.config_text.delete("1.0", tk.END)
            batch = self.data.get("latest_batch") or {}
            current = self.data.get("current_view") or {}
            payload = {
                "config_version": batch.get("config_version"),
                "stale_versions": current.get("stale_versions", []),
                "version_history": current.get("version_history", []),
                "config": current.get("config") or batch.get("config"),
                "pools": (batch.get("runtime") or {}).get("pools", []),
                "rules": (current.get("config") or {}).get("rules", []),
                "replacement_reports": current.get("replacement_reports", []),
            }
            self.config_text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

        def _fill_ledger(self) -> None:
            self.ledger_tree.delete(*self.ledger_tree.get_children())
            for batch in sorted(
                self.data.get("ledger", {}).get("batches_table", []),
                key=lambda item: (item["start_cursor"], item["batch_id"]),
            ):
                self.ledger_tree.insert(
                    "",
                    tk.END,
                    values=(
                        batch["batch_id"],
                        batch["config_version"],
                        batch["ledger_format_version"],
                        batch["start_cursor"],
                        batch["end_cursor"],
                        batch["content_digest"][:16],
                        batch["epoch_status"],
                    ),
                )

    try:
        app = DiagnosticsApp()
    except Exception as exc:  # pragma: no cover - GUI environment specific
        print(f"Unable to create Tkinter diagnostics window: {exc}")
        return 2
    try:
        app.mainloop()
    except Exception as exc:  # pragma: no cover - GUI environment specific
        try:
            messagebox.showerror("Diagnostics error", str(exc))
        finally:
            app.destroy()
        return 2
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="state/arbiter.sqlite")
    parser.add_argument("--snapshots", default="snapshots")
    args = parser.parse_args()
    raise SystemExit(launch(args.ledger, args.snapshots))
