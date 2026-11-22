"""Tkinter dashboard for page outcomes, slots, estimates, and ledger metadata."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from .errors import EReplay
from .ledger import Ledger
from .logging_utils import default_log_path


def last_business_code(log_path: str | Path | None = None) -> str:
    target = Path(log_path) if log_path else default_log_path()
    if not target.exists():
        return "NONE"
    last_code = "NONE"
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "business_error" and record.get("code"):
            last_code = str(record["code"])
    return last_code


def load_dashboard_data(db_path: str | Path) -> dict[str, Any]:
    with Ledger(db_path) as ledger:
        snapshot = ledger.get_active_snapshot()
        if snapshot is None:
            raise EReplay("ledger has no active snapshot", field="snapshot")
        snapshot_id = int(snapshot["snapshot_id"])
        result = json.loads(snapshot["final_result_json"]) if snapshot.get("final_result_json") else {"groups": []}
        return {
            "snapshot": dict(snapshot),
            "route_decision": json.loads(snapshot["route_decision_json"]),
            "page_outcomes": [json.loads(row["payload_json"]) for row in ledger.read_page_outcomes(snapshot_id)],
            "evictions": [json.loads(row["payload_json"]) for row in ledger.read_evictions(snapshot_id)],
            "rescans": [json.loads(row["payload_json"]) for row in ledger.read_rescans(snapshot_id)],
            "result": result,
            "ledger": ledger.ledger_summary(),
            "unknown_attempts": ledger.unknown_attempts(),
            "integrity": ledger.sqlite_integrity_check(),
        }


class Dashboard:
    PALETTE = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3", "#ede9fe", "#cffafe", "#fee2e2", "#e5e7eb"]

    def __init__(self, root: tk.Tk, data: dict[str, Any], *, error_code: str | None = None) -> None:
        self.root = root
        self.data = data
        self.error_code = error_code or "NONE"
        self.root.title("PageAgg Ledger Dashboard")
        self.root.geometry("1180x760")
        self.style = ttk.Style(root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self._build_header()
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._build_outcomes_tab(notebook)
        self._build_slots_tab(notebook)
        self._build_groups_tab(notebook)
        self._build_route_tab(notebook)
        self._build_events_tab(notebook)

    def _build_header(self) -> None:
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="x")
        snapshot = self.data["snapshot"]
        ledger = self.data["ledger"]
        items = [
            ("Plan version", str(snapshot["plan_version"])),
            ("Route", str(snapshot["route"])),
            ("Ledger format", str(ledger["format_version"])),
            ("Status", str(snapshot["status"])),
            ("Cursor", str(snapshot["cursor"])),
            ("Error code", self.error_code),
        ]
        for index, (label, value) in enumerate(items):
            ttk.Label(frame, text=label, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=index * 2, sticky="w", padx=(0, 3)
            )
            ttk.Label(frame, text=value).grid(row=0, column=index * 2 + 1, sticky="w", padx=(0, 16))
        ttk.Label(frame, text="Catalog digest").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(frame, text=snapshot["catalog_digest"]).grid(
            row=1, column=1, columnspan=7, sticky="w", pady=(6, 0)
        )
        ttk.Label(frame, text="Ledger digest").grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Label(frame, text=str(ledger["content_digest"])).grid(
            row=2, column=1, columnspan=7, sticky="w", pady=(2, 0)
        )

    def _tree(self, parent: ttk.Frame, columns: list[tuple[str, str, int]], *, height: int = 16) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        names = [name for name, _, _ in columns]
        tree = ttk.Treeview(frame, columns=names, show="headings", height=height)
        for name, heading, width in columns:
            tree.heading(name, text=heading)
            tree.column(name, width=width, anchor="w", stretch=True)
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _build_outcomes_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Page outcomes")
        tree = self._tree(
            tab,
            [
                ("access_seq", "Access", 60),
                ("page_id", "Page", 60),
                ("outcome", "Outcome", 120),
                ("cache_hit", "Hit", 50),
                ("physical_scan", "Scan", 50),
                ("slot", "Slot", 50),
                ("canonical_hit", "Plan hit", 65),
                ("canonical_slot", "Plan slot", 70),
                ("evicted", "Evicted page", 90),
                ("occupancy", "Slot occupancy after", 300),
                ("kind", "Occupancy kind", 120),
            ],
        )
        for row in self.data["page_outcomes"]:
            tree.insert(
                "",
                "end",
                values=(
                    row.get("access_seq"),
                    row.get("page_id"),
                    row.get("outcome"),
                    row.get("cache_hit"),
                    row.get("physical_scan"),
                    row.get("slot"),
                    row.get("canonical_hit"),
                    row.get("canonical_slot"),
                    row.get("evicted_page_id"),
                    row.get("slot_occupancy_after"),
                    row.get("occupancy_kind"),
                ),
            )

    def _page_color(self, page_id: int | None) -> str:
        if page_id is None:
            return "#ffffff"
        return self.PALETTE[page_id % len(self.PALETTE)]

    def _build_slots_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Slot timeline")
        label = (
            "Actual resident slots"
            if self.data["snapshot"]["route"] == "RESIDENT"
            else "Shadow plan slots (RESCAN keeps no physical residency)"
        )
        ttk.Label(tab, text=label, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
        canvas = tk.Canvas(tab, background="#f8fafc", highlightthickness=0)
        xscroll = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
        yscroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        cell_w, cell_h = 96, 27
        outcomes = self.data["page_outcomes"]
        max_slots = max((len(row.get("slot_occupancy_after") or []) for row in outcomes), default=1)
        canvas.configure(scrollregion=(0, 0, 130 + max_slots * cell_w, 28 + len(outcomes) * cell_h))
        canvas.create_text(8, 14, text="access / page", anchor="w", font=("TkDefaultFont", 9, "bold"))
        for slot in range(max_slots):
            canvas.create_text(130 + slot * cell_w + cell_w // 2, 14, text=f"slot {slot}")
        for row_index, row in enumerate(outcomes):
            y = 28 + row_index * cell_h
            canvas.create_text(
                8,
                y + cell_h // 2,
                anchor="w",
                text=f"#{row['access_seq']}  p{row['page_id']}  {row['outcome']}",
            )
            for slot, page_id in enumerate(row.get("slot_occupancy_after") or []):
                x = 130 + slot * cell_w
                color = self._page_color(page_id)
                canvas.create_rectangle(x, y, x + cell_w - 3, y + cell_h - 2, fill=color, outline="#94a3b8")
                canvas.create_text(
                    x + (cell_w - 3) // 2,
                    y + (cell_h - 2) // 2,
                    text="-" if page_id is None else f"page {page_id}",
                    fill="#0f172a",
                )

    def _group_metric_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for group in self.data["result"].get("groups", []):
            for metric in group.get("metrics", []):
                rows.append(
                    {
                        "group_key": group["group_key"],
                        "count_rows": group["count_rows"],
                        **metric,
                    }
                )
        return rows

    def _build_groups_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Estimates & error heatmap")
        paned = ttk.Panedwindow(tab, orient="vertical")
        paned.pack(fill="both", expand=True)
        table_frame = ttk.Frame(paned)
        heat_frame = ttk.Frame(paned)
        paned.add(table_frame, weight=3)
        paned.add(heat_frame, weight=2)
        tree = self._tree(
            table_frame,
            [
                ("group", "Group", 60),
                ("rows", "Rows", 60),
                ("column", "Column", 100),
                ("type", "Type", 55),
                ("values", "Values", 65),
                ("nulls", "Nulls", 55),
                ("sum", "Sum decimal", 145),
                ("mean", "Mean decimal", 145),
                ("sum_error", "Sum error µ", 85),
                ("mean_error", "Mean error µ", 85),
            ],
            height=9,
        )
        for row in self._group_metric_rows():
            quant = row["quantization"]
            tree.insert(
                "",
                "end",
                values=(
                    row["group_key"],
                    row["count_rows"],
                    row["column"],
                    row["column_type"],
                    row["value_count"],
                    row["null_count"],
                    row["sum_decimal"],
                    row["mean_decimal"],
                    quant["sum_error_bound_units"],
                    quant["mean_error_bound_units"],
                ),
            )
        ttk.Label(heat_frame, text="Quantization error upper bound heatmap (scaled µ-units)").pack(anchor="w")
        canvas = tk.Canvas(heat_frame, height=210, background="#f8fafc", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        rows = self._group_metric_rows()
        groups = sorted({row["group_key"] for row in rows})
        columns = sorted({row["column"] for row in rows})
        if not groups or not columns:
            return
        lookup = {(row["group_key"], row["column"]): row for row in rows}
        max_error = max(
            (row["quantization"]["mean_error_bound_units"] for row in rows),
            default=0,
        ) or 1
        left, top, cell_w, cell_h = 90, 28, 105, 30
        for column_index, column in enumerate(columns):
            canvas.create_text(left + column_index * cell_w + cell_w // 2, 14, text=column)
        for row_index, group in enumerate(groups):
            canvas.create_text(45, top + row_index * cell_h + cell_h // 2, text=f"group {group}")
            for column_index, column in enumerate(columns):
                metric = lookup.get((group, column))
                if metric is None:
                    continue
                value = metric["quantization"]["mean_error_bound_units"]
                ratio = min(1.0, value / max_error)
                red = 235
                green = int(245 - 145 * ratio)
                blue = int(225 - 185 * ratio)
                color = f"#{red:02x}{green:02x}{blue:02x}"
                x = left + column_index * cell_w
                y = top + row_index * cell_h
                canvas.create_rectangle(x, y, x + cell_w - 4, y + cell_h - 4, fill=color, outline="#cbd5e1")
                canvas.create_text(
                    x + (cell_w - 4) // 2,
                    y + (cell_h - 4) // 2,
                    text=f"{value} µ",
                    fill="#7f1d1d" if ratio > 0.5 else "#14532d",
                )

    def _build_route_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text="Route decision")
        decision = self.data["route_decision"]
        text = tk.Text(tab, wrap="word", font=("TkFixedFont", 10), background="#f8fafc")
        scroll = ttk.Scrollbar(tab, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        text.insert("1.0", json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
        text.configure(state="disabled")

    def _build_events_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Ledger events")
        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True)
        tree = self._tree(
            frame,
            [
                ("kind", "Kind", 90),
                ("seq", "Event", 65),
                ("access", "Access", 65),
                ("page", "Page/Victim", 90),
                ("incoming", "Incoming", 90),
                ("slot", "Slot", 50),
                ("reason", "Reason", 260),
            ],
            height=12,
        )
        for event in self.data["evictions"]:
            tree.insert(
                "",
                "end",
                values=(
                    "EVICTION",
                    event["event_seq"],
                    event["access_seq"],
                    event["victim_page_id"],
                    event["incoming_page_id"],
                    event["slot"],
                    event["reason"],
                ),
            )
        for event in self.data["rescans"]:
            tree.insert(
                "",
                "end",
                values=(
                    "RESCAN",
                    event["event_seq"],
                    event["access_seq"],
                    event["page_id"],
                    "",
                    "",
                    event["reason"],
                ),
            )
        status = ttk.Label(
            tab,
            text=(
                f"SQLite integrity: {self.data['integrity']}    "
                f"Unknown batch attempts: {len(self.data['unknown_attempts'])}    "
                f"Plan versions: {self.data['ledger']['plan_version_counter']}"
            ),
        )
        status.pack(anchor="w", pady=(6, 0))


def launch_ui(
    db_path: str | Path,
    *,
    catalog_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> None:
    del catalog_path  # The ledger already records the catalog path and digest.
    data = load_dashboard_data(db_path)
    root = tk.Tk()
    Dashboard(root, data, error_code=last_business_code(log_path))
    root.mainloop()
