from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .viewmodel import build_state_view_model

try:  # pragma: no cover - exercised on Windows desktop
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TK_AVAILABLE = True
    TK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - report any native Tk load failure
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    TK_AVAILABLE = False
    TK_IMPORT_ERROR = exc


class GuiUnavailableError(DiagnosticError):
    code = "E_GUI"


def _bool_text(value: Any) -> str:
    if value in (1, True, "1"):
        return "yes"
    if value in (0, False, "0"):
        return "no"
    return str(value)


if TK_AVAILABLE:  # pragma: no cover - GUI code is intended for Windows

    class DiagnosticTkApp:
        def __init__(
            self,
            root: "tk.Tk",
            *,
            state_dir: str | Path | None = None,
            report_path: str | Path | None = None,
        ) -> None:
            self.root = root
            self.state_dir = Path(state_dir or "state/demo")
            self.report_path = Path(report_path) if report_path else None
            self.view_model: dict[str, Any] = {}
            root.title("Columnar Page WAL Recovery Diagnostics")
            root.geometry("1280x820")

            self.summary_var = tk.StringVar(value="No data loaded")
            ttk.Label(root, textvariable=self.summary_var, padding=8).pack(anchor="w")

            toolbar = ttk.Frame(root, padding=(8, 0))
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="Reload State", command=self.reload_state).pack(side="left")
            ttk.Button(toolbar, text="Open Report JSON", command=self.choose_report).pack(
                side="left", padx=6
            )
            ttk.Button(toolbar, text="Open Ledger Dir", command=self.choose_state_dir).pack(
                side="left"
            )

            self.notebook = ttk.Notebook(root)
            self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
            self.trees: dict[str, ttk.Treeview] = {}
            self._build_pages_tab()
            self._build_wal_tab()
            self._build_timeline_tab()
            self._build_budget_tab()
            self._build_generations_tab()
            self._build_checkpoint_tab()
            self._build_crash_tab()
            self._build_ledger_tab()

            self.detail_text = tk.Text(root, height=8, wrap="none")
            self.detail_text.pack(fill="x", padx=8, pady=(0, 8))

            if self.report_path:
                self.load_report(self.report_path)
            else:
                self.reload_state()

        def _make_tree_tab(
            self, key: str, title: str, columns: list[tuple[str, str, int]]
        ) -> "ttk.Treeview":
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            tree = ttk.Treeview(
                frame,
                columns=[column[0] for column in columns],
                show="headings",
            )
            for column_id, heading, width in columns:
                tree.heading(column_id, text=heading)
                tree.column(column_id, width=width, anchor="w")
            y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            self.trees[key] = tree
            return tree

        def _build_pages_tab(self) -> None:
            tree = self._make_tree_tab(
                "pages",
                "Pages / Versions",
                [
                    ("page_id", "Page", 110),
                    ("version", "Version", 80),
                    ("lsn", "LSN", 70),
                    ("visible", "Visible", 80),
                    ("after_crash", "After crash", 90),
                    ("row_count", "Rows", 70),
                    ("column_count", "Cols", 60),
                    ("digest", "Content digest", 180),
                ],
            )
            tree.bind("<<TreeviewSelect>>", lambda _event: self.show_selected_digest("pages"))

        def _build_wal_tab(self) -> None:
            self._make_tree_tab(
                "wal",
                "WAL Ranges",
                [
                    ("lsn", "LSN", 70),
                    ("tx_id", "Tx", 90),
                    ("page_id", "Page", 100),
                    ("range", "Row range", 100),
                    ("seed", "Seed", 90),
                    ("commit_flag", "Commit", 80),
                    ("persist_flag", "Persist", 80),
                ],
            )

        def _build_timeline_tab(self) -> None:
            self._make_tree_tab(
                "timeline",
                "Schedule Timeline",
                [
                    ("lsn", "LSN", 70),
                    ("tx_id", "Tx", 90),
                    ("page_id", "Page", 100),
                    ("status", "Status", 90),
                    ("global", "Global active", 100),
                    ("tx_active", "Tx active", 90),
                    ("wait_ms", "Wait ms", 90),
                    ("run_ms", "Run ms", 90),
                    ("thread", "Worker", 180),
                ],
            )

        def _build_budget_tab(self) -> None:
            self._make_tree_tab(
                "budget",
                "Budget Occupancy",
                [
                    ("page_id", "Page", 120),
                    ("used", "Replayed bytes", 130),
                    ("limit", "Page byte budget", 140),
                    ("ratio", "Used %", 90),
                    ("last_batch_id", "Last batch", 90),
                ],
            )

        def _build_generations_tab(self) -> None:
            self._make_tree_tab(
                "generations",
                "Ledger Generations / Stale",
                [
                    ("generation", "Generation", 110),
                    ("current", "Current", 90),
                    ("stale", "Stale", 80),
                    ("slots", "Global slots", 100),
                    ("tx_quota", "Tx quota", 90),
                    ("checkpoint_interval", "CP interval", 110),
                    ("spec_hash", "Spec hash", 220),
                ],
            )

        def _build_checkpoint_tab(self) -> None:
            tree = self._make_tree_tab(
                "checkpoints",
                "Checkpoints",
                [
                    ("checkpoint_no", "#", 60),
                    ("at_lsn", "Applied LSN", 110),
                    ("digest", "Aggregate digest", 220),
                    ("stale", "Stale", 70),
                ],
            )
            tree.bind("<<TreeviewSelect>>", lambda _event: self.show_checkpoint_pages())
            self.checkpoint_pages_tree = self._make_tree_tab(
                "checkpoint_pages",
                "Checkpoint Page Digests",
                [
                    ("page_id", "Page", 120),
                    ("version", "Version", 80),
                    ("lsn", "LSN", 70),
                    ("content_digest", "Content digest", 260),
                ],
            )

        def _build_crash_tab(self) -> None:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text="Crash / Invisible")
            self.crash_text = tk.Text(frame, height=12, wrap="word")
            self.crash_text.pack(fill="x", padx=8, pady=8)
            invisible_frame = ttk.Frame(frame)
            invisible_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.trees["invisible"] = ttk.Treeview(
                invisible_frame,
                columns=("page_id", "version", "lsn", "digest"),
                show="headings",
            )
            for column_id, heading, width in (
                ("page_id", "Page", 140),
                ("version", "Version", 100),
                ("lsn", "LSN", 100),
                ("digest", "Digest", 360),
            ):
                self.trees["invisible"].heading(column_id, text=heading)
                self.trees["invisible"].column(column_id, width=width)
            self.trees["invisible"].pack(fill="both", expand=True)

        def _build_ledger_tab(self) -> None:
            pane = ttk.PanedWindow(self.notebook, orient="horizontal")
            self.notebook.add(pane, text="Ledger Versions / Snapshots")
            revisions_frame = ttk.Frame(pane)
            snapshots_frame = ttk.Frame(pane)
            pane.add(revisions_frame, weight=1)
            pane.add(snapshots_frame, weight=1)
            ttk.Label(revisions_frame, text="Ledger revisions").pack(anchor="w")
            self.trees["revisions"] = ttk.Treeview(
                revisions_frame,
                columns=("revision_no", "round", "cursor", "digest"),
                show="headings",
            )
            for col, text, width in (
                ("revision_no", "Rev", 60),
                ("round", "Round/attempt", 110),
                ("cursor", "Cursor LSN", 100),
                ("digest", "Aggregate", 220),
            ):
                self.trees["revisions"].heading(col, text=text)
                self.trees["revisions"].column(col, width=width)
            self.trees["revisions"].pack(fill="both", expand=True)

            ttk.Label(snapshots_frame, text="Snapshots").pack(anchor="w")
            self.trees["snapshots"] = ttk.Treeview(
                snapshots_frame,
                columns=("snapshot_id", "kind", "label", "lsn", "stale"),
                show="headings",
            )
            for col, text, width in (
                ("snapshot_id", "ID", 60),
                ("kind", "Kind", 110),
                ("label", "Label", 220),
                ("lsn", "LSN", 70),
                ("stale", "Stale", 60),
            ):
                self.trees["snapshots"].heading(col, text=text)
                self.trees["snapshots"].column(col, width=width)
            self.trees["snapshots"].pack(fill="both", expand=True)

        def choose_report(self) -> None:  # pragma: no cover - native dialog
            path = filedialog.askopenfilename(
                title="Select demo_report.json",
                filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            )
            if path:
                self.load_report(Path(path))

        def choose_state_dir(self) -> None:  # pragma: no cover - native dialog
            path = filedialog.askdirectory(title="Select state directory")
            if path:
                self.state_dir = Path(path)
                self.reload_state()

        def load_report(self, path: Path) -> None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                messagebox.showerror("Cannot load report", str(exc))
                return
            self.report_path = path
            self.view_model = payload.get("state_view") or payload
            self.render()

        def reload_state(self) -> None:
            self.view_model = build_state_view_model(self.state_dir)
            self.render()

        def _clear_tree(self, key: str) -> None:
            tree = self.trees[key]
            for item in tree.get_children():
                tree.delete(item)

        def _insert(self, key: str, values: tuple[Any, ...]) -> str:
            return self.trees[key].insert(
                "",
                "end",
                values=tuple("" if value is None else value for value in values),
            )

        def render(self) -> None:
            model = self.view_model
            if not model.get("available", False):
                self.summary_var.set(f"No ledger available under {self.state_dir}")
                return
            budget = model.get("budget", {})
            view = model.get("current_view") or {}
            stale = model.get("stale_counts", {})
            self.summary_var.set(
                f"Generation G{model['current_generation']} | "
                f"slots={budget.get('global_parallel_slots', budget.get('global_slots'))} "
                f"tx_quota={budget.get('per_transaction_replay_quota', budget.get('per_tx_quota'))} "
                f"page_budget={budget.get('single_page_replay_byte_budget', budget.get('page_byte_budget'))} "
                f"cp_interval={budget.get('checkpoint_interval_entries', budget.get('checkpoint_interval'))} | "
                f"view={view.get('view_snapshot_id', '-')} cursor={view.get('cursor_lsn', '-')} | "
                f"stale generations={stale.get('generations', 0)}, "
                f"checkpoints={stale.get('checkpoints', 0)}, views={stale.get('view_snapshots', 0)}"
            )
            self._render_pages(model)
            self._render_wal(model)
            self._render_timeline(model)
            self._render_budget(model)
            self._render_generations(model)
            self._render_checkpoints(model)
            self._render_crash(model)
            self._render_ledger(model)

        def _render_pages(self, model: dict[str, Any]) -> None:
            self._clear_tree("pages")
            for page in model.get("pages", []):
                self._insert(
                    "pages",
                    (
                        page["page_id"],
                        page["version"],
                        page["lsn"],
                        "visible" if page["visible"] else "INVISIBLE",
                        _bool_text(page["after_crash"]),
                        page.get("row_count", ""),
                        page.get("column_count", ""),
                        page["content_digest"][:20],
                    ),
                )
            self._clear_tree("invisible")
            for page in model.get("invisible_versions", []):
                self._insert(
                    "invisible",
                    (page["page_id"], page["version"], page["lsn"], page["content_digest"]),
                )

        def _render_wal(self, model: dict[str, Any]) -> None:
            self._clear_tree("wal")
            for entry in model.get("wal", []):
                self._insert(
                    "wal",
                    (
                        entry["lsn"],
                        entry["tx_id"],
                        entry["page_id"],
                        f"[{entry['row_start']},{entry['row_end']})",
                        entry["seed"],
                        _bool_text(entry["commit_flag"]),
                        _bool_text(entry["persist_flag"]),
                    ),
                )

        def _render_timeline(self, model: dict[str, Any]) -> None:
            self._clear_tree("timeline")
            for event in model.get("timeline", []):
                self._insert(
                    "timeline",
                    (
                        event["lsn"],
                        event["tx_id"],
                        event["page_id"],
                        event["status"],
                        event.get("global_active_at_start"),
                        event.get("tx_active_at_start"),
                        None if event.get("wait_ms") is None else round(event["wait_ms"], 3),
                        None if event.get("run_ms") is None else round(event["run_ms"], 3),
                        event.get("thread_name") or "",
                    ),
                )

        def _render_budget(self, model: dict[str, Any]) -> None:
            self._clear_tree("budget")
            for usage in model.get("budget_usage", []):
                limit = usage["byte_budget"]
                used = usage["bytes_consumed"]
                ratio = f"{(100.0 * used / limit):.2f}%" if limit else "-"
                self._insert(
                    "budget",
                    (
                        usage["page_id"],
                        used,
                        limit,
                        ratio,
                        usage.get("last_batch_id"),
                    ),
                )

        def _render_generations(self, model: dict[str, Any]) -> None:
            self._clear_tree("generations")
            current_generation = model.get("current_generation")
            for generation in model.get("generations", []):
                budget = generation.get("budget", {})
                generation_id = generation["generation"]
                self._insert(
                    "generations",
                    (
                        f"G{generation_id}",
                        "yes" if generation_id == current_generation else "no",
                        _bool_text(generation["stale"]),
                        budget.get("global_slots", ""),
                        budget.get("per_tx_quota", ""),
                        budget.get("checkpoint_interval", ""),
                        generation.get("spec_hash_short", ""),
                    ),
                )

        def _render_checkpoints(self, model: dict[str, Any]) -> None:
            self._clear_tree("checkpoints")
            self.checkpoint_pages_by_item: dict[str, list[dict[str, Any]]] = {}
            for checkpoint in model.get("checkpoints", []):
                item_id = self._insert(
                    "checkpoints",
                    (
                        checkpoint["checkpoint_no"],
                        checkpoint["at_lsn"],
                        checkpoint["aggregate_digest"],
                        _bool_text(checkpoint["stale"]),
                    ),
                )
                self.checkpoint_pages_by_item[item_id] = checkpoint.get("pages", [])

        def show_checkpoint_pages(self) -> None:
            selection = self.trees["checkpoints"].selection()
            self._clear_tree("checkpoint_pages")
            if not selection:
                return
            pages = getattr(self, "checkpoint_pages_by_item", {}).get(selection[0], [])
            for page in pages:
                self._insert(
                    "checkpoint_pages",
                    (
                        page["page_id"],
                        page["version"],
                        page["lsn"],
                        page["content_digest"],
                    ),
                )

        def _render_crash(self, model: dict[str, Any]) -> None:
            crash = model.get("crash")
            self.crash_text.delete("1.0", "end")
            if not crash:
                self.crash_text.insert("1.0", "No crash decision recorded.")
                return
            self.crash_text.insert(
                "1.0",
                json.dumps(crash.get("decision", crash), indent=2, ensure_ascii=False),
            )

        def _render_ledger(self, model: dict[str, Any]) -> None:
            self._clear_tree("revisions")
            for revision in model.get("revisions", []):
                self._insert(
                    "revisions",
                    (
                        revision["revision_no"],
                        f"R{revision['round_no']}.{revision['attempt_no']}",
                        revision["cursor_lsn"],
                        revision["aggregate_digest"],
                    ),
                )
            self._clear_tree("snapshots")
            for snapshot in model.get("snapshots", []):
                self._insert(
                    "snapshots",
                    (
                        snapshot["snapshot_id"],
                        snapshot["kind"],
                        snapshot["label"],
                        snapshot["at_lsn"],
                        _bool_text(snapshot["stale"]),
                    ),
                )

        def show_selected_digest(self, tree_key: str) -> None:
            tree = self.trees[tree_key]
            selection = tree.selection()
            if not selection:
                return
            values = tree.item(selection[0], "values")
            self.detail_text.delete("1.0", "end")
            self.detail_text.insert("1.0", "\n".join(str(value) for value in values))


def launch_gui(
    *,
    state_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> int:
    if not TK_AVAILABLE or tk is None:
        raise GuiUnavailableError(
            "Tkinter could not be loaded. On Windows install/reinstall Python 3.11+ "
            "with tcl/tk enabled, then run `python -m pip install numpy==1.26.4`.",
            details={"import_error": repr(TK_IMPORT_ERROR)},
        )
    root = tk.Tk()
    DiagnosticTkApp(root, state_dir=state_dir, report_path=report_path)
    root.mainloop()
    return 0
