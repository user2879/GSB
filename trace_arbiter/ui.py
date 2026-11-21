from __future__ import annotations

import json

from .engine import view_current
from .errors import ArbiterError
from .ledger import LedgerDB


def run_gui(db: LedgerDB) -> None:
    """Show the cached ledger view; no authoritative service is contacted."""

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # pragma: no cover - Windows standard installs Tk
        raise ArbiterError(
            "E_UI",
            "Tkinter is unavailable; use `view --json` for the headless cache view",
        ) from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise ArbiterError("E_UI", f"cannot open Tkinter display: {exc}") from exc
    root.title("Trace Arbitration Ledger - Cached Quick View")
    root.geometry("1180x760")

    header = tk.StringVar()
    replacement = tk.StringVar()
    status = tk.StringVar(value="Cached ledger view. Run `review` for authoritative verification.")
    ttk.Label(root, textvariable=header, font=("Segoe UI", 14, "bold")).pack(
        anchor="w", padx=12, pady=(10, 2)
    )
    ttk.Label(root, textvariable=replacement).pack(anchor="w", padx=12)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=8)

    traces_tab = ttk.Frame(notebook)
    changes_tab = ttk.Frame(notebook)
    stats_tab = ttk.Frame(notebook)
    notebook.add(traces_tab, text="Traces")
    notebook.add(changes_tab, text="Replacement / Flips")
    notebook.add(stats_tab, text="Fixed-point Latency")

    trace_columns = ("trace", "root", "sampled_nodes", "rules", "entry_digest_valid")
    trace_tree = ttk.Treeview(traces_tab, columns=trace_columns, show="headings", height=18)
    headings = {
        "trace": "Trace ID",
        "root": "Root Span",
        "sampled_nodes": "Sampled Nodes",
        "rules": "Rule Versions",
        "entry_digest_valid": "Digest Valid",
    }
    widths = {
        "trace": 210,
        "root": 190,
        "sampled_nodes": 260,
        "rules": 150,
        "entry_digest_valid": 100,
    }
    for column in trace_columns:
        trace_tree.heading(column, text=headings[column])
        trace_tree.column(column, width=widths[column], anchor="w")
    trace_scroll = ttk.Scrollbar(traces_tab, orient="vertical", command=trace_tree.yview)
    trace_tree.configure(yscrollcommand=trace_scroll.set)
    trace_tree.pack(side="left", fill="both", expand=True)
    trace_scroll.pack(side="right", fill="y")

    change_columns = ("trace", "kind", "before_rules", "after_rules", "diff")
    change_tree = ttk.Treeview(changes_tab, columns=change_columns, show="headings")
    change_headings = {
        "trace": "Trace ID",
        "kind": "Change",
        "before_rules": "Before Rules",
        "after_rules": "After Rules",
        "diff": "Diff Fields",
    }
    for column in change_columns:
        change_tree.heading(column, text=change_headings[column])
        change_tree.column(column, width=230 if column != "diff" else 300, anchor="w")
    change_tree.pack(fill="both", expand=True)

    stats_text = tk.Text(stats_tab, wrap="none", font=("Consolas", 10))
    stats_text.pack(fill="both", expand=True)

    def refresh() -> None:
        try:
            view = view_current(db)
        except ArbiterError as exc:
            status.set(f"{exc.code}: {exc.message}")
            return
        header.set(
            f"Ruleset {view['current_ruleset_version']} | snapshot {view['snapshot_id']} | "
            f"{len(view['traces'])} traces | query {view['query_elapsed_us']} us"
        )
        replacement.set(
            f"Whole-view replacement. Previous ruleset: {view['previous_ruleset_version']}; "
            f"stale sources excluded: {view['stale_sources_excluded']}; "
            f"changed traces: {len(view['changed_traces'])}; "
            f"snapshot digest valid: {view['snapshot_digest_valid']}"
        )
        trace_tree.delete(*trace_tree.get_children())
        for row in sorted(view["traces"], key=lambda item: item["trace_id"]):
            entry = row["entry"]
            trace_tree.insert(
                "",
                "end",
                values=(
                    row["trace_id"],
                    entry["root_span_id"],
                    ", ".join(entry["sampled_node_ids"]) or "-",
                    ", ".join(str(item) for item in row["rule_versions"]),
                    "yes" if row["entry_digest_valid"] else "NO",
                ),
            )
        change_tree.delete(*change_tree.get_children())
        for row in view["changed_traces"]:
            change_tree.insert(
                "",
                "end",
                values=(
                    row["trace_id"],
                    row["change_kind"],
                    json.dumps(row["before_rule_versions"], ensure_ascii=False),
                    json.dumps(row["after_rule_versions"], ensure_ascii=False),
                    ", ".join(row["diff_fields"]),
                ),
            )
        stats_text.delete("1.0", "end")
        stats_text.insert(
            "end",
            json.dumps(view["latency_stats"], ensure_ascii=False, indent=2, sort_keys=True),
        )
        status.set(
            "Cached ledger view only. Summaries shown here are not the authoritative source; "
            "run `python -m trace_arbiter review`."
        )

    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=12, pady=(0, 4))
    ttk.Button(controls, text="Refresh cached view", command=refresh).pack(side="left")
    ttk.Label(controls, textvariable=status).pack(side="left", padx=12)
    refresh()
    root.mainloop()
