#!/usr/bin/env python3
"""Tkinter viewer for an aggwin run directory.

Shows: query list with canonical identifiers, per-group aggregation
state, per-partition window contents, processed cursors and the snapshot
list.  Reads only local files and the local SQLite catalog.

Usage: python gui.py --run-dir runs/run1
"""

import argparse
import json
import os
import sqlite3


def build_model(run_dir):
    """Pure data model for the UI (headless-testable)."""
    model = {"run_dir": run_dir, "queries": [], "aggs": [], "windows": [],
             "cursors": [], "snapshots": []}
    with open(os.path.join(run_dir, "results.json"),
              encoding="utf-8") as fh:
        results = json.load(fh)
    with open(os.path.join(run_dir, "state_dump.json"),
              encoding="utf-8") as fh:
        state = json.load(fh)

    canonical = results.get("canonical_ids", {})
    for qid, canonical_id in sorted(canonical.items()):
        kind = "aggregate" if qid in results.get("aggs", {}) else "window"
        model["queries"].append({
            "query_id": qid, "canonical_id": canonical_id, "kind": kind,
            "param_version": results.get("param_version")})

    for qid, groups in sorted(results.get("aggs", {}).items()):
        for key, body in sorted(groups.items()):
            model["aggs"].append({
                "query": qid, "group_key": key, "count": body["count"],
                "sum": body["sum"], "min": body["min"], "max": body["max"],
                "mean": body.get("mean"),
                "rows_processed": body["rows_processed"],
                "group_version": body["group_version"]})

    for qid, parts in sorted(state.get("windows", {}).items()):
        for key, part in sorted(parts.items()):
            model["windows"].append({
                "query": qid, "partition_key": key,
                "ring": part["ring"]["items"],
                "pending": len(part["pending"]),
                "emitted": len(part["results"]),
                "late": part["late"], "dropped": part["dropped"],
                "watermark": part["watermark"]})

    for table, cursor in sorted(state.get("cursors", {}).items()):
        applied = state.get("applied", {}).get(table, [])
        model["cursors"].append({
            "table": table, "cursor": cursor,
            "applied_rows": sum(end - start for start, end in applied)})

    catalog_path = os.path.join(run_dir, "catalog.db")
    if os.path.exists(catalog_path):
        conn = sqlite3.connect(catalog_path)
        try:
            for row in conn.execute(
                    "SELECT snapshot_id, path, query_param_version,"
                    " format_version, digest, stale FROM snapshots"
                    " ORDER BY created_at"):
                model["snapshots"].append({
                    "snapshot_id": row[0], "path": row[1],
                    "param_version": row[2], "format_version": row[3],
                    "digest": row[4], "stale": bool(row[5])})
        finally:
            conn.close()
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description="aggwin run viewer")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    model = build_model(args.run_dir)

    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(f"aggwin viewer - {os.path.abspath(args.run_dir)}")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    def add_tree(title, columns, rows):
        frame = ttk.Frame(notebook)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor="w")
        for row in rows:
            tree.insert("", "end", values=[row.get(c) for c in columns])
        tree.pack(fill="both", expand=True)
        notebook.add(frame, text=title)
        return tree

    add_tree("Queries", ["query_id", "canonical_id", "kind",
                         "param_version"], model["queries"])
    add_tree("Aggregations", ["query", "group_key", "count", "sum", "min",
                              "max", "mean", "rows_processed",
                              "group_version"], model["aggs"])
    add_tree("Windows", ["query", "partition_key", "ring", "pending",
                         "emitted", "late", "dropped", "watermark"],
             model["windows"])
    add_tree("Cursors", ["table", "cursor", "applied_rows"],
             model["cursors"])
    add_tree("Snapshots", ["snapshot_id", "param_version", "format_version",
                           "digest", "stale", "path"], model["snapshots"])
    root.mainloop()


if __name__ == "__main__":
    main()
