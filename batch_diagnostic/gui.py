"""Tkinter diagnostic dashboard (headless-safe import).

All data preparation lives in pure functions; Tkinter is imported lazily inside
:func:`launch_gui`, so environments without a display can still import the
package and call the diagnostic builders directly.
"""

from __future__ import annotations

import json

from .invalidation import propagate
from .planning import build_plan
from .receipt import decode_receipt
from .spec import load_spec

GUI_TITLE = "Batch Attribution Diagnostics"


def build_overview(spec) -> list[tuple[str, str]]:
    return [
        ("requests", str(len(spec.requests))),
        ("workers", str(len(spec.workers))),
        ("input_dim", str(spec.input_dim())),
        ("global_max_in_flight", str(spec.budget.global_max_in_flight)),
        ("worker_set_version", spec.worker_set_fingerprint()[:16]),
        ("batch_size_version", spec.batch_size_fingerprint()[:16]),
        ("content_digest", spec.content_fingerprint()[:16]),
    ]


def build_plan_rows(spec) -> list[dict]:
    plan = build_plan(spec)
    return [batch.to_dict() for batch in plan.batches]


def build_normalization_rows(receipt_dicts: list[dict], known_ids: set[str]) -> list[dict]:
    rows = []
    for raw in receipt_dicts:
        normalized = decode_receipt(raw, known_request_ids=known_ids)
        rows.append(normalized.to_dict())
    return rows


def build_invalidation_rows(old_spec, new_spec) -> dict:
    plan = build_plan(new_spec)
    report = propagate(old_spec, new_spec, plan)
    return report.to_dict()


def launch_gui(spec_path_map: dict[str, str] | None = None) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tkinter is unavailable in this environment; the package remains fully "
            "usable via its CLI and library API. Original error: " + str(exc)
        ) from exc

    spec_path_map = spec_path_map or {}
    root = tk.Tk()
    root.title(GUI_TITLE)
    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    overview_tab = ttk.Frame(notebook)
    plan_tab = ttk.Frame(notebook)
    notebook.add(overview_tab, text="Overview")
    notebook.add(plan_tab, text="Batch Plan")

    if spec_path_map:
        spec = load_spec(
            spec_path_map["requests"],
            spec_path_map["workers"],
            spec_path_map["budget"],
        )
        for row_index, (key, value) in enumerate(build_overview(spec)):
            ttk.Label(overview_tab, text=key, width=24).grid(
                row=row_index, column=0, sticky="w", padx=6, pady=2
            )
            ttk.Label(overview_tab, text=value, foreground="#0a6").grid(
                row=row_index, column=1, sticky="w", padx=6, pady=2
            )
        columns = ("batch_id", "worker_id", "batch_index", "remainder", "request_ids")
        tree = ttk.Treeview(plan_tab, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
        for batch in build_plan_rows(spec):
            tree.insert(
                "",
                "end",
                values=(
                    batch["batch_id"],
                    batch["worker_id"],
                    batch["batch_index"],
                    batch["remainder"],
                    ",".join(batch["request_ids"]),
                ),
            )
        tree.grid(row=0, column=0, sticky="nsew")

    status = ttk.Label(root, text="loopback diagnostic session ready")
    status.grid(row=1, column=0, sticky="w", padx=8, pady=4)
    root.mainloop()
    return 0


def render_overview_json(spec) -> str:
    return json.dumps(
        {
            "overview": dict(build_overview(spec)),
            "plan": build_plan_rows(spec),
        },
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(launch_gui())
