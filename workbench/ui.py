"""Tkinter diagnostic workbench UI.

The UI is a pure rendering layer over the ledger view model: all business
logic lives in the engine/ledger modules, so the same stable, ordered data
drives both the GUI and the headless ``ui-dump`` JSON.  When Tkinter or a
display server is unavailable the error is reported with guidance instead of
a raw traceback.
"""

from __future__ import annotations

import json
from typing import Any

from .canonical import short_hexdigest
from .errors import WorkbenchError
from .ledger import Ledger


def _build_text(model: dict[str, Any]) -> str:
    if not model.get("run"):
        return "PIPELINE DIAGNOSTIC WORKBENCH\n\nNo runs recorded yet."
    run = model["run"]
    evolution = json.loads(run["evolution_json"])
    blocks: list[str] = []
    blocks.append("PIPELINE DIAGNOSTIC WORKBENCH")
    blocks.append(
        f"Run: {run['id']}    Status: {run['status']}    "
        f"Fingerprint version: {run['fingerprint_version']}")
    blocks.append(f"Fingerprint: {run['spec_fingerprint']}")
    blocks.append(f"Samples digest: {run['samples_digest']}")
    blocks.append(
        "Evolution: reasons=" + ",".join(evolution.get("reasons", [])) +
        f" | affected={','.join(evolution.get('affected_stage_ids', []))}"
        f" | invalidated_shards={evolution.get('invalidated_shards', 0)}")
    blocks.append("")
    blocks.append("STAGE DEPENDENCIES AND STATUS")
    for stage in model["stages"]:
        blocks.append(
            f"[{stage['ordinal'] + 1:02d}] {stage['stage_id']} "
            f"({stage['stage_type']}) <- {stage['depends_on']}  "
            f"status={stage['status']} rows={stage['rows_in']}->"
            f"{stage['rows_out']} digest={short_hexdigest(stage['output_digest'])}")
    blocks.append("")
    blocks.append("STAGE SUMMARIES (COLUMN STATISTICS)")
    for stage in model["stages"]:
        blocks.append(f"# {stage['stage_id']}")
        for column in sorted(stage["column_stats"]):
            stats = stage["column_stats"][column]
            blocks.append("  " + column + " " + json.dumps(
                stats, sort_keys=True, separators=(",", ":")))
    blocks.append("")
    blocks.append("SHARD DISTRIBUTION")
    if model["shards"]:
        for shard in model["shards"]:
            blocks.append(
                f"shard {shard['shard_index']:04d}: "
                f"rows=[{shard['row_start']},{shard['row_end']}) "
                f"count={shard['row_count']} "
                f"fp={short_hexdigest(shard['shard_fingerprint'])}")
    else:
        blocks.append("  (no shards)")
    blocks.append("")
    blocks.append("FINGERPRINT LEDGER")
    for fingerprint in model["fingerprints"]:
        blocks.append(
            f"v{fingerprint['version']}: {fingerprint['fingerprint']}")
    blocks.append("")
    blocks.append("BUSINESS EVENTS / LEDGER VIEW")
    if model["events"]:
        for event in model["events"]:
            blocks.append(
                f"seq={event['sequence']} {event['code']} "
                f"stage={event['stage_id'] or '-'} "
                f"path={event['field_path']} :: {event['message']}")
    else:
        blocks.append("  (no events)")
    return "\n".join(blocks)


def launch_ui(workspace: str) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise WorkbenchError(
            "E_SPEC",
            "Tkinter unavailable; use 'ui-dump' for headless diagnostics "
            f"({exc})", field_path="$.ui")
    ledger = Ledger(workspace)
    model = ledger.view_model()
    text = _build_text(model)
    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - no display server
        ledger.close()
        raise WorkbenchError(
            "E_SPEC",
            f"cannot open display; headless alternative is 'ui-dump' ({exc})",
            field_path="$.ui")
    root.title("Pipeline Diagnostic Workbench")
    root.geometry("980x720")
    frame = ttk.Frame(root, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)
    header = ttk.Label(
        frame,
        text=f"Workspace: {workspace}    Run: "
             f"{model['run']['id'] if model.get('run') else '-'}",
        font=("TkDefaultFont", 10, "bold"))
    header.pack(anchor=tk.W, pady=(0, 6))
    text_widget = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 9))
    y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                             command=text_widget.yview)
    x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL,
                             command=text_widget.xview)
    text_widget.configure(yscrollcommand=y_scroll.set,
                          xscrollcommand=x_scroll.set)
    text_widget.insert(tk.END, text)
    text_widget.configure(state=tk.DISABLED)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    root.mainloop()
    ledger.close()
