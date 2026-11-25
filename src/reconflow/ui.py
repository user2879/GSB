from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .incremental import read_fast_once
from .ledger import Ledger
from .workspace import Workspace


def collect_diagnostic_state(workspace: Workspace) -> dict[str, Any]:
    """Read-only diagnostic projection used by Tk and headless smoke output."""

    read = read_fast_once(workspace)
    view = read["view"]
    conclusion = view["conclusion"]
    ledger = Ledger(workspace.ledger_path)
    ledger_state: dict[str, Any] = {"available": workspace.ledger_path.exists()}
    if ledger_state["available"]:
        ledger_state["user_version"] = ledger.user_version()
        ledger_state["pending_batches"] = ledger.pending_batches()
        ledger_state["recovery_events"] = ledger.recovery_events()[-20:]
        ledger_state["snapshots"] = ledger.list_snapshots()
        ledger_state["ui_reads"] = ledger.ui_reads(10)
    return {
        "read": {key: value for key, value in read.items() if key != "view"},
        "config_version": view.get("config_version"),
        "cache_digest": view.get("cache_digest"),
        "conclusion": conclusion,
        "last_invalidation": view.get("last_invalidation"),
        "ledger": ledger_state,
    }


def headless_diagnostic_text(workspace: Workspace) -> str:
    state = collect_diagnostic_state(workspace)
    conclusion = state["conclusion"]
    lines = [
        "ReconFlow Diagnostic View",
        f"config_version={state['config_version']} digest={state['cache_digest']}",
        f"read_elapsed_ms={state['read']['elapsed_ms']} bytes={state['read']['bytes_read']}",
        "",
        "[weighted_sampling]",
    ]
    for domain in sorted(conclusion["weighted_sampling"]["domains"]):
        item = conclusion["weighted_sampling"]["domains"][domain]
        lines.append(f"  {domain}: quota={item['quota_points']} selected={item['selected_count']} saturated={item['quota_saturated']}")
    lines.append("[dedup_shards]")
    for shard in sorted(conclusion["shards"], key=lambda item: item["shard_id"]):
        lines.append(
            f"  {shard['shard_id']}: docs={shard['doc_count']} kept={shard['kept_count']} dropped={shard['dropped_count']} domains={','.join(shard['domains'])}"
        )
    lines.append("[ratio]")
    for item in sorted(conclusion["ratio_normalization"]["domains"], key=lambda row: row["domain_id"]):
        lines.append(f"  {item['domain_id']}: actual={item['actual_bp']} target={item['target_bp']} deviation={item['deviation_bp']}bp")
    error = conclusion["error_attribution"]
    lines.append("[error_attribution]")
    for stage in error["stages"]:
        lines.append(f"  {stage['stage']}: bound_u={stage['bound_micro_bp']} share={stage['share_permille']}/1000")
    lines.append(f"  dominant={error['dominant_stage']} threshold={error['dominance_threshold_permille']}/1000 above={error['dominant_above_threshold']}")
    propagation = state.get("last_invalidation")
    lines.append("[invalidation]")
    if propagation:
        lines.append(f"  shards={','.join(propagation['affected_shards'])}")
        lines.append(f"  ratio={','.join(propagation['affected_ratio_domains'])}")
        for step in propagation["steps"]:
            lines.append(f"  {step['seq']:03d} {step['from']} -> {step['to']} via {step['edge']}")
    else:
        lines.append("  <initial cache: no incremental edit trace>")
    ledger = state["ledger"]
    lines.append("[ledger]")
    if ledger.get("available"):
        lines.append(f"  format_version={ledger.get('user_version')} unknown_or_pending={len(ledger.get('pending_batches', []))}")
        for batch in ledger.get("pending_batches", []):
            lines.append(f"  UNKNOWN {batch['batch_id']} status={batch['status']}")
        lines.append(f"  snapshots={len(ledger.get('snapshots', []))}")
        for snapshot in ledger.get("snapshots", []):
            lines.append(f"  {snapshot['snapshot_id']} sha256={snapshot['file_sha256'][:16]}")
    else:
        lines.append("  ledger not created")
    return "\n".join(lines) + "\n"


class DiagnosticApp:
    def __init__(self, root: Any, workspace: Workspace) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.workspace = workspace
        self.root.title("ReconFlow 确定性复核诊断台")
        self.root.geometry("1180x760")
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self._build_stage_json()
        self._build_overview()
        self._build_invalidation()
        self._build_errors()
        self._build_ledger()
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x")
        self.refresh_button = ttk.Button(bottom, text="只读刷新 (不写账本)", command=self.refresh)
        self.refresh_button.pack(side="left", padx=8, pady=6)
        self.status = ttk.Label(bottom, text="ready")
        self.status.pack(side="left", padx=8)
        self.refresh()

    def _tree_tab(self, title: str, columns: list[str]) -> tuple[Any, Any]:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=180, anchor="w")
        yscroll = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        return frame, tree

    def _build_stage_json(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="阶段中间量JSON")
        self.stage_json_text = self.tk.Text(frame, wrap="none")
        yscroll = self.ttk.Scrollbar(frame, orient="vertical", command=self.stage_json_text.yview)
        xscroll = self.ttk.Scrollbar(frame, orient="horizontal", command=self.stage_json_text.xview)
        self.stage_json_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.stage_json_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_overview(self) -> None:
        _, self.overview_tree = self._tree_tab("逐阶段结论", ["section", "id", "field", "value", "digest"])

    def _build_invalidation(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="失效集合/传播依据")
        self.invalidation_text = self.tk.Text(frame, wrap="none")
        yscroll = self.ttk.Scrollbar(frame, orient="vertical", command=self.invalidation_text.yview)
        self.invalidation_text.configure(yscrollcommand=yscroll.set)
        self.invalidation_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

    def _build_errors(self) -> None:
        _, self.error_tree = self._tree_tab("误差归因/裁决", ["stage_or_candidate", "metric", "value", "share_or_status"])

    def _build_ledger(self) -> None:
        _, self.ledger_tree = self._tree_tab("未知批次/快照/读取", ["kind", "id", "status", "digest_or_time"])

    @staticmethod
    def _clear(tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def refresh(self) -> None:
        state = collect_diagnostic_state(self.workspace)
        conclusion = state["conclusion"]
        self.stage_json_text.delete("1.0", "end")
        stage_payload = {
            "weighted_sampling": conclusion["weighted_sampling"],
            "shards": conclusion["shards"],
            "ratio_normalization": conclusion["ratio_normalization"],
            "error_attribution": conclusion["error_attribution"],
            "candidate_adjudication": conclusion.get("candidate_adjudication"),
        }
        self.stage_json_text.insert("1.0", json.dumps(stage_payload, indent=2, sort_keys=True, ensure_ascii=False))
        self._clear(self.overview_tree)
        for domain in sorted(conclusion["weighted_sampling"]["domains"]):
            item = conclusion["weighted_sampling"]["domains"][domain]
            self.overview_tree.insert("", "end", values=("weighted", domain, "quota/selected", f"{item['quota_points']}/{item['selected_count']}", ""))
        for shard in sorted(conclusion["shards"], key=lambda x: x["shard_id"]):
            self.overview_tree.insert(
                "",
                "end",
                values=("dedup", shard["shard_id"], "docs/kept/dropped", f"{shard['doc_count']}/{shard['kept_count']}/{shard['dropped_count']}", shard["shard_digest"][:16]),
            )
        for item in sorted(conclusion["ratio_normalization"]["domains"], key=lambda x: x["domain_id"]):
            self.overview_tree.insert("", "end", values=("ratio", item["domain_id"], "actual/target/dev", f"{item['actual_bp']}/{item['target_bp']}/{item['deviation_bp']}", ""))

        self.invalidation_text.delete("1.0", "end")
        propagation = state.get("last_invalidation")
        self.invalidation_text.insert("end", json.dumps(propagation or {"note": "initial cache"}, indent=2, sort_keys=True, ensure_ascii=False))

        self._clear(self.error_tree)
        error = conclusion["error_attribution"]
        for stage in error["stages"]:
            self.error_tree.insert("", "end", values=("stage", stage["stage"], stage["bound_micro_bp"], f"{stage['share_permille']}/1000"))
        self.error_tree.insert("", "end", values=("dominant", error["dominant_stage"], error["total_bound_micro_bp"], f"threshold={error['dominance_threshold_permille']}"))
        adjudication = conclusion.get("candidate_adjudication")
        if adjudication:
            for candidate in adjudication["candidates"]:
                self.error_tree.insert("", "end", values=("candidate", candidate["candidate_id"], candidate["max_abs_deviation_micro_bp"], f"within={candidate['within_error_bound']}"))
            self.error_tree.insert("", "end", values=("selected", adjudication["selected_candidate_id"], adjudication["score_gap_micro_bp"], f"affected={adjudication['error_affected']}"))

        self._clear(self.ledger_tree)
        ledger = state["ledger"]
        if ledger.get("available"):
            for batch in ledger.get("pending_batches", []):
                self.ledger_tree.insert("", "end", values=("unknown_or_pending", batch["batch_id"], batch["status"], ""))
            for snapshot in ledger.get("snapshots", []):
                self.ledger_tree.insert("", "end", values=("snapshot", snapshot["snapshot_id"], "committed", snapshot["file_sha256"][:24]))
            for read in ledger.get("ui_reads", []):
                self.ledger_tree.insert("", "end", values=("ui_read", str(read["id"]), f"{read['elapsed_ms']}ms", read["created_utc"]))
        else:
            self.ledger_tree.insert("", "end", values=("ledger", "missing", "run ledger-init", ""))
        self.status.configure(text=f"refreshed config_version={state['config_version']} read={state['read']['elapsed_ms']}ms (ledger unchanged)")


def run_tk_app(workspace: Workspace, *, smoke_ms: int = 0) -> int:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - environment dependent
        print(json.dumps({"code": "E_UI", "message": f"Tkinter unavailable: {exc}"}, ensure_ascii=False))
        return 8
    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - headless environment
        print(json.dumps({"code": "E_UI_DISPLAY", "message": f"cannot open Tk display: {exc}"}, ensure_ascii=False))
        return 8
    app = DiagnosticApp(root, workspace)
    if smoke_ms > 0:
        root.after(smoke_ms, root.destroy)
    root.mainloop()
    _ = app
    return 0
