"""Tkinter diagnostic interface with deterministic text rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .certfamily import Family, load_family
from .errors import E_UI, WorkbenchError
from .ledger import Ledger
from .verify import compare_results


def _family_text(family: Family) -> str:
    lines = ["CERTIFICATE FAMILY", ""]
    lines.append("ROOTS")
    for item in sorted(family.data["roots"], key=lambda value: value["id"]):
        lines.append(f"  {item['id']}  {item['subject']}  ca={item['ca']} path_len={item['path_len']}")
        lines.append(f"    fp={item['fingerprint']}")
    lines.append("INTERMEDIATES")
    for item in sorted(family.data["intermediates"], key=lambda value: value["id"]):
        lines.append(f"  {item['id']}  {item['subject']}  path_len={item['path_len']} parent={item['parent_id']}")
        lines.append(f"    fp={item['fingerprint']} serial={item['serial_number']}")
    lines.append("LEAVES")
    for item in sorted(family.data["leaves"], key=lambda value: value["id"]):
        lines.append(
            f"  {item['id']}  profile={item['profile']} valid={item['not_before_ms']}..{item['not_after_ms']}"
        )
        lines.append(f"    fp={item['fingerprint']} serial={item['serial_number']} chain_depth={len(item['chain_fingerprints'])}")
    return "\n".join(lines)


def _handshake_text(view: dict[str, Any]) -> str:
    lines = ["REAL HANDSHAKE CHAINS", ""]
    for run in sorted(view.get("runs", []), key=lambda item: item["run_seq"]):
        handshake = run.get("handshake")
        lines.append(f"RUN {run['run_id']}  status={run['status']}  trust_version={run['trust_version']}")
        if not handshake:
            lines.append("  handshake unavailable")
            continue
        negotiated = handshake.get("negotiated", {})
        lines.append(
            f"  leaf={handshake.get('leaf_id')} tls={negotiated.get('tls_version')} "
            f"cipher={negotiated.get('cipher_name')} capture={handshake.get('snapshot_digest')}"
        )
        for position, fingerprint in enumerate(handshake.get("chain_fingerprints", [])):
            lines.append(f"    [{position}] {fingerprint}")
    return "\n".join(lines)


def _revision_text(view: dict[str, Any]) -> str:
    lines = ["REVISION RESULTS AND FLIP POINTS", ""]
    lines.append(f"trust_version={view.get('trust_version')} revision_chain_digest={view.get('revision_chain_digest')}")
    for run in sorted(view.get("runs", []), key=lambda item: item["run_seq"]):
        lines.append("")
        lines.append(f"RUN {run['run_id']}  status={run['status']}")
        for row in sorted(run.get("revisions", []), key=lambda item: item["revision"]):
            first = "none"
            if row.get("committed") and row.get("result"):
                failure = row["result"].get("first_failure")
                if failure:
                    first = f"layer={failure.get('layer')} check={failure.get('check')} reason={failure.get('reason')}"
            lines.append(
                f"  r{row['revision']:03d}  {row['conclusion']:<7} "
                f"category={str(row.get('failure_category')):<18} first={first}"
            )
            if row.get("result_index"):
                lines.append(f"        index={row['result_index']}")
        for flip in sorted(run.get("flips", []), key=lambda item: (item["from_revision"], item["to_revision"], item["kind"])):
            lines.append(
                f"  FLIP {flip['from_revision']}->{flip['to_revision']} {flip['kind']} "
                f"{flip.get('from_failure_category')} -> {flip.get('to_failure_category')}"
            )
    return "\n".join(lines)


def _index_text(view: dict[str, Any]) -> str:
    lines = ["INDEX COMPARISON", ""]
    for run in sorted(view.get("runs", []), key=lambda item: item["run_seq"]):
        committed = [row for row in run.get("revisions", []) if row.get("committed") and row.get("result")]
        committed.sort(key=lambda item: item["revision"])
        for left, right in zip(committed, committed[1:]):
            comparison = compare_results(left["result"], right["result"])
            if comparison.get("equal"):
                detail = "equal"
            else:
                detail = (
                    f"first-difference layer={comparison.get('layer')} "
                    f"item={comparison.get('item')} field={comparison.get('field')} "
                    f"expected={comparison.get('expected')!r} actual={comparison.get('actual')!r}"
                )
            lines.append(
                f"  {run['run_id']}  r{left['revision']}->r{right['revision']}  "
                f"{left['result_index']} <> {right['result_index']}"
            )
            lines.append(f"    {detail}")
    return "\n".join(lines)


def _snapshot_text(view: dict[str, Any]) -> str:
    lines = ["SESSION SNAPSHOTS AND REPLACEMENT AUDIT", ""]
    lines.append(f"current_trust_version={view.get('trust_version')}")
    for item in sorted(view.get("session_snapshots", []), key=lambda value: value["snapshot_seq"]):
        lines.append(
            f"  {item['content_digest']}  format={item['format_version']} trust_version={item['trust_version']} "
            f"complete={bool(item['complete'])} stale={bool(item['stale'])}"
        )
    audit = view.get("replacement_audit")
    if audit:
        summary = audit.get("summary", {})
        lines.append("")
        lines.append(
            f"LATEST REPLACEMENT operation={summary.get('operation')} "
            f"old={summary.get('old_trust_version')} new={summary.get('new_trust_version')}"
        )
        lines.append(f"  affected_chains={len(summary.get('affected_chains', []))}")
        lines.append(f"  old_failure_categories={summary.get('old_failure_categories', [])}")
        for flip in sorted(
            summary.get("flip_points", []),
            key=lambda item: (item.get("snapshot_digest", ""), item.get("from_revision", 0), item.get("to_revision", 0), item.get("kind", "")),
        ):
            lines.append(
                f"  old-flip {flip.get('from_revision')}->{flip.get('to_revision')} {flip.get('kind')} "
                f"{flip.get('from_failure_category')}->{flip.get('to_failure_category')}"
            )
    return "\n".join(lines)


def render_text_view(view: dict[str, Any], family: Family) -> str:
    sections = [
        _family_text(family),
        _handshake_text(view),
        _revision_text(view),
        _index_text(view),
        _snapshot_text(view),
    ]
    return "\n\n".join(section.rstrip() for section in sections) + "\n"


def run_ui(workdir: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # pragma: no cover - depends on host Tk installation
        raise WorkbenchError(
            E_UI,
            "tkinter is unavailable; install the Windows Python Tcl/Tk component",
            field_path="tkinter",
        ) from exc

    root_path = Path(workdir)
    family_path = root_path / "family" / "family.json"
    ledger_path = root_path / "ledger.sqlite3"
    if not family_path.exists() or not ledger_path.exists():
        raise WorkbenchError(
            E_UI,
            "workbench data is missing; run `python -m tlstrust demo --workdir <path>` first",
            field_path="workdir",
        )
    family = load_family(family_path)
    ledger = Ledger(ledger_path)
    if ledger.current_trust_version() is None:
        ledger.close()
        raise WorkbenchError(E_UI, "ledger has no current trust version", field_path="trust_version")

    window = tk.Tk()
    window.title("TLS Trust Rotation Diagnosis")
    window.geometry("1180x760")
    header = tk.Label(
        window,
        anchor="w",
        text="127.0.0.1 TLS trust-anchor rotation · real ssl handshake · deterministic ledger",
    )
    header.pack(fill="x", padx=8, pady=(8, 2))
    status = tk.StringVar(value="loaded")
    status_label = tk.Label(window, anchor="w", textvariable=status)
    status_label.pack(fill="x", padx=8)
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)
    tabs: dict[str, tk.Text] = {}
    for name in ("Family", "Handshake", "Revisions", "Index Compare", "Snapshots"):
        frame = ttk.Frame(notebook)
        text = tk.Text(frame, wrap="none", font=("Consolas", 9))
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        notebook.add(frame, text=name)
        tabs[name] = text

    def set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def refresh() -> None:
        view = ledger.build_current_view()
        set_text(tabs["Family"], _family_text(family))
        set_text(tabs["Handshake"], _handshake_text(view))
        set_text(tabs["Revisions"], _revision_text(view))
        set_text(tabs["Index Compare"], _index_text(view))
        set_text(tabs["Snapshots"], _snapshot_text(view))
        status.set(
            f"trust_version={view['trust_version']} view_digest={view['view_digest']} "
            f"runs={len(view['runs'])} snapshots={len(view['session_snapshots'])}"
        )

    def replay_latest() -> None:
        snapshots = ledger.list_session_snapshots(current_only=True)
        if not snapshots:
            messagebox.showinfo("Replay", "No current session snapshot is available.")
            return
        payload = ledger.load_session_snapshot(snapshots[-1]["content_digest"])
        result = ledger.replay_session(payload, family)
        status.set(
            f"replay_ok={result['ok']} run={result['run_id']} revisions={len(result['replayed_revisions'])}"
        )
        messagebox.showinfo("Replay verified", json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))

    controls = ttk.Frame(window)
    controls.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(controls, text="Refresh current view", command=refresh).pack(side="left")
    ttk.Button(controls, text="Replay latest snapshot", command=replay_latest).pack(side="left", padx=8)
    ttk.Button(controls, text="Close", command=window.destroy).pack(side="right")
    refresh()
    try:
        window.mainloop()
    finally:
        ledger.close()

