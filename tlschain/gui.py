from __future__ import annotations

import argparse
from typing import Any

from .gui_model import build_diagnostics_model, model_digest
from .session import build_full_session
from .spec import load_spec


def _clear(tree: Any) -> None:
    for item in tree.get_children():
        tree.delete(item)


def _insert_rows(tree: Any, rows: list[dict[str, Any]], columns: list[str]) -> None:
    _clear(tree)
    for index, row in enumerate(rows):
        tree.insert(
            "",
            "end",
            iid=str(index),
            values=[row.get(column, "") for column in columns],
        )


def _insert_chain_tree(tree: Any, rows: list[dict[str, Any]], columns: list[str]) -> None:
    _clear(tree)
    by_id = {row["id"]: row for row in rows}
    children: dict[str, list[str]] = {cert_id: [] for cert_id in by_id}
    roots: list[str] = []
    for cert_id in sorted(by_id):
        issuer = by_id[cert_id]["issuer"]
        if issuer == cert_id or issuer not in by_id:
            roots.append(cert_id)
        else:
            children[issuer].append(cert_id)

    def add(cert_id: str, parent: str) -> None:
        row = by_id[cert_id]
        values = [row.get(column, "") for column in columns]
        tree.insert(parent, "end", iid=cert_id, text=cert_id, values=values)
        for child_id in sorted(children[cert_id]):
            add(child_id, cert_id)

    for root_id in roots:
        add(root_id, "")


def run_gui(cert_path: str, handshake_path: str, ledger_path: str | None = None) -> int:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("TLS Fixed-Point Chain Diagnostics")
    root.geometry("1320x820")
    status_var = tk.StringVar(value="Ready")
    summary_var = tk.StringVar(value="")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    overview_frame = ttk.Frame(notebook)
    chain_frame = ttk.Frame(notebook)
    verification_frame = ttk.Frame(notebook)
    invalidation_frame = ttk.Frame(notebook)
    ledger_frame = ttk.Frame(notebook)
    notebook.add(overview_frame, text="Overview")
    notebook.add(chain_frame, text="Certificate Chain")
    notebook.add(verification_frame, text="Verification")
    notebook.add(invalidation_frame, text="Invalidation")
    notebook.add(ledger_frame, text="Ledger")

    ttk.Label(overview_frame, textvariable=summary_var, justify="left").pack(anchor="w", padx=12, pady=12)
    overview_tree = ttk.Treeview(
        overview_frame,
        columns=("seq", "sender", "decision", "affected", "selected", "reason"),
        show="headings",
        height=18,
    )
    for column in ("seq", "sender", "decision", "affected", "selected", "reason"):
        overview_tree.heading(column, text=column)
        overview_tree.column(column, width=170)
    overview_tree.pack(fill="both", expand=True, padx=12, pady=8)

    chain_columns = ("id", "issuer", "serial", "not_before_ms", "not_after_ms", "modulus", "algorithm", "children")
    chain_tree = ttk.Treeview(chain_frame, columns=chain_columns, show="tree headings")
    for column in chain_columns:
        chain_tree.heading(column, text=column)
        chain_tree.column(column, width=150)
    chain_tree.pack(fill="both", expand=True, padx=12, pady=12)

    verification_columns = (
        "seq",
        "cert_id",
        "step",
        "signature_q",
        "digest_q",
        "residual",
        "window_state",
        "state",
        "modulus_bound",
        "time_bound",
        "path_bound",
        "cumulative_bound",
        "message_state",
        "affected",
    )
    verification_tree = ttk.Treeview(
        verification_frame, columns=verification_columns, show="headings"
    )
    for column in verification_columns:
        verification_tree.heading(column, text=column)
        verification_tree.column(column, width=110)
    verification_tree.pack(fill="both", expand=True, padx=12, pady=12)

    invalidation_columns = ("order", "kind", "id", "reason", "field")
    invalidation_tree = ttk.Treeview(
        invalidation_frame, columns=invalidation_columns, show="headings"
    )
    for column in invalidation_columns:
        invalidation_tree.heading(column, text=column)
        invalidation_tree.column(column, width=240)
    invalidation_tree.pack(fill="both", expand=True, padx=12, pady=12)

    ledger_columns = ("batch_index", "through_seq", "progress_cursor", "spec_digest", "payload_digest")
    ledger_tree = ttk.Treeview(ledger_frame, columns=ledger_columns, show="headings")
    for column in ledger_columns:
        ledger_tree.heading(column, text=column)
        ledger_tree.column(column, width=220)
    ledger_tree.pack(fill="both", expand=True, padx=12, pady=12)

    unknown_frame = ttk.LabelFrame(ledger_frame, text="Unknown / uncommitted messages")
    unknown_frame.pack(fill="x", padx=12, pady=8)
    unknown_var = tk.StringVar(value="")
    ttk.Label(unknown_frame, textvariable=unknown_var).pack(anchor="w", padx=8, pady=8)

    def refresh() -> None:
        spec = load_spec(cert_path, handshake_path)
        session = build_full_session(spec)
        model = build_diagnostics_model(spec, session, ledger_path)
        counts = model["decision_counts"]
        summary_var.set(
            "decision counts: "
            f"PASS={counts['PASS']} FAIL={counts['FAIL']} INDETERMINATE={counts['INDETERMINATE']}\n"
            f"affected_by_error={model['affected_by_error_count']}  "
            f"spec_digest={model['spec_digest']}\n"
            f"ledger_cursor={model['ledger']['progress_cursor']}  "
            f"ledger_digest={model['ledger_content_digest']}"
        )
        overview_rows = [
            {
                "seq": message.seq,
                "sender": message.sender,
                "decision": session.message_records[message.seq]["decision"],
                "affected": session.message_records[message.seq]["affected_by_error"],
                "selected": ",".join(session.message_records[message.seq]["selected_cert_ids"]),
                "reason": session.message_records[message.seq]["selection_reason"],
            }
            for message in sorted(spec.messages, key=lambda item: item.seq)
        ]
        _insert_rows(overview_tree, overview_rows, ["seq", "sender", "decision", "affected", "selected", "reason"])
        _insert_chain_tree(chain_tree, model["certificates"], list(chain_columns))
        verification_rows = [
            {
                **row,
                "residual": row["signature_residual_q"],
                "modulus_bound": row["modulus_error_bound"],
                "time_bound": row["time_error_bound"],
                "path_bound": row["path_error_bound"],
                "cumulative_bound": row["cumulative_error_bound"],
            }
            for row in model["verification_rows"]
        ]
        _insert_rows(verification_tree, verification_rows, list(verification_columns))
        _insert_rows(invalidation_tree, model["invalidation_trace"], list(invalidation_columns))
        _insert_rows(ledger_tree, model["ledger"]["batches"], list(ledger_columns))
        unknown_var.set(
            ", ".join(str(seq) for seq in model["unknown_message_seqs"])
            if model["unknown_message_seqs"]
            else "none"
        )
        status_var.set(f"Refreshed; diagnostics digest={model_digest(model)}")

    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=12, pady=6)
    ttk.Button(controls, text="Refresh (read-only ledger)", command=refresh).pack(side="left")
    ttk.Label(controls, textvariable=status_var).pack(side="left", padx=12)
    refresh()
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tkinter TLS fixed-point diagnostics")
    parser.add_argument("--certs", required=True)
    parser.add_argument("--handshake", required=True)
    parser.add_argument("--ledger")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_gui(args.certs, args.handshake, args.ledger)


if __name__ == "__main__":
    raise SystemExit(main())
