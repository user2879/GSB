"""Read-only diagnostics model shared by the CLI and Tkinter UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import STATES
from .snapshots import backups_directory, list_snapshots
from .store import Ledger


def build_diagnostics(
    ledger: Ledger,
    workdir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    conclusion = ledger.conclusion()
    transactions = ledger.transactions()
    partitions = ledger.partitions()
    latest = ledger.latest_states()
    unknown = {item.transaction_id: item for item in conclusion.unknown_transactions}
    partition_rows: list[dict[str, Any]] = []
    for partition in sorted(partitions, key=lambda item: item.partition_id):
        prefix = next(
            item for item in conclusion.partitions if item.partition_id == partition.partition_id
        )
        distribution = {state: 0 for state in STATES}
        distribution["pending"] = 0
        partition_transactions = sorted(
            (
                transaction
                for transaction in transactions
                if transaction.partition_id == partition.partition_id
            ),
            key=lambda transaction: (transaction.txn_seq, transaction.transaction_id),
        )
        unknown_ids: list[str] = []
        for transaction in partition_transactions:
            state = latest.get(transaction.transaction_id)
            distribution["pending" if state is None else state] += 1
            if transaction.transaction_id in unknown:
                unknown_ids.append(transaction.transaction_id)
        partition_rows.append(
            {
                "partition_id": partition.partition_id,
                "partition_version": partition.partition_version,
                "max_position": partition.max_position,
                "prefix": prefix.prefix,
                "truncated_position": prefix.truncated_position,
                "truncated_reason": prefix.truncated_reason,
                "truncated_transaction_id": prefix.truncated_transaction_id,
                "state_distribution": distribution,
                "transaction_ids": [
                    transaction.transaction_id for transaction in partition_transactions
                ],
                "unknown_transaction_ids": sorted(unknown_ids),
            }
        )

    base = Path(workdir) if workdir is not None else ledger.path.parent
    snapshots = list_snapshots(base)
    backup_rows: list[dict[str, Any]] = []
    backup_dir = backups_directory(base)
    if backup_dir.exists():
        for path in sorted(backup_dir.glob("ledger-*.sqlite"), key=lambda item: item.name):
            try:
                stat = path.stat()
            except OSError:
                continue
            backup_rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size": stat.st_size,
                    "readonly": (stat.st_mode & 0o222) == 0,
                }
            )

    return {
        "ledger_format_version": conclusion.ledger_format_version,
        "partition_set_version": conclusion.partition_set_version,
        "transaction_set_version": conclusion.transaction_set_version,
        "event_cursor": conclusion.event_cursor,
        "ledger_digest": conclusion.ledger_digest,
        "partitions": partition_rows,
        "unknown_transactions": [
            item.as_dict() for item in conclusion.unknown_transactions
        ],
        "snapshots": [item.as_dict() for item in snapshots],
        "backups": backup_rows,
    }


def render_text(view: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        "LEDGER format={ledger_format_version} partition_set={partition_set_version} "
        "transaction_set={transaction_set_version} cursor={event_cursor}".format(**view)
    )
    lines.append("DIGEST " + str(view["ledger_digest"]))
    lines.append("PARTITIONS")
    for row in view["partitions"]:
        distribution = row["state_distribution"]
        lines.append(
            "  {partition_id} v{partition_version} max={max_position} prefix={prefix} "
            "truncated={truncated_position} reason={truncated_reason} "
            "states=prepared:{prepared},committing:{committing},committed:{committed},aborted:{aborted},pending:{pending} "
            "unknown={unknown}".format(
                **row,
                unknown=",".join(row["unknown_transaction_ids"]) or "-",
                **distribution,
            )
        )
    lines.append("UNKNOWN")
    if not view["unknown_transactions"]:
        lines.append("  -")
    else:
        for item in view["unknown_transactions"]:
            intervals = ",".join(f"[{start},{end}]" for start, end in item["intervals"])
            lines.append(
                f"  {item['transaction_id']} partition={item['partition_id']} intervals={intervals} rerun={item['requires_rerun']}"
            )
    lines.append("SNAPSHOTS")
    if not view["snapshots"]:
        lines.append("  -")
    else:
        for item in view["snapshots"]:
            lines.append(
                f"  {item['name']} ledger_format={item['ledger_format_version']} cursor={item['event_cursor']} readable={item['readable']}"
            )
    lines.append("BACKUPS")
    if not view["backups"]:
        lines.append("  -")
    else:
        for item in view["backups"]:
            lines.append(
                f"  {item['name']} readonly={item['readonly']} size={item['size']}"
            )
    return "\n".join(lines) + "\n"
