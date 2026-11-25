from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .spec import SpecBundle, load_spec_dir


@dataclass(frozen=True)
class NodeWaterRow:
    node_id: str
    quota_version: int
    capacity_slots: int
    effective_cap: int
    active_leases: int
    available_slots: int
    watermark: str


@dataclass(frozen=True)
class ShardRow:
    shard_id: str
    node_id: str
    status: str
    unknown: bool
    digest: str
    content_digest: str
    source_format: str


@dataclass(frozen=True)
class ReceiptRow:
    receipt_id: int
    quota_version: int
    shard_id: str
    node_id: str
    source_format: str
    source_line: int
    adopted: bool
    current: bool
    reason: str
    filled_fields: str
    stale: bool


@dataclass(frozen=True)
class DashboardModel:
    ledger_path: str
    ledger_format_version: str
    content_digest: str
    spec_quota_version: int
    node_water: tuple[NodeWaterRow, ...]
    shards: tuple[ShardRow, ...]
    receipts: tuple[ReceiptRow, ...]
    unknown_shards: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_path": self.ledger_path,
            "ledger_format_version": self.ledger_format_version,
            "content_digest": self.content_digest,
            "spec_quota_version": self.spec_quota_version,
            "node_water": [row.__dict__ for row in self.node_water],
            "shards": [row.__dict__ for row in self.shards],
            "receipts": [row.__dict__ for row in self.receipts],
            "unknown_shards": list(self.unknown_shards),
        }


def _short(value: str | None, length: int = 12) -> str:
    if not value:
        return "-"
    return value[:length]


def build_dashboard_model(spec_dir: str | Path, ledger_path: str | Path) -> DashboardModel:
    spec = load_spec_dir(spec_dir)
    path = Path(ledger_path)
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM meta")}
        active_rows = list(conn.execute("SELECT node_id, COUNT(*) AS c FROM leases WHERE state='active' GROUP BY node_id"))
        active_by_node = {row["node_id"]: int(row["c"]) for row in active_rows}
        node_rows: list[NodeWaterRow] = []
        for node in sorted(spec.nodes, key=lambda n: n.node_id):
            active = active_by_node.get(node.node_id, 0)
            available = max(0, node.capacity_slots - active)
            ratio = 0 if node.capacity_slots == 0 else active / node.capacity_slots
            mark = "high" if ratio >= 0.8 else "normal"
            node_rows.append(
                NodeWaterRow(
                    node_id=node.node_id,
                    quota_version=spec.quota_version,
                    capacity_slots=node.capacity_slots,
                    effective_cap=node.effective_lease_cap,
                    active_leases=active,
                    available_slots=available,
                    watermark=f"{active}/{node.capacity_slots} {mark}",
                )
            )

        result_rows = list(
            conn.execute(
                "SELECT * FROM shard_results WHERE quota_version=? AND stale=0 ORDER BY shard_id,node_id",
                (spec.quota_version,),
            )
        )
        results_by_shard: dict[tuple[str, str], sqlite3.Row] = {
            (row["shard_id"], row["node_id"]): row for row in result_rows
        }
        shard_rows: list[ShardRow] = []
        for seq in range(spec.content.shard_count):
            shard_id = spec.content.shard_id(seq)
            matching = [row for row in result_rows if row["shard_id"] == shard_id]
            if not matching:
                shard_rows.append(ShardRow(shard_id, "-", "missing", True, "-", "-", "-"))
                continue
            for row in sorted(matching, key=lambda r: r["node_id"]):
                unknown = bool(int(row["unknown"]))
                shard_rows.append(
                    ShardRow(
                        shard_id=shard_id,
                        node_id=row["node_id"],
                        status=row["status"],
                        unknown=unknown,
                        digest=_short(row["digest"]),
                        content_digest=_short(row["content_bytes_digest"]),
                        source_format=row["source_format"],
                    )
                )

        receipt_rows_raw = list(
            conn.execute(
                "SELECT * FROM receipts ORDER BY quota_version, shard_id, node_id, source_format, receipt_id"
            )
        )
        receipt_rows: list[ReceiptRow] = []
        for row in receipt_rows_raw:
            canonical = json.loads(row["canonical_json"])
            filled = ",".join(canonical.get("filled_fields", [])) or "-"
            receipt_rows.append(
                ReceiptRow(
                    receipt_id=int(row["receipt_id"]),
                    quota_version=int(row["quota_version"]),
                    shard_id=row["shard_id"],
                    node_id=row["node_id"],
                    source_format=row["source_format"],
                    source_line=int(row["source_line"]),
                    adopted=bool(row["adopted"]),
                    current=bool(row["current"]),
                    reason=row["reason"],
                    filled_fields=filled,
                    stale=bool(row["stale"]) if "stale" in row.keys() else False,
                )
            )

        unknown = sorted(
            row["shard_id"]
            for row in conn.execute("SELECT DISTINCT shard_id FROM shard_results WHERE unknown=1 AND stale=0")
        )
        return DashboardModel(
            ledger_path=str(path),
            ledger_format_version=meta.get("ledger_format_version", "unknown"),
            content_digest=_short(meta.get("content_digest"), 16),
            spec_quota_version=spec.quota_version,
            node_water=tuple(node_rows),
            shards=tuple(shard_rows),
            receipts=tuple(receipt_rows),
            unknown_shards=tuple(unknown),
        )
    finally:
        conn.close()


def render_text(model: DashboardModel) -> str:
    lines = [
        "EDGE LEDGER DIAGNOSTIC SNAPSHOT",
        f"ledger={model.ledger_path}",
        f"format_version={model.ledger_format_version} content_digest={model.content_digest} quota_version={model.spec_quota_version}",
        "",
        "[node lease watermarks]",
        "node_id | quota | active/cap | effective_cap | available | watermark",
    ]
    for node in model.node_water:
        lines.append(
            f"{node.node_id} | {node.quota_version} | {node.active_leases}/{node.capacity_slots} | "
            f"{node.effective_cap} | {node.available_slots} | {node.watermark}"
        )
    lines.extend(["", "[shard conclusions]", "shard_id | node | status | unknown | digest | content | format"])
    for shard in model.shards:
        flag = "UNKNOWN/RERUN" if shard.unknown else "known"
        lines.append(
            f"{shard.shard_id} | {shard.node_id} | {shard.status} | {flag} | {shard.digest} | "
            f"{shard.content_digest} | {shard.source_format}"
        )
    lines.extend(["", "[cross-version receipts: audit incl. stale epochs]", "id | q | shard | node | format:line | adopted/current/stale | reason | filled"])
    for receipt in model.receipts:
        lines.append(
            f"{receipt.receipt_id} | {receipt.quota_version} | {receipt.shard_id} | {receipt.node_id} | "
            f"{receipt.source_format}:{receipt.source_line} | adopted={receipt.adopted},current={receipt.current},stale={receipt.stale} | "
            f"{receipt.reason} | {receipt.filled_fields}"
        )
    lines.extend(["", f"unknown_shards={','.join(model.unknown_shards) or '-'}"])
    return "\n".join(lines) + "\n"
