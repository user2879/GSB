from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .aggregate import aggregate_samples
from .canonical import canonical_dumps, digest
from .errors import VerifyError
from .fixedpoint import quantize_units
from .graph import effective_node_id
from .samples import generate_samples
from .spec import load_spec


def verify_ledger(spec_path: str | Path, db_path: str | Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    db_path = Path(db_path)
    if not db_path.exists():
        raise VerifyError(f"ledger database does not exist: {db_path}", field="ledger", position=str(db_path))

    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if user_version != 1:
            raise VerifyError(
                f"unexpected ledger format version {user_version}",
                field="PRAGMA.user_version",
                expected=1,
                actual=user_version,
            )
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM ledger_meta")}
        if int(meta.get("ledger_format_version", "0")) != 1:
            raise VerifyError("missing integer ledger_format_version", field="ledger_meta")

        rows = {
            row["shard_id"]: dict(row)
            for row in conn.execute("SELECT * FROM outcomes ORDER BY topo_index,cursor")
        }
        for shard_id in spec["topo_order"]:
            if shard_id not in rows:
                raise VerifyError(
                    f"shard {shard_id} has no complete committed outcome",
                    field="outcomes",
                    position=shard_id,
                    unknown_shard=shard_id,
                )
            row = rows[shard_id]
            shard = spec["shard_by_id"][shard_id]
            node_id = effective_node_id(shard, spec)
            node = spec["node_by_id"][node_id]
            expected_summary = aggregate_samples(
                generate_samples(spec, shard),
                int(node["fixed_scale"]),
            )
            expected_water = quantize_units(int(shard["resource_units"]), int(spec["scale_factor"])).q
            checks = [
                ("summary_digest", row["summary_digest"], expected_summary["summary_digest"]),
                ("summary_json", row["summary_json"], canonical_dumps(expected_summary)),
                ("node_id", row["node_id"], node_id),
                ("capacity_version", int(row["capacity_version"]), int(node["capacity_version"])),
                ("resource_units", int(row["resource_units"]), int(shard["resource_units"])),
                ("water_q", int(row["water_q"]), int(expected_water)),
                ("scale_factor", int(row["scale_factor"]), int(spec["scale_factor"])),
                ("topo_index", int(row["topo_index"]), int(spec["topo_index"][shard_id])),
                ("status", row["status"], "committed"),
            ]
            for field_name, actual, expected in checks:
                if actual != expected:
                    raise VerifyError(
                        f"replayed outcome first differs at {shard_id}.{field_name}",
                        field=f"outcomes.{shard_id}.{field_name}",
                        position=shard_id,
                        expected=expected,
                        actual=actual,
                        first_difference={"shard_id": shard_id, "field": field_name},
                    )

        active_units = int(
            conn.execute("SELECT COALESCE(SUM(delta_units),0) FROM occupancy_events").fetchone()[0]
        )
        if active_units != 0:
            raise VerifyError(
                "occupancy events do not net to zero after all endings/recovery",
                field="occupancy_events",
                actual_active_units=active_units,
                expected=0,
            )

        stored_digest = meta.get("content_digest", "")
        outcome_rows = [dict(row) for row in conn.execute("SELECT * FROM outcomes ORDER BY cursor")]
        lease_rows = [dict(row) for row in conn.execute("SELECT * FROM lease_records ORDER BY lease_id")]
        actual_digest = digest(
            {"format_version": 1, "outcomes": outcome_rows, "leases": lease_rows}
        )
        if stored_digest != actual_digest:
            raise VerifyError(
                "independent ledger content digest mismatch",
                field="content_digest",
                expected=actual_digest,
                actual=stored_digest,
            )
        return {
            "ok": True,
            "verified_shards": len(spec["topo_order"]),
            "content_digest": actual_digest,
            "active_units": active_units,
            "ledger_format_version": user_version,
        }
    finally:
        conn.close()
