from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def build_state_view_model(state_dir: str | Path) -> dict[str, Any]:
    db_path = Path(state_dir) / "recovery.db"
    if not db_path.exists():
        return {"available": False, "state_dir": str(state_dir), "generations": []}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        meta = {row["key"]: row["value"] for row in _rows(conn, "SELECT key,value FROM meta")}
        current_generation = int(meta.get("current_generation", "0"))
        generations = _rows(conn, "SELECT * FROM spec_generation ORDER BY generation")
        for generation in generations:
            generation["budget"] = json.loads(generation.pop("budget_json"))
            generation["pages_declared"] = json.loads(generation.pop("pages_json"))
            generation["spec_hash_short"] = generation["spec_hash"][:12]

        budget: dict[str, Any] = {}
        page_declarations: list[dict[str, Any]] = []
        current_spec = next(
            (item for item in generations if item["generation"] == current_generation),
            None,
        )
        if current_spec:
            budget = current_spec["budget"]
            payload = current_spec["pages_declared"]
            page_declarations = payload.get("pages", payload if isinstance(payload, list) else [])

        view_row = conn.execute(
            "SELECT * FROM view_snapshot WHERE generation=? AND status='current' "
            "ORDER BY view_snapshot_id DESC LIMIT 1",
            (current_generation,),
        ).fetchone()
        current_view: dict[str, Any] | None = None
        pages: list[dict[str, Any]] = []
        if view_row is not None:
            current_view = dict(view_row)
            current_view["summary"] = json.loads(current_view.pop("summary_json"))
            pages = _rows(
                conn,
                "SELECT * FROM page_view WHERE view_snapshot_id=? ORDER BY page_id",
                (current_view["view_snapshot_id"],),
            )

        declared_by_id = {
            item.get("page_id") or item.get("id"): item for item in page_declarations
        }
        for page in pages:
            declaration = declared_by_id.get(page["page_id"], {})
            page["row_count"] = declaration.get("row_count")
            page["column_count"] = len(declaration.get("columns", []))
            page["digest_short"] = page["content_digest"][:12]

        wal = _rows(
            conn,
            "SELECT lsn,seq_index,tx_id,page_id,row_start,row_end,seed,"
            "commit_flag,persist_flag FROM wal_entry WHERE generation=? ORDER BY lsn",
            (current_generation,),
        )
        batches = _rows(
            conn,
            "SELECT batch_id,round_no,attempt_no,status,start_lsn,end_lsn,entry_count,"
            "slot_peak,byte_consumed,wait_total_ms,run_total_ms,unknown_reason,stale "
            "FROM replay_batch WHERE generation=? ORDER BY round_no,attempt_no",
            (current_generation,),
        )
        timeline = _rows(
            conn,
            "SELECT t.lsn,t.tx_id,t.page_id,t.thread_name,t.wait_ms,t.run_ms,"
            "t.global_active_at_start,t.tx_active_at_start,t.byte_count,t.status,"
            "t.wait_started_ms,t.wait_ended_ms FROM task_outcome t "
            "JOIN replay_batch b ON t.batch_id=b.batch_id "
            "WHERE b.generation=? ORDER BY t.lsn,t.page_id",
            (current_generation,),
        )
        usage = _rows(
            conn,
            "SELECT page_id,bytes_consumed,byte_budget,last_batch_id "
            "FROM budget_usage WHERE generation=? ORDER BY page_id",
            (current_generation,),
        )
        checkpoints = _rows(
            conn,
            "SELECT checkpoint_id,checkpoint_no,at_lsn,batch_id,aggregate_digest,stale "
            "FROM checkpoint WHERE generation=? ORDER BY checkpoint_no,at_lsn",
            (current_generation,),
        )
        for checkpoint in checkpoints:
            checkpoint["digest_short"] = checkpoint["aggregate_digest"][:12]
            checkpoint["pages"] = _rows(
                conn,
                "SELECT page_id,version,lsn,content_digest,version_digest,content_path "
                "FROM checkpoint_page WHERE checkpoint_id=? ORDER BY page_id",
                (checkpoint["checkpoint_id"],),
            )
        revisions = _rows(
            conn,
            "SELECT revision_id,revision_no,batch_id,round_no,attempt_no,cursor_lsn,"
            "aggregate_digest,created_unix_ms FROM ledger_revision "
            "WHERE generation=? ORDER BY revision_no",
            (current_generation,),
        )
        snapshots = _rows(
            conn,
            "SELECT snapshot_id,kind,label,at_lsn,digest,stale,created_unix_ms "
            "FROM snapshot WHERE generation=? ORDER BY snapshot_id",
            (current_generation,),
        )
        crash_row = conn.execute(
            "SELECT * FROM crash_decision WHERE generation=?", (current_generation,)
        ).fetchone()
        crash = None
        if crash_row is not None:
            crash = dict(crash_row)
            crash["decision"] = json.loads(crash.pop("decision_json"))

        invisible_versions = _rows(
            conn,
            "SELECT page_id,version,lsn,tx_id,after_crash,content_digest,"
            "version_digest,batch_id,round_no,attempt_no FROM page_version "
            "WHERE generation=? AND visible=0 ORDER BY page_id,version,lsn",
            (current_generation,),
        )
        all_page_versions = _rows(
            conn,
            "SELECT page_id,version,lsn,tx_id,visible,after_crash,content_digest,"
            "version_digest,byte_length,batch_id,round_no,attempt_no "
            "FROM page_version WHERE generation=? AND version>0 "
            "ORDER BY page_id,version,lsn",
            (current_generation,),
        )

        stale_generation_ids = [
            item["generation"] for item in generations if item["stale"]
        ]
        stale_counts = {
            "generations": len(stale_generation_ids),
            "checkpoints": conn.execute(
                "SELECT COUNT(*) AS count FROM checkpoint WHERE stale=1"
            ).fetchone()["count"],
            "batches": conn.execute(
                "SELECT COUNT(*) AS count FROM replay_batch WHERE stale=1"
            ).fetchone()["count"],
            "view_snapshots": conn.execute(
                "SELECT COUNT(*) AS count FROM view_snapshot WHERE status='stale'"
            ).fetchone()["count"],
        }
        return {
            "available": True,
            "state_dir": str(state_dir),
            "db_path": str(db_path),
            "meta": meta,
            "current_generation": current_generation,
            "generations": generations,
            "budget": budget,
            "page_declarations": page_declarations,
            "pages": pages,
            "wal": wal,
            "batches": batches,
            "timeline": timeline,
            "budget_usage": usage,
            "checkpoints": checkpoints,
            "revisions": revisions,
            "snapshots": snapshots,
            "crash": crash,
            "current_view": current_view,
            "invisible_versions": invisible_versions,
            "page_versions": all_page_versions,
            "stale_counts": stale_counts,
        }
    finally:
        conn.close()


def render_plain_text_summary(view_model: dict[str, Any]) -> str:
    if not view_model.get("available"):
        return f"No ledger found under {view_model.get('state_dir')}"
    lines = [
        f"generation=G{view_model['current_generation']} "
        f"pages={len(view_model['pages'])} wal={len(view_model['wal'])} "
        f"checkpoints={len(view_model['checkpoints'])}",
        "pages:",
    ]
    for page in view_model["pages"]:
        visibility = "visible" if page["visible"] else "INVISIBLE"
        lines.append(
            f"  {page['page_id']:<8} v{page['version']:<3} lsn={page['lsn']:<3} "
            f"{visibility:<9} {page['digest_short']}"
        )
    lines.append("batches:")
    for batch in view_model["batches"]:
        lines.append(
            f"  R{batch['round_no']}.{batch['attempt_no']} {batch['status']:<9} "
            f"lsn {batch['start_lsn']}-{batch['end_lsn']} peak={batch['slot_peak']} "
            f"bytes={batch['byte_consumed']}"
        )
    return "\n".join(lines)
