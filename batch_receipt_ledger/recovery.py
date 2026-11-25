"""强制终止后的确定性恢复（需求 8）。

恢复原则：只承认已提交事务；计划中存在但账本缺失/unknown 的批一律
标记 unknown，并先回收它们占用的配额（held -> recovered_quota）。
"""

from .errors import LedgerError
from .ledger import Ledger
from .plan import build_plan


def recover(db_path, spec):
    ledger = Ledger(db_path)
    snapshot_version = ledger.current_snapshot()
    if snapshot_version is None:
        ledger.close()
        raise LedgerError("账本中没有任何已提交快照，无法恢复", field="snapshot")

    snapshot = ledger.load_snapshot(snapshot_version)
    plans = build_plan(spec)
    committed = {
        row["batch_id"]: row for row in snapshot["batches"]
        if row["status"] == "committed"
    }
    unknown_rows = [row for row in snapshot["batches"] if row["status"] == "unknown"]

    recovered = []
    for plan in plans:
        row = committed.get(plan.batch_id)
        if row is None:
            ledger.mark_unknown(
                snapshot_version, plan, _now_ms(),
                reason="no_commit_record_after_forced_termination")
            recovered.append({
                "batch_id": plan.batch_id,
                "worker_id": plan.worker_id,
                "action": "quota_reclaimed",
                "state": "unknown",
            })

    for row in unknown_rows:
        recovered.append({
            "batch_id": row["batch_id"],
            "worker_id": row["worker_id"],
            "action": "quota_reclaimed_existing_unknown",
            "state": "unknown",
        })

    replay = ledger.replay_verify(snapshot_version)
    restored = ledger.load_snapshot(snapshot_version)
    ledger.close()

    committed_receipts = {}
    for row in restored["receipts"]:
        committed_receipts.setdefault(row["batch_id"], []).append(row)

    return {
        "snapshot_version": snapshot_version,
        "committed_batches": sorted(committed),
        "unknown_batches": sorted(
            row["batch_id"] for row in restored["batches"]
            if row["status"] == "unknown"),
        "quota_reclaimed": recovered,
        "replay": replay,
        "deterministic_state": {
            "committed_receipt_batch_count": len(committed),
            "resumable": True,
            "rule": "仅已提交批可恢复输出摘要；unknown 批需重算后才能复核",
        },
    }


def _now_ms():
    import time
    return int(time.monotonic() * 1000.0)
