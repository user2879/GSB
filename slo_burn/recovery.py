"""强制终止后的恢复：只承认完整提交，未提交批次标记未知并回收租约配额。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ledger import Ledger


@dataclass
class RecoveryReport:
    recovered_commits: int
    unknown_batches: list[dict]
    reclaimed_lease_quota: list[dict]
    last_commit_seq: int | None
    note: str


def recover(ledger_path: str | Path) -> RecoveryReport:
    """重启恢复入口。

    - 窗口状态与预算只从 conclusions/batches 中已提交记录恢复；
    - pending_batches 中的批次状态为 unknown，必须重跑；
    - 重跑前先回收其租约配额（租约权威不持久化，重启即视为全部到期）。
    不生成、也不承诺任何“强制终止时刻退出报告”。
    """
    ledger = Ledger(ledger_path)
    ledger.verify_format()
    latest = ledger.latest_commit()
    pending = ledger.pending_batches()
    reclaimed = [
        {
            "service_id": item["service_id"],
            "minute": item["minute"],
            "action": "release-lease-quota-before-rerun",
        }
        for item in pending
    ]
    commit_count = ledger.connection.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    ledger.close()
    return RecoveryReport(
        recovered_commits=commit_count,
        unknown_batches=pending,
        reclaimed_lease_quota=reclaimed,
        last_commit_seq=latest["commit_seq"] if latest else None,
        note=(
            "仅依据完整提交的账本记录恢复；未提交批次为 unknown 须重跑，"
            "且重跑前回收租约配额；不生成强制终止退出报告"
        ),
    )
