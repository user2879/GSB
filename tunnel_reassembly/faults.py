"""异常注入：通道声明损坏 / 偏差样本不足 / 快照摘要不符。

注入方式为显式参数，确定性可复现；三类异常分别映射
E_CHANNEL、E_ESTIMATE、E_SNAPSHOT 并写入结构化日志。
"""

from __future__ import annotations

import copy
import json

from .config import canonical_digest, load_channels
from .errors import BusinessError, E_CHANNEL, E_ESTIMATE, E_SNAPSHOT


def inject_bad_channel(config_obj: dict) -> None:
    """把一个通道的 window_ms 篡改为 0，触发 E_CHANNEL 整体拒绝。"""
    corrupted = copy.deepcopy(config_obj)
    corrupted["channels"][0]["window_ms"] = 0
    load_channels(corrupted)


def force_estimate_pending(spec, groups: list, logger=None) -> int:
    """把最小样本下限抬高到样本组数之上，使估计始终未定。

    返回被标记为 PENDING 的组数；真实裁决会拒绝据此开启窗口。
    """
    needed = len(groups) + 1
    inflated = spec.__class__(**{**spec.__dict__, "min_estimate_samples": needed})
    from .engine import estimate_versions

    versions = estimate_versions(groups, inflated.min_estimate_samples)
    pending = sum(1 for v in versions if v.skew_ms is None) - 1
    if not any(v.skew_ms is not None for v in versions):
        if logger is not None:
            logger.event(
                "ERROR",
                E_ESTIMATE,
                f"注入：样本下限抬高到 {needed}，估计始终未定，拒绝开启窗口",
                channel_id=spec.channel_id,
                groups=len(groups),
            )
    return max(pending, 0)


def corrupt_snapshot_digest(ledger) -> None:
    """直接篡改快照内容摘要，恢复时必须以 E_SNAPSHOT 整体替换。"""
    row = ledger.conn.execute("SELECT snapshot_id FROM snapshots ORDER BY created_ms DESC LIMIT 1").fetchone()
    if row is None:
        raise BusinessError(E_SNAPSHOT, "无快照可注入摘要损坏")
    ledger.conn.execute(
        "UPDATE snapshots SET content_digest=? WHERE snapshot_id=?",
        ("sha256:" + "0" * 64, row[0]),
    )
    ledger.conn.commit()
