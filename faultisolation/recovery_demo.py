"""强制终止与账本恢复演示。

流程：

1. 用前半段事件跑实时仲裁并按批次提交账本；
2. 在中途强制 ``kill -9`` 仲裁子进程（模拟进程被强制终止）；
3. 重启新子进程，仅依据账本中 *已完整提交* 的记录恢复租约集合与槽位占用；
4. 未提交批次对应的游标标记为 ``unknown_from_seq``，要求从中断点重跑；
5. 重跑全量后账本摘要与独立重放一致。
"""

from __future__ import annotations

import json
from pathlib import Path

from .clock import LogicalClock
from .engine import Engine, canonical_event_order
from .errors import LedgerError
from .ledger import Ledger
from .process import ArbiterProcessGroup, LiveScheduler
from .replay import replay_path, verify_ledger
from .spec import SpecBundle, load_bundle


def run_recovery_demo(spec_path: str, session_path: str, ledger_path: str,
                      split_ratio: float = 0.5) -> dict:
    bundle = load_bundle(spec_path, session_path)
    ledger_file = Path(ledger_path)
    if ledger_file.exists():
        ledger_file.unlink()
    for suffix in ("-wal", "-shm"):
        extra = Path(str(ledger_file) + suffix)
        if extra.exists():
            extra.unlink()

    ordered = canonical_event_order(bundle.events, {i.instance_id: i.priority for i in bundle.instances})
    split = max(1, int(len(ordered) * split_ratio))
    head_events = ordered[:split]
    tail_events = ordered[split:]

    # 阶段一：仅处理前半段，强制终止服务进程（不发送 shutdown）
    head_bundle = _clone_with_events(bundle, head_events)
    group = ArbiterProcessGroup(head_bundle)
    group.start()
    ledger = Ledger(ledger_file)
    try:
        scheduler = LiveScheduler(head_bundle, ledger, path="event")
        scheduler.run()
        group.kill_all()  # 强制终止，模拟 kill -9
    finally:
        ledger.close()

    # 阶段二：从已提交账本恢复（不依赖任何内存状态）
    ledger = Ledger(ledger_file)
    recovery = ledger.recover()
    committed_seq = recovery.last_committed_seq
    ledger.close()

    # 阶段三：重启服务并重跑全量，验证账本可重建为完整一致状态
    full_ledger_path = str(Path(ledger_path).with_name(Path(ledger_path).stem + "-rerun.sqlite"))
    full_ledger = Ledger(full_ledger_path, replace=True)
    group2 = ArbiterProcessGroup(bundle)
    group2.start()
    try:
        scheduler2 = LiveScheduler(bundle, full_ledger, path="event")
        rerun_result = scheduler2.run()
    finally:
        group2.stop()
        full_ledger.close()

    verify = verify_ledger(bundle, full_ledger_path)
    return {
        "total_events": len(ordered),
        "forced_kill_after_event": head_events[-1].seq,
        "committed_seq_before_kill": committed_seq,
        "unknown_from_seq": recovery.unknown_from_seq,
        "recovered_active_leases": len(recovery.active_leases),
        "recovered_slot_owners": recovery.slot_owners,
        "digest_ok_at_recovery": recovery.digest_ok,
        "rerun_matched_telemetry": sum(1 for t in rerun_result["telemetry"] if t["matched"]),
        "rerun_verify": verify,
    }


def _clone_with_events(bundle: SpecBundle, events) -> SpecBundle:
    return SpecBundle(
        domains=bundle.domains, instances=bundle.instances, events=events,
        session=bundle.session, spec_path=bundle.spec_path,
        events_path=bundle.events_path, session_path=bundle.session_path,
    )
