"""恢复后的一致性重放校验（需求 8）。

从快照恢复后，按账本记录的预算参数与进度游标重新推进剩余批次，
将以下各项同账本逐项比较：
  - 批次结局（state）、块摘要（digest）、带宽占用（bandwidth_kbps）；
  - 紧迫度排序（ranking order）；
  - 误差归因（三段误差界）。
差异定位到首个不一致的批次、块与字段，以 E_REPLAY 报明依据。
等待时长为单调时钟量，非确定性，不参与逐项比较（仅校验非负）。
"""

from __future__ import annotations

import os
import tempfile

from .content import BlockRegistry
from .errors import E_REPLAY, StructuredLog, WorkbenchError
from .fixedpoint import ErrorLedger, FixedPoint
from .ledger import Ledger
from .scheduler import Scheduler
from .verdict import Candidate, UrgencyRanker
from .workbench import Session

_COMPARE_FIELDS = ("block_id", "node_id", "start_byte", "end_byte", "digest",
                   "bandwidth_kbps", "state", "err_bound_quanta")


def _fail(batch_id: str, block_id: str, fld: str, expected, actual) -> WorkbenchError:
    return WorkbenchError(
        E_REPLAY,
        f"重放与账本不一致: 批次 {batch_id} 字段 {fld}",
        batch_id=batch_id,
        block_id=block_id,
        field=fld,
        expected=expected,
        actual=actual,
    )


def replay_group(session: Session, group_id: str) -> dict:
    """对单个副本组执行重放校验，返回比对报告。"""
    ctx = session.groups[group_id]
    spec = ctx.spec
    assert session.ledger is not None
    # 仅比较本组、当前版本的已提交批次；旧版本记录已整体标记陈旧，不参与比较。
    recorded = {
        row["batch_id"]: row
        for row in session.ledger.committed_batches()
        if row["state"] != "UNKNOWN"
        and row["batch_id"].startswith(f"{group_id}:")
        and row["spec_version"] == ctx.version
    }
    progress = session.ledger.progress()

    # 依据账本进度游标与预算参数，在临时账本中重新推进。
    fp = FixedPoint(spec.params.scale_exponent)
    err_ledger = ErrorLedger(fp)
    registry = BlockRegistry(os.path.join(session.workdir, "blocks", group_id))
    for block in spec.blocks:
        registry.register_existing(block)
    ranker = UrgencyRanker(fp)
    candidates = [Candidate(b, spec.node(b.node_id)) for b in spec.blocks]
    ranking = ranker.adjudicate_all(candidates)

    tmpdir = tempfile.mkdtemp(prefix="rb-replay-")
    replay_ledger = Ledger(os.path.join(tmpdir, "replay.db"), "replay", ctx.version)
    try:
        scheduler = Scheduler(
            spec, registry, err_ledger,
            ledger=replay_ledger, spec_version=ctx.version,
            ranked_order=ranking["order"],
        )
        summary = scheduler.run()
    finally:
        replayed = {r["batch_id"]: r for r in replay_ledger.committed_batches()}
        replay_ledger.close()

    # 1) 批次逐项比较（确定性字段）。账本中已提交批次必须在重放中一致重现；
    # 恢复后重跑的批次（账本中为 UNKNOWN/缺失）允许仅出现在重放侧。
    rerun_batches = sorted(set(replayed) - set(recorded))
    for batch_id in sorted(recorded):
        rec = recorded[batch_id]
        rep = replayed.get(batch_id)
        block_id = rec["block_id"]
        if rep is None:
            raise _fail(batch_id, block_id, "presence", "committed", "missing")
        for fld in _COMPARE_FIELDS:
            if rec[fld] != rep[fld]:
                raise _fail(batch_id, block_id, fld, rec[fld], rep[fld])
        if rec["wait_ms"] < 0:
            raise _fail(batch_id, block_id, "wait_ms", ">=0", rec["wait_ms"])

    # 2) 紧迫度排序比较。
    if ctx.last_ranking is not None:
        recorded_order = [i["block_id"] for i in ctx.last_ranking["order"]]
        replay_order = [i["block_id"] for i in ranking["order"]]
        if recorded_order != replay_order:
            raise _fail("(ranking)", "(ranking)", "urgency_order",
                        recorded_order, replay_order)

    # 3) 误差归因比较（仅确定段：量化截断与批次切分取整；
    #    排队延迟折算依赖单调时钟，与等待时长一样不参与逐项比较）。
    replay_bound = err_ledger.global_bound()
    current_bound = ctx.err_ledger.global_bound() if ctx.err_ledger else None
    if current_bound is not None and current_bound.total() > 0:
        for fld in ("quantization_trunc", "batch_split_rounding"):
            if getattr(current_bound, fld) != getattr(replay_bound, fld):
                raise _fail("(error_attribution)", "(error_attribution)", fld,
                            getattr(current_bound, fld), getattr(replay_bound, fld))

    report = {
        "group_id": group_id,
        "batches_compared": len(recorded),
        "rerun_batches": rerun_batches,
        "cursor": progress["cursor"],
        "consumed_bytes": progress["consumed_bytes"],
        "replayed_consumed_bytes": summary["consumed_bytes"],
        "consistent": True,
    }
    session.log.emit("REPLAY", "重放一致", **report)
    return report
