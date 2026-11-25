"""分片切分：按驱动块组行区间确定性切出批次。"""

from typing import Any, Dict, List

from .invalidation import shard_batch_digest
from .model import BlockGroupSpec


def split_batches(
    plan: Dict[str, Any],
    driver_group: BlockGroupSpec,
    shard_rows: int,
) -> List[Dict[str, Any]]:
    if shard_rows <= 0:
        raise ValueError("shard_rows 必须为正整数")
    batches: List[Dict[str, Any]] = []
    start = 0
    ordinal = 0
    while start < driver_group.rows:
        end = min(driver_group.rows, start + shard_rows)
        batch = {
            "shard_id": f"{plan['plan_id']}:{ordinal:04d}",
            "plan_id": plan["plan_id"],
            "ordinal": ordinal,
            "start": start,
            "end": end,
            "row_count": end - start,
        }
        batch["digest"] = shard_batch_digest(plan, batch)
        batches.append(batch)
        start = end
        ordinal += 1
    if not batches:
        batch = {
            "shard_id": f"{plan['plan_id']}:0000",
            "plan_id": plan["plan_id"],
            "ordinal": 0,
            "start": 0,
            "end": 0,
            "row_count": 0,
        }
        batch["digest"] = shard_batch_digest(plan, batch)
        batches.append(batch)
    return batches
