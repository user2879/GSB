"""批次规划、多版本回执采纳与灰度结论合成。

采纳规则（全序、确定性，无字典/哈希遍历依赖）：
  1. 参数版本序号降序；
  2. 回执格式版本降序（r3 > r2 > r1）；
  3. 归并序号升序（同一规则内格式声明顺序给出的稳定序号）。
"""

from dataclasses import dataclass

from .codec import format_rank
from .deterministic import request_destination
from .digest import canonical_json, content_digest


@dataclass(frozen=True)
class BatchTask:
    batch_id: str
    rule_id: str
    index: int
    worker_id: str
    worker_ids: tuple
    param_version: int
    receipt_formats: tuple
    retry_policy: str
    max_retries: int
    socket_timeout_ms: int
    requests: tuple
    request_seqs: tuple
    destinations: dict

    def context(self, param_digest: str, quant: str) -> dict:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "destination": self.worker_id,
            "quant": quant,
            "param_version": self.param_version,
            "request_seqs": list(self.request_seqs),
            "param_digest": param_digest,
        }


def build_batch_plan(spec) -> tuple:
    """按规则顺序、请求到达顺序与批大小切分，规划稳定的批次序列。"""
    plan = []
    for rule in spec.rules:
        ordered_workers = tuple(rule.worker_ids)
        for start in range(0, len(spec.requests), rule.batch_size):
            chunk = spec.requests[start:start + rule.batch_size]
            index = start // rule.batch_size
            batch_id = f"{rule.rule_id}-b{index:04d}"
            worker_id = ordered_workers[index % len(ordered_workers)]
            destinations = {
                req.seq: request_destination(req.client_key, ordered_workers)
                for req in chunk
            }
            plan.append(
                BatchTask(
                    batch_id=batch_id, rule_id=rule.rule_id, index=index,
                    worker_id=worker_id, worker_ids=ordered_workers,
                    param_version=rule.param_version,
                    receipt_formats=rule.receipt_formats,
                    retry_policy=rule.retry_policy,
                    max_retries=rule.max_retries,
                    socket_timeout_ms=rule.socket_timeout_ms,
                    requests=tuple(chunk),
                    request_seqs=tuple(req.seq for req in chunk),
                    destinations=dict(sorted(destinations.items())),
                )
            )
    return tuple(plan)


def merge_receipts(normalized_receipts: list) -> tuple:
    """从同批多个版本回执选出采纳回执，返回 (采纳, 否决列表)。

    每个输入字典需含 merge_seq；输出的否决项携带稳定否决理由。
    """
    if not normalized_receipts:
        raise ValueError("归并至少需要一个回执")
    ordered = sorted(
        normalized_receipts,
        key=lambda item: (
            -item["param_version"],
            -format_rank(item["source_format"]),
            item["merge_seq"],
        ),
    )
    winner = ordered[0]
    rejected = []
    for item in ordered[1:]:
        if item["param_version"] < winner["param_version"]:
            reason = "superseded:lower-param-version"
        elif format_rank(item["source_format"]) < format_rank(winner["source_format"]):
            reason = "superseded:lower-format-rank"
        else:
            reason = "superseded:later-merge-seq"
        rejected.append(
            {
                "batch_id": item["batch_id"],
                "source_format": item["source_format"],
                "param_version": item["param_version"],
                "merge_seq": item["merge_seq"],
                "reason": reason,
                "output_digest": item["output_digest"],
            }
        )
    rejected.sort(key=lambda item: item["merge_seq"])
    return winner, rejected


def destination_distribution(task: BatchTask) -> list:
    """逐批去向分布：按去向标识排序的 [去向, 计数] 列表。"""
    counts = {}
    for destination in task.destinations.values():
        counts[destination] = counts.get(destination, 0) + 1
    return [[key, counts[key]] for key in sorted(counts)]


def synthesize_conclusion(task: BatchTask, winner: dict,
                          per_request_outputs: list) -> dict:
    """合成批次归并结论；content_digest 覆盖全部结论字段，保证可逐字节复核。"""
    conclusion = {
        "batch_id": task.batch_id,
        "rule_id": task.rule_id,
        "worker_id": task.worker_id,
        "param_version": winner["param_version"],
        "source_format": winner["source_format"],
        "param_digest": winner["param_digest"],
        "request_seqs": list(task.request_seqs),
        "destination_distribution": destination_distribution(task),
        "request_destinations": [
            [seq, task.destinations[seq]] for seq in task.request_seqs
        ],
        "output_digest": winner["output_digest"],
        "per_request_outputs": [
            [item[0], item[1]] for item in sorted(per_request_outputs)
        ],
        "filled_fields": winner["filled_fields"],
    }
    conclusion["content_digest"] = content_digest(
        conclusion, prefix="conclusion"
    )
    return conclusion


def conclusion_bytes(conclusion: dict) -> bytes:
    """结论的规范化字节形态，快速视图与权威重算据此逐字节比对。"""
    return canonical_json(conclusion)

