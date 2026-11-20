"""候选集 JSONL 加载与逐行校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import RerankConfig
from .errors import RecordInvalidError
from .models import Candidate, Failure


def _reject_constant(value: str) -> None:
    raise ValueError(f"非有限 JSON 常量: {value}")


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_record(raw: dict, line_no: int, config: RerankConfig) -> Candidate:
    request_id = raw.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise RecordInvalidError(
            "请求 ID 为空", details={"line_no": line_no}
        )

    item_id = raw.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise RecordInvalidError(
            f"候选物品 ID 非法（行 {line_no}）",
            details={"line_no": line_no, "request_id": request_id},
        )

    recall = raw.get("recall_scores")
    if not isinstance(recall, dict) or set(recall.keys()) != set(config.recall_channels):
        raise RecordInvalidError(
            f"召回通道不匹配（行 {line_no}）",
            details={"line_no": line_no, "request_id": request_id, "item_id": item_id},
        )
    for channel, score in recall.items():
        if not _is_finite_number(score):
            raise RecordInvalidError(
                f"召回分数非有限数: {channel}（行 {line_no}）",
                details={"line_no": line_no, "request_id": request_id, "item_id": item_id},
            )

    dense = raw.get("dense")
    if not isinstance(dense, list) or len(dense) != config.dense_dim:
        raise RecordInvalidError(
            f"稠密特征维度不匹配（行 {line_no}）",
            details={"line_no": line_no, "request_id": request_id, "item_id": item_id},
        )
    for value in dense:
        if not _is_finite_number(value):
            raise RecordInvalidError(
                f"稠密特征含非有限值（行 {line_no}）",
                details={"line_no": line_no, "request_id": request_id, "item_id": item_id},
            )

    sparse = raw.get("sparse")
    if not isinstance(sparse, list) or len(sparse) != config.sparse_dim:
        raise RecordInvalidError(
            f"稀疏计数维度不匹配（行 {line_no}）",
            details={"line_no": line_no, "request_id": request_id, "item_id": item_id},
        )
    for value in sparse:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RecordInvalidError(
                f"稀疏计数必须是非负整数（行 {line_no}）",
                details={"line_no": line_no, "request_id": request_id, "item_id": item_id},
            )

    return Candidate(
        request_id=request_id,
        item_id=item_id,
        recall_scores={channel: float(recall[channel]) for channel in config.recall_channels},
        dense=[float(value) for value in dense],
        sparse=[int(value) for value in sparse],
        line_no=line_no,
    )


def load_candidates(
    path: str | Path, config: RerankConfig
) -> tuple[dict[str, list[Candidate]], list[Failure]]:
    """逐行解析 JSONL，返回按请求聚合的候选与失败清单。

    非法记录被隔离（跳过），合法记录继续参与；请求保持首次出现顺序，
    候选保持文件行顺序，保证全量与增量运行的确定性。
    """
    path = Path(path)
    requests: dict[str, list[Candidate]] = {}
    order: list[str] = []
    failures: list[Failure] = []
    seen_items: dict[str, set[str]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                failures.append(
                    Failure("E_RECORD_INVALID", "request", f"line:{line_no}",
                            f"JSON 解析失败或含非有限常量: {exc}", line_no)
                )
                continue
            if not isinstance(raw, dict):
                failures.append(
                    Failure("E_RECORD_INVALID", "request", f"line:{line_no}",
                            "记录不是 JSON 对象", line_no)
                )
                continue
            try:
                candidate = _validate_record(raw, line_no, config)
            except RecordInvalidError as exc:
                target = exc.details.get("request_id") or f"line:{line_no}"
                failures.append(
                    Failure(exc.code, "request", str(target), exc.message, line_no)
                )
                continue

            bucket = seen_items.setdefault(candidate.request_id, set())
            if candidate.item_id in bucket:
                failures.append(
                    Failure(
                        "E_RECORD_INVALID",
                        "request",
                        candidate.request_id,
                        f"候选物品 ID 在请求内重复: {candidate.item_id}",
                        line_no,
                    )
                )
                continue
            bucket.add(candidate.item_id)
            if candidate.request_id not in requests:
                requests[candidate.request_id] = []
                order.append(candidate.request_id)
            requests[candidate.request_id].append(candidate)

    ordered = {request_id: requests[request_id] for request_id in order}
    return ordered, failures
