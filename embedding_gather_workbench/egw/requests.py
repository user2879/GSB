"""Gather 请求加载与校验。"""
import json

from .errors import E_AGG_MODE_UNKNOWN, E_ROW_INVALID, failure


class GatherRequest:
    def __init__(self, request_id, id_sequence, agg_mode):
        self.request_id = request_id
        self.id_sequence = list(id_sequence)
        self.agg_mode = agg_mode


def load_requests(path, allowed_modes):
    requests = []
    failures = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    failure(E_ROW_INVALID, f"请求 JSON 解析失败: {exc}", source=path, line=line_no)
                )
                continue
            request_id = record.get("request_id")
            seq = record.get("id_sequence")
            agg_mode = record.get("agg_mode")
            if request_id is None or seq is None or agg_mode is None:
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        "缺少 request_id/id_sequence/agg_mode 字段",
                        source=path,
                        line=line_no,
                        request_id=request_id,
                    )
                )
                continue
            if request_id in seen:
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        f"请求 ID {request_id} 重复",
                        source=path,
                        line=line_no,
                        request_id=request_id,
                    )
                )
                continue
            if not isinstance(seq, list) or not seq or not all(
                isinstance(x, int) and not isinstance(x, bool) for x in seq
            ):
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        "id_sequence 必须是非空整数数组",
                        source=path,
                        line=line_no,
                        request_id=request_id,
                    )
                )
                continue
            if agg_mode not in allowed_modes:
                failures.append(
                    failure(
                        E_AGG_MODE_UNKNOWN,
                        f"未知聚合模式 {agg_mode!r}，允许值 {allowed_modes}",
                        source=path,
                        line=line_no,
                        request_id=request_id,
                    )
                )
                continue
            seen.add(request_id)
            requests.append(GatherRequest(request_id, seq, agg_mode))
    requests.sort(key=lambda item: str(item.request_id))
    return requests, failures
