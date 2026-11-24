"""独立异常注入配置。

注入只通过显式传入的注入 JSON 生效，与正常样本目录隔离；每条注入带稳定
编号，编号会贯穿解析失败清单与最终报告，用于证明故障命中。
"""

from __future__ import annotations

import json
import os

from .errors import E_INJECTION, Failure

INJECTION_TYPES = frozenset(
    {
        "non_finite_duration",
        "unknown_param_version",
        "truncated_line",
        "duplicate_request_id",
    }
)


def load_injection(path: str) -> tuple[dict, list[Failure]]:
    source = os.path.basename(path)
    failures: list[Failure] = []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = payload.get("injections", [])
    receipt_index: dict[str, dict[int, str]] = {}
    duplicate_requests: list[dict] = []
    unknown_versions: set[tuple[str, int]] = set()
    valid_entries: list[dict] = []
    seen_ids: set[str] = set()

    for index, entry in enumerate(entries, start=1):
        injection_id = entry.get("id") or f"INJ-{index:03d}"
        injection_type = entry.get("type")
        if injection_id in seen_ids:
            failures.append(Failure(E_INJECTION, source, index, f"注入编号重复: {injection_id}"))
            continue
        if injection_type not in INJECTION_TYPES:
            failures.append(
                Failure(E_INJECTION, source, index, f"未知注入类型: {injection_type!r}")
            )
            continue
        seen_ids.add(injection_id)
        entry = {**entry, "id": injection_id}
        valid_entries.append(entry)

        if injection_type in ("non_finite_duration", "truncated_line"):
            receipt_index.setdefault(entry["file"], {})[int(entry["line"])] = injection_id
        elif injection_type == "unknown_param_version":
            unknown_versions.add((entry["worker_id"], int(entry["param_version"])))
            if "file" in entry and "line" in entry:
                receipt_index.setdefault(entry["file"], {})[int(entry["line"])] = injection_id
        elif injection_type == "duplicate_request_id":
            duplicate_requests.append(entry)

    plan = {
        "entries": valid_entries,
        "receipt_index": receipt_index,
        "duplicate_requests": duplicate_requests,
        "unknown_versions": unknown_versions,
        "source": source,
    }
    return plan, failures


def apply_duplicate_requests(
    requests: tuple,
    duplicate_entries: list[dict],
) -> tuple[tuple, list[Failure], list[str]]:
    """按注入计划重放重复请求标识，命中编号进入失败清单。"""

    from .models import Request

    failures: list[Failure] = []
    hit_ids: list[str] = []
    existing = {request.request_id: request for request in requests}
    extended = list(requests)
    next_seq = max((request.seq for request in requests), default=0)
    for entry in duplicate_entries:
        injection_id = entry["id"]
        target_id = entry["request_id"]
        if target_id not in existing:
            failures.append(
                Failure(
                    E_INJECTION,
                    "injection",
                    0,
                    f"{injection_id} 目标请求不存在: {target_id}",
                    injection_id,
                )
            )
            continue
        next_seq += 1
        extended.append(
            Request(
                seq=next_seq,
                request_id=target_id,
                arrival_ms=int(entry.get("arrival_ms", 0)),
                token_count=1,
            )
        )
        failures.append(
            Failure(
                "E_DUPLICATE_REQUEST_ID",
                "injection",
                next_seq,
                f"{injection_id} 命中：重复请求标识 {target_id}",
                injection_id,
            )
        )
        hit_ids.append(injection_id)
    return tuple(extended), failures, hit_ids
