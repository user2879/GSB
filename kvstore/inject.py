"""异常注入：仅通过生成被篡改的输入文件或传递参数实现，绝不修改源码。

三类注入对应三个业务码：

* ``trace_corrupt``   破坏轨迹 JSON / 序号连续性 -> E_TRACE
* ``index_rollback``  快照内索引条目版本回退     -> E_VERSION
* ``snapshot_digest`` 快照内容摘要不匹配         -> E_SNAPSHOT
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import codec
from .trace import write_trace


def inject_trace_corrupt(trace_path: str, out_dir: str) -> str:
    with open(trace_path, "r", encoding="utf-8") as handle:
        operations = [json.loads(line) for line in handle if line.strip()]
    # 让第二条操作的序号跳号，破坏连续性
    if len(operations) >= 2:
        operations[1]["op_seq"] = operations[1]["op_seq"] + 5
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trace_corrupt.jsonl")
    write_trace(out_path, operations)
    return out_path


def inject_trace_bad_json(trace_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trace_bad_json.jsonl")
    with open(trace_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) >= 2:
        lines[1] = lines[1].rstrip("\n")[: max(5, len(lines[1]) // 2)] + "\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
    return out_path


def inject_index_rollback(snapshot_path: str, out_dir: str) -> str:
    """把快照中某个索引条目的 entry_version 改小，制造版本回退输入。"""
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload["index_entries"]:
        raise RuntimeError("快照中没有索引条目可注入版本回退")
    target = payload["index_entries"][0]
    # 注入负的条目版本号，确定性触发版本回退/非法版本 E_VERSION
    target["entry_version"] = -1
    # 重新计算摘要使文件本身自洽，确保失败点是版本回退而非摘要
    from .snapshot import compute_digest

    payload["content_digest"] = compute_digest(payload)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "snapshot_index_rollback.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return out_path


def inject_snapshot_digest(snapshot_path: str, out_dir: str) -> str:
    """只篡改摘要字段，内容保持不变 -> E_SNAPSHOT。"""
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["content_digest"] = "0" * 64
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "snapshot_bad_digest.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return out_path


INJECTORS = {
    "trace_corrupt": inject_trace_corrupt,
    "trace_bad_json": inject_trace_bad_json,
    "index_rollback": inject_index_rollback,
    "snapshot_digest": inject_snapshot_digest,
}
