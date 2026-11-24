"""工作器、请求流、归并规则与命令流的加载校验层。

每个加载器都返回 ``(对象, 失败清单)``：非法项进入失败清单并被跳过，
不抛出异常，保证整轮运行不被单条坏输入中断。
"""

from __future__ import annotations

import json
import os

from .errors import (
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_DUPLICATE_REQUEST_ID,
    E_REQUEST_INVALID,
    E_WORKER_UNKNOWN,
    Failure,
)
from .models import MergeRule, Request, Worker

MIN_PORT = 1024
MAX_PORT = 65535
ALLOWED_QUANTS = frozenset({"int8", "int4", "fp16"})
COMMAND_TYPES = frozenset(
    {
        "replace_receipt",
        "add_batch",
        "switch_param_version",
        "adjust_batch_size",
        "modify_merge_rule",
        "convert_receipt",
    }
)


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def load_worker_config(path: str) -> tuple[dict, list[Failure]]:
    source = os.path.basename(path)
    failures: list[Failure] = []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_workers = payload.get("workers")
    if not isinstance(raw_workers, list):
        raise ValueError(f"{source}: workers 必须是数组")

    workers: list[Worker] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_workers, start=1):
        worker_id = item.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            failures.append(Failure(E_CONFIG_INVALID, source, index, "工作器标识缺失或非字符串"))
            continue
        if worker_id in seen_ids:
            failures.append(Failure(E_CONFIG_INVALID, source, index, f"工作器标识重复: {worker_id}"))
            continue
        port = item.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not MIN_PORT <= port <= MAX_PORT:
            failures.append(
                Failure(E_CONFIG_INVALID, source, index, f"回环端口越界: {port!r}")
            )
            continue
        if not _is_positive_int(item.get("model_dim")) or not _is_positive_int(
            item.get("hidden_dim")
        ):
            failures.append(
                Failure(E_CONFIG_INVALID, source, index, "模型维度或隐藏层维度不是正整数")
            )
            continue
        param_version = item.get("param_version")
        if not isinstance(param_version, int) or isinstance(param_version, bool) or param_version < 0:
            failures.append(Failure(E_CONFIG_INVALID, source, index, "参数版本序号不是非负整数"))
            continue
        quant = item.get("quant", "int8")
        if quant not in ALLOWED_QUANTS:
            failures.append(Failure(E_CONFIG_INVALID, source, index, f"未知量化口径: {quant!r}"))
            continue
        seen_ids.add(worker_id)
        workers.append(
            Worker(
                worker_id=worker_id,
                port=port,
                model_dim=item["model_dim"],
                hidden_dim=item["hidden_dim"],
                param_version=param_version,
                quant=quant,
            )
        )

    batch_size = payload.get("batch_size")
    if not _is_positive_int(batch_size):
        failures.append(Failure(E_CONFIG_INVALID, source, 0, "批大小必须为正整数"))
        batch_size = 1

    return {"workers": tuple(workers), "batch_size": batch_size}, failures


def load_request_stream(path: str) -> tuple[tuple[Request, ...], list[Failure]]:
    source = os.path.basename(path)
    failures: list[Failure] = []
    requests: list[Request] = []
    seen_ids: set[str] = set()
    previous_seq: int | None = None
    with open(path, encoding="utf-8") as handle:
        for line, text in enumerate(handle, start=1):
            text = text.strip()
            if not text:
                continue
            item = json.loads(text)
            seq = item.get("seq")
            request_id = item.get("request_id")
            arrival_ms = item.get("arrival_ms")
            token_count = item.get("token_count", 1)
            if isinstance(request_id, str) and request_id in seen_ids:
                failures.append(
                    Failure(
                        E_DUPLICATE_REQUEST_ID,
                        source,
                        line,
                        f"重复请求标识，保留首次出现: {request_id}",
                    )
                )
                continue
            if not isinstance(seq, int) or isinstance(seq, bool):
                failures.append(Failure(E_REQUEST_INVALID, source, line, "请求序号不是整数"))
                continue
            if previous_seq is not None and seq <= previous_seq:
                failures.append(
                    Failure(E_REQUEST_INVALID, source, line, f"请求序号未严格递增: {seq}")
                )
                continue
            if not isinstance(arrival_ms, int) or isinstance(arrival_ms, bool) or arrival_ms < 0:
                failures.append(Failure(E_REQUEST_INVALID, source, line, "到达毫秒不是非负整数"))
                continue
            if not isinstance(request_id, str) or not request_id:
                failures.append(Failure(E_REQUEST_INVALID, source, line, "请求标识缺失"))
                continue
            if not _is_positive_int(token_count):
                failures.append(Failure(E_REQUEST_INVALID, source, line, "令牌数不是正整数"))
                continue
            previous_seq = seq
            seen_ids.add(request_id)
            requests.append(Request(seq, request_id, arrival_ms, token_count))
    return tuple(requests), failures


def load_merge_rules(path: str, workers: tuple[Worker, ...]) -> tuple[tuple[MergeRule, ...], list[Failure]]:
    source = os.path.basename(path)
    failures: list[Failure] = []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    declared = {worker.worker_id for worker in workers}
    rules: list[MergeRule] = []
    for index, item in enumerate(payload.get("rules", []), start=1):
        rule_id = item.get("rule_id")
        worker_ids = item.get("worker_ids")
        batch_size = item.get("batch_size")
        if not isinstance(rule_id, str) or not rule_id:
            failures.append(Failure(E_CONFIG_INVALID, source, index, "归并规则标识缺失"))
            continue
        if not isinstance(worker_ids, list) or not worker_ids:
            failures.append(Failure(E_CONFIG_INVALID, source, index, "工作器集合必须是非空数组"))
            continue
        unknown = [worker_id for worker_id in worker_ids if worker_id not in declared]
        if unknown:
            for worker_id in unknown:
                failures.append(
                    Failure(
                        E_WORKER_UNKNOWN,
                        source,
                        index,
                        f"归并规则 {rule_id} 引用未声明工作器: {worker_id}",
                    )
                )
            continue
        if not _is_positive_int(batch_size):
            failures.append(Failure(E_CONFIG_INVALID, source, index, f"归并规则 {rule_id} 批大小非正"))
            continue
        rules.append(
            MergeRule(rule_id=rule_id, worker_ids=tuple(sorted(worker_ids)), batch_size=batch_size)
        )
    return tuple(rules), failures


def load_commands(path: str) -> tuple[tuple[dict, ...], list[Failure]]:
    source = os.path.basename(path)
    failures: list[Failure] = []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    commands: list[dict] = []
    for index, item in enumerate(payload.get("commands", []), start=1):
        command_type = item.get("type")
        if command_type not in COMMAND_TYPES:
            failures.append(Failure(E_COMMAND_INVALID, source, index, f"未知命令类型: {command_type!r}"))
            continue
        if "cursor" not in item:
            item = {**item, "cursor": index}
        commands.append(item)
    return tuple(commands), failures
