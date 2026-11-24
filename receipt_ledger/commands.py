"""命令流的确定性应用。

所有命令都在内存状态上做纯函数式变换并返回新状态与失败清单，顺序固定，
不依赖线程调度；重复运行同一命令序列必然得到同一最终状态。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace
import json

from .convert import conversion_record, downgrade, receipt_ref, upgrade
from .errors import E_COMMAND_INVALID, Failure
from .models import MergeRule, Receipt
from .receipts import parse_receipt_line

if TYPE_CHECKING:
    from .engine import PipelineState


def apply_commands(
    state: "PipelineState",
    commands: tuple[dict, ...],
) -> tuple["PipelineState", list[Failure], list[dict], list[str]]:
    failures: list[Failure] = []
    conversion_log: list[dict] = list(state.conversion_log)
    todo_queue: list[str] = list(state.todo_queue)
    receipts = list(state.receipts)
    workers = list(state.workers)
    rules = list(state.rules)
    batch_size = state.batch_size

    for command in commands:
        command_type = command["type"]
        cursor = command.get("cursor")
        if command_type == "replace_receipt":
            receipts, failures = _replace_receipt(receipts, command, failures)
            todo_queue.append(f"c{cursor}:reparse:{command['file']}")
        elif command_type == "add_batch":
            receipts, failures = _add_batch(receipts, command, failures)
            todo_queue.append(f"c{cursor}:add_batch:{command['payload']['batch_id']}")
        elif command_type == "switch_param_version":
            workers = _switch_param_version(workers, command)
            todo_queue.append(f"c{cursor}:invalidate:worker:{command['worker_id']}")
        elif command_type == "adjust_batch_size":
            if not isinstance(command.get("batch_size"), int) or command["batch_size"] <= 0:
                failures.append(
                    Failure(E_COMMAND_INVALID, "commands", cursor or 0, "批大小调整值非正整数")
                )
            else:
                batch_size = command["batch_size"]
            todo_queue.append(f"c{cursor}:batch_size:{batch_size}")
        elif command_type == "modify_merge_rule":
            rules, failures = _modify_rule(rules, workers, command, failures)
            todo_queue.append(f"c{cursor}:rule:{command['rule_id']}")
        elif command_type == "convert_receipt":
            receipts, conversion_log, failures = _convert(
                receipts, conversion_log, command, failures
            )
            todo_queue.append(f"c{cursor}:convert:{command['target']}")

    new_state = replace(
        state,
        receipts=tuple(receipts),
        workers=tuple(workers),
        rules=tuple(rules),
        batch_size=batch_size,
        conversion_log=tuple(conversion_log),
        todo_queue=tuple(todo_queue),
    )
    return new_state, failures, conversion_log, todo_queue


def _replace_receipt(receipts: list[Receipt], command: dict, failures: list[Failure]):
    file_name = command["file"]
    line = int(command["line"])
    receipt, failure = parse_receipt_line(
        json.dumps(command["payload"], ensure_ascii=False, sort_keys=True),
        file_name,
        line,
        command.get("version"),
    )
    if failure is not None:
        failures.append(failure)
        return receipts, failures
    replaced = False
    for index, existing in enumerate(receipts):
        if existing.source_file == file_name and existing.line == line:
            receipts[index] = receipt
            replaced = True
            break
    if not replaced:
        receipts.append(receipt)
    return receipts, failures


def _add_batch(receipts: list[Receipt], command: dict, failures: list[Failure]):
    payload = command["payload"]
    receipt, failure = parse_receipt_line(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        command.get("file", "commands.jsonl"),
        int(command.get("line", 0)),
        payload.get("version"),
    )
    if failure is not None:
        failures.append(failure)
        return receipts, failures
    receipts.append(receipt)
    return receipts, failures


def _switch_param_version(workers: list, command: dict) -> list:
    updated = []
    for worker in workers:
        if worker.worker_id == command["worker_id"]:
            updated.append(replace(worker, param_version=int(command["param_version"])))
        else:
            updated.append(worker)
    return updated


def _modify_rule(rules: list, workers: list, command: dict, failures: list[Failure]):
    declared = {worker.worker_id for worker in workers}
    unknown = [worker_id for worker_id in command["worker_ids"] if worker_id not in declared]
    if unknown:
        for worker_id in unknown:
            failures.append(
                Failure(
                    "E_WORKER_UNKNOWN",
                    "commands",
                    command.get("cursor", 0),
                    f"归并规则 {command['rule_id']} 引用未声明工作器: {worker_id}",
                )
            )
        return rules, failures
    updated = []
    found = False
    for rule in rules:
        if rule.rule_id == command["rule_id"]:
            found = True
            updated.append(
                MergeRule(
                    rule_id=rule.rule_id,
                    worker_ids=tuple(sorted(command["worker_ids"])),
                    batch_size=int(command["batch_size"]),
                )
            )
        else:
            updated.append(rule)
    if not found:
        updated.append(
            MergeRule(
                rule_id=command["rule_id"],
                worker_ids=tuple(sorted(command["worker_ids"])),
                batch_size=int(command["batch_size"]),
            )
        )
    return updated, failures


def _convert(receipts: list[Receipt], log: list[dict], command: dict, failures: list[Failure]):
    target = command["target"]
    ref = command.get("receipt_ref")
    order = ("r1", "r2", "r3")
    updated = list(receipts)
    for index, receipt in enumerate(receipts):
        if ref is not None and receipt_ref(receipt) != ref:
            continue
        if target == receipt.source_version:
            continue
        if order.index(target) > order.index(receipt.source_version):
            new_receipt = upgrade(receipt, target)
            updated[index] = new_receipt
            log.append(conversion_record("upgrade", new_receipt, target, receipt_ref(new_receipt)))
        else:
            new_receipt, sidecar = downgrade(receipt, target)
            updated[index] = new_receipt
            record = conversion_record("downgrade", new_receipt, target, receipt_ref(receipt))
            record["sidecar"] = sidecar
            log.append(record)
    if ref is not None and all(receipt_ref(receipt) != ref for receipt in receipts):
        failures.append(
            Failure(
                E_COMMAND_INVALID,
                "commands",
                command.get("cursor", 0),
                f"转换命令找不到回执引用: {ref}",
            )
        )
    return updated, log, failures
