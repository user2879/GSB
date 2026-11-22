"""Strict local-JSON loaders for partitions and transaction batches."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .canonical import message_digest
from .errors import BusinessError, OrderError, SpecError
from .models import FORMAT_VERSION, Message, Partition, Transaction
from .observability import StructuredLogger


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json(path: str | os.PathLike[str]) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SpecError(
            f"无法读取合法 JSON 文件: {exc}",
            field="file",
            details={"path": str(path), "source": type(exc).__name__},
        ) from exc


def _txn_reference(raw: Any, index: int) -> str:
    if isinstance(raw, dict):
        value = raw.get("transaction_id")
        if isinstance(value, str) and value:
            return value
    return f"<transaction-index:{index}>"


def _check_payload_json(value: Any, path: tuple[str | int, ...] = ()) -> None:
    """Reject JSON values that cannot round-trip through the canonical encoder."""

    if isinstance(value, float) and not math.isfinite(value):
        field = "payload" if not path else "payload" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in path
        )
        raise SpecError("消息载荷包含非有限浮点数", field=field)
    if isinstance(value, dict):
        for key, child in value.items():
            _check_payload_json(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_payload_json(child, path + (index,))


def parse_partitions(data: Any) -> tuple[Partition, ...]:
    if isinstance(data, dict):
        if "format_version" in data:
            version = data["format_version"]
            if not _is_int(version) or version < 1:
                raise SpecError(
                    "分区声明 format_version 必须为正整数",
                    field="format_version",
                )
        raw_items = data.get("partitions")
    else:
        raw_items = data
    if not isinstance(raw_items, list):
        raise SpecError("分区声明必须是数组或包含 partitions 数组", field="partitions")
    if not raw_items:
        raise SpecError("至少需要一个分区", field="partitions")

    seen: set[str] = set()
    partitions: list[Partition] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise SpecError(
                "分区项必须是对象",
                field=f"partitions[{index}]",
            )
        partition_id = raw.get("partition_id")
        if not isinstance(partition_id, str) or not partition_id.strip():
            raise SpecError(
                "partition_id 必须为非空字符串",
                field=f"partitions[{index}].partition_id",
            )
        if partition_id in seen:
            raise SpecError(
                "partition_id 必须唯一",
                partition_id=partition_id,
                field=f"partitions[{index}].partition_id",
            )
        version = raw.get("partition_version")
        if not _is_int(version) or version < 0:
            raise SpecError(
                "partition_version 必须为非负整数",
                partition_id=partition_id,
                field=f"partitions[{index}].partition_version",
            )
        maximum = raw.get("max_position")
        if not _is_int(maximum) or maximum < 0:
            raise SpecError(
                "max_position 必须为非负整数",
                partition_id=partition_id,
                field=f"partitions[{index}].max_position",
            )
        seen.add(partition_id)
        partitions.append(
            Partition(
                partition_id=partition_id,
                partition_version=version,
                max_position=maximum,
            )
        )
    return tuple(partitions)


def load_partitions(
    path: str | os.PathLike[str],
    logger: StructuredLogger | None = None,
) -> tuple[Partition, ...]:
    try:
        return parse_partitions(_read_json(path))
    except BusinessError as error:
        if logger is not None:
            logger.write_error(error)
        raise


def parse_transactions(
    data: Any,
    partitions: Iterable[Partition],
) -> tuple[Transaction, ...]:
    partition_map = {partition.partition_id: partition for partition in partitions}
    if not partition_map:
        raise SpecError("事务批次缺少可用分区声明", field="partitions")

    if isinstance(data, dict):
        if "format_version" in data:
            version = data["format_version"]
            if not _is_int(version) or version < 1:
                raise SpecError(
                    "事务批次 format_version 必须为正整数",
                    field="format_version",
                )
        raw_items = data.get("transactions")
    else:
        raw_items = data
    if not isinstance(raw_items, list):
        raise SpecError("事务批次必须是数组或包含 transactions 数组", field="transactions")
    if not raw_items:
        raise SpecError("事务批次不能为空", field="transactions")

    transactions: list[Transaction] = []
    seen_transactions: set[str] = set()
    previous_txn_seq: int | None = None
    for index, raw in enumerate(raw_items):
        transaction_id = _txn_reference(raw, index)
        if not isinstance(raw, dict):
            raise SpecError(
                "事务项必须是对象",
                transaction_id=transaction_id,
                field=f"transactions[{index}]",
            )
        raw_transaction_id = raw.get("transaction_id")
        if not isinstance(raw_transaction_id, str) or not raw_transaction_id.strip():
            raise SpecError(
                "transaction_id 必须为非空字符串",
                transaction_id=transaction_id,
                field=f"transactions[{index}].transaction_id",
            )
        if raw_transaction_id in seen_transactions:
            raise SpecError(
                "transaction_id 必须唯一",
                transaction_id=raw_transaction_id,
                field=f"transactions[{index}].transaction_id",
            )
        partition_id = raw.get("partition_id")
        if not isinstance(partition_id, str) or partition_id not in partition_map:
            raise SpecError(
                "目标分区不存在",
                transaction_id=raw_transaction_id,
                partition_id=partition_id if isinstance(partition_id, str) else None,
                field=f"transactions[{index}].partition_id",
            )
        txn_seq = raw.get("txn_seq")
        if not _is_int(txn_seq) or txn_seq <= 0:
            raise SpecError(
                "txn_seq 必须为正整数",
                transaction_id=raw_transaction_id,
                partition_id=partition_id,
                field=f"transactions[{index}].txn_seq",
            )
        if previous_txn_seq is not None and txn_seq <= previous_txn_seq:
            raise SpecError(
                "事务序号必须按批次顺序严格递增",
                transaction_id=raw_transaction_id,
                partition_id=partition_id,
                field=f"transactions[{index}].txn_seq",
                details={"previous_txn_seq": previous_txn_seq, "txn_seq": txn_seq},
            )
        local_version = raw.get("local_txn_version")
        if not _is_int(local_version) or local_version <= 0:
            raise SpecError(
                "local_txn_version 必须为正整数",
                transaction_id=raw_transaction_id,
                partition_id=partition_id,
                field=f"transactions[{index}].local_txn_version",
            )
        commit_time_ms = raw.get("commit_time_ms")
        if not _is_int(commit_time_ms) or commit_time_ms < 0:
            raise SpecError(
                "commit_time_ms 必须为非负整数",
                transaction_id=raw_transaction_id,
                partition_id=partition_id,
                field=f"transactions[{index}].commit_time_ms",
            )
        raw_messages = raw.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise SpecError(
                "messages 必须为非空数组",
                transaction_id=raw_transaction_id,
                partition_id=partition_id,
                field=f"transactions[{index}].messages",
            )

        messages: list[Message] = []
        previous_position: int | None = None
        for message_index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, dict):
                raise SpecError(
                    "消息项必须是对象",
                    transaction_id=raw_transaction_id,
                    partition_id=partition_id,
                    field=f"transactions[{index}].messages[{message_index}]",
                )
            position = raw_message.get("position")
            if not _is_int(position) or position < 0:
                raise SpecError(
                    "消息位点必须为非负整数",
                    transaction_id=raw_transaction_id,
                    partition_id=partition_id,
                    field=f"transactions[{index}].messages[{message_index}].position",
                )
            if previous_position is not None and position <= previous_position:
                raise SpecError(
                    "同一事务的消息位点必须严格递增",
                    transaction_id=raw_transaction_id,
                    partition_id=partition_id,
                    field=f"transactions[{index}].messages[{message_index}].position",
                    details={"previous_position": previous_position, "position": position},
                )
            if position > partition_map[partition_id].max_position:
                raise SpecError(
                    "消息位点超过分区顺序位点上限",
                    transaction_id=raw_transaction_id,
                    partition_id=partition_id,
                    field=f"transactions[{index}].messages[{message_index}].position",
                    details={"max_position": partition_map[partition_id].max_position},
                )
            if "payload" not in raw_message:
                raise SpecError(
                    "消息缺少 payload",
                    transaction_id=raw_transaction_id,
                    partition_id=partition_id,
                    field=f"transactions[{index}].messages[{message_index}].payload",
                )
            payload = raw_message["payload"]
            try:
                _check_payload_json(payload)
            except SpecError as error:
                raise SpecError(
                    error.message,
                    transaction_id=raw_transaction_id,
                    partition_id=partition_id,
                    field=f"transactions[{index}].messages[{message_index}].payload",
                ) from error
            computed_digest = message_digest(partition_id, position, payload)
            supplied_digest = raw_message.get("message_digest")
            if supplied_digest is not None and supplied_digest != computed_digest:
                raise SpecError(
                    "message_digest 与规范化消息内容不一致",
                    transaction_id=raw_transaction_id,
                    partition_id=partition_id,
                    field=f"transactions[{index}].messages[{message_index}].message_digest",
                )
            messages.append(
                Message(
                    position=position,
                    payload=payload,
                    message_digest=computed_digest,
                )
            )
            previous_position = position

        seen_transactions.add(raw_transaction_id)
        previous_txn_seq = txn_seq
        transactions.append(
            Transaction(
                transaction_id=raw_transaction_id,
                txn_seq=txn_seq,
                partition_id=partition_id,
                local_txn_version=local_version,
                messages=tuple(messages),
                commit_time_ms=commit_time_ms,
            )
        )

    positions: dict[tuple[str, int], str] = {}
    for transaction in transactions:
        for message in transaction.messages:
            key = (transaction.partition_id, message.position)
            previous_transaction = positions.get(key)
            if previous_transaction is not None:
                raise OrderError(
                    "同一分区内消息位点重复",
                    transaction_id=transaction.transaction_id,
                    partition_id=transaction.partition_id,
                    field=f"messages[position={message.position}]",
                    details={
                        "position": message.position,
                        "first_transaction_id": previous_transaction,
                    },
                )
            positions[key] = transaction.transaction_id
    return tuple(transactions)


def load_transactions(
    path: str | os.PathLike[str],
    partitions: Iterable[Partition],
    logger: StructuredLogger | None = None,
) -> tuple[Transaction, ...]:
    try:
        return parse_transactions(_read_json(path), partitions)
    except BusinessError as error:
        if logger is not None:
            logger.write_error(error)
        raise


def load_and_validate(
    partitions_path: str | os.PathLike[str],
    transactions_path: str | os.PathLike[str],
    logger: StructuredLogger | None = None,
) -> tuple[tuple[Partition, ...], tuple[Transaction, ...]]:
    partitions = load_partitions(partitions_path, logger)
    transactions = load_transactions(transactions_path, partitions, logger)
    return partitions, transactions


def validate_partition_version_supported(format_version: int) -> None:
    if format_version < 1:
        raise SpecError("分区格式版本非法", field="format_version")
    if format_version > FORMAT_VERSION:
        raise SpecError(
            "当前程序不支持该分区格式版本",
            field="format_version",
            details={"supported": FORMAT_VERSION, "supplied": format_version},
        )
