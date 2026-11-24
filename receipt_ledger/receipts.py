"""r1/r2/r3 三版本回执解析器。

三版本统一为 :class:`~receipt_ledger.models.Receipt`。解析器严格区分：

* 来源版本天然缺失的字段 —— 记录在 ``missing`` 中（合法）；
* 显式缺失标记占位 —— 允许但标记为缺失；
* 字段错位、多余键、类型错误、截断行 —— E_RECEIPT_MALFORMED /
  E_RECEIPT_TRUNCATED，记录来源文件、物理行号与版本。
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Any

from .errors import (
    E_NON_FINITE_DURATION,
    E_RECEIPT_MALFORMED,
    E_RECEIPT_TRUNCATED,
    Failure,
)
from .models import MISSING, Receipt
from .util import is_finite_number, sha256_hex

BASE_FIELDS = ("version", "batch_id", "request_id", "output_digest")
R2_FIELDS = ("param_version", "duration_ms", "worker_id")
R3_FIELDS = ("rank_in_batch", "supersedes", "param_summary")

VERSION_FIELDS: dict[str, tuple[str, ...]] = {
    "r1": BASE_FIELDS,
    "r2": BASE_FIELDS + R2_FIELDS,
    "r3": BASE_FIELDS + R2_FIELDS + R3_FIELDS,
}
OPTIONAL_BY_VERSION: dict[str, frozenset[str]] = {
    "r1": frozenset(),
    "r2": frozenset(R2_FIELDS),
    "r3": frozenset(R2_FIELDS + R3_FIELDS),
}


def integrity_tag(payload: dict) -> str:
    """对不含完整性标签的规范化载荷计算稳定标签。"""

    material = {key: value for key, value in payload.items() if key != "integrity_tag"}
    return "sha256:" + sha256_hex(material)[:12]


def parse_receipt_line(
    text: str,
    source_file: str,
    line: int,
    expected_version: str | None = None,
    injection_id: str | None = None,
) -> tuple[Receipt | None, Failure | None]:
    source = os.path.basename(source_file)
    try:
        item = json.loads(text)
    except json.JSONDecodeError as exc:
        truncated = exc.msg.lower().startswith("expecting") or "unterminated" in exc.msg.lower()
        code = E_RECEIPT_TRUNCATED if truncated else E_RECEIPT_MALFORMED
        return None, Failure(
            code,
            source,
            line,
            f"{expected_version or 'unknown'} 回执行无法解析: {exc.msg}",
            injection_id,
        )

    if not isinstance(item, dict):
        return None, Failure(
            E_RECEIPT_MALFORMED, source, line, "回执行不是 JSON 对象", injection_id
        )

    version = item.get("version", expected_version)
    if version not in VERSION_FIELDS:
        return None, Failure(
            E_RECEIPT_MALFORMED,
            source,
            line,
            f"未知回执版本: {version!r}",
            injection_id,
        )
    if expected_version and version != expected_version:
        return None, Failure(
            E_RECEIPT_MALFORMED,
            source,
            line,
            f"版本字段 {version} 与文件声明 {expected_version} 不一致（字段错位）",
            injection_id,
        )

    allowed = set(VERSION_FIELDS[version]) | {"integrity_tag"}
    extra_keys = sorted(set(item) - allowed)
    if extra_keys:
        return None, Failure(
            E_RECEIPT_MALFORMED,
            source,
            line,
            f"{version} 回执出现错位/多余字段: {extra_keys}",
            injection_id,
        )

    missing: list[str] = []
    for field_name in BASE_FIELDS:
        value = item.get(field_name)
        if field_name == "version":
            continue
        if not isinstance(value, str) or not value:
            return None, Failure(
                E_RECEIPT_MALFORMED,
                source,
                line,
                f"{version} 基础字段缺失或类型错误: {field_name}",
                injection_id,
            )

    values: dict[str, Any] = {}
    optional = OPTIONAL_BY_VERSION[version]
    for field_name in R2_FIELDS + R3_FIELDS:
        present = field_name in item
        value = item.get(field_name)
        if not present and field_name not in optional:
            missing.append(field_name)
            values[field_name] = None
            continue
        if not present:
            if field_name in optional:
                missing.append(field_name)
                values[field_name] = None
            continue
        if value == MISSING:
            missing.append(field_name)
            values[field_name] = None
            continue
        error = _validate_optional_field(field_name, value)
        if error is not None:
            code = E_NON_FINITE_DURATION if error == "non_finite_duration" else E_RECEIPT_MALFORMED
            message = (
                f"{version} 字段 {field_name} 出现非有限数值"
                if code == E_NON_FINITE_DURATION
                else f"{version} 字段 {field_name} 类型/取值错误: {error}"
            )
            return None, Failure(code, source, line, message, injection_id)
        values[field_name] = value

    if not missing and "missing" in item:
        return None, Failure(
            E_RECEIPT_MALFORMED, source, line, "缺失位非空但回执字段完整", injection_id
        )

    tag = item.get("integrity_tag") or integrity_tag(item)
    receipt = Receipt(
        source_version=version,
        source_file=source,
        line=line,
        batch_id=item["batch_id"],
        request_id=item["request_id"],
        output_digest=item["output_digest"],
        param_version=values.get("param_version"),
        duration_ms=values.get("duration_ms"),
        worker_id=values.get("worker_id"),
        rank_in_batch=values.get("rank_in_batch"),
        supersedes=tuple(values.get("supersedes") or ()),
        param_summary=values.get("param_summary"),
        missing=tuple(missing),
        integrity_tag=tag,
        injection_id=injection_id,
    )
    return receipt, None


def _validate_optional_field(field_name: str, value: Any) -> str | None:
    if field_name == "worker_id" or field_name == "param_summary":
        if not isinstance(value, str) or not value:
            return "必须是非空字符串"
        return None
    if field_name == "param_version" or field_name == "duration_ms" or field_name == "rank_in_batch":
        if field_name == "duration_ms" and not is_finite_number(value):
            return "non_finite_duration"
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return "必须是非负整数"
        return None
    if field_name == "supersedes":
        if not isinstance(value, list) or any(
            not isinstance(ref, str) or not ref for ref in value
        ):
            return "必须是字符串引用数组"
        return None
    return "未知字段"


def discover_receipt_files(samples_dir: str) -> list[str]:
    paths = sorted(
        glob.glob(os.path.join(samples_dir, "receipts", "*.jsonl"))
        + glob.glob(os.path.join(samples_dir, "**", "receipt_*.jsonl"), recursive=True)
    )
    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = os.path.abspath(path)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path)
    return unique


def parse_receipt_file(
    path: str,
    injection_lines: dict[int, str] | None = None,
) -> tuple[list[Receipt], list[Failure]]:
    injection_lines = injection_lines or {}
    expected_version = _version_from_filename(path)
    receipts: list[Receipt] = []
    failures: list[Failure] = []
    with open(path, encoding="utf-8") as handle:
        for line, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            receipt, failure = parse_receipt_line(
                text, path, line, expected_version, injection_lines.get(line)
            )
            if failure is not None:
                failures.append(failure)
            else:
                receipts.append(receipt)
    return receipts, failures


def parse_all(
    samples_dir: str,
    injection_index: dict[str, dict[int, str]] | None = None,
) -> tuple[list[Receipt], list[Failure], list[dict]]:
    injection_index = injection_index or {}
    all_receipts: list[Receipt] = []
    all_failures: list[Failure] = []
    fingerprints: list[dict] = []
    for path in discover_receipt_files(samples_dir):
        base = os.path.basename(path)
        receipts, failures = parse_receipt_file(path, injection_index.get(base))
        all_receipts.extend(receipts)
        all_failures.extend(failures)
        with open(path, "rb") as handle:
            data = handle.read()
            fingerprints.append(
                {
                    "file": base,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return all_receipts, all_failures, fingerprints


def _version_from_filename(path: str) -> str | None:
    base = os.path.basename(path)
    for version in ("r1", "r2", "r3"):
        if version in base:
            return version
    return None
