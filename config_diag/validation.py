from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, spec_error
from .graph import build_propagation, validate_acyclic
from .model import (
    ChangeEdit,
    ChangeOperation,
    ChangeSequence,
    ConfigSpec,
    Formula,
    FormulaTerm,
    KeySpec,
    SubscriptionClient,
    SubscriptionSample,
)

KEY_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
CLIENT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
OPERATION_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class _DuplicateFieldError(Exception):
    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(field)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateFieldError(key)
        result[key] = value
    return result


def load_json_file(path: Path | str, purpose: str) -> tuple[Any, Path]:
    json_path = Path(path)
    try:
        text = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise spec_error(f"无法读取{purpose}文件: {exc}", key=None, field="$file", path=str(json_path)) from exc
    try:
        value = json.loads(text, object_pairs_hook=_object_pairs)
    except _DuplicateFieldError as exc:
        raise spec_error(
            f"{purpose} JSON 含重复字段",
            key=None,
            field=exc.field,
            path=str(json_path),
        ) from exc
    except json.JSONDecodeError as exc:
        raise spec_error(
            f"{purpose} JSON 解析失败: {exc.msg}",
            key=None,
            field="$json",
            path=str(json_path),
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    return value, json_path


def _as_int(value: Any, key: str | None, field: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise spec_error("字段必须为整数（布尔值不视为整数）", key=key, field=field)
    if positive and value <= 0:
        raise spec_error("字段必须为正整数", key=key, field=field, actual=value)
    if nonnegative and value < 0:
        raise spec_error("字段必须为非负整数", key=key, field=field, actual=value)
    return value


def _as_str(value: Any, key: str | None, field: str) -> str:
    if not isinstance(value, str):
        raise spec_error("字段必须为字符串", key=key, field=field)
    return value


def _reject_unknown(obj: dict[str, Any], allowed: set[str], prefix: str, key: str | None = None) -> None:
    for field in obj:
        if field not in allowed:
            raise spec_error("包含未声明字段", key=key, field=f"{prefix}.{field}" if prefix else field)


def _parse_refs(value: Any, key: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise spec_error("派生引用必须为字符串数组", key=key, field=field)
    refs: list[str] = []
    for index, ref in enumerate(value):
        if not isinstance(ref, str) or not KEY_NAME_RE.fullmatch(ref):
            raise spec_error("派生引用键名非法", key=key, field=f"{field}[{index}]", actual=ref)
        if ref in refs:
            raise spec_error("派生引用不得重复", key=key, field=f"{field}[{index}]", actual=ref)
        refs.append(ref)
    return tuple(sorted(refs))


def _parse_formula(value: Any, key: str, refs: tuple[str, ...]) -> Formula:
    if not isinstance(value, dict):
        raise spec_error("formula 必须为对象", key=key, field="formula")
    op = _as_str(value.get("op"), key, "formula.op")
    if op == "identity":
        _reject_unknown(value, {"op"}, "formula", key=key)
        if refs:
            raise spec_error("identity 公式不得包含派生引用", key=key, field="refs")
        return Formula(op="identity")
    if op not in ("linear", "product"):
        raise spec_error("仅支持 identity/linear/product 公式", key=key, field="formula.op", actual=op)

    _reject_unknown(value, {"op", "terms", "divisor", "bias"}, "formula", key=key)
    terms_raw = value.get("terms")
    if not isinstance(terms_raw, list) or not terms_raw:
        raise spec_error(f"{op} 公式至少需要一个 term", key=key, field="formula.terms")
    if op == "product" and len(terms_raw) < 2:
        raise spec_error("product 公式至少需要两个 term", key=key, field="formula.terms")
    terms: list[FormulaTerm] = []
    seen: set[str] = set()
    for index, item in enumerate(terms_raw):
        field = f"formula.terms[{index}]"
        if not isinstance(item, dict):
            raise spec_error("term 必须为对象", key=key, field=field)
        _reject_unknown(item, {"ref", "weight"}, field, key=key)
        ref = _as_str(item.get("ref"), key, f"{field}.ref")
        if not KEY_NAME_RE.fullmatch(ref):
            raise spec_error("term 引用键名非法", key=key, field=f"{field}.ref", actual=ref)
        if ref in seen:
            raise spec_error("term 引用不得重复", key=key, field=f"{field}.ref", actual=ref)
        seen.add(ref)
        weight = _as_int(item.get("weight"), key, f"{field}.weight")
        terms.append(FormulaTerm(ref=ref, weight=weight))
    term_refs = {term.ref for term in terms}
    if term_refs != set(refs):
        raise spec_error(
            "formula.terms 必须与 refs 完全一致",
            key=key,
            field="formula.terms",
            refs=sorted(refs),
            terms=sorted(term_refs),
        )
    divisor = _as_int(value.get("divisor"), key, "formula.divisor", positive=True)
    bias = _as_int(value.get("bias"), key, "formula.bias")
    return Formula(
        op=op,
        terms=tuple(sorted(terms, key=lambda item: item.ref)),
        divisor=divisor,
        bias=bias,
    )


def _parse_key_spec(raw: Any, index: int, seen: set[str]) -> KeySpec:
    if not isinstance(raw, dict):
        raise spec_error("配置键声明必须为对象", key=None, field=f"keys[{index}]")
    _reject_unknown(
        raw,
        {"key", "value", "version", "ttl_ms", "refs", "formula"},
        f"keys[{index}]",
        key=raw.get("key") if isinstance(raw.get("key"), str) else None,
    )
    key = _as_str(raw.get("key"), None, f"keys[{index}].key")
    if not KEY_NAME_RE.fullmatch(key):
        raise spec_error("键名不合法", key=key, field="key", pattern=KEY_NAME_RE.pattern)
    if key in seen:
        raise spec_error("键名重复", key=key, field="key", index=index)
    seen.add(key)
    value = _as_int(raw.get("value"), key, "value")
    version = _as_int(raw.get("version"), key, "version", positive=True)
    ttl_ms = _as_int(raw.get("ttl_ms"), key, "ttl_ms", positive=True)
    refs = _parse_refs(raw.get("refs"), key, "refs")
    formula = _parse_formula(raw.get("formula"), key, refs)
    return KeySpec(
        key=key,
        value=value,
        version=version,
        ttl_ms=ttl_ms,
        refs=refs,
        formula=formula,
    )


def load_config_spec(path: Path | str) -> ConfigSpec:
    raw, json_path = load_json_file(path, "配置声明")
    if not isinstance(raw, dict):
        raise spec_error("配置声明顶层必须为对象", key=None, field="$")
    _reject_unknown(raw, {"schema_version", "derivation_version", "keys"}, "")
    schema_version = _as_int(raw.get("schema_version"), None, "schema_version", positive=True)
    if schema_version != 1:
        raise spec_error("不支持的 schema_version", key=None, field="schema_version", actual=schema_version)
    derivation_version = _as_int(raw.get("derivation_version"), None, "derivation_version", positive=True)
    keys_raw = raw.get("keys")
    if not isinstance(keys_raw, list) or not keys_raw:
        raise spec_error("keys 必须为非空数组", key=None, field="keys", path=str(json_path))
    seen: set[str] = set()
    keys = tuple(_parse_key_spec(item, index, seen) for index, item in enumerate(keys_raw))
    refs_by_key = {item.key: item.refs for item in keys}
    for item in keys:
        for ref in item.refs:
            if ref not in refs_by_key:
                raise spec_error("派生引用目标不存在", key=item.key, field="refs", ref=ref)
    validate_acyclic(refs_by_key)
    return ConfigSpec(schema_version=schema_version, derivation_version=derivation_version, keys=keys)


def load_subscription_sample(path: Path | str, known_keys: set[str]) -> SubscriptionSample:
    raw, json_path = load_json_file(path, "订阅客户端样本")
    if not isinstance(raw, dict):
        raise spec_error("订阅样本顶层必须为对象", key=None, field="$", path=str(json_path))
    _reject_unknown(raw, {"schema_version", "clients"}, "")
    schema_version = _as_int(raw.get("schema_version"), None, "schema_version", positive=True)
    if schema_version != 1:
        raise spec_error("不支持的订阅样本 schema_version", key=None, field="schema_version")
    clients_raw = raw.get("clients")
    if not isinstance(clients_raw, list) or not clients_raw:
        raise spec_error("clients 必须为非空数组", key=None, field="clients")
    seen: set[str] = set()
    clients: list[SubscriptionClient] = []
    for index, item in enumerate(clients_raw):
        prefix = f"clients[{index}]"
        if not isinstance(item, dict):
            raise spec_error("订阅客户端必须为对象", key=None, field=prefix)
        _reject_unknown(
            item,
            {"client_id", "seed", "poll_interval_ms", "keys", "sample_size"},
            prefix,
            key=item.get("client_id") if isinstance(item.get("client_id"), str) else None,
        )
        client_id = _as_str(item.get("client_id"), None, f"{prefix}.client_id")
        if not CLIENT_ID_RE.fullmatch(client_id):
            raise spec_error("client_id 不合法", key=client_id, field="client_id")
        if client_id in seen:
            raise spec_error("client_id 重复", key=client_id, field="client_id")
        seen.add(client_id)
        seed = _as_int(item.get("seed"), client_id, "seed", nonnegative=True)
        if seed > 2**32 - 1:
            raise spec_error("seed 超出 uint32 范围", key=client_id, field="seed", actual=seed)
        interval = _as_int(item.get("poll_interval_ms"), client_id, "poll_interval_ms", positive=True)
        keys = _parse_refs(item.get("keys"), client_id, "keys")
        if not keys:
            raise spec_error("订阅客户端至少选择一个键", key=client_id, field="keys")
        for key in keys:
            if key not in known_keys:
                raise spec_error("订阅客户端引用了未知键", key=client_id, field="keys", ref=key)
        sample_size = item.get("sample_size", len(keys))
        sample_size = _as_int(sample_size, client_id, "sample_size", positive=True)
        if sample_size > len(keys):
            raise spec_error("sample_size 不得大于订阅键数量", key=client_id, field="sample_size")
        clients.append(
            SubscriptionClient(
                client_id=client_id,
                seed=seed,
                poll_interval_ms=interval,
                keys=keys,
                sample_size=sample_size,
            )
        )
    clients.sort(key=lambda item: item.client_id)
    return SubscriptionSample(schema_version=schema_version, clients=tuple(clients))


def _parse_change_edit(
    raw: Any,
    index: int,
    spec_by_key: dict[str, KeySpec],
    seen: set[str],
) -> ChangeEdit:
    prefix = f"changes[{index}]"
    if not isinstance(raw, dict):
        raise spec_error("变更项必须为对象", key=None, field=prefix)
    _reject_unknown(
        raw,
        {"key", "version", "value", "ttl_ms", "refs", "formula"},
        prefix,
        key=raw.get("key") if isinstance(raw.get("key"), str) else None,
    )
    key = _as_str(raw.get("key"), None, f"{prefix}.key")
    if key not in spec_by_key:
        raise spec_error("变更引用未知键", key=key, field="key")
    if key in seen:
        raise spec_error("同一操作内不得重复修改同一键", key=key, field="key")
    seen.add(key)
    version = _as_int(raw.get("version"), key, "version", positive=True)
    mutable = [name for name in ("value", "ttl_ms", "refs", "formula") if name in raw]
    if not mutable:
        raise spec_error("变更必须至少修改 value/ttl_ms/refs/formula 之一", key=key, field="version")
    value = None
    ttl_ms = None
    refs = None
    formula = None
    is_root = not spec_by_key[key].refs
    if "value" in raw:
        value = _as_int(raw["value"], key, "value")
        if not is_root:
            raise spec_error("派生键的取值由公式计算，不允许直接修改", key=key, field="value")
    if "ttl_ms" in raw:
        ttl_ms = _as_int(raw["ttl_ms"], key, "ttl_ms", positive=True)
    if ("refs" in raw) != ("formula" in raw):
        raise spec_error("修改派生公式时必须同时提供 refs 与 formula", key=key, field="refs" if "refs" not in raw else "formula")
    if "refs" in raw:
        refs = _parse_refs(raw["refs"], key, "refs")
        formula = _parse_formula(raw["formula"], key, refs)
        if is_root:
            if refs or formula.op != "identity":
                raise spec_error("根配置键不得改为派生键", key=key, field="refs")
    return ChangeEdit(key=key, version=version, value=value, ttl_ms=ttl_ms, refs=refs, formula=formula)


def load_change_sequence(path: Path | str, spec: ConfigSpec) -> ChangeSequence:
    raw, json_path = load_json_file(path, "变更序列")
    if not isinstance(raw, dict):
        raise spec_error("变更序列顶层必须为对象", key=None, field="$", path=str(json_path))
    _reject_unknown(raw, {"schema_version", "operations"}, "")
    schema_version = _as_int(raw.get("schema_version"), None, "schema_version", positive=True)
    if schema_version != 1:
        raise spec_error("不支持的变更序列 schema_version", key=None, field="schema_version")
    operations_raw = raw.get("operations")
    if not isinstance(operations_raw, list):
        raise spec_error("operations 必须为数组", key=None, field="operations")
    spec_by_key = spec.by_key
    current_refs = {key: list(value.refs) for key, value in spec_by_key.items()}
    current_versions = {key: value.version for key, value in spec_by_key.items()}
    operation_ids: set[str] = set()
    sequences: set[int] = set()
    last_sequence = 0
    operations: list[ChangeOperation] = []
    for op_index, raw_op in enumerate(operations_raw):
        prefix = f"operations[{op_index}]"
        if not isinstance(raw_op, dict):
            raise spec_error("变更操作必须为对象", key=None, field=prefix)
        _reject_unknown(
            raw_op,
            {"operation_id", "sequence", "description", "changes"},
            prefix,
            key=raw_op.get("operation_id") if isinstance(raw_op.get("operation_id"), str) else None,
        )
        operation_id = _as_str(raw_op.get("operation_id"), None, f"{prefix}.operation_id")
        if not OPERATION_ID_RE.fullmatch(operation_id):
            raise spec_error("operation_id 不合法", key=operation_id, field="operation_id")
        if operation_id in operation_ids:
            raise spec_error("operation_id 重复", key=operation_id, field="operation_id")
        operation_ids.add(operation_id)
        sequence = _as_int(raw_op.get("sequence"), None, f"{prefix}.sequence", positive=True)
        if sequence in sequences or sequence <= last_sequence:
            raise spec_error("operation sequence 必须严格递增且唯一", key=operation_id, field="sequence", actual=sequence)
        sequences.add(sequence)
        last_sequence = sequence
        description = _as_str(raw_op.get("description", ""), None, f"{prefix}.description")
        changes_raw = raw_op.get("changes")
        if not isinstance(changes_raw, list) or not changes_raw:
            raise spec_error("changes 必须为非空数组", key=operation_id, field="changes")
        seen_changes: set[str] = set()
        edits = tuple(
            _parse_change_edit(item, index, spec_by_key, seen_changes)
            for index, item in enumerate(changes_raw)
        )

        candidate_refs = {key: list(value) for key, value in current_refs.items()}
        for edit in edits:
            if edit.refs is not None:
                candidate_refs[edit.key] = list(edit.refs)
        candidate_refs_tuples = {key: tuple(value) for key, value in candidate_refs.items()}
        try:
            validate_acyclic(candidate_refs_tuples)
        except DiagnosticError as exc:
            exc.context.update({"operation_id": operation_id, "sequence": sequence})
            raise

        for edit in edits:
            if edit.version <= current_versions[edit.key]:
                raise spec_error(
                    "键版本号必须单调递增",
                    key=edit.key,
                    field="version",
                    current=current_versions[edit.key],
                    incoming=edit.version,
                    operation_id=operation_id,
                )
        direct = {edit.key for edit in edits}
        affected, _ = build_propagation(candidate_refs_tuples, direct, invalid_status="recompute")
        for edit in edits:
            current_versions[edit.key] = edit.version
        for key in affected:
            if key not in direct:
                current_versions[key] += 1
        current_refs = candidate_refs
        operations.append(
            ChangeOperation(
                operation_id=operation_id,
                sequence=sequence,
                description=description,
                changes=edits,
            )
        )
    return ChangeSequence(schema_version=schema_version, operations=tuple(operations))


def load_bundle(samples_dir: Path | str) -> tuple[ConfigSpec, SubscriptionSample, ChangeSequence]:
    base = Path(samples_dir)
    spec = load_config_spec(base / "config_keys.json")
    clients = load_subscription_sample(base / "subscription_clients.json", set(spec.by_key))
    changes = load_change_sequence(base / "change_sequence.json", spec)
    return spec, clients, changes
