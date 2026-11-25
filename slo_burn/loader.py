"""本地 JSON/NDJSON 加载与强校验。

非法输入统一抛出 E_SPEC，并定位首个非法服务与批次行号（1 基物理行）。
校验在任何业务计算之前完成，保证“坏数据不进账本”。
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import ErrorLocation, SpecError
from .params import ParamEdit
from .models import (
    BUDGET_FLOOR_MIN,
    BURN_MAX,
    BURN_MIN,
    TARGET_MAX,
    TARGET_MIN,
    WINDOW_MIN,
    MinuteBatch,
    ServiceSpec,
)

_SPEC_FIELDS = {
    "service_id",
    "target_permyriad",
    "window_minutes",
    "burn_threshold",
    "budget_floor",
}
_BATCH_FIELDS = {"service_id", "minute", "successes", "failures"}
_EDIT_FIELDS = {
    "service_id",
    "param_version",
    "target_permyriad",
    "window_minutes",
    "burn_threshold",
    "budget_floor",
    "effective_after_minute",
}


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _spec_error(message: str, loc: ErrorLocation) -> SpecError:
    return SpecError(message, **loc.as_context())


def _validate_service_dict(raw: object, index: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise _spec_error(
            "服务声明必须是 JSON 对象",
            ErrorLocation(service_id=None, line=index + 1, field_name="service"),
        )
    unknown = set(raw) - _SPEC_FIELDS
    if unknown:
        raise _spec_error(
            f"服务声明含未知字段 {sorted(unknown)}",
            ErrorLocation(
                service_id=raw.get("service_id") if isinstance(raw.get("service_id"), str) else None,
                line=index + 1,
                field=sorted(unknown)[0],
            ),
        )
    service_id = raw.get("service_id")
    if not isinstance(service_id, str) or not service_id.strip():
        raise _spec_error(
            "service_id 必须是非空字符串",
            ErrorLocation(line=index + 1, field_name="service_id"),
        )
    target = raw.get("target_permyriad")
    if not _is_plain_int(target) or not (TARGET_MIN <= target <= TARGET_MAX):
        raise _spec_error(
            f"target_permyriad 必须是 [{TARGET_MIN},{TARGET_MAX}] 内的万分比整数",
            ErrorLocation(service_id=service_id, line=index + 1, field_name="target_permyriad"),
        )
    window = raw.get("window_minutes")
    if not _is_plain_int(window) or window < WINDOW_MIN:
        raise _spec_error(
            "window_minutes 必须是正整数",
            ErrorLocation(service_id=service_id, line=index + 1, field_name="window_minutes"),
        )
    burn = raw.get("burn_threshold")
    if not _is_plain_int(burn) or not (BURN_MIN <= burn <= BURN_MAX):
        raise _spec_error(
            f"burn_threshold 必须是 [{BURN_MIN},{BURN_MAX}] 内的万分比整数",
            ErrorLocation(service_id=service_id, line=index + 1, field_name="burn_threshold"),
        )
    floor = raw.get("budget_floor")
    if not _is_plain_int(floor) or floor < BUDGET_FLOOR_MIN:
        raise _spec_error(
            "budget_floor 必须是非负整数",
            ErrorLocation(service_id=service_id, line=index + 1, field_name="budget_floor"),
        )
    return raw


def load_service_specs(path: str | Path) -> list[ServiceSpec]:
    """加载并强校验服务声明 JSON，返回按 service_id 稳定排序的列表。"""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"服务声明文件不存在: {path}", path=str(path))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"服务声明 JSON 解析失败: {exc.msg}",
            path=str(path),
            line=exc.lineno,
            column=exc.colno,
        )
    if not isinstance(document, dict) or "services" not in document:
        raise SpecError("服务声明文件顶层必须是含 services 数组的对象", path=str(path))
    raw_services = document["services"]
    if not isinstance(raw_services, list) or not raw_services:
        raise SpecError("services 必须是非空数组", path=str(path))

    specs: list[ServiceSpec] = []
    seen: set[str] = set()
    first_by_id: dict[str, int] = {}
    for index, raw in enumerate(raw_services):
        validated = _validate_service_dict(raw, index)
        service_id = validated["service_id"]
        if service_id in seen:
            raise _spec_error(
                "服务标识必须唯一，发现重复 service_id",
                ErrorLocation(service_id=service_id, line=index + 1, field_name="service_id"),
            )
        seen.add(service_id)
        first_by_id[service_id] = index + 1
        specs.append(
            ServiceSpec(
                service_id=service_id,
                target_permyriad=validated["target_permyriad"],
                window_minutes=validated["window_minutes"],
                burn_threshold=validated["burn_threshold"],
                budget_floor=validated["budget_floor"],
            )
        )
    specs.sort(key=lambda item: item.service_id)
    return specs


def _read_ndjson_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line_number, text in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not text.strip():
            continue
        lines.append((line_number, text))
    return lines


def _validate_batch_dict(
    raw: object,
    line: int,
    known_services: set[str],
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise _spec_error("批次必须是 JSON 对象", ErrorLocation(line=line))
    unknown = set(raw) - _BATCH_FIELDS
    if unknown:
        raise _spec_error(
            f"批次含未知字段 {sorted(unknown)}",
            ErrorLocation(
                service_id=raw.get("service_id") if isinstance(raw.get("service_id"), str) else None,
                line=line,
                field=sorted(unknown)[0],
            ),
        )
    service_id = raw.get("service_id")
    if not isinstance(service_id, str) or not service_id.strip():
        raise _spec_error("批次 service_id 必须是非空字符串", ErrorLocation(line=line, field_name="service_id"))
    if service_id not in known_services:
        raise _spec_error(
            "批次引用了未声明的 service_id",
            ErrorLocation(service_id=service_id, line=line, field_name="service_id"),
        )
    minute = raw.get("minute")
    if not _is_plain_int(minute) or minute < 0:
        raise _spec_error(
            "minute 必须是非负整数",
            ErrorLocation(service_id=service_id, line=line, field_name="minute"),
        )
    successes = raw.get("successes")
    if not _is_plain_int(successes) or successes < 0:
        raise _spec_error(
            "successes 必须是非负整数",
            ErrorLocation(service_id=service_id, line=line, field_name="successes"),
        )
    failures = raw.get("failures")
    if not _is_plain_int(failures) or failures < 0:
        raise _spec_error(
            "failures 必须是非负整数",
            ErrorLocation(service_id=service_id, line=line, field_name="failures"),
        )
    return raw


def load_batches(
    path: str | Path,
    specs: list[ServiceSpec],
    stream: str = "regular",
) -> list[MinuteBatch]:
    """加载单个 NDJSON 批次流并强校验。

    regular 流：分钟序号必须严格单调递增且分钟唯一；
    late    流：允许乱序，但同一流内分钟同样不得重复（重复由归并层按稳定规则处理）。
    行号为物理文件行号，跨文件精确定位首个非法批次。
    """
    path = Path(path)
    known_services = {spec.service_id for spec in specs}
    try:
        raw_lines = _read_ndjson_lines(path)
    except FileNotFoundError:
        raise SpecError(f"批次文件不存在: {path}", path=str(path))

    batches: list[MinuteBatch] = []
    last_minute_by_service: dict[str, int] = {}
    seen_minutes: dict[str, set[int]] = {spec.service_id: set() for spec in specs}
    batch_counter: dict[str, int] = {spec.service_id: 0 for spec in specs}

    for line, text in raw_lines:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"NDJSON 解析失败: {exc.msg}",
                line=line,
                column=exc.colno,
            )
        validated = _validate_batch_dict(raw, line, known_services)
        service_id = validated["service_id"]
        minute = validated["minute"]
        if stream == "regular":
            previous = last_minute_by_service.get(service_id)
            if previous is not None and minute <= previous:
                raise _spec_error(
                    "常规流分钟序号必须严格单调递增",
                    ErrorLocation(service_id=service_id, line=line, minute=minute, field_name="minute"),
                )
            last_minute_by_service[service_id] = minute
        if minute in seen_minutes[service_id]:
            if stream == "regular":
                raise _spec_error(
                    "不同批不得重复同一分钟",
                    ErrorLocation(service_id=service_id, line=line, minute=minute, field_name="minute"),
                )
        seen_minutes[service_id].add(minute)
        batch_counter[service_id] += 1
        batches.append(
            MinuteBatch(
                service_id=service_id,
                line=line,
                batch_seq=batch_counter[service_id],
                minute=minute,
                successes=validated["successes"],
                failures=validated["failures"],
                stream=stream,
            )
        )
    return batches


def load_batch_directory(
    directory: str | Path,
    specs: list[ServiceSpec],
) -> tuple[list[MinuteBatch], list[MinuteBatch]]:
    """加载目录内的常规流与迟到流。

    约定文件名：regular.ndjson（或 <service_id>.ndjson）与 late.ndjson。
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SpecError(f"批次目录不存在: {directory}", path=str(directory))
    regular: list[MinuteBatch] = []
    late: list[MinuteBatch] = []

    regular_path = directory / "regular.ndjson"
    late_path = directory / "late.ndjson"
    if regular_path.exists():
        regular.extend(load_batches(regular_path, specs, "regular"))
    if late_path.exists():
        late.extend(load_batches(late_path, specs, "late"))

    service_files = sorted(
        p for p in directory.glob("*.ndjson") if p.name not in {"regular.ndjson", "late.ndjson"}
    )
    for service_file in service_files:
        stem = service_file.stem
        if stem not in {spec.service_id for spec in specs}:
            raise SpecError(
                "批次文件名对应未声明的 service_id",
                service_id=stem,
                path=str(service_file),
            )
        regular.extend(load_batches(service_file, specs, "regular"))
    regular.sort(key=lambda b: (b.service_id, b.minute, b.line))
    late.sort(key=lambda b: (b.service_id, b.minute, b.line))
    return regular, late


def load_param_edits(
    path: str | Path,
    specs: list[ServiceSpec],
) -> list[ParamEdit]:
    """加载参数编辑脚本并强校验：版本号单调递增、字段合法区间。"""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"参数编辑文件不存在: {path}", path=str(path))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"参数编辑 JSON 解析失败: {exc.msg}",
            path=str(path),
            line=exc.lineno,
            column=exc.colno,
        )
    if not isinstance(document, dict) or "edits" not in document:
        raise SpecError("参数编辑文件顶层必须是含 edits 数组的对象", path=str(path))
    raw_edits = document["edits"]
    if not isinstance(raw_edits, list):
        raise SpecError("edits 必须是数组", path=str(path))

    known = {spec.service_id for spec in specs}
    edits: list[ParamEdit] = []
    last_version_by_service = {spec.service_id: spec.param_version for spec in specs}
    for index, raw in enumerate(raw_edits):
        if not isinstance(raw, dict):
            raise _spec_error("参数编辑必须是 JSON 对象", ErrorLocation(line=index + 1))
        unknown = set(raw) - _EDIT_FIELDS
        if unknown:
            raise _spec_error(
                f"参数编辑含未知字段 {sorted(unknown)}",
                ErrorLocation(
                    service_id=raw.get("service_id") if isinstance(raw.get("service_id"), str) else None,
                    line=index + 1,
                    field=sorted(unknown)[0],
                ),
            )
        service_id = raw.get("service_id")
        if not isinstance(service_id, str) or service_id not in known:
            raise _spec_error(
                "参数编辑引用了未声明的 service_id",
                ErrorLocation(service_id=service_id, line=index + 1, field_name="service_id"),
            )
        version = raw.get("param_version")
        if not _is_plain_int(version) or version <= last_version_by_service[service_id]:
            raise _spec_error(
                "param_version 必须为整数且对同一服务严格单调递增",
                ErrorLocation(service_id=service_id, line=index + 1, field_name="param_version"),
            )
        changes = {
            key: raw[key]
            for key in (
                "target_permyriad",
                "window_minutes",
                "burn_threshold",
                "budget_floor",
            )
            if key in raw
        }
        if not changes:
            raise _spec_error(
                "参数编辑至少修改一个参数字段",
                ErrorLocation(service_id=service_id, line=index + 1),
            )
        field_ranges = {
            "target_permyriad": (TARGET_MIN, TARGET_MAX),
            "window_minutes": (WINDOW_MIN, 10**9),
            "burn_threshold": (BURN_MIN, BURN_MAX),
            "budget_floor": (BUDGET_FLOOR_MIN, 10**18),
        }
        for field_name, (low, high) in field_ranges.items():
            if field_name in changes and (
                not _is_plain_int(changes[field_name])
                or not (low <= changes[field_name] <= high)
            ):
                raise _spec_error(
                    f"{field_name} 超出合法整数区间 [{low},{high}]",
                    ErrorLocation(service_id=service_id, line=index + 1, field_name=field_name),
                )
        effective_after = raw.get("effective_after_minute", 0)
        if not _is_plain_int(effective_after) or effective_after < 0:
            raise _spec_error(
                "effective_after_minute 必须是非负整数",
                ErrorLocation(service_id=service_id, line=index + 1, field_name="effective_after_minute"),
            )
        last_version_by_service[service_id] = version
        edits.append(
            ParamEdit(
                service_id=service_id,
                param_version=version,
                effective_after_minute=effective_after,
                changes=changes,
                line=index + 1,
            )
        )
    return edits
