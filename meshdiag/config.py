from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import read_json
from .errors import MeshError, diag, fail_rule
from .models import RequestSpec, Service, parse_service_rows


@dataclass(frozen=True)
class Workspace:
    workspace_id: str
    config_path: Path
    directory: Path
    services: tuple[Service, ...]
    global_concurrency: int
    defaults: dict[str, Any]
    ledger_path: Path
    snapshot_dir: Path
    revision_path: Path
    requests_path: Path


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_workspace(path: str | Path) -> Workspace:
    config_path = Path(path).resolve()
    try:
        payload = read_json(config_path)
    except (OSError, ValueError) as exc:
        raise MeshError(
            "E_RULE",
            "工作区配置无法读取",
            diagnostics=[diag("E_RULE", str(config_path), str(exc))],
        ) from exc
    if not isinstance(payload, dict):
        fail_rule([diag("E_RULE", "$", "工作区配置必须是 JSON 对象")])
    workspace_id = payload.get("workspace_id")
    if not isinstance(workspace_id, str) or not workspace_id:
        fail_rule(
            [diag("E_RULE", "workspace_id", "workspace_id 缺失或不是非空字符串")]
        )
    global_concurrency = payload.get("global_concurrency")
    if (
        not isinstance(global_concurrency, int)
        or isinstance(global_concurrency, bool)
        or global_concurrency < 1
    ):
        fail_rule(
            [
                diag(
                    "E_RULE",
                    "global_concurrency",
                    "global_concurrency 必须是正整数",
                    global_concurrency,
                )
            ]
        )
    services = parse_service_rows(payload.get("services"))
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        fail_rule([diag("E_RULE", "defaults", "defaults 必须是对象")])
    allowed_defaults = {"timeout_ms", "retries", "cooldown_ms"}
    for key, value in sorted(defaults.items()):
        if key not in allowed_defaults:
            fail_rule([diag("E_RULE", f"defaults.{key}", "未知兼容默认值字段")])
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail_rule(
                [diag("E_RULE", f"defaults.{key}", "兼容默认值必须是非负整数", value)]
            )
    base = config_path.parent
    revision_file = payload.get("revision_file")
    requests_file = payload.get("requests_file")
    ledger_file = payload.get("ledger_path", "var/ledger.sqlite3")
    snapshot_dir = payload.get("snapshot_dir", "var/snapshots")
    for field_name, value in (
        ("revision_file", revision_file),
        ("requests_file", requests_file),
        ("ledger_path", ledger_file),
        ("snapshot_dir", snapshot_dir),
    ):
        if not isinstance(value, str) or not value:
            fail_rule([diag("E_RULE", field_name, f"{field_name} 必须是非空路径字符串")])
    return Workspace(
        workspace_id=workspace_id,
        config_path=config_path,
        directory=base,
        services=services,
        global_concurrency=global_concurrency,
        defaults=defaults,
        ledger_path=_resolve(base, ledger_file),
        snapshot_dir=_resolve(base, snapshot_dir),
        revision_path=_resolve(base, revision_file),
        requests_path=_resolve(base, requests_file),
    )


def load_revision_document(path: str | Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise MeshError(
            "E_RULE",
            "规则修订文件无法读取",
            diagnostics=[diag("E_RULE", str(path), str(exc))],
        ) from exc
    if not isinstance(payload, dict):
        fail_rule([diag("E_RULE", "$", "规则修订文件必须是 JSON 对象")])
    return payload


def load_requests(path: str | Path) -> tuple[RequestSpec, ...]:
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise MeshError(
            "E_RULE",
            "请求文件无法读取",
            diagnostics=[diag("E_RULE", str(path), str(exc))],
        ) from exc
    rows = payload.get("requests") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        fail_rule([diag("E_RULE", "requests", "requests 必须是非空数组")])
    errors = []
    seen: set[str] = set()
    requests = []
    for index, row in enumerate(rows):
        path_text = f"requests[{index}]"
        if not isinstance(row, dict):
            errors.append(diag("E_RULE", path_text, "请求必须是对象"))
            continue
        request_id = row.get("id")
        if not isinstance(request_id, str) or not request_id:
            errors.append(diag("E_RULE", f"{path_text}.id", "请求 id 缺失或为空"))
            continue
        if request_id in seen:
            errors.append(diag("E_RULE", f"{path_text}.id", "请求 id 重复", request_id))
            continue
        seen.add(request_id)
        method = row.get("method", "GET")
        if not isinstance(method, str) or method.upper() not in {"GET", "HEAD"}:
            errors.append(
                diag("E_RULE", f"{path_text}.method", "method 仅支持 GET/HEAD", method)
            )
            method = "GET"
        request_path = row.get("path")
        if not isinstance(request_path, str) or not request_path.startswith("/"):
            errors.append(
                diag(
                    "E_RULE",
                    f"{path_text}.path",
                    "path 必须是 / 开头的字符串",
                    request_path,
                )
            )
            request_path = "/"
        requests.append(
            RequestSpec(
                request_id=request_id,
                method=method.upper(),
                path=request_path,
                index=index,
            )
        )
    if errors:
        fail_rule(errors, "请求序列校验失败")
    return tuple(sorted(requests, key=lambda item: item.index))
