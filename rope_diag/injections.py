from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import SCHEME_ALIASES
from .errors import ErrorCode, Failure
from .util import read_json, sha256_file, sha256_json


INJECTION_STAGES = (
    "position",
    "scoring",
    "statistics",
    "trace",
    "sensitivity",
)
INJECTION_ACTIONS = ("raise", "nonfinite", "truncate")


@dataclass(frozen=True)
class Injection:
    injection_id: str
    enabled: bool
    scheme: str | None
    sample_id: str | None
    stage: str
    action: str
    message: str
    parameters: dict[str, Any]

    def matches(self, *, scheme: str | None, sample_id: str | None, stage: str) -> bool:
        if not self.enabled or self.stage != stage:
            return False
        if self.scheme is not None and scheme != self.scheme:
            return False
        if self.sample_id is not None and sample_id != self.sample_id:
            return False
        return True


def _failure(message: str, details: dict[str, Any] | None = None) -> Failure:
    return Failure(
        code=ErrorCode.CONFIG_INVALID,
        message=f"injections: {message}",
        stage="injection_validation",
        details=details or {},
    )


def parse_injections(raw: Any) -> tuple[list[Injection], list[Failure]]:
    failures: list[Failure] = []
    if isinstance(raw, dict):
        raw = raw.get("injections")
    if not isinstance(raw, list):
        return [], [_failure("必须为数组或含 injections 的对象")]

    injections: list[Injection] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            failures.append(_failure(f"injections[{index}] 必须为对象"))
            continue
        injection_id = item.get("injection_id", item.get("id"))
        if not isinstance(injection_id, str) or not injection_id.strip():
            failures.append(_failure(f"injections[{index}].injection_id 必须为非空字符串"))
            continue
        injection_id = injection_id.strip()
        if injection_id in seen:
            failures.append(_failure(f"injection_id 重复: {injection_id}"))
            continue
        seen.add(injection_id)

        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            failures.append(_failure(f"{injection_id}.enabled 必须为布尔值"))
            continue

        scheme_raw = item.get("scheme")
        scheme = None
        if scheme_raw is not None:
            if not isinstance(scheme_raw, str) or scheme_raw.lower() not in SCHEME_ALIASES:
                failures.append(_failure(f"{injection_id}.scheme 非法", {"scheme": scheme_raw}))
                continue
            scheme = SCHEME_ALIASES[scheme_raw.lower()]

        sample_id = item.get("sample_id")
        if sample_id is not None and (not isinstance(sample_id, str) or not sample_id.strip()):
            failures.append(_failure(f"{injection_id}.sample_id 必须为非空字符串或 null"))
            continue
        if isinstance(sample_id, str):
            sample_id = sample_id.strip()

        stage = item.get("stage")
        if stage not in INJECTION_STAGES:
            failures.append(
                _failure(f"{injection_id}.stage 必须属于 {list(INJECTION_STAGES)}")
            )
            continue
        action = item.get("action")
        if action not in INJECTION_ACTIONS:
            failures.append(
                _failure(f"{injection_id}.action 必须属于 {list(INJECTION_ACTIONS)}")
            )
            continue
        message = item.get("message", f"injected fault at {stage}")
        if not isinstance(message, str):
            failures.append(_failure(f"{injection_id}.message 必须为字符串"))
            continue
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            failures.append(_failure(f"{injection_id}.parameters 必须为对象"))
            continue

        injections.append(
            Injection(
                injection_id=injection_id,
                enabled=enabled,
                scheme=scheme,
                sample_id=sample_id,
                stage=stage,
                action=action,
                message=message,
                parameters=parameters,
            )
        )
    injections.sort(key=lambda item: item.injection_id)
    return injections, failures


def load_injections(path: Path) -> tuple[list[Injection], list[Failure], str]:
    try:
        raw = read_json(path)
        file_hash = sha256_file(path)
    except (OSError, ValueError) as exc:
        failure = Failure(
            code=ErrorCode.IO_ERROR,
            message=f"无法读取异常注入配置: {exc}",
            stage="injection_validation",
            details={"path": path.as_posix()},
        )
        return [], [failure], ""
    injections, failures = parse_injections(raw)
    digest = sha256_json(
        {"content": raw, "file_sha256": file_hash, "schema": "rope-diag-injection-v1"}
    )
    return injections, failures, digest


def first_match(
    injections: Iterable[Injection],
    *,
    scheme: str | None,
    sample_id: str | None,
    stage: str,
) -> Injection | None:
    for injection in injections:
        if injection.matches(scheme=scheme, sample_id=sample_id, stage=stage):
            return injection
    return None
