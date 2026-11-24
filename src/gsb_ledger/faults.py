"""Isolated, explicitly identified receipt fault injection."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_FAULT_MISSED, E_RECEIPT_MALFORMED, E_UNKNOWN_PARAM_VERSION, E_DUPLICATE_RANK, Issue
from .util import read_json


SUPPORTED_FAULTS = {
    "truncate_line",
    "unknown_param_version",
    "non_finite_duration",
    "duplicate_rank",
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    kind: str
    path: str
    line: int
    expected_code: str
    enabled: bool = True
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "kind": self.kind,
            "path": self.path,
            "line": self.line,
            "expected_code": self.expected_code,
            "enabled": self.enabled,
            "value": self.value,
        }


@dataclass
class FaultEffect:
    text: str
    obj: dict[str, Any] | None
    force_code: str = ""
    injection_ids: tuple[str, ...] = ()
    message: str = ""


@dataclass
class InjectionReport:
    injection: Injection
    hit_count: int = 0
    observed_codes: list[str] = field(default_factory=list)
    observed_details: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.hit_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection.injection_id,
            "kind": self.injection.kind,
            "path": self.injection.path,
            "line": self.injection.line,
            "expected_code": self.injection.expected_code,
            "enabled": self.injection.enabled,
            "hit": self.hit,
            "hit_count": self.hit_count,
            "observed_codes": sorted(set(self.observed_codes)),
            "observed_details": sorted(set(self.observed_details)),
            "messages": sorted(set(self.messages)),
        }


class FaultInjector:
    """Applies configured faults without mutating source files."""

    def __init__(
        self,
        injections: tuple[Injection, ...] = (),
        *,
        project_root: Path | None = None,
        source_path: str = "",
    ) -> None:
        self.injections = injections
        self.project_root = project_root
        self.source_path = source_path
        self._by_target: dict[tuple[str, int], list[Injection]] = {}
        self._last_rank: dict[str, int] = {}
        self.reports: dict[str, InjectionReport] = {
            item.injection_id: InjectionReport(item) for item in injections
        }
        for item in injections:
            if item.enabled:
                self._by_target.setdefault((self._normalize_path(item.path), item.line), []).append(item)

    def _normalize_path(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() and self.project_root is not None:
            path = self.project_root / path
        return path.resolve().as_posix()

    def at_line(self, path: Path, line: int) -> tuple[Injection, ...]:
        return tuple(self._by_target.get((path.resolve().as_posix(), line), ()))

    def transform_text(self, text: str, path: Path, line: int) -> FaultEffect:
        injections = self.at_line(path, line)
        if not injections:
            return FaultEffect(text=text, obj=None)
        injection_ids = tuple(item.injection_id for item in injections)
        message_parts: list[str] = []
        force_code = ""
        current_text = text
        parsed_obj: dict[str, Any] | None = None
        for injection in injections:
            if injection.kind == "truncate_line":
                current_text = current_text[: max(1, len(current_text) // 2)]
                force_code = injection.expected_code or E_RECEIPT_MALFORMED
                message_parts.append("receipt row truncated")
            elif injection.kind == "unknown_param_version":
                if parsed_obj is None:
                    try:
                        candidate = json.loads(current_text)
                        parsed_obj = copy.deepcopy(candidate) if isinstance(candidate, dict) else None
                    except json.JSONDecodeError:
                        parsed_obj = None
                if parsed_obj is not None:
                    parsed_obj["param_version"] = (
                        injection.value if isinstance(injection.value, int) else 999_999
                    )
                message_parts.append("unknown parameter version injected")
            elif injection.kind == "non_finite_duration":
                if parsed_obj is None:
                    try:
                        candidate = json.loads(current_text)
                        parsed_obj = copy.deepcopy(candidate) if isinstance(candidate, dict) else None
                    except json.JSONDecodeError:
                        parsed_obj = None
                if parsed_obj is not None:
                    parsed_obj["duration_ms"] = float("nan")
                message_parts.append("non-finite duration injected")
            elif injection.kind == "duplicate_rank":
                if parsed_obj is None:
                    try:
                        candidate = json.loads(current_text)
                        parsed_obj = copy.deepcopy(candidate) if isinstance(candidate, dict) else None
                    except json.JSONDecodeError:
                        parsed_obj = None
                if parsed_obj is not None:
                    key = path.resolve().as_posix()
                    previous_rank = self._last_rank.get(key)
                    if previous_rank is not None:
                        parsed_obj["rank"] = previous_rank
                    elif isinstance(parsed_obj.get("rank"), int):
                        self._last_rank[key] = int(parsed_obj["rank"])
                message_parts.append("duplicate in-batch rank injected")
            else:
                message_parts.append(f"unsupported fault {injection.kind!r} ignored")
        if parsed_obj is not None:
            current_text = json.dumps(parsed_obj, ensure_ascii=False, sort_keys=True)
        return FaultEffect(
            text=current_text,
            obj=parsed_obj,
            force_code=force_code,
            injection_ids=injection_ids,
            message="; ".join(message_parts),
        )

    def observe_rank(self, path: Path, rank: Any) -> None:
        if isinstance(rank, int) and not isinstance(rank, bool):
            self._last_rank[path.resolve().as_posix()] = rank

    def observe_issue(self, issue: Issue) -> None:
        if not issue.injection_id:
            return
        for injection_id in issue.injection_id.split(","):
            report = self.reports.get(injection_id)
            if report is None:
                continue
            report.hit_count += 1
            report.observed_codes.append(issue.code)
            if issue.detail_code:
                report.observed_details.append(issue.detail_code)
            report.messages.append(issue.message)

    def reports_for(self, issues: list[Issue]) -> list[dict[str, Any]]:
        for report in self.reports.values():
            report.hit_count = 0
            report.observed_codes = []
            report.observed_details = []
            report.messages = []
        for issue in issues:
            self.observe_issue(issue)
        result = [report.to_dict() for report in self.reports.values()]
        for report in result:
            if report["enabled"] and not report["hit"]:
                report["observed_codes"] = sorted(
                    set(report["observed_codes"]) | {E_FAULT_MISSED}
                )
        return sorted(result, key=lambda item: item["injection_id"])


def load_fault_injector(
    path: str | Path,
    *,
    project_root: Path | None = None,
) -> FaultInjector:
    fault_path = Path(path)
    try:
        raw = read_json(fault_path)
    except (OSError, json.JSONDecodeError):
        return FaultInjector(project_root=project_root, source_path=str(fault_path))
    items_raw = raw.get("injections", []) if isinstance(raw, dict) else []
    injections: list[Injection] = []
    for index, item in enumerate(items_raw if isinstance(items_raw, list) else []):
        if not isinstance(item, dict):
            continue
        injection_id = item.get("injection_id") or item.get("id") or f"INJ-UNNAMED-{index + 1}"
        kind = item.get("kind", "")
        if kind not in SUPPORTED_FAULTS:
            continue
        target_path = item.get("path", "")
        line = item.get("line", 0)
        if not isinstance(target_path, str) or not isinstance(line, int) or line < 1:
            continue
        expected_code = item.get("expected_code", "")
        if expected_code not in {
            E_RECEIPT_MALFORMED,
            E_UNKNOWN_PARAM_VERSION,
            E_DUPLICATE_RANK,
        }:
            expected_code = {
                "truncate_line": E_RECEIPT_MALFORMED,
                "non_finite_duration": E_RECEIPT_MALFORMED,
                "unknown_param_version": E_UNKNOWN_PARAM_VERSION,
                "duplicate_rank": E_DUPLICATE_RANK,
            }[kind]
        injections.append(
            Injection(
                injection_id=str(injection_id),
                kind=kind,
                path=target_path.replace("\\", "/"),
                line=int(line),
                expected_code=expected_code,
                enabled=bool(item.get("enabled", True)),
                value=item.get("value"),
            )
        )
    return FaultInjector(
        tuple(injections),
        project_root=project_root,
        source_path=str(fault_path),
    )
