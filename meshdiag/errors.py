from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Diagnostic:
    code: str
    path: str
    message: str
    value: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.value is not None:
            result["value"] = self.value
        result.update(self.extra)
        return result


class MeshError(Exception):
    """A deterministic, machine-readable diagnostic failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        diagnostics: Iterable[Diagnostic] = (),
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.diagnostics = list(diagnostics)
        self.extra = dict(extra or {})

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "errors": [entry.as_dict() for entry in self.diagnostics],
        }
        payload.update(self.extra)
        return payload

    def __str__(self) -> str:
        if not self.diagnostics:
            return f"{self.code}: {self.message}"
        details = "; ".join(
            f"{entry.code} {entry.path}: {entry.message}" for entry in self.diagnostics
        )
        return f"{self.code}: {self.message}; {details}"


def diag(code: str, path: str, message: str, value: Any = None, **extra: Any) -> Diagnostic:
    return Diagnostic(code=code, path=path, message=message, value=value, extra=extra)


def fail_rule(errors: Iterable[Diagnostic], message: str = "规则修订校验失败") -> None:
    diagnostics = sorted(errors, key=lambda item: (item.path, item.code, item.message))
    if diagnostics:
        raise MeshError("E_RULE", message, diagnostics=diagnostics)


def fail_ref(errors: Iterable[Diagnostic], message: str = "规则引用校验失败") -> None:
    diagnostics = sorted(errors, key=lambda item: (item.path, item.code, item.message))
    if diagnostics:
        raise MeshError("E_REF", message, diagnostics=diagnostics)


def fail_cycle(cycle: Iterable[str]) -> None:
    sequence = list(cycle)
    raise MeshError(
        "E_CYCLE",
        "规则依赖图存在环",
        diagnostics=[
            diag(
                "E_CYCLE",
                "rules.depends_on",
                "依赖成环：" + " -> ".join(sequence),
            )
        ],
        extra={"cycle": sequence},
    )


def fail_replay(
    message: str,
    *,
    request_id: str | None = None,
    rule_id: str | None = None,
    field_name: str | None = None,
    expected: Any = None,
    actual: Any = None,
    basis: str | None = None,
) -> None:
    diagnostics = []
    if request_id is not None or field_name is not None:
        path = ".".join(
            part
            for part in (
                f"requests[{request_id}]" if request_id is not None else "",
                field_name or "",
            )
            if part
        )
        diagnostics.append(
            diag(
                "E_REPLAY",
                path or "replay",
                basis or "重放结果与快照账本不一致",
                actual,
                expected=expected,
                actual=actual,
                request_id=request_id,
                rule_id=rule_id,
                field=field_name,
                basis=basis,
            )
        )
    raise MeshError("E_REPLAY", message, diagnostics=diagnostics)
