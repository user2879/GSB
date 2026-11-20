from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


E_INPUT_INVALID = "E_INPUT_INVALID"
E_DECODE_PARAM_INVALID = "E_DECODE_PARAM_INVALID"
E_RESPONSE_BUDGET_PENDING = "E_RESPONSE_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SESSION_NOT_FOUND = "E_SESSION_NOT_FOUND"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_INTERNAL = "E_INTERNAL"


@dataclass
class AppError(Exception):
    code: str
    message: str
    stage: str = "input"
    session_id: Optional[str] = None
    shard_seq: Optional[int] = None
    injection_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        prefix = f"[{self.code}]"
        if self.session_id is not None:
            prefix += f"[{self.session_id}]"
        if self.shard_seq is not None:
            prefix += f"[shard={self.shard_seq}]"
        return f"{prefix} {self.message}"

    def as_failure(self) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "business_code": self.code,
            "stage": self.stage,
            "session_id": self.session_id,
            "shard_seq": self.shard_seq,
            "message": self.message,
            "injection_id": self.injection_id,
            "details": self.details,
        }
        return item


def failure(
    code: str,
    stage: str,
    message: str,
    *,
    session_id: Optional[str] = None,
    shard_seq: Optional[int] = None,
    injection_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "business_code": code,
        "stage": stage,
        "session_id": session_id,
        "shard_seq": shard_seq,
        "message": message,
        "injection_id": injection_id,
        "details": details or {},
    }


def sort_failures(items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            item.get("session_id") or "",
            -1 if item.get("shard_seq") is None else int(item["shard_seq"]),
            item.get("stage") or "",
            item.get("business_code") or "",
            item.get("injection_id") or "",
            item.get("message") or "",
        ),
    )
