"""Failure and event helpers with deterministic ordering."""

from __future__ import annotations

from typing import Any, Iterable


STAGE_ORDER = {
    "config": 0,
    "snapshot": 1,
    "load": 2,
    "normalize": 3,
    "fuse": 4,
    "truncate": 5,
    "report": 6,
}


def failure(
    stage: str,
    code: str,
    detail: str,
    *,
    channel: str | None = None,
    candidate_id: str | None = None,
    injection_id: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "channel": channel,
        "code": code,
        "detail": detail,
        "injection_id": injection_id,
        "stage": stage,
    }


def failure_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        STAGE_ORDER.get(str(item.get("stage")), 99),
        str(item.get("stage") or ""),
        str(item.get("channel") or ""),
        str(item.get("candidate_id") or ""),
        str(item.get("code") or ""),
        str(item.get("injection_id") or ""),
        str(item.get("detail") or ""),
    )


def sort_failures(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=failure_sort_key)


def event_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("channel") or ""),
        str(item.get("code") or ""),
        str(item.get("candidate_id") or ""),
        str(item.get("detail") or ""),
    )


def sort_events(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=event_sort_key)
