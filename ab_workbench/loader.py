"""JSONL loading and strict validation of exposure events."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .errors import E_LOG_INVALID, FailureRecord

REQUIRED_FIELDS = (
    "request_id",
    "group",
    "user_id",
    "item_id",
    "position",
    "score",
    "clicked",
    "converted",
    "event_ts_ms",
)


@dataclass(frozen=True)
class Exposure:
    request_id: str
    group: str
    user_id: str
    item_id: str
    position: int
    score: float
    clicked: int
    converted: int
    event_ts_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoadResult:
    exposures: List[Exposure]
    failures: List[FailureRecord]
    file_fingerprint: str

    @property
    def sorted_exposures(self) -> List[Exposure]:
        return sorted(
            self.exposures, key=lambda e: (e.request_id, e.item_id, e.position)
        )


def _validate_record(
    raw: Dict[str, Any], line_no: int, valid_groups
) -> Tuple[Optional[Exposure], Optional[FailureRecord]]:
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        return None, FailureRecord(
            stage="load",
            code=E_LOG_INVALID,
            reason="missing_fields",
            request_id=str(raw.get("request_id", "")),
            item_id=str(raw.get("item_id", "")),
            detail=",".join(missing),
        )
    request_id = raw["request_id"]
    item_id = raw["item_id"]
    group = raw["group"]

    def fail(reason: str, detail: str = "") -> FailureRecord:
        return FailureRecord(
            stage="load",
            code=E_LOG_INVALID,
            reason=reason,
            request_id=str(request_id),
            item_id=str(item_id),
            group=str(group),
            detail=detail,
        )

    if not isinstance(request_id, str) or not request_id:
        return None, fail("request_id_not_string")
    if not isinstance(item_id, str) or not item_id:
        return None, fail("item_id_not_string")
    if not isinstance(raw["user_id"], str) or not raw["user_id"]:
        return None, fail("user_id_not_string")
    if group not in valid_groups:
        return None, fail("group_out_of_range", repr(group))
    position = raw["position"]
    if not isinstance(position, int) or isinstance(position, bool) or position <= 0:
        return None, fail("position_not_positive_int", repr(position))
    score = raw["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None, fail("score_not_numeric", repr(score))
    if not math.isfinite(float(score)):
        return None, fail("score_not_finite", repr(score))
    for flag_name in ("clicked", "converted"):
        flag = raw[flag_name]
        if not isinstance(flag, int) or isinstance(flag, bool) or flag not in (0, 1):
            return None, fail(f"{flag_name}_not_binary", repr(flag))
    ts = raw["event_ts_ms"]
    if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
        return None, fail("timestamp_negative_or_invalid", repr(ts))
    return (
        Exposure(
            request_id=request_id,
            group=group,
            user_id=raw["user_id"],
            item_id=item_id,
            position=position,
            score=float(score),
            clicked=int(raw["clicked"]),
            converted=int(raw["converted"]),
            event_ts_ms=ts,
        ),
        None,
    )


def load_exposures(path: str, valid_groups) -> LoadResult:
    from .fingerprint import fingerprint_file

    fp = fingerprint_file(path)
    exposures: List[Exposure] = []
    failures: List[FailureRecord] = []
    seen_pairs: Dict[Tuple[str, str], int] = {}
    request_ts: Dict[str, List[Tuple[int, int, str]]] = {}
    request_group: Dict[str, str] = {}
    request_user: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    FailureRecord(
                        stage="load",
                        code=E_LOG_INVALID,
                        reason="json_parse_error",
                        detail=f"line={line_no}:{exc.msg}",
                    )
                )
                continue
            exposure, failure = _validate_record(raw, line_no, valid_groups)
            if failure is not None:
                failures.append(failure)
                continue
            pair = (exposure.request_id, exposure.item_id)
            if pair in seen_pairs:
                failures.append(
                    FailureRecord(
                        stage="load",
                        code=E_LOG_INVALID,
                        reason="duplicate_request_item",
                        request_id=exposure.request_id,
                        item_id=exposure.item_id,
                        group=exposure.group,
                    )
                )
                continue
            seen_pairs[pair] = line_no
            prior_group = request_group.get(exposure.request_id)
            if prior_group is not None and prior_group != exposure.group:
                failures.append(
                    FailureRecord(
                        stage="load",
                        code=E_LOG_INVALID,
                        reason="inconsistent_group_within_request",
                        request_id=exposure.request_id,
                        item_id=exposure.item_id,
                        group=exposure.group,
                    )
                )
                continue
            prior_user = request_user.get(exposure.request_id)
            if prior_user is not None and prior_user != exposure.user_id:
                failures.append(
                    FailureRecord(
                        stage="load",
                        code=E_LOG_INVALID,
                        reason="inconsistent_user_within_request",
                        request_id=exposure.request_id,
                        item_id=exposure.item_id,
                        group=exposure.group,
                    )
                )
                continue
            request_group[exposure.request_id] = exposure.group
            request_user[exposure.request_id] = exposure.user_id
            request_ts.setdefault(exposure.request_id, []).append(
                (exposure.event_ts_ms, line_no, exposure.item_id)
            )
            exposures.append(exposure)

    for request_id, entries in request_ts.items():
        ordered = sorted(entries, key=lambda x: (x[1],))
        previous = -1
        for ts, line_no, item_id in ordered:
            if ts < previous:
                failures.append(
                    FailureRecord(
                        stage="load",
                        code=E_LOG_INVALID,
                        reason="timestamp_not_monotonic_within_request",
                        request_id=request_id,
                        item_id=item_id,
                    )
                )
            previous = ts

    valid_request_ids = {e.request_id for e in exposures}
    monotonic_bad = {
        f.request_id for f in failures if f.reason == "timestamp_not_monotonic_within_request"
    }
    if monotonic_bad:
        exposures = [e for e in exposures if e.request_id not in monotonic_bad]
    _ = valid_request_ids
    failures.sort(key=lambda f: f.sort_key())
    return LoadResult(exposures=exposures, failures=failures, file_fingerprint=fp)


def load_attributes(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    attributes: Dict[str, Dict[str, str]] = {}
    for user_id, attrs in raw.items():
        attributes[user_id] = {str(k): str(v) for k, v in sorted(attrs.items())}
    return attributes


def load_injections(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return list(raw.get("injections", []))
