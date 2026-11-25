from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import Config
from .errors import (
    E_AUX_NON_INTEGER,
    E_CAND_COUNT_OVER,
    E_CANDIDATE_ID_DUPLICATE,
    E_CHANNEL_UNKNOWN,
    E_REQUEST_ID_DUPLICATE,
    E_SCORE_NON_FINITE,
    Failure,
)
from .models import Candidate, Request


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def parse_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], list[Failure]]:
    rows: list[dict[str, Any]] = []
    failures: list[Failure] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                failures.append(
                    Failure(
                        "E_JSON_LINE_INVALID",
                        str(exc),
                        source=str(path),
                        line_no=line_no,
                    )
                )
                continue
            if not isinstance(value, dict):
                failures.append(
                    Failure(
                        "E_JSON_LINE_INVALID",
                        "JSONL row must be an object",
                        source=str(path),
                        line_no=line_no,
                    )
                )
                continue
            value["_line_no"] = line_no
            rows.append(value)
    return rows, failures


def load_requests_jsonl(
    path: str | Path, config: Config
) -> tuple[dict[str, Request], list[Failure]]:
    rows, failures = parse_jsonl(path)
    requests: dict[str, Request] = {}
    source = str(path)

    for row in rows:
        line_no = int(row.get("_line_no", -1))
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            failures.append(
                Failure(
                    "E_REQUEST_ID_INVALID",
                    "request_id must be a non-empty string",
                    source=source,
                    line_no=line_no,
                )
            )
            continue
        if request_id in requests:
            failures.append(
                Failure(
                    E_REQUEST_ID_DUPLICATE,
                    f"duplicate request_id {request_id!r}",
                    source=source,
                    request_id=request_id,
                    line_no=line_no,
                )
            )
            continue

        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list):
            failures.append(
                Failure(
                    "E_CANDIDATES_INVALID",
                    "candidates must be a list",
                    source=source,
                    request_id=request_id,
                    line_no=line_no,
                )
            )
            continue

        request_failures: list[Failure] = []
        seen_ids: set[int] = set()
        candidates: list[Candidate] = []
        for index, raw in enumerate(raw_candidates):
            candidate_line = line_no if len(raw_candidates) == 1 else None
            if not isinstance(raw, dict):
                request_failures.append(
                    Failure(
                        "E_CANDIDATE_INVALID",
                        f"candidate at index {index} must be an object",
                        source=source,
                        request_id=request_id,
                        line_no=candidate_line,
                    )
                )
                continue
            candidate_id = raw.get("candidate_id")
            if not _is_int(candidate_id) or candidate_id <= 0:
                request_failures.append(
                    Failure(
                        "E_CANDIDATE_ID_INVALID",
                        "candidate_id must be a positive integer",
                        source=source,
                        request_id=request_id,
                        line_no=candidate_line,
                    )
                )
            elif candidate_id in seen_ids:
                request_failures.append(
                    Failure(
                        E_CANDIDATE_ID_DUPLICATE,
                        f"candidate_id {candidate_id} is duplicated in request",
                        source=source,
                        request_id=request_id,
                        candidate_id=candidate_id,
                        line_no=candidate_line,
                        injection_id=raw.get("_injection_id"),
                    )
                )
            else:
                seen_ids.add(candidate_id)

            score = raw.get("raw_score", raw.get("score"))
            if not _is_finite_number(score):
                request_failures.append(
                    Failure(
                        E_SCORE_NON_FINITE,
                        "raw_score must be a finite number",
                        source=source,
                        request_id=request_id,
                        candidate_id=candidate_id if _is_int(candidate_id) else None,
                        line_no=candidate_line,
                        injection_id=raw.get("_injection_id"),
                    )
                )

            aux = raw.get("aux_feature", 0)
            if aux is None:
                aux = 0
            if not _is_int(aux):
                request_failures.append(
                    Failure(
                        E_AUX_NON_INTEGER,
                        "aux_feature must be an integer",
                        source=source,
                        request_id=request_id,
                        candidate_id=candidate_id if _is_int(candidate_id) else None,
                        line_no=candidate_line,
                    )
                )
                aux = 0

            channel = raw.get("channel")
            if not isinstance(channel, str) or channel not in config.raw["channels"]:
                request_failures.append(
                    Failure(
                        E_CHANNEL_UNKNOWN,
                        f"channel {channel!r} is not present in config.channels",
                        source=source,
                        request_id=request_id,
                        candidate_id=candidate_id if _is_int(candidate_id) else None,
                        line_no=candidate_line,
                    )
                )

            if request_failures and request_failures[-1].candidate_id == (
                candidate_id if _is_int(candidate_id) else None
            ) and request_failures[-1].code in {
                E_CANDIDATE_ID_DUPLICATE,
                E_SCORE_NON_FINITE,
            }:
                # Still avoid appending a structurally unusable duplicate/bad row.
                if (
                    request_failures[-1].code == E_CANDIDATE_ID_DUPLICATE
                    or request_failures[-1].code == E_SCORE_NON_FINITE
                ):
                    continue

            if _is_int(candidate_id) and _is_finite_number(score) and isinstance(channel, str):
                candidates.append(
                    Candidate(
                        candidate_id=int(candidate_id),
                        raw_score=float(score),
                        channel=channel,
                        aux_feature=int(aux) if _is_int(aux) else 0,
                        content_key=raw.get("content_key"),
                        source_line=candidate_line,
                    )
                )

        if len(raw_candidates) > config.max_candidates:
            request_failures.append(
                Failure(
                    E_CAND_COUNT_OVER,
                    f"candidate count {len(raw_candidates)} exceeds max {config.max_candidates}",
                    source=source,
                    request_id=request_id,
                    line_no=line_no,
                    business_code=E_CAND_COUNT_OVER,
                )
            )

        failures.extend(request_failures)
        blocking = {
            E_CAND_COUNT_OVER,
            E_CANDIDATE_ID_DUPLICATE,
            E_SCORE_NON_FINITE,
            E_AUX_NON_INTEGER,
            E_CHANNEL_UNKNOWN,
            "E_CANDIDATE_ID_INVALID",
            "E_CANDIDATES_INVALID",
        }
        if not any(item.code in blocking for item in request_failures):
            requests[request_id] = Request(request_id, candidates, line_no)

    failures.sort(key=lambda item: (item.line_no if item.line_no is not None else 10**12,
                                    item.request_id or "", item.candidate_id or 0, item.code))
    return dict(sorted(requests.items())), failures


def load_command_stream(path: str | Path) -> tuple[list[dict[str, Any]], list[Failure]]:
    return parse_jsonl(path)
