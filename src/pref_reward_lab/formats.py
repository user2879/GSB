from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .config import LabConfig
from .models import ParseResult, RawPair, V1Record, V2Record

V1_PAIR_ID_WIDTH = 24
V1_INT_WIDTH = 12
V1_LINE_WIDTH = V1_PAIR_ID_WIDTH + V1_INT_WIDTH * 2
PAIR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
INT_MIN = -99_999_999_999
INT_MAX = 99_999_999_999


def _failure(
    message: str,
    *,
    stage: str,
    source_version: str,
    pair_id: str | None = None,
    line: int | None = None,
    field: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": "E_FORMAT_INVALID",
        "message": message,
        "stage": stage,
        "source_version": source_version,
        "severity": "error",
    }
    if pair_id is not None:
        result["pair_id"] = pair_id
    if line is not None:
        result["line"] = line
    if field is not None:
        result["field"] = field
    return result


def _valid_pair_id(value: Any) -> bool:
    return isinstance(value, str) and bool(PAIR_ID_PATTERN.match(value))


def raw_pair_to_dict(pair: RawPair) -> dict[str, str]:
    return pair.content_dict()


def raw_pair_from_dict(value: dict[str, Any]) -> RawPair:
    return RawPair(
        pair_id=str(value["pair_id"]),
        prompt=str(value["prompt"]),
        chosen=str(value["chosen"]),
        rejected=str(value["rejected"]),
        annotator_id=str(value["annotator_id"]),
        timestamp=str(value["timestamp"]),
    )


def parse_raw_pairs(path: str | Path) -> ParseResult:
    records: dict[str, RawPair] = {}
    failures: list[dict[str, Any]] = []
    required_text = ("pair_id", "prompt", "chosen", "rejected", "annotator_id", "timestamp")
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                failures.append(
                    _failure(
                        f"invalid JSON: {exc.msg}",
                        stage="parse_raw",
                        source_version="raw",
                        line=line_number,
                    )
                )
                continue
            if not isinstance(value, dict):
                failures.append(
                    _failure(
                        "raw pair line must be a JSON object",
                        stage="parse_raw",
                        source_version="raw",
                        line=line_number,
                    )
                )
                continue
            bad_field = next(
                (field for field in required_text if not isinstance(value.get(field), str)),
                None,
            )
            if bad_field is None:
                bad_field = next(
                    (
                        field
                        for field in required_text
                        if not str(value.get(field, "")).strip()
                    ),
                    None,
                )
            if bad_field is not None:
                failures.append(
                    _failure(
                        f"raw pair field {bad_field} must be a non-empty string",
                        stage="parse_raw",
                        source_version="raw",
                        pair_id=str(value.get("pair_id", "")) or None,
                        line=line_number,
                        field=bad_field,
                    )
                )
                continue
            pair_id = str(value["pair_id"])
            if not _valid_pair_id(pair_id):
                failures.append(
                    _failure(
                        "pair_id contains unsupported characters or is too long",
                        stage="parse_raw",
                        source_version="raw",
                        pair_id=pair_id,
                        line=line_number,
                        field="pair_id",
                    )
                )
                continue
            if pair_id in records:
                failures.append(
                    _failure(
                        "duplicate pair_id in raw JSONL",
                        stage="parse_raw",
                        source_version="raw",
                        pair_id=pair_id,
                        line=line_number,
                        field="pair_id",
                    )
                )
                continue
            records[pair_id] = RawPair(
                pair_id=pair_id,
                prompt=str(value["prompt"]),
                chosen=str(value["chosen"]),
                rejected=str(value["rejected"]),
                annotator_id=str(value["annotator_id"]),
                timestamp=str(value["timestamp"]),
            )
    return ParseResult(records=records, failures=failures)


def encode_v1_record(record: V1Record) -> str:
    if not _valid_pair_id(record.pair_id):
        raise ValueError(f"invalid v1 pair_id: {record.pair_id!r}")
    if len(record.pair_id) > V1_PAIR_ID_WIDTH:
        raise ValueError(f"v1 pair_id exceeds {V1_PAIR_ID_WIDTH} characters")
    if not INT_MIN <= record.score_milli <= INT_MAX:
        raise ValueError("v1 score_milli out of supported range")
    if not INT_MIN <= record.delta_milli <= INT_MAX:
        raise ValueError("v1 delta_milli out of supported range")
    return (
        f"{record.pair_id:<{V1_PAIR_ID_WIDTH}}"
        f"{record.score_milli:>{V1_INT_WIDTH}d}"
        f"{record.delta_milli:>{V1_INT_WIDTH}d}"
    )


def parse_v1_text(text: str) -> ParseResult:
    records: dict[str, V1Record] = {}
    failures: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if len(line) != V1_LINE_WIDTH:
            failures.append(
                _failure(
                    f"v1 line width is {len(line)}, expected {V1_LINE_WIDTH}",
                    stage="parse_v1",
                    source_version="v1",
                    line=line_number,
                )
            )
            continue
        pair_id = line[:V1_PAIR_ID_WIDTH].strip()
        score_text = line[V1_PAIR_ID_WIDTH : V1_PAIR_ID_WIDTH + V1_INT_WIDTH].strip()
        delta_text = line[V1_PAIR_ID_WIDTH + V1_INT_WIDTH :].strip()
        if not _valid_pair_id(pair_id):
            failures.append(
                _failure(
                    "invalid v1 pair_id field",
                    stage="parse_v1",
                    source_version="v1",
                    pair_id=pair_id or None,
                    line=line_number,
                    field="pair_id",
                )
            )
            continue
        try:
            score_milli = int(score_text, 10)
        except ValueError:
            failures.append(
                _failure(
                    "v1 score field is not a base-10 integer",
                    stage="parse_v1",
                    source_version="v1",
                    pair_id=pair_id,
                    line=line_number,
                    field="score_milli",
                )
            )
            continue
        try:
            delta_milli = int(delta_text, 10)
        except ValueError:
            failures.append(
                _failure(
                    "v1 delta field is not a base-10 integer",
                    stage="parse_v1",
                    source_version="v1",
                    pair_id=pair_id,
                    line=line_number,
                    field="delta_milli",
                )
            )
            continue
        if not INT_MIN <= score_milli <= INT_MAX or not INT_MIN <= delta_milli <= INT_MAX:
            failures.append(
                _failure(
                    "v1 integer field outside supported range",
                    stage="parse_v1",
                    source_version="v1",
                    pair_id=pair_id,
                    line=line_number,
                )
            )
            continue
        if pair_id in records:
            failures.append(
                _failure(
                    "duplicate pair_id in v1 exchange",
                    stage="parse_v1",
                    source_version="v1",
                    pair_id=pair_id,
                    line=line_number,
                    field="pair_id",
                )
            )
            continue
        records[pair_id] = V1Record(pair_id, score_milli, delta_milli)
    return ParseResult(records=records, failures=failures)


def parse_v1_file(path: str | Path) -> ParseResult:
    return parse_v1_text(Path(path).read_text(encoding="utf-8"))


def _v2_objects(text: str) -> tuple[list[Any], list[tuple[int, str]]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        objects: list[Any] = []
        failures: list[tuple[int, str]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as exc:
                failures.append((line_number, f"invalid JSON: {exc.msg}"))
        return objects, failures
    if isinstance(parsed, list):
        return list(parsed), []
    if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
        return list(parsed["records"]), []
    if isinstance(parsed, dict):
        return [parsed], []
    return [], [(1, "v2 root must be an object, array, or records object")]


def parse_v2_text(text: str, config: LabConfig | None = None) -> ParseResult:
    if config is None:
        config = LabConfig.from_dict({})
    if not text.strip():
        return ParseResult(
            records={},
            failures=[
                _failure(
                    "v2 exchange file is empty",
                    stage="parse_v2",
                    source_version="v2",
                    line=1,
                )
            ],
        )
    objects, structural_failures = _v2_objects(text)
    records: dict[str, V2Record] = {}
    failures: list[dict[str, Any]] = [
        _failure(
            message,
            stage="parse_v2",
            source_version="v2",
            line=line,
        )
        for line, message in structural_failures
    ]
    for index, value in enumerate(objects, 1):
        if not isinstance(value, dict):
            failures.append(
                _failure(
                    "v2 record must be a JSON object",
                    stage="parse_v2",
                    source_version="v2",
                    line=index,
                )
            )
            continue
        pair_id_value = value.get("pair_id", value.get("id"))
        pair_id = str(pair_id_value) if pair_id_value is not None else ""
        if not _valid_pair_id(pair_id):
            failures.append(
                _failure(
                    "invalid v2 pair_id field",
                    stage="parse_v2",
                    source_version="v2",
                    pair_id=pair_id or None,
                    line=index,
                    field="pair_id",
                )
            )
            continue
        score_value = value.get("reward_score", value.get("score"))
        delta_value = value.get("reward_delta", value.get("delta", 0.0))
        try:
            if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                raise ValueError
            if isinstance(delta_value, bool) or not isinstance(delta_value, (int, float)):
                raise ValueError
            reward_score = float(score_value)
            reward_delta = float(delta_value)
            if not math.isfinite(reward_score) or not math.isfinite(reward_delta):
                raise ValueError
        except (TypeError, ValueError):
            failures.append(
                _failure(
                    "v2 reward_score and reward_delta must be finite numbers",
                    stage="parse_v2",
                    source_version="v2",
                    pair_id=pair_id,
                    line=index,
                    field="reward_score/reward_delta",
                )
            )
            continue
        reward_label = value.get("reward_label", value.get("label"))
        if not isinstance(reward_label, str) or reward_label not in config.reward_labels:
            failures.append(
                _failure(
                    f"reward_label {reward_label!r} is not in configured enum",
                    stage="parse_v2",
                    source_version="v2",
                    pair_id=pair_id,
                    line=index,
                    field="reward_label",
                )
            )
            continue
        precision_label = value.get("precision_label", value.get("precision"))
        if not isinstance(precision_label, str) or precision_label not in config.precision_labels:
            failures.append(
                _failure(
                    f"precision_label {precision_label!r} is not in configured enum",
                    stage="parse_v2",
                    source_version="v2",
                    pair_id=pair_id,
                    line=index,
                    field="precision_label",
                )
            )
            continue
        if pair_id in records:
            failures.append(
                _failure(
                    "duplicate pair_id in v2 exchange",
                    stage="parse_v2",
                    source_version="v2",
                    pair_id=pair_id,
                    line=index,
                    field="pair_id",
                )
            )
            continue
        records[pair_id] = V2Record(
            pair_id=pair_id,
            reward_score=reward_score,
            reward_delta=reward_delta,
            reward_label=reward_label,
            precision_label=precision_label,
        )
    return ParseResult(records=records, failures=failures)


def parse_v2_file(path: str | Path, config: LabConfig | None = None) -> ParseResult:
    return parse_v2_text(Path(path).read_text(encoding="utf-8"), config)


def write_v1_file(path: str | Path, records: Iterable[V1Record]) -> None:
    ordered = sorted(records, key=lambda record: record.pair_id)
    text = "".join(encode_v1_record(record) + "\n" for record in ordered)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def v2_record_to_dict(record: V2Record) -> dict[str, Any]:
    return {
        "pair_id": record.pair_id,
        "reward_score": record.reward_score,
        "reward_delta": record.reward_delta,
        "reward_label": record.reward_label,
        "precision_label": record.precision_label,
    }


def write_v2_file(path: str | Path, records: Iterable[V2Record]) -> None:
    ordered = sorted(records, key=lambda record: record.pair_id)
    text = json.dumps(
        [v2_record_to_dict(record) for record in ordered],
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text + "\n", encoding="utf-8", newline="\n")
