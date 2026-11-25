from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .errors import E_FIELD_TRUNCATED, E_SELECT_CODE_UNKNOWN, Failure
from .fingerprinting import derive_cache_key, fingerprint
from .pipeline import RequestResult


SELECTION_CODES: dict[int, dict[str, str]] = {
    1: {"name": "SAMPLED", "description": "in TopK and selected by layer sampling"},
    2: {"name": "SELECTED_NOT_SAMPLED", "description": "in TopK but not selected by layer sampling"},
    3: {"name": "TRUNCATED", "description": "outside TopK"},
}
NAME_TO_CODE = {value["name"]: code for code, value in SELECTION_CODES.items()}


def selection_code_for(candidate_id: int, result: RequestResult) -> int:
    if candidate_id in set(result.sampled_ids):
        return 1
    if candidate_id in set(result.top_k_ids):
        return 2
    return 3


def make_v1_line(record: dict[str, Any], config: Config) -> tuple[str | None, Failure | None]:
    widths = config.raw["v1_fields"]
    request_id = str(record["request_id"])
    seq = int(record["seq"])
    candidate_id = int(record["candidate_id"])
    score = int(record["score_permille"])
    code = int(record["selection_code"])
    values = [request_id, str(seq), str(candidate_id), str(score), str(code)]
    names = ["request_id", "seq", "candidate_id", "score_permille", "select_code"]
    width_values = [int(widths[name]) for name in names]
    for name, value, width in zip(names, values, width_values):
        if len(value) > width:
            return None, Failure(
                E_FIELD_TRUNCATED,
                f"{name} value {value!r} exceeds fixed width {width}",
                source="v1",
                request_id=request_id,
                candidate_id=candidate_id,
                business_code=E_FIELD_TRUNCATED,
            )

    padded = [
        request_id.ljust(width_values[0]),
        str(seq).zfill(width_values[1]),
        str(candidate_id).zfill(width_values[2]),
        str(score).rjust(width_values[3]),
        str(code).zfill(width_values[4]),
    ]
    separator = str(widths.get("separator", "|"))
    return separator.join(padded) if separator else "".join(padded), None


def result_to_records(result: RequestResult, config: Config) -> list[dict[str, Any]]:
    score_by_id = {item["candidate_id"]: item for item in result.scores}
    layer_by_candidate: dict[int, dict[str, Any]] = {}
    for layer in result.layers.values():
        for cid in layer["population_ids"]:
            layer_by_candidate[cid] = layer

    top_set = set(result.top_k_ids)
    sample_set = set(result.sampled_ids)
    shuffle_set = set(result.shuffle_order)
    records: list[dict[str, Any]] = []
    for seq, cid in enumerate(result.candidate_ids, 1):
        score_item = score_by_id[cid]
        code = 1 if cid in sample_set else 2 if cid in top_set else 3
        layer = layer_by_candidate.get(cid)
        weight = None
        if code == 1 and layer and layer.get("quota"):
            weight = round(layer["population"] / layer["quota"], 6)
        normalized = float(score_item["normalized_float"])
        score_mille = int(
            Decimal(repr(normalized)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        records.append(
            {
                "schema": "candidate-conclusion-v2",
                "request_id": result.request_id,
                "seq": seq,
                "candidate_id": cid,
                "score": round(normalized / 1000.0, 9),
                "score_permille": score_mille,
                "fixed_score": score_item["fixed_score"],
                "dequant_score": score_item["dequant_score"],
                "selection_code": code,
                "selection_result": SELECTION_CODES[code]["name"],
                "sampling_scheme": config.raw["sampling"]["scheme"],
                "sampling_weight": weight,
                "precision_label": (
                    f"fixed-d{score_item['fixed_decimals']}-gear"
                    f"{config.raw['quantization']['gear_exponent']}"
                ),
                "channel": score_item["channel"],
                "aux_feature": score_item["aux_feature"],
                "rank": result.ranked_ids.index(cid) + 1,
                "top_k": cid in top_set,
                "boundary": {
                    "boundary_selected": cid in result.boundary.get("boundary_selected_ids", []),
                    "boundary_rejected": cid in result.boundary.get("boundary_rejected_ids", []),
                    "nearest_rejected": cid == result.boundary.get("nearest_rejected_id"),
                },
                "shuffle_position": (
                    result.shuffle_order.index(cid) + 1 if cid in shuffle_set else None
                ),
            }
        )
    return records


def serialize_v1(
    records: Iterable[dict[str, Any]], config: Config
) -> tuple[list[str], list[Failure]]:
    lines: list[str] = []
    failures: list[Failure] = []
    for record in records:
        line, failure = make_v1_line(record, config)
        if failure:
            failures.append(failure)
        elif line is not None:
            lines.append(line)
    return lines, failures


def parse_v1_lines(
    lines: Iterable[str], config: Config, *, source: str = "<v1>"
) -> tuple[list[dict[str, Any]], list[Failure]]:
    widths = config.raw["v1_fields"]
    separator = str(widths.get("separator", "|"))
    names = ["request_id", "seq", "candidate_id", "score_permille", "select_code"]
    width_values = [int(widths[name]) for name in names]
    records: list[dict[str, Any]] = []
    failures: list[Failure] = []

    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        request_id = None
        candidate_id = None
        try:
            if separator and separator in line:
                parts = line.split(separator)
                if len(parts) != 5:
                    raise ValueError("expected five fields separated by configured separator")
                for name, part, width in zip(names, parts, width_values):
                    if name == "score_permille" and len(part) > width:
                        failures.append(
                            Failure(
                                E_FIELD_TRUNCATED,
                                f"score field length {len(part)} exceeds width {width}",
                                source=source,
                                line_no=line_no,
                                business_code=E_FIELD_TRUNCATED,
                            )
                        )
                        raise _FieldTruncated
                request_id = parts[0].strip()
                seq = int(parts[1])
                candidate_id = int(parts[2])
                score = int(parts[3].strip())
                code = int(parts[4])
            else:
                pos = 0
                values: list[str] = []
                for name, width in zip(names, width_values):
                    piece = line[pos : pos + width]
                    if name == "score_permille" and len(piece.strip()) > width:
                        raise _FieldTruncated
                    if len(piece) < width:
                        failures.append(
                            Failure(
                                E_FIELD_TRUNCATED,
                                f"line ends inside field {name}",
                                source=source,
                                line_no=line_no,
                                business_code=E_FIELD_TRUNCATED,
                            )
                        )
                        raise _FieldTruncated
                    values.append(piece)
                    pos += width
                request_id = values[0].strip()
                seq = int(values[1])
                candidate_id = int(values[2])
                score = int(values[3].strip())
                code = int(values[4])
        except _FieldTruncated:
            continue
        except ValueError as exc:
            failures.append(
                Failure(
                    "E_V1_LINE_INVALID",
                    f"cannot parse v1 line {line_no}: {exc}",
                    source=source,
                    line_no=line_no,
                    request_id=request_id,
                    candidate_id=candidate_id,
                )
            )
            continue

        if code not in SELECTION_CODES:
            failures.append(
                Failure(
                    E_SELECT_CODE_UNKNOWN,
                    f"unknown selection code {code}",
                    source=source,
                    line_no=line_no,
                    request_id=request_id,
                    candidate_id=candidate_id,
                    business_code=E_SELECT_CODE_UNKNOWN,
                )
            )
            continue
        records.append(_v1_record(request_id, seq, candidate_id, score, code, config))
    return records, failures


class _FieldTruncated(Exception):
    pass


def _v1_record(
    request_id: str, seq: int, candidate_id: int, score: int, code: int, config: Config
) -> dict[str, Any]:
    return {
        "schema": "candidate-conclusion-v2",
        "request_id": request_id,
        "seq": seq,
        "candidate_id": candidate_id,
        "score_permille": score,
        "score": float(Decimal(score) / Decimal(1000)),
        "selection_code": code,
        "selection_result": SELECTION_CODES[code]["name"],
        "sampling_scheme": config.raw["sampling"]["scheme"],
        "precision_label": "v1-permille-exact",
        "precision_source": "exact_integer_permille",
        "source_version": "v1",
    }


def v1_to_v2(
    lines: Iterable[str], config: Config, sidecar: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[Failure]]:
    records, failures = parse_v1_lines(lines, config)
    restored = (sidecar or {}).get("records", {})
    for record in records:
        key = f"{record['request_id']}:{record['candidate_id']}"
        extra = restored.get(key)
        if extra:
            fields = dict(extra.get("v2_fields", {}))
            record.update(fields)
            record["sidecar_restored"] = True
            record["restored_fields"] = sorted(fields)
        record["conversion"] = "v1_to_v2_exact_permille"
    return records, failures


def parse_v2_object(
    payload: Any, config: Config, *, source: str = "<v2>"
) -> tuple[list[dict[str, Any]], list[Failure]]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        raw_records = payload["records"]
    elif isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, dict):
        raw_records = [payload]
    else:
        return [], [Failure("E_V2_JSON_INVALID", "v2 payload must be object or list", source=source)]

    records: list[dict[str, Any]] = []
    failures: list[Failure] = []
    for index, raw in enumerate(raw_records, 1):
        if not isinstance(raw, dict):
            failures.append(Failure("E_V2_JSON_INVALID", f"record {index} is not object", source=source))
            continue
        record = dict(raw)
        name = record.get("selection_result")
        code = record.get("selection_code")
        if code is None and name in NAME_TO_CODE:
            record["selection_code"] = NAME_TO_CODE[name]
        elif code not in SELECTION_CODES and name not in NAME_TO_CODE:
            failures.append(
                Failure(
                    E_SELECT_CODE_UNKNOWN,
                    f"unknown v2 selection result/code: {name!r}/{code!r}",
                    source=source,
                    request_id=record.get("request_id"),
                    candidate_id=record.get("candidate_id"),
                    business_code=E_SELECT_CODE_UNKNOWN,
                )
            )
            continue
        if "score_permille" not in record and "score" in record:
            score_decimal = Decimal(str(record["score"]))
            if not score_decimal.is_finite():
                failures.append(
                    Failure(
                        "E_SCORE_NON_FINITE",
                        "v2 score must be finite",
                        source=source,
                        request_id=record.get("request_id"),
                        candidate_id=record.get("candidate_id"),
                    )
                )
                continue
            if score_decimal <= 1:
                record["score_permille"] = int(
                    (score_decimal * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                )
        record.setdefault("schema", "candidate-conclusion-v2")
        records.append(record)
    return records, failures


def parse_v2_text(
    text: str, config: Config, *, source: str = "<v2>"
) -> tuple[list[dict[str, Any]], list[Failure]]:
    stripped = text.strip()
    if not stripped:
        return [], []
    try:
        payload = json.loads(stripped)
        return parse_v2_object(payload, config, source=source)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        failures: list[Failure] = []
        for line_no, line in enumerate(stripped.splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(Failure("E_V2_JSON_INVALID", str(exc), source=source, line_no=line_no))
                continue
            line_records, line_failures = parse_v2_object(payload, config, source=source)
            records.extend(line_records)
            failures.extend(line_failures)
        return records, failures


def load_v2_file(path: str | Path, config: Config) -> tuple[list[dict[str, Any]], list[Failure]]:
    path = Path(path)
    return parse_v2_text(path.read_text(encoding="utf-8"), config, source=str(path))


def v2_to_v1(
    records: Iterable[dict[str, Any]], config: Config
) -> tuple[list[str], dict[str, Any], list[Failure]]:
    normalized_records: list[dict[str, Any]] = []
    sidecar_records: dict[str, Any] = {}
    failures: list[Failure] = []
    for original in records:
        record = dict(original)
        if "selection_code" not in record:
            name = record.get("selection_result")
            if name not in NAME_TO_CODE:
                failures.append(
                    Failure(
                        E_SELECT_CODE_UNKNOWN,
                        f"unknown selection_result {name!r}",
                        source="v2",
                        request_id=record.get("request_id"),
                        candidate_id=record.get("candidate_id"),
                        business_code=E_SELECT_CODE_UNKNOWN,
                    )
                )
                continue
            record["selection_code"] = NAME_TO_CODE[name]
        if "score_permille" not in record:
            score_decimal = Decimal(str(record["score"]))
            if score_decimal <= 1:
                mille = (score_decimal * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            else:
                mille = score_decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            record["score_permille"] = int(mille)
        if "score" in record and record.get("score_permille") is not None:
            score_decimal = Decimal(str(record["score"]))
            exact_mille = score_decimal * 1000 if score_decimal <= 1 else score_decimal
            record["truncation_residual"] = float(
                exact_mille - Decimal(int(record["score_permille"]))
            )
        else:
            record.setdefault("truncation_residual", 0.0)
        record.setdefault("seq", 0)
        normalized_records.append(record)
        key = f"{record['request_id']}:{record['candidate_id']}"
        dropped_keys = [
            "score",
            "selection_result",
            "sampling_scheme",
            "sampling_weight",
            "precision_label",
            "truncation_residual",
            "boundary",
            "rank",
            "top_k",
            "channel",
            "aux_feature",
            "shuffle_position",
            "fixed_score",
            "dequant_score",
        ]
        sidecar_records[key] = {
            "v2_fields": {key_name: record[key_name] for key_name in dropped_keys if key_name in record},
            "candidate_set_fingerprint": fingerprint(
                [record.get("request_id"), record.get("candidate_id")], 16
            ),
        }
    lines, line_failures = serialize_v1(normalized_records, config)
    failures.extend(line_failures)
    sidecar = {
        "schema": "v1-sidecar-v1",
        "config_hash": config.config_hash,
        "records": sidecar_records,
    }
    return lines, sidecar, failures


def serialization_cache_key(
    version: str,
    records: list[dict[str, Any]],
    config: Config,
    *,
    channel_weights_fingerprint: str | None = None,
) -> str:
    candidate_set = sorted(
        (str(record.get("request_id")), int(record["candidate_id"]))
        for record in records
        if record.get("candidate_id") is not None
    )
    return derive_cache_key(
        f"serialization_{version}",
        candidate_set=candidate_set,
        upstream_content=fingerprint(records, 20),
        channel_weights=channel_weights_fingerprint or config.channel_weights_fingerprint,
        sampling=config.sampling_fingerprint,
        version=config.derive_version_fingerprint(version),
    )


def write_text(path: str | Path, content: str | Iterable[str]) -> None:
    if isinstance(content, str):
        text = content if content.endswith("\n") else content + "\n"
    else:
        text = "\n".join(content) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
