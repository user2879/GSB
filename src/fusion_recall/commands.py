from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import (
    ALLOWED_FRAC_BITS,
    ALLOWED_QUANT_TIERS,
    config_fingerprint,
    fixed_point_fingerprint,
    full_config_fingerprint,
    is_positive_int,
    language_weights_fingerprint,
    rational_pair,
)
from .errors import E_INPUT_INVALID
from .loading import validate_document
from .vocab import Vocabulary


DOCUMENT_OPS = {"add_document", "replace_document", "upsert_document"}
CONFIG_OPS = {
    "set_language_weight",
    "set_field_weight",
    "set_saturation_k",
    "set_quant_tier",
    "set_frac_bits",
    "set_top_k",
}


def _fail(line_number: int | None, message: str) -> dict[str, Any]:
    return {
        "code": E_INPUT_INVALID,
        "source": "commands",
        "line": line_number,
        "item_id": None,
        "message": message,
    }


def parse_commands(
    config: dict[str, Any],
    vocab: Vocabulary,
    *,
    path: str | Path | None = None,
    lines: Iterable[str] | None = None,
) -> dict[str, Any]:
    if lines is None:
        if path is None:
            raise ValueError("path or lines required")
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line_number, line in enumerate(list(lines), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(_fail(line_number, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(raw, dict):
            failures.append(_fail(line_number, "command must be an object"))
            continue
        op = raw.get("op")
        if op not in DOCUMENT_OPS | CONFIG_OPS | {"delete_document"}:
            failures.append(_fail(line_number, f"unknown or unsupported op: {op!r}"))
            continue

        command = dict(raw)
        command["line"] = line_number
        if op in DOCUMENT_OPS:
            document, failure = validate_document(
                raw.get("document"), config, vocab, source="commands", line_number=line_number
            )
            if failure is not None:
                failures.append(failure.to_dict())
                continue
            command["document"] = document
        elif op == "delete_document":
            if not isinstance(raw.get("doc_id"), str) or not raw["doc_id"]:
                failures.append(_fail(line_number, "delete_document requires non-empty doc_id"))
                continue
        elif op == "set_language_weight":
            view = raw.get("view")
            if view not in config["view_names"]:
                failures.append(_fail(line_number, f"unknown language view: {view!r}"))
                continue
            try:
                rational_pair(raw.get("weight"))
            except Exception as exc:  # noqa: BLE001 - validation boundary
                failures.append(_fail(line_number, f"invalid language weight: {exc}"))
                continue
        elif op == "set_field_weight":
            field = raw.get("field")
            if field not in config["field_weights"]:
                failures.append(_fail(line_number, f"unknown field: {field!r}"))
                continue
            if not is_positive_int(raw.get("weight")):
                failures.append(_fail(line_number, "field weight must be a positive integer"))
                continue
        elif op == "set_saturation_k":
            if not is_positive_int(raw.get("value")):
                failures.append(_fail(line_number, "saturation_k must be a positive integer"))
                continue
        elif op == "set_top_k":
            if not is_positive_int(raw.get("value")):
                failures.append(_fail(line_number, "top_k must be a positive integer"))
                continue
        elif op == "set_quant_tier":
            if raw.get("value") not in ALLOWED_QUANT_TIERS:
                failures.append(_fail(line_number, f"quant_tier must be one of {ALLOWED_QUANT_TIERS}"))
                continue
        elif op == "set_frac_bits":
            if raw.get("value") not in ALLOWED_FRAC_BITS:
                failures.append(_fail(line_number, f"frac_bits must be one of {ALLOWED_FRAC_BITS}"))
                continue
        valid.append(command)
    return {"commands": valid, "failures": failures}


def apply_config_command(config: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    op = command["op"]
    if op == "set_language_weight":
        numerator, denominator = rational_pair(command["weight"])
        view = command["view"]
        config["language_weights"][view] = f"{numerator}/{denominator}"
        config["language_weight_rational"][view] = [numerator, denominator]
        config["language_weights_fingerprint"] = language_weights_fingerprint(config)
    elif op == "set_field_weight":
        config["field_weights"][command["field"]] = int(command["weight"])
        config["config_fingerprint"] = config_fingerprint(config)
    elif op == "set_saturation_k":
        config["saturation_k"] = int(command["value"])
        config["config_fingerprint"] = config_fingerprint(config)
    elif op == "set_top_k":
        config["top_k"] = int(command["value"])
        config["config_fingerprint"] = config_fingerprint(config)
    elif op == "set_quant_tier":
        config["fixed_point"]["quant_tier"] = command["value"]
        config["fixed_point_fingerprint"] = fixed_point_fingerprint(config)
    elif op == "set_frac_bits":
        config["fixed_point"]["frac_bits"] = int(command["value"])
        config["fixed_point_fingerprint"] = fixed_point_fingerprint(config)
    else:
        return config
    config["full_config_fingerprint"] = full_config_fingerprint(config)
    return config
