from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from .loader import (
    InputBundle,
    load_lexicon,
    validate_sentence_payload,
)
from .model import Failure, ScoreParams


def _extract_items(raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    items = raw.get("injections", raw.get("items", []))
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _lexicon_injection(item: dict[str, Any], failures: list[Failure]) -> str | None:
    line = item.get("line")
    if line is None:
        source = item.get("source", "inject")
        target = item.get("target", "bad")
        frequency = item.get("frequency", 0)
        confidence = item.get("alignment_confidence", 1.5)
        line = f"{source} {target} {frequency} {confidence}"
    with tempfile.TemporaryDirectory(prefix="mtcomp-inject-") as directory:
        path = Path(directory) / "lexicon.tsv"
        path.write_text(str(line) + "\n", encoding="utf-8")
        local: list[Failure] = []
        load_lexicon(path, local)
    failures.extend(local)
    return local[0].code if local else None


def _nonfinite_injection(item: dict[str, Any], base: InputBundle) -> tuple[str | None, str]:
    if base.state is None or base.state.params is None:
        return "E_CONFIG_INVALID", "base parameters unavailable"
    field_name = str(item.get("field", "alignment_weight"))
    raw_value = item.get("value", "NaN")
    if isinstance(raw_value, str) and raw_value.lower() in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
    }:
        value = float(raw_value)
    else:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = math.nan
    mapping = base.state.params.to_dict(include_enums=True)
    mapping[field_name] = value
    try:
        params = ScoreParams.from_mapping(mapping)
        errors = params.validate()
    except (KeyError, TypeError, ValueError) as exc:
        errors = [str(exc)]
    code = "E_CONFIG_INVALID" if errors else None
    return code, "; ".join(errors) if errors else "unexpectedly accepted"


def _overlong_injection(item: dict[str, Any], base: InputBundle) -> tuple[str | None, str]:
    if base.state is None or base.state.config is None:
        return "E_SENT_INVALID", "base config unavailable"
    config = base.state.config
    token = str(item.get("token", config.vocabulary[0] if config.vocabulary else "x"))
    count = int(item.get("token_count", config.max_sentence_tokens + 1))
    payload = {
        "id": item.get("sentence_id", item.get("id", "INJECT-LONG")),
        "batch": item.get("batch", 999),
        "tokens": [token] * count,
    }
    _sentence, error, _details = validate_sentence_payload(payload, config)
    code = "E_SENT_INVALID" if error else None
    return code, error or "unexpectedly accepted"


def run_injection_diagnostics(bundle: InputBundle) -> dict[str, Any]:
    items = _extract_items(bundle.injection_data)
    reports: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        item_id = str(item.get("id", f"INJ-{index:03d}"))
        kind = str(item.get("type", item.get("kind", "")))
        failure_code: str | None = None
        message = ""
        if kind == "lexicon_invalid":
            local_failures: list[Failure] = []
            failure_code = _lexicon_injection(item, local_failures)
            if local_failures:
                message = local_failures[0].message
            else:
                message = "unexpectedly accepted"
        elif kind == "nonfinite_weight":
            failure_code, message = _nonfinite_injection(item, bundle)
        elif kind == "overlong_sentence":
            failure_code, message = _overlong_injection(item, bundle)
        else:
            failure_code = "E_INJECTION_UNKNOWN"
            message = f"unknown injection type: {kind}"

        control_ok = False
        if bundle.state is not None and bundle.state.config is not None:
            if bundle.state.sentences:
                first = next(iter(bundle.state.sentences.values()))
                control_ok = all(
                    token in set(bundle.state.config.vocabulary)
                    for token in first.tokens
                )
            else:
                control_ok = True
        accepted_injection = failure_code in {None, "E_INJECTION_UNKNOWN"}
        reports.append(
            {
                "item_number": index,
                "item_id": item_id,
                "kind": kind,
                "isolated": not accepted_injection,
                "accepted": accepted_injection,
                "code": failure_code,
                "message": message,
                "control_state_valid": control_ok,
                "effect_on_engine_state": "none",
            }
        )
    return {
        "schema_version": 1,
        "injection_count": len(reports),
        "all_isolated": all(item["isolated"] for item in reports),
        "control_state_valid": all(item["control_state_valid"] for item in reports),
        "injections": reports,
    }
