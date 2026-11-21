from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .errors import ConfigAbort, Failure
from .loading import load_workspace, validate_config
from .simulation import apply_batch

EXPECTED = {
    "non_finite_model_score": {"E_LOG_INVALID"},
    "unknown_group": {"E_LOG_INVALID"},
    "missing_spec_version": {"E_CONFIG_INVALID"},
    "invalid_attribution_order": {"E_CONFIG_INVALID"},
    "zero_tolerance": {"E_CONFIG_INVALID"},
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare_state_before_batch(initial_state: Any, target_batch: int) -> Any:
    state = initial_state
    for batch in sorted(state.plan.get("batches", []), key=lambda item: item.get("batch", 0)):
        if int(batch.get("batch", 0)) >= target_batch:
            break
        state, _ = apply_batch(state, batch)
    return state


def _invalid_event(state: Any, injection: dict[str, Any]) -> dict[str, Any]:
    sample = state.sorted_exposures()[0]
    event = {
        "request_id": injection.get("request_id", f"inj-{injection['injection_id']}"),
        "experiment_group": sample.experiment_group,
        "user_id": sample.user_id,
        "item_id": "inj-item-001",
        "rank": 1,
        "model_score": 0.5,
        "clicked": 0,
        "converted": 0,
        "event_ms": sample.event_ms,
    }
    if injection["type"] == "non_finite_model_score":
        event["model_score"] = float("nan")
    elif injection["type"] == "unknown_group":
        event["experiment_group"] = "__unknown_group__"
    return event


def _run_one(initial_state: Any, injection: dict[str, Any]) -> dict[str, Any]:
    state = _prepare_state_before_batch(initial_state, int(injection["batch"]))
    injection_type = injection["type"]
    observed_failures: list[Failure] = []
    evidence: dict[str, Any] = {
        "batch": injection["batch"],
        "type": injection_type,
        "reloaded_from_disk": True,
    }
    with tempfile.TemporaryDirectory(prefix=f"driftbench-inject-{injection['injection_id']}-") as tmp:
        root = Path(tmp)
        rows = [event.to_dict() for event in state.sorted_exposures()]
        raw_config = state.config.to_dict()
        if injection_type in {"non_finite_model_score", "unknown_group"}:
            bad_event = _invalid_event(state, injection)
            rows.append(bad_event)
            evidence["injected_request_id"] = bad_event["request_id"]
            evidence["injected_field"] = (
                "model_score" if injection_type == "non_finite_model_score" else "experiment_group"
            )
        elif injection_type == "missing_spec_version":
            raw_config["active_spec_version"] = injection.get("value", "v-missing")
            evidence["field_path"] = "active_spec_version"
        elif injection_type == "invalid_attribution_order":
            raw_config["attribution_order"] = injection.get("value", ["bogus"])
            evidence["field_path"] = "attribution_order"
        elif injection_type == "zero_tolerance":
            raw_config["drift_tolerance"] = injection.get("value", "0")
            evidence["field_path"] = "drift_tolerance"
        else:
            observed_failures.append(
                Failure("E_INJECTION_MISS", f"unknown injection type {injection_type!r}")
            )

        _write_jsonl(root / "exposures.jsonl", rows)
        _write_jsonl(root / "specs.jsonl", [spec.to_dict() for spec in state.specs.values()])
        (root / "config.json").write_text(
            json.dumps(raw_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "user_properties.json").write_text(
            json.dumps({"users": state.properties}, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            loaded = load_workspace(root)
            observed_failures.extend(loaded.failures)
        except ConfigAbort as exc:
            observed_failures.extend(exc.failures)

    expected = EXPECTED.get(injection_type, {"E_INJECTION_MISS"})
    observed_codes = sorted({failure.code for failure in observed_failures})
    matched = [
        failure
        for failure in observed_failures
        if failure.code in expected
        and (
            "injected_request_id" not in evidence
            or failure.request_id == evidence["injected_request_id"]
        )
        and (
            "field_path" not in evidence
            or failure.field_path == evidence["field_path"]
        )
    ]
    hit = bool(matched)
    if not hit:
        observed_failures.append(
            Failure(
                "E_INJECTION_MISS",
                f"injection {injection['injection_id']} did not produce expected code/path",
                injection_id=injection["injection_id"],
                context={"expected": sorted(expected), "observed": observed_codes},
            )
        )
    return {
        "injection_id": injection["injection_id"],
        "batch": injection["batch"],
        "type": injection_type,
        "hit": hit,
        "expected_codes": sorted(expected),
        "observed_codes": observed_codes,
        "evidence": evidence,
        "matched_failure": matched[0].to_dict() if matched else None,
        "failures": [failure.to_dict() for failure in observed_failures],
    }


def run_injection_suite(initial_state: Any, injections: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_run_one(initial_state, injection) for injection in injections]
    return {
        "schema_version": 1,
        "report_type": "fault_injection",
        "all_hit": all(row["hit"] for row in rows),
        "hit_count": sum(1 for row in rows if row["hit"]),
        "injection_count": len(rows),
        "injections": rows,
    }


def load_injections(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("injections") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("injections file must contain an array or an object with 'injections'")
    return items
