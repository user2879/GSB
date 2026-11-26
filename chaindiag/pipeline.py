from __future__ import annotations

import copy
from typing import Any

from .canonical import canonical_json
from .engine import ChainEngine
from .ledger import CacheStore
from .model import normalize_boundaries, normalize_tokens, validate_chain, validate_config


TOKEN_CONCLUSION_FIELDS = (
    "token_seq",
    "token_uid",
    "token_value",
    "segment_index",
    "segment_count",
    "status",
    "vector",
    "boundary_token",
    "cache_keys",
    "stage_residuals",
    "fault_ids",
)


def injections_list(value: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, dict):
        return list(value.get("injections", []))
    return list(value)


def engine_from_data(*, config: dict[str, Any], chain: Any, tokens: Any, segments: Any,
                     injections: Any = None, store: CacheStore | None = None,
                     serial: bool = False, exchange_version: str | None = None,
                     overrides: dict[str, Any] | None = None) -> ChainEngine:
    cfg = validate_config(copy.deepcopy(config))
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
        cfg = validate_config(cfg)
    operators, _meta = validate_chain(copy.deepcopy(chain))
    token_list = normalize_tokens(copy.deepcopy(tokens), cfg.get("vocab_size"))
    if isinstance(segments, list):
        segment_data = {"boundaries": segments}
    else:
        segment_data = copy.deepcopy(segments)
    boundaries = normalize_boundaries(segment_data, len(token_list))
    version = exchange_version or cfg.get("exchange_version", "e2")
    return ChainEngine(
        cfg,
        operators,
        token_list,
        boundaries,
        injections=injections_list(injections),
        exchange_version=version,
        store=store or CacheStore(),
        serial=serial,
    )


def state_to_engine(state: dict[str, Any], *, store: CacheStore | None = None,
                    serial: bool = False, overrides: dict[str, Any] | None = None) -> ChainEngine:
    chain = state.get("chain_e2") or state.get("chain") or {"operators": state.get("operators", [])}
    segments = state.get("segments")
    if segments is None and "boundaries" in state:
        segments = {"boundaries": state["boundaries"]}
    return engine_from_data(
        config=state["config"],
        chain=chain,
        tokens=state.get("tokens_data", state.get("tokens")),
        segments=segments,
        injections=state.get("injections"),
        store=store,
        serial=serial,
        exchange_version=state.get("exchange_version"),
        overrides=overrides,
    )


def token_conclusions(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: token.get(key) for key in TOKEN_CONCLUSION_FIELDS} for token in report["token_results"]]


def ledger_order(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": entry["ordinal"],
            "op_seq": entry["op_seq"],
            "kind": entry["kind"],
            "token_seq": entry["token_seq"],
            "token_uid": entry["token_uid"],
            "cache_key": entry["cache_key"],
        }
        for entry in report["reuse_ledger"]
    ]


def scheduler_signature(report: dict[str, Any]) -> dict[str, Any]:
    scheduler = report.get("scheduler", {})
    return {
        "status": scheduler.get("status"),
        "point_budget": scheduler.get("point_budget"),
        "operator_quota": scheduler.get("operator_quota"),
        "consumed_budget": scheduler.get("consumed_budget"),
        "remaining_budget": scheduler.get("remaining_budget"),
        "consumed_by_op": scheduler.get("consumed_by_op"),
        "pending": scheduler.get("pending"),
        "quota_aborted": [{k: v for k, v in item.items() if k != "reason"} for item in scheduler.get("quota_aborted", [])],
        "diagnostics": sorted(
            scheduler.get("diagnostics", []),
            key=lambda item: (item.get("code", ""), item.get("op_seq", -1), item.get("task_id", "")),
        ),
        "execution_order": [
            {k: v for k, v in item.items() if k != "ordinal"}
            for item in scheduler.get("execution_order", [])
        ],
    }


def conclusion_signature(report: dict[str, Any], *, include_scheduler: bool = False) -> dict[str, Any]:
    signature = {
        "status": report.get("status"),
        "config_hash": report.get("config_hash"),
        "chain_fingerprint": report.get("chain_fingerprint"),
        "exchange_version": report.get("exchange_version"),
        "token_results": token_conclusions(report),
        "cache_records": report.get("cache_records"),
        "ledger_order": ledger_order(report),
        "fault_report": report.get("fault_report"),
        "stage_keys": [
            {"op_seq": stage["op_seq"], "kind": stage["kind"], "stage_key": stage["stage_key"],
             "residual_sum": stage["residual_sum"], "rounding_steps_recorded": stage["rounding_steps_recorded"]}
            for stage in report.get("stage_summaries", [])
        ],
    }
    if include_scheduler:
        signature["scheduler"] = scheduler_signature(report)
    return signature


def first_diff(a: Any, b: Any, path: str = "$") -> dict[str, Any] | None:
    if type(a) is not type(b):
        # ints and floats are compared by value below; bool type mismatch still caught by repr/type.
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or isinstance(a, bool) or isinstance(b, bool):
            return {"path": path, "left_type": type(a).__name__, "right_type": type(b).__name__, "left": a, "right": b}
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        for key in keys:
            if key not in a or key not in b:
                return {"path": f"{path}.{key}", "issue": "missing_key", "left": key in a, "right": key in b}
            diff = first_diff(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return {"path": path, "issue": "length", "left": len(a), "right": len(b)}
        for index, (left, right) in enumerate(zip(a, b)):
            diff = first_diff(left, right, f"{path}[{index}]")
            if diff:
                return diff
        return None
    if a != b:
        return {"path": path, "left": a, "right": b}
    return None


def compare_reports(left: dict[str, Any], right: dict[str, Any], *, scheduler: bool = False) -> dict[str, Any]:
    left_sig = conclusion_signature(left, include_scheduler=scheduler)
    right_sig = conclusion_signature(right, include_scheduler=scheduler)
    mismatch = first_diff(left_sig, right_sig)
    return {
        "equal": mismatch is None,
        "first_mismatch": mismatch,
        "left_canonical_sha256": __import__("hashlib").sha256(canonical_json(left_sig).encode()).hexdigest(),
        "right_canonical_sha256": __import__("hashlib").sha256(canonical_json(right_sig).encode()).hexdigest(),
    }
