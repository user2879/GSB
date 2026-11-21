"""Isolated fault-injection runner with unique-id diagnostic reporting."""
from __future__ import annotations

import copy
import os

from .editing import EditApplier, EditConflict
from .errors import (
    E_CONFIG_INVALID,
    E_INJECTION_ILLEGAL_TOKEN,
    E_INJECTION_MIGRATION_CONFLICT,
    E_THRESHOLD_NOT_FINITE,
)
from .io_utils import write_json
from .loader import load_samples, load_vocabulary
from .models import ConfigError, HierarchyConfig


def _check_illegal_token(item, vocab):
    tokens = item.get("tokens", [])
    illegal = [token for token in tokens if token not in vocab.tokens]
    if illegal:
        return {
            "code": E_INJECTION_ILLEGAL_TOKEN,
            "message": f"illegal token(s) rejected at validation boundary: {illegal}",
            "illegal_tokens": illegal,
        }
    return None


def _check_migration_conflict(item, raw_config):
    applier = EditApplier(copy.deepcopy(raw_config), vocab=None)
    try:
        applier.vocab = None  # migration path does not need vocab
        applier.apply_migrations({"intent_migrations": [item]})
    except EditConflict as exc:
        return {"code": E_INJECTION_MIGRATION_CONFLICT,
                "message": str(exc), "detail": exc.code}
    return None


def _check_threshold(item, vocab, raw_config):
    value = item.get("value")
    from .fixedpoint import is_finite_number
    from decimal import Decimal
    if not is_finite_number(value):
        return {"code": E_THRESHOLD_NOT_FINITE,
                "message": f"non-finite threshold rejected: {value!r}"}
    if not (Decimal("0") <= Decimal(str(value)) <= Decimal("1")):
        return {"code": E_CONFIG_INVALID,
                "message": f"threshold outside [0,1] rejected: {value}"}
    probe = copy.deepcopy(raw_config)
    probe["thresholds"][item["level"]] = str(value)
    try:
        HierarchyConfig.from_dict(probe, vocab)
    except ConfigError as exc:
        return {"code": E_CONFIG_INVALID,
                "message": "threshold probe failed validation",
                "detail": exc.errors}
    return None


def run_injections(vocab_path: str, config_path: str, injection_manifest: dict,
                   report_path: str, samples_path: str | None = None) -> dict:
    """Execute every injection in an isolated side context; main state untouched."""
    vocab = load_vocabulary(vocab_path)
    import json
    with open(config_path, encoding="utf-8") as handle:
        raw_config = json.load(handle)
    base_config = HierarchyConfig.from_dict(raw_config, vocab)

    results = []
    seen_ids = set()
    for index, item in enumerate(injection_manifest.get("injections", [])):
        injection_id = item.get("id") or f"INJ-{index + 1:04d}"
        if injection_id in seen_ids:
            raise AssertionError(f"duplicate injection id: {injection_id}")
        seen_ids.add(injection_id)
        kind = item["kind"]
        detail = None
        if kind == "illegal_token":
            detail = _check_illegal_token(item, vocab)
        elif kind == "migration_conflict":
            detail = _check_migration_conflict(item, raw_config)
        elif kind == "non_finite_threshold":
            detail = _check_threshold(item, vocab, raw_config)
        else:
            detail = {"code": E_CONFIG_INVALID,
                      "message": f"unknown injection kind: {kind}"}
        results.append({
            "id": injection_id,
            "kind": kind,
            "hit": detail is not None,
            "code": detail["code"] if detail else "NO_ERROR_HIT",
            "detail": detail or {"message": "injection was unexpectedly accepted"},
        })

    failure_codes = {}
    if samples_path:
        _, failures = load_samples(samples_path, vocab, base_config)
        for failure in failures:
            failure_codes.setdefault(failure["code"], 0)
            failure_codes[failure["code"]] += 1

    report = {
        "manifest_id": injection_manifest.get("manifest_id", "injection-manifest"),
        "total": len(results),
        "all_hit": bool(results) and all(row["hit"] for row in results),
        "unique_ids": sorted(seen_ids),
        "results": results,
        "sample_validation_failures": failure_codes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    write_json(report_path, report)
    return report
