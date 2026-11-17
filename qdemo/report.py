"""Experiment record + JSON report builders (fully deterministic content)."""
from __future__ import annotations

from .common import OK, fingerprint, to_jsonable

RECORD_VERSION = 1


def build_record(cfg: dict, weights, calib, results: dict) -> dict:
    return to_jsonable({
        "record_version": RECORD_VERSION,
        "config": cfg,
        "config_fingerprint": fingerprint(cfg),
        "weights_fingerprint": weights.fingerprint,
        "calib_fingerprint": calib.fingerprint,
        "rounding": {"scale_sig_digits": cfg["rounding"]["scale_sig_digits"]},
        "layers": results["layers"],
        "distill": {
            "temperature": cfg["distill"]["temperature"],
            "soft_label_weight": cfg["distill"]["soft_label_weight"],
            "stable_layers": results["summary"]["distill_stable_layers"],
            "unstable_layers": results["summary"]["distill_unstable_layers"],
        },
        "failures": results["failures"],
    })


def build_report(cfg: dict, weights, calib, results: dict) -> dict:
    record = build_record(cfg, weights, calib, results)
    return {
        "business_code": OK if not results["failures"] else "OK_WITH_FAILURES",
        "experiment": cfg.get("experiment", "qdemo"),
        "config_fingerprint": record["config_fingerprint"],
        "weights_fingerprint": record["weights_fingerprint"],
        "calib_fingerprint": record["calib_fingerprint"],
        "rounding": record["rounding"],
        "summary": results["summary"],
        "invalidation": results["invalidation"],
        "layers": results["layers"],
        "distill": record["distill"],
        "failures": results["failures"],
    }
