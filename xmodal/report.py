"""确定性 JSON 验证报告组装。"""

import statistics

from .errors import ErrorCode
from .hashing import digest_json


def _score_distribution(scores):
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "stdev": None, "histogram": []}
    values = [float(v) for _, v in sorted(scores.items())]
    bins = 5
    lo, hi = min(values), max(values)
    histogram = [0] * bins
    if hi > lo:
        for value in values:
            idx = min(bins - 1, int((value - lo) / (hi - lo) * bins))
            histogram[idx] += 1
    else:
        histogram[-1] = len(values)
    return {
        "count": len(values),
        "min": round(lo, 9),
        "max": round(hi, 9),
        "mean": round(statistics.fmean(values), 9),
        "stdev": round(statistics.pstdev(values), 9),
        "histogram": histogram,
    }


def _cross_format_summary(deviations, sidecars):
    if not deviations and not sidecars:
        return {"samples": 0, "image_max_abs": None, "text_max_abs": None,
                "residuals": []}
    image_values = [d["image_max_abs"] for d in deviations]
    text_values = [d["text_max_abs"] for d in deviations]
    residuals = []
    for sample in sorted(sidecars):
        sidecar = sidecars[sample]
        residuals.append({
            "sample": sample,
            "direction": sidecar.get("direction"),
            "precision_labels": sidecar.get("precision_labels", {}),
            "truncation_residual": sidecar.get("truncation_residual", {}),
            "lossy_keys": sidecar.get("lossy_keys", []),
        })
    return {
        "samples": len(set(d["sample"] for d in deviations)
                       | set(sidecars)),
        "image_max_abs": round(max(image_values), 12) if image_values
        else 0.0,
        "text_max_abs": round(max(text_values), 12) if text_values
        else 0.0,
        "residuals": residuals,
    }


def _failure_summary(failures):
    ordered = sorted(failures, key=lambda f: (f["sample"], f["stage"],
                                              f["code"]))
    by_code = {}
    for failure in ordered:
        by_code.setdefault(failure["code"], 0)
        by_code[failure["code"]] += 1
    return {"ordered": ordered, "count": len(ordered), "by_code": by_code}


def build_report(**kwargs):
    failures = _failure_summary(kwargs["failures"])
    injection_index = {item["id"]: item for item in kwargs["injections"]}
    hits = sorted(kwargs["injection_hits"],
                  key=lambda h: (h["id"], h["stage"], h.get("sample") or ""))
    for hit in hits:
        source = injection_index.get(hit["id"], {})
        hit["configured_type"] = source.get("type")
        hit["expected_code"] = source.get(
            "expected_code") or _expected_code(source.get("type"))
        hit["hit"] = bool(hit.get("code")) or hit["type"] == "stage_delay"
    plan = kwargs["plan"]
    invalidation = {
        stage: sorted(plan[stage]["recompute"]) for stage in plan
    }
    reuse = {stage: sorted(plan[stage]["reuse"]) for stage in plan}
    retrieval = []
    for result in kwargs["retrieval"]:
        retrieval.append({
            "query": result["query"],
            "top_score": result["top_score"],
            "threshold": result["threshold"],
            "code": result["code"],
            "candidates": result["candidates"],
            "audit_candidates": result["all_candidates"],
        })
    manifest = kwargs["manifest"]
    return {
        "report_version": 1,
        "run_id": manifest.get("run_id"),
        "format_version": kwargs["format_version"],
        "snapshot": {
            "global_version": manifest.get("global_version", 0),
            "config_hash": kwargs["config_hash"],
            "spec_hash": kwargs["spec_hash"],
            "stage_versions": manifest.get("stage_versions", {}),
            "completed_samples": manifest.get("completed_samples", []),
            "recovery_point": kwargs["recovery"],
        },
        "interrupted": kwargs["interrupted"],
        "shutdown": kwargs["shutdown_info"],
        "retrieval_threshold": kwargs["threshold"],
        "retrieval": retrieval,
        "alignment": {
            "scores": {k: round(float(v), 9)
                       for k, v in sorted(kwargs["scores"].items())},
            "components": {k: kwargs["score_details"][k]
                           for k in sorted(kwargs["score_details"])},
            "distribution": _score_distribution(kwargs["scores"]),
        },
        "cross_format": _cross_format_summary(
            kwargs["deviations"], kwargs["sidecars"]),
        "roundtrip_fingerprints": {
            sample: kwargs["roundtrip"][sample]["pack_fingerprint"]
            for sample in sorted(kwargs["roundtrip"])
        },
        "invalidation": invalidation,
        "reuse": reuse,
        "reuse_stats": kwargs["reuse_stats"],
        "injections": hits,
        "failures": failures,
        "content_hash": digest_json({
            "retrieval": retrieval,
            "scores": kwargs["scores"],
            "failures": failures["ordered"],
            "cross_format": _cross_format_summary(
                kwargs["deviations"], kwargs["sidecars"]),
            "roundtrip": {
                sample: kwargs["roundtrip"][sample]["pack_fingerprint"]
                for sample in sorted(kwargs["roundtrip"])
            },
        }, prefix="report-content"),
    }


def _expected_code(inject_type):
    mapping = {
        "truncate_png": ErrorCode.SAMPLE_INVALID,
        "missing_text_field": ErrorCode.SAMPLE_INVALID,
        "offset_table_oob": ErrorCode.SAMPLE_INVALID,
        "non_finite_float": ErrorCode.SAMPLE_INVALID,
        "threshold_boundary": ErrorCode.RETRIEVAL_LOW_SCORE,
        "unknown_format_version": ErrorCode.SAMPLE_INVALID,
        "stage_delay": None,
    }
    return mapping.get(inject_type)
