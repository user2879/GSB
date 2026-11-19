"""Diagnostic report writing, dependency graph export and run comparison."""

from __future__ import annotations

import json
import os
from typing import Any

from .hashing import sha256_hex


def write_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2,
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(payload)
    os.replace(tmp, path)


def export_graph_dot(graph: dict, path: str) -> None:
    lines = ["digraph dbench_pipeline {", "  rankdir=LR;"]
    for stage, info in sorted(graph.items()):
        shape = "box" if info.get("kind") == "denoise" else "ellipse"
        lines.append(f'  "{stage}" [shape={shape}];')
    for stage, info in sorted(graph.items()):
        for dep in info["deps"]:
            lines.append(f'  "{dep}" -> "{stage}";')
    lines.append("}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def compare_runs(report_a: dict, report_b: dict, label_a: str,
                 label_b: str) -> dict[str, Any]:
    """Compare two runs stage-by-stage; locate the first divergent stage."""
    stages_a = {s["stage"]: s for s in report_a.get("stages", [])}
    stages_b = {s["stage"]: s for s in report_b.get("stages", [])}
    all_stages = sorted(set(stages_a) | set(stages_b),
                        key=lambda s: report_a.get("dependency_graph", {})
                        .get(s, {}).get("sort_key", [9, 9]))
    mismatches: list[dict] = []
    first_mismatch = None
    for stage in all_stages:
        sa, sb = stages_a.get(stage), stages_b.get(stage)
        fp_a = sa and sa.get("content_fingerprint")
        fp_b = sb and sb.get("content_fingerprint")
        if fp_a != fp_b:
            mismatches.append({
                "stage": stage,
                label_a: fp_a,
                label_b: fp_b,
            })
            if first_mismatch is None:
                first_mismatch = stage
    pixel_a = _pixel_fp(report_a)
    pixel_b = _pixel_fp(report_b)
    return {
        "schema": "dbench.comparison.v1",
        "labels": [label_a, label_b],
        "equivalent": not mismatches and pixel_a == pixel_b,
        "first_mismatch_stage": first_mismatch,
        "stage_mismatches": mismatches,
        "pixel_fingerprints": {label_a: pixel_a, label_b: pixel_b},
        "pixels_equal": pixel_a == pixel_b,
    }


def _pixel_fp(report: dict) -> str | None:
    for stage in report.get("stages", []):
        if stage["stage"] == "decode":
            summary = stage.get("array_summary") or {}
            return summary.get("content_fingerprint")
    return None


def file_sha256(path: str) -> str:
    digest = sha256_hex(b"")
    with open(path, "rb") as handle:
        import hashlib

        digest = hashlib.sha256(handle.read()).hexdigest()
    return digest
