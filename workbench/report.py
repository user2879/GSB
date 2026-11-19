"""Reproducible JSON reports and visualization diagnostics."""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .config import STAGE_NAMES
from .utils import atomic_write_json, digest_obj, png_bytes


def build_report(report: dict, serial_report: dict | None = None) -> dict:
    """Assemble the canonical, JSON-serializable diagnostic report."""
    comparison = None
    if serial_report is not None:
        comparison = compare_runs(report, serial_report)
    failure_codes = {}
    for failure in report["failures"]:
        failure_codes[failure["code"]] = failure_codes.get(failure["code"], 0) + 1
    pixel_hashes = sorted(
        (
            sample["final_pixel_hash"]
            for sample in report["samples"]
            if sample["status"] == "ok"
        )
    )
    return {
        "schema": "workbench-report-v1",
        "run_id": report["run_id"],
        "tier": report["tier"],
        "run_mode": report["run_mode"],
        "workers": report["workers"],
        "interrupted": report["interrupted"],
        "config_fingerprint": report["config_fingerprint"],
        "final_manifest_version": report["final_version"],
        "strategy_decision": report["global_decision"],
        "stage_timing": {
            "deterministic_cost_units": report["stage_cost_units"],
            "wall_time_ms_total": round(sum(report["wall_time_ms"].values()), 3),
            "note": "wall_time_ms is diagnostic-only; cost units are the canonical timing field",
        },
        "reuse_audit": {
            "reused_per_stage": report["reuse_counts"],
            "recomputed_per_stage": report["recompute_counts"],
            "reused_samples": [
                {"sample": s["sample"], "stages": s["reused"]}
                for s in report["samples"]
                if s.get("reused")
            ],
            "recomputed_samples": [
                {"sample": s["sample"], "stages": s["recomputed"]}
                for s in report["samples"]
                if s.get("recomputed")
            ],
            "pruned_samples": report["pruned_samples"],
        },
        "concurrency_comparison": comparison,
        "failures": {
            "count": len(report["failures"]),
            "codes": failure_codes,
            "items": report["failures"],
        },
        "reproducibility": {
            "sample_order": [s["sample"] for s in report["samples"]],
            "completed_order": report["completed_samples"],
            "final_pixel_hashes": pixel_hashes,
            "report_fingerprint": digest_obj(
                [
                    report["config_fingerprint"],
                    report["tier"],
                    pixel_hashes,
                    report["reuse_counts"],
                    report["recompute_counts"],
                ]
            ),
        },
        "sample_results": [
            {
                "sample": s["sample"],
                "status": s["status"],
                "decision": s.get("decision"),
                "stage_summaries": s.get("stage_summaries"),
                "final_pixel_hash": s.get("final_pixel_hash"),
                "reused": s.get("reused", []),
                "recomputed": s.get("recomputed", []),
                "failure": s.get("failure"),
            }
            for s in report["samples"]
        ],
    }


def _comparable_blob(report: dict) -> dict:
    """Canonical execution-agnostic fields.

    Reuse statistics are deliberately excluded: a warm-cache verification run
    legitimately differs there, while pixels, decisions and summaries must not.
    """
    return {
        "samples": [
            {
                "sample": s["sample"],
                "status": s["status"],
                "decision": s.get("decision"),
                "stage_summaries": s.get("stage_summaries"),
                "final_pixel_hash": s.get("final_pixel_hash"),
            }
            for s in report["samples"]
        ],
        "completed": report["completed_samples"],
        "failures": report["failures"],
        "cost": report["stage_cost_units"],
    }


def compare_runs(parallel_report: dict, serial_report: dict) -> dict:
    """Field-by-field comparison of parallel vs serial canonical outputs."""
    parallel_blob = _comparable_blob(parallel_report)
    serial_blob = _comparable_blob(serial_report)
    mismatches = []
    if digest_obj(parallel_blob) != digest_obj(serial_blob):
        for key in parallel_blob:
            if digest_obj(parallel_blob[key]) != digest_obj(serial_blob[key]):
                mismatches.append(key)
    return {
        "parallel_run_id": parallel_report["run_id"],
        "serial_run_id": serial_report["run_id"],
        "field_identical": not mismatches,
        "mismatched_fields": mismatches,
        "compared_fields": sorted(parallel_blob.keys()),
    }


def render_strip(images: list, labels: list, cell_height: int = 96) -> np.ndarray:
    """Horizontal strip of source/final images for quick visual diagnosis."""
    from PIL import Image, ImageDraw

    if not images:
        return np.zeros((cell_height, cell_height, 3), dtype=np.uint8)
    label_band = 14
    strip_pil = Image.new(
        "RGB", (cell_height * len(images), cell_height + label_band), (255, 255, 255)
    )
    draw = ImageDraw.Draw(strip_pil)
    for index, (image, label) in enumerate(zip(images, labels)):
        x0 = index * cell_height
        tile = Image.fromarray(image, mode="RGB").resize(
            (cell_height, cell_height), Image.Resampling.NEAREST
        )
        strip_pil.paste(tile, (x0, label_band))
        draw.text((x0 + 2, 2), label[:12], fill=(0, 0, 0))
    return np.asarray(strip_pil, dtype=np.uint8)


def render_dashboard(report_doc: dict) -> np.ndarray:
    """Static PNG dashboard summarizing decision, reuse and failures."""
    from PIL import Image, ImageDraw

    width, height = 720, 360
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    selected = report_doc["strategy_decision"]["selected"]
    lines = [
        f"run {report_doc['run_id']} | tier {report_doc['tier']} | {report_doc['run_mode']}",
        f"decision: {selected['name']} score={selected['score']} est={selected['estimated_ms']}ms",
        f"manifest version: {report_doc['final_manifest_version']}",
        "reuse: " + str(report_doc["reuse_audit"]["reused_per_stage"]),
        "recompute: " + str(report_doc["reuse_audit"]["recomputed_per_stage"]),
        f"failures: {report_doc['failures']['count']} {report_doc['failures']['codes']}",
        f"interrupted: {report_doc['interrupted']}",
    ]
    comparison = report_doc.get("concurrency_comparison")
    if comparison:
        lines.append(
            "parallel==serial: " + str(comparison["field_identical"])
        )
    for index, line in enumerate(lines):
        draw.text((16, 16 + index * 28), line, fill=(20, 20, 20))

    # Bar chart of deterministic cost per stage.
    costs = report_doc["stage_timing"]["deterministic_cost_units"]
    max_cost = max(costs[stage] for stage in STAGE_NAMES) or 1.0
    chart_top, chart_left = 210, 40
    bar_width = 90
    for index, stage in enumerate(STAGE_NAMES):
        bar_height = int(120 * costs[stage] / max_cost)
        x0 = chart_left + index * (bar_width + 18)
        draw.rectangle(
            [x0, chart_top + 120 - bar_height, x0 + bar_width, chart_top + 120],
            fill=(70, 110, 200),
        )
        draw.text((x0, chart_top + 124), stage, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)


def write_run_artifacts(
    output_dir: str | Path,
    report_doc: dict,
    strip_image: np.ndarray | None,
    final_images: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    atomic_write_json(report_path, report_doc)

    dashboard_path = output_dir / "dashboard.png"
    dashboard_path.write_bytes(png_bytes(render_dashboard(report_doc)))

    strip_path = None
    if strip_image is not None:
        strip_path = output_dir / "sample_strip.png"
        strip_path.write_bytes(png_bytes(strip_image))

    finals_dir = output_dir / "finals"
    finals_dir.mkdir(exist_ok=True)
    for name, data in final_images.items():
        (finals_dir / f"{name}.png").write_bytes(data)

    index_html = output_dir / "index.html"
    index_html.write_text(_render_html(report_doc), encoding="utf-8")
    return {
        "report": str(report_path),
        "dashboard": str(dashboard_path),
        "strip": str(strip_path) if strip_path else None,
        "html": str(index_html),
        "finals_dir": str(finals_dir),
    }


def _render_html(report_doc: dict) -> str:
    rows = []
    for sample in report_doc["sample_results"]:
        rows.append(
            "<tr><td>{sample}</td><td>{status}</td><td>{hash}</td><td>{reused}</td></tr>".format(
                sample=sample["sample"],
                status=sample["status"],
                hash=sample.get("final_pixel_hash") or "-",
                reused=",".join(sample.get("reused", [])) or "-",
            )
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Workbench report {report_doc['run_id']}</title>
<style>body{{font-family:monospace;margin:24px}} td,th{{border:1px solid #ccc;padding:4px 8px}}</style>
</head><body>
<h2>Generative Editing Workbench - {report_doc['run_id']}</h2>
<img src="dashboard.png" alt="dashboard">
<h3>Samples</h3><table><tr><th>sample</th><th>status</th><th>pixel hash</th><th>reused stages</th></tr>
{''.join(rows)}</table>
<p>Full machine-readable data: <a href="report.json">report.json</a></p>
</body></html>"""
