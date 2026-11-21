"""End-to-end local demonstration and acceptance comparison helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .conversion import convert_file
from .edits import apply_edit_script
from .errors import InterruptedRun
from .generator import generate_workspace
from .levels import LevelMap
from .pipeline import EvaluationPipeline
from .util import atomic_write_json, sha256_json


def _clone_state_workspace(source: str | Path, target: str | Path) -> Path:
    source = Path(source)
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for filename in ("config.json", "level_map.json", "state.json"):
        if (source / filename).exists():
            shutil.copyfile(source / filename, target / filename)
    for directory in ("samples", "eval"):
        src = source / directory
        if src.exists():
            shutil.copytree(src, target / directory)
    return target


def _product_signature(report: dict[str, Any]) -> dict[str, Any]:
    canonical = []
    for item in report.get("canonical_results", []):
        canonical.append(
            {
                "file_id": item["file_id"],
                "total_micro": item["total_micro"],
                "dimensions_micro": item["dimensions_micro"],
                "level_code": item["level_code"],
                "level_name": item["level_name"],
                "precision_tag": item["precision_tag"],
                "frame_summary_count": item["frame_summary_count"],
                "frame_fingerprints": [
                    frame["frame_fingerprint"] for frame in item.get("frame_summaries", [])
                ],
            }
        )
    return {
        "canonical": canonical,
        "deviations": report.get("deviations", []),
        "consistency": {
            key: value
            for key, value in report.get("consistency", {}).items()
            if key not in ("canonical_digest",)
        },
        "degradations": report.get("degradations", {}).get("entries", []),
        "boundary_samples": report.get("boundary_samples", []),
        "failures": report.get("failures", []),
        "ledger_order": [
            {"order": entry["order"], "task_key": entry["task_key"]}
            for entry in report.get("cache_ledger", {}).get("entries", [])
        ],
        "output_sha256": report.get("outputs", {}).get("sha256"),
        "active_version": report.get("run", {}).get("active_version"),
    }


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_sig = _product_signature(left)
    right_sig = _product_signature(right)
    field_results: dict[str, bool] = {}
    for key in left_sig:
        field_results[key] = sha256_json(left_sig[key]) == sha256_json(right_sig[key])
    return {
        "equal": all(field_results.values()),
        "fields": field_results,
        "left_digest": sha256_json(left_sig),
        "right_digest": sha256_json(right_sig),
        "left_reuse_count": left.get("cache_ledger", {}).get("reuse_count"),
        "left_recompute_count": left.get("cache_ledger", {}).get("recompute_count"),
        "right_reuse_count": right.get("cache_ledger", {}).get("reuse_count"),
        "right_recompute_count": right.get("cache_ledger", {}).get("recompute_count"),
    }


def run_conversion_matrix(workspace: str | Path) -> list[dict[str, Any]]:
    workspace = Path(workspace)
    level_map = LevelMap.load(workspace / "level_map.json")
    config = json.loads((workspace / "config.json").read_text(encoding="utf-8"))
    fixed_bits = int(config["fixed_decimal_bits"])
    matrix = [
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
    ]
    source_paths = {
        1: workspace / "eval/v1/results_v1.txt",
        2: workspace / "eval/v2/results_v2.json",
        3: workspace / "eval/v3/results_v3.bin",
    }
    output_dir = workspace / "conversions"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for source_version, target_version in matrix:
        extension = {1: "txt", 2: "json", 3: "bin"}[target_version]
        target_path = output_dir / f"v{source_version}_to_v{target_version}.{extension}"
        bypass_path = output_dir / f"v{source_version}_to_v{target_version}_bypass.json"
        summary = convert_file(
            source_version,
            target_version,
            source_paths[source_version],
            target_path,
            level_map,
            fixed_bits,
            bypass_path,
        )
        summary["degradation_codes"] = sorted(
            {item["code"] for item in summary["degradations"]}
        )
        results.append(summary)
    return results


def run_demo(workspace: str | Path = "validation_workspace") -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    generated = generate_workspace(workspace, clean=True)
    initial_report = EvaluationPipeline(workspace).run()
    conversions = run_conversion_matrix(workspace)
    edit_summary = apply_edit_script(
        workspace,
        workspace / "edits" / "engineering_sequence.jsonl",
        run_after_each=True,
        run_final=False,
    )
    final_report = json.loads((workspace / "reports" / "diagnostic.json").read_text(encoding="utf-8"))

    resume_workspace = workspace.parent / f"{workspace.name}_resume"
    generate_workspace(resume_workspace, clean=True)
    interrupted = False
    interruption_error = None
    try:
        EvaluationPipeline(resume_workspace, stop_after_stage="parse").run()
    except InterruptedRun as exc:
        interrupted = True
        interruption_error = str(exc)
    manifest_after_interrupt = json.loads(
        (resume_workspace / "snapshots" / "manifest.json").read_text(encoding="utf-8")
    )
    resumed_report = EvaluationPipeline(resume_workspace).run()

    full_workspace = workspace.parent / f"{workspace.name}_full_rebuild"
    _clone_state_workspace(workspace, full_workspace)
    full_report = EvaluationPipeline(full_workspace).run()
    full_comparison = compare_reports(final_report, full_report)

    serial_workspace = workspace.parent / f"{workspace.name}_serial"
    _clone_state_workspace(workspace, serial_workspace)
    serial_report = EvaluationPipeline(serial_workspace, serial_mode=True).run()
    serial_comparison = compare_reports(final_report, serial_report)

    conversion_summary = []
    for item in conversions:
        conversion_summary.append(
            {
                "pair": f"v{item['source_version']}->v{item['target_version']}",
                "records": item["converted_records"],
                "failures": len(item["failures"]),
                "degradations": len(item["degradations"]),
                "degradation_codes": item["degradation_codes"],
                "output_sha256": item["output_sha256"],
            }
        )
    result = {
        "schema": "aqeval.demo.v1",
        "workspace": str(workspace),
        "generated": generated,
        "initial_run": {
            key: initial_report[key]
            for key in ("summary", "concurrency", "outputs")
        },
        "conversions": conversion_summary,
        "edit_sequence": edit_summary,
        "final_run": {
            "summary": final_report["summary"],
            "snapshot": final_report["snapshot"],
            "concurrency": final_report["concurrency"],
            "outputs": final_report["outputs"],
            "injections": final_report["injections"],
            "degradation_statistics": final_report["degradations"]["statistics"],
        },
        "snapshot_resume": {
            "interrupted": interrupted,
            "interruption_error": interruption_error,
            "manifest_after_interrupt": {
                "complete": manifest_after_interrupt.get("complete"),
                "stages": sorted(manifest_after_interrupt.get("stages", {})),
            },
            "resumed_stages": resumed_report["snapshot"]["resumed_stages"],
            "parse_resumed": resumed_report["stages"]["parse"]["resumed"],
        },
        "full_rebuild_comparison": full_comparison,
        "serial_vs_concurrent_comparison": serial_comparison,
    }
    atomic_write_json(workspace / "demo_summary.json", result)
    return result


def print_demo_summary(result: dict[str, Any]) -> None:
    print("AQEVAL_DEMO_BEGIN")
    print(f"workspace={result['workspace']}")
    print(
        "initial: "
        f"files={result['initial_run']['summary']['canonical_count']} "
        f"parsed={result['initial_run']['summary']['parsed_count']} "
        f"reuse={result['initial_run']['summary']['reuse_count']} "
        f"recompute={result['initial_run']['summary']['recompute_count']}"
    )
    print("conversions:")
    for item in result["conversions"]:
        print(
            f"  {item['pair']:<8} records={item['records']} "
            f"failures={item['failures']} degradations={item['degradations']} "
            f"codes={','.join(item['degradation_codes']) or '-'}"
        )
    print(
        "edits: "
        f"operations={result['edit_sequence']['operation_count']} "
        f"runs={len(result['edit_sequence']['runs'])}"
    )
    print(
        "final: "
        f"failures={result['final_run']['summary']['failure_count']} "
        f"degradations={result['final_run']['summary']['degradation_count']} "
        f"boundaries={result['final_run']['summary']['boundary_count']} "
        f"reuse={result['final_run']['summary']['reuse_count']} "
        f"recompute={result['final_run']['summary']['recompute_count']}"
    )
    print(
        "injections: "
        + ", ".join(
            f"{item['id']}={item['business_code']}:{'HIT' if item['hit'] else 'MISS'}"
            for item in result["final_run"]["injections"]
        )
    )
    resume = result["snapshot_resume"]
    print(
        "snapshot_resume: "
        f"interrupted={resume['interrupted']} "
        f"parse_resumed={resume['parse_resumed']} "
        f"resumed_stages={','.join(resume['resumed_stages'])}"
    )
    print(
        "full_rebuild_equivalence: "
        f"equal={result['full_rebuild_comparison']['equal']} "
        f"fields={json.dumps(result['full_rebuild_comparison']['fields'], sort_keys=True)}"
    )
    print(
        "serial_vs_concurrent: "
        f"equal={result['serial_vs_concurrent_comparison']['equal']} "
        f"fields={json.dumps(result['serial_vs_concurrent_comparison']['fields'], sort_keys=True)}"
    )
    print(f"summary_json={Path(result['workspace']) / 'demo_summary.json'}")
    print("AQEVAL_DEMO_END")

