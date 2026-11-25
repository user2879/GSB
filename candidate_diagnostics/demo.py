from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .attribution import attribute_many, attribute_request
from .concurrency import analyze_many_concurrent
from .config import Config, deep_merge, load_config
from .engine import IncrementalEngine, compare_result_fields
from .formats import (
    load_v2_file,
    parse_v1_lines,
    v1_to_v2,
    v2_to_v1,
    write_json,
    write_text,
)
from .injections import run_injection_plan
from .loader import load_command_stream, load_requests_jsonl
from .models import Candidate, Request
from .pipeline import analyze_many
from .scheduler import SnapshotStore, run_scheduled_snapshot
from .stats_receipts import (
    adjudicate_receipts,
    load_three_receipt_formats,
)


def _read_pytest_summary(root: Path) -> dict[str, Any]:
    path = root / "artifacts" / "pytest-output.txt"
    if not path.exists():
        return {"available": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    summary_lines = [
        line
        for line in lines
        if "passed" in line or "failed" in line or "error" in line
    ]
    return {
        "available": True,
        "path": str(path),
        "tail": lines[-20:],
        "summary_lines": summary_lines[-5:],
    }


def _apply_command(
    command: dict[str, Any], requests: dict[str, Request], engine: IncrementalEngine
) -> None:
    op = command["op"]
    if op == "set_channel_weight":
        engine.update_config({"channels": {command["channel"]: float(command["weight"])}})
    elif op == "set_top_k":
        engine.update_config({"top_k": int(command["top_k"])})
    elif op == "set_sampling_ratio":
        patch: dict[str, Any] = {"sampling": {}}
        if "ratio" in command:
            patch["sampling"]["ratio"] = float(command["ratio"])
        if "ratio_permille" in command:
            patch["sampling"]["ratio_permille"] = int(command["ratio_permille"])
        engine.update_config(patch)
    elif op == "set_quantization":
        engine.update_config({"quantization": {k: v for k, v in command.items()
                                              if k in {"fixed_decimals", "gear_exponent", "rounding"}}})
    elif op == "set_reference_width":
        engine.update_config(
            {
                "quantization": {
                    "reference_fixed_decimals": int(command["reference_fixed_decimals"]),
                    "reference_bit_width": int(command["reference_bit_width"]),
                }
            }
        )
    elif op == "replace_candidates":
        rid = command["request_id"]
        candidates = [Candidate(**candidate) for candidate in command["candidates"]]
        requests[rid] = Request(rid, candidates)
    else:
        raise ValueError(f"unknown command op: {op}")


def _roundtrip_check(all_records: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    lines, sidecar, failures = v2_to_v1(all_records, config)
    restored, restore_failures = v1_to_v2(lines, config, sidecar)
    original_by_key = {
        (r["request_id"], r["candidate_id"]): r for r in all_records
    }
    mismatches: list[dict[str, Any]] = []
    checked_fields = [
        "score",
        "selection_result",
        "sampling_scheme",
        "sampling_weight",
        "precision_label",
        "boundary",
        "rank",
        "top_k",
        "channel",
        "aux_feature",
        "shuffle_position",
        "fixed_score",
        "dequant_score",
    ]
    for record in restored:
        key = (record["request_id"], record["candidate_id"])
        original = original_by_key[key]
        for field_name in checked_fields:
            if field_name in original and original.get(field_name) != record.get(field_name):
                mismatches.append(
                    {"key": key, "field": field_name,
                     "original": original.get(field_name),
                     "restored": record.get(field_name)}
                )
    return {
        "v1_line_count": len(lines),
        "failure_count": len(failures) + len(restore_failures),
        "failures": [f.to_dict() for f in failures + restore_failures],
        "restored_count": len(restored),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "sidecar_key_count": len(sidecar["records"]),
        "exact_without_sidecar_example": {
            "score": restored[0]["score"],
            "precision_label": restored[0]["precision_label"],
        },
        "lines": lines,
        "sidecar": sidecar,
        "restored": restored,
    }


def run_demo(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or Path.cwd())
    samples = root / "samples"
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    config = load_config(samples / "config.json")
    requests, load_failures = load_requests_jsonl(samples / "candidates.jsonl", config)
    commands, command_parse_failures = load_command_stream(samples / "commands.jsonl")

    baseline_engine = IncrementalEngine(config)
    baseline_run = baseline_engine.run(requests)
    serial_results = analyze_many(requests, config)
    concurrent_results, concurrency_report = analyze_many_concurrent(requests, config)
    serializable_serial = {rid: result.to_dict() for rid, result in serial_results.items()}
    serializable_concurrent = {rid: result.to_dict() for rid, result in concurrent_results.items()}
    concurrency_match = serializable_serial == serializable_concurrent

    command_trace: list[dict[str, Any]] = []
    incremental_engine = IncrementalEngine(config)
    working_requests = dict(requests)
    incremental_run = incremental_engine.run(working_requests)
    for index, command in enumerate(commands, 1):
        _apply_command(command, working_requests, incremental_engine)
        incremental_run = incremental_engine.run(working_requests)
        command_trace.append(
            {
                "index": index,
                "command": {k: v for k, v in command.items() if not k.startswith("_")},
                "config_hash": incremental_engine.config.config_hash,
                "ledger": {
                    "hit_requests": incremental_run.ledger["hit_requests"],
                    "recomputed_requests": incremental_run.ledger["recomputed_requests"],
                    "hit_segments": incremental_run.ledger["hit_segments"],
                    "recomputed_segments": incremental_run.ledger["recomputed_segments"],
                    "hit_products": incremental_run.ledger["hit_products"],
                    "recomputed_products": incremental_run.ledger["recomputed_products"],
                },
            }
        )

    blank_final = IncrementalEngine(incremental_engine.config).run(working_requests)
    incremental_vs_blank = compare_result_fields(incremental_run, blank_final)

    all_records = [
        record
        for rid in sorted(baseline_run.results)
        for record in baseline_run.records[rid]
    ]
    all_v1_lines = [
        line
        for rid in sorted(baseline_run.v1_lines)
        for line in baseline_run.v1_lines[rid]
    ]
    roundtrip = _roundtrip_check(all_records, config)

    generated_v1_records, generated_v1_failures = parse_v1_lines(
        (samples / "conclusions.generated.v1.txt").read_text(encoding="utf-8").splitlines(),
        config,
        source=str(samples / "conclusions.generated.v1.txt"),
    )
    v2_sample_records, v2_sample_failures = load_v2_file(
        samples / "conclusions.generated.v2.json", config
    )
    bad_v1_lines = (samples / "conclusions.bad.v1.txt").read_text(encoding="utf-8").splitlines()
    bad_v1_records, bad_v1_failures = parse_v1_lines(
        bad_v1_lines, config, source=str(samples / "conclusions.bad.v1.txt")
    )

    receipts = load_three_receipt_formats(
        samples / "stats_receipts_v1_legacy.txt",
        samples / "stats_receipts_v2_nested.json",
        samples / "stats_receipts_v3_columnar.csv",
    )
    stats_view = adjudicate_receipts(receipts)

    attribution = attribute_many(requests, config)
    qflip_config = Config(
        deep_merge(config.raw, {"quantization": {"fixed_decimals": 0, "gear_exponent": 0}})
    )
    qflip_attribution = attribute_request(requests["req-qflip"], qflip_config, config)
    quota_config = Config(
        deep_merge(config.raw, {"sampling": {"ratio": 0.8, "ratio_permille": 800}})
    )
    quota_attribution = attribute_request(requests["req-001"], config, quota_config)

    state_dir = artifacts / "state"
    store = SnapshotStore(state_dir, config)
    scheduled_engine = IncrementalEngine(config)
    scheduled_run, schedule_plan, schedule_manifest = run_scheduled_snapshot(
        scheduled_engine, requests, config, store
    )
    resume_manifest = SnapshotStore(state_dir, config).begin_run()
    version_cfg = Config(deep_merge(config.raw, {"top_k": config.top_k + 1}))
    version_dir = artifacts / "state-version-change"
    SnapshotStore(version_dir, config).begin_run()
    version_manifest = SnapshotStore(version_dir, version_cfg).begin_run()

    injection_report = run_injection_plan(samples / "injections.json", config)
    pytest_summary = _read_pytest_summary(root)

    write_text(artifacts / "conclusions.v1.txt", all_v1_lines)
    write_json(artifacts / "conclusions.v2.json", {
        "schema": "candidate-conclusion-v2-batch",
        "records": all_records,
    })
    write_text(artifacts / "conclusions.roundtrip.v1.txt", roundtrip.pop("lines"))
    write_json(artifacts / "conclusions.sidecar.json", roundtrip.pop("sidecar"))
    roundtrip_restored = roundtrip.pop("restored")
    write_json(artifacts / "conclusions.restored.v2.json", {"records": roundtrip_restored})
    write_json(artifacts / "baseline-results.json", baseline_run.to_dict())
    write_json(artifacts / "incremental-trace.json", {
        "final_ledger": incremental_run.ledger,
        "command_trace": command_trace,
        "comparison_to_blank": incremental_vs_blank,
    })
    write_json(artifacts / "attribution.json", {
        "baseline_reference": attribution,
        "quantization_flip_case": qflip_attribution,
        "quota_rounding_case": quota_attribution,
    })
    write_json(artifacts / "stats-view.json", stats_view)
    write_json(artifacts / "injection-report.json", injection_report)

    report = {
        "schema": "candidate-diagnostics-report-v1",
        "sample_request_count": len(requests),
        "load_failure_count": len(load_failures),
        "load_failures": [f.to_dict() for f in load_failures],
        "command_parse_failure_count": len(command_parse_failures),
        "baseline": {
            rid: {
                "ranked_ids": result.ranked_ids,
                "top_k_ids": result.top_k_ids,
                "boundary": result.boundary,
                "sampled_ids": result.sampled_ids,
                "shuffle_order": result.shuffle_order,
                "layer_quotas": {
                    channel: layer["quota"] for channel, layer in result.layers.items()
                },
            }
            for rid, result in baseline_run.results.items()
        },
        "concurrency": {
            "business_fields_match_serial": concurrency_match,
            **concurrency_report,
        },
        "incremental": {
            "command_count": len(commands),
            "final_matches_blank_rebuild": incremental_vs_blank["identical"],
            "comparison": incremental_vs_blank,
            "first_weight_edit": command_trace[0]["ledger"] if command_trace else None,
            "candidate_replacement_edit": (
                command_trace[1]["ledger"] if len(command_trace) > 1 else None
            ),
            "last_command_summary": command_trace[-1]["ledger"] if command_trace else None,
        },
        "format_roundtrip": roundtrip,
        "sample_parse": {
            "generated_v1_records": len(generated_v1_records),
            "generated_v1_failures": [f.to_dict() for f in generated_v1_failures],
            "generated_v2_records": len(v2_sample_records),
            "generated_v2_failures": [f.to_dict() for f in v2_sample_failures],
            "bad_v1_failures": [f.to_dict() for f in bad_v1_failures],
            "bad_v1_record_count": len(bad_v1_records),
        },
        "stats": {
            "receipt_count": stats_view["receipt_count"],
            "authoritative_count": stats_view["authoritative_count"],
            "decisions": stats_view["decisions"],
        },
        "attribution_summary": {
            "baseline_first_stage_counts": attribution["first_exceeded_stage_counts"],
            "quantization_case": {
                "request_id": "req-qflip",
                "first_exceeded_stage": qflip_attribution["first_exceeded_stage"],
                "causes": [item["cause"] for item in qflip_attribution["flip_attribution"]],
            },
            "boundary_case": {
                "request_id": "req-bcmp",
                "first_exceeded_stage": next(
                    report["first_exceeded_stage"]
                    for report in attribution["reports"]
                    if report["request_id"] == "req-bcmp"
                ),
                "causes": [
                    item["cause"]
                    for report in attribution["reports"]
                    if report["request_id"] == "req-bcmp"
                    for item in report["flip_attribution"]
                    if item["stage"] == "truncation"
                ],
            },
            "quota_case": {
                "request_id": "req-001",
                "causes": [item["cause"] for item in quota_attribution["flip_attribution"]],
                "quota_diffs": quota_attribution["stages"]["stratified_sampling"]["quota_diffs"],
            },
        },
        "scheduler": {
            "plan": schedule_plan,
            "manifest": schedule_manifest,
            "resume_manifest_decision": resume_manifest["startup_decision"],
            "version_change_manifest_decision": version_manifest["startup_decision"],
            "version_change_reason": version_manifest.get("version_change_reason"),
        },
        "injections": injection_report,
        "pytest": pytest_summary,
    }
    write_json(artifacts / "diagnostic-report.json", report)
    write_text(artifacts / "diagnostic-report.md", render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 候选截断与采样诊断报告",
        "",
        f"- 样本请求数：`{report['sample_request_count']}`",
        f"- 加载失败数：`{report['load_failure_count']}`",
        f"- 并发/串行业务字段一致：`{report['concurrency']['business_fields_match_serial']}`",
        f"- 并发观测峰值：`{report['concurrency']['observed_max_concurrency']}` / 配置 `{report['concurrency']['configured_max_workers']}`",
        f"- 增量终态与空白重建一致：`{report['incremental']['final_matches_blank_rebuild']}`",
        f"- 首次权重编辑命中请求：`{report['incremental']['first_weight_edit']['hit_requests']}`",
        f"- 首次权重编辑重算请求：`{report['incremental']['first_weight_edit']['recomputed_requests']}`",
        f"- v2→v1→v2 不匹配字段数：`{report['format_roundtrip']['mismatch_count']}`",
        f"- 注入命中：`{report['injections']['hit_count']}/{report['injections']['total']}`",
        "",
        "## TopK / 边界 / 抽样",
    ]
    for rid, item in report["baseline"].items():
        boundary = item["boundary"]
        lines.append(
            f"- `{rid}`: TopK=`{item['top_k_ids']}`; "
            f"边界入选=`{boundary.get('boundary_selected_id')}`; "
            f"紧邻落选=`{boundary.get('nearest_rejected_id')}`; "
            f"抽样=`{item['sampled_ids']}`; 打散=`{item['shuffle_order']}`"
        )
    lines.extend(["", "## 注入故障映射"])
    for item in report["injections"]["results"]:
        lines.append(
            f"- `{item['injection_id']}` {'命中' if item['hit'] else '未命中'} → "
            f"`{', '.join(item['observed_codes']) or item.get('expected_code')}`"
        )
    sched = report["scheduler"]["plan"]
    lines.extend([
        "",
        "## 预算与快照",
        f"- 已处理：`{sched['processed']}`",
        f"- 待办：`{sched['pending']}`",
        f"- 游标：`{sched['completed_cursor']}`; 剩余预算：`{sched['remaining_budget']}`",
        f"- 同版本重启：`{report['scheduler']['resume_manifest_decision']}`",
        f"- 配置变化重启：`{report['scheduler']['version_change_manifest_decision']}`",
        "",
        "## 验证",
    ])
    if report["pytest"].get("available"):
        for item in report["pytest"].get("summary_lines", []):
            lines.append(f"- `{item}`")
    else:
        lines.append("- 尚未发现 `artifacts/pytest-output.txt`；请运行 `python -m pytest | tee artifacts/pytest-output.txt`。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = run_demo(Path.cwd())
    print(json.dumps({
        "report": "artifacts/diagnostic-report.json",
        "markdown": "artifacts/diagnostic-report.md",
        "requests": report["sample_request_count"],
        "injections_hit": f"{report['injections']['hit_count']}/{report['injections']['total']}",
        "incremental_matches_blank": report["incremental"]["final_matches_blank_rebuild"],
        "concurrency_matches_serial": report["concurrency"]["business_fields_match_serial"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
