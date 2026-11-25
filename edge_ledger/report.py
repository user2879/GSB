from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .engine import BuildResult


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def compare_quick_view(result: Mapping[str, Any]) -> dict[str, Any]:
    quick = result.get("quick_view", {})
    stats = result.get("stats", {})
    total = stats.get("total", {})
    authoritative = {
        "sample_count": total.get("input_samples", 0),
        "rendered_count": total.get("rendered_samples", 0),
        "removed_duplicates": total.get("removed_duplicates", 0),
        "render_points_consumed": total.get("render_points_consumed", 0),
        "clusters": total.get("clusters", 0),
        "code_counts": stats.get("code_counts", {}),
    }
    differences: list[dict[str, Any]] = []
    stage_by_key = {
        "sample_count": "load_validation",
        "rendered_count": "render",
        "removed_duplicates": "dedupe",
        "render_points_consumed": "schedule",
        "clusters": "dedupe",
        "code_counts": "load_validation",
    }
    for key in sorted(authoritative):
        if quick.get(key) != authoritative[key]:
            differences.append(
                {
                    "key": key,
                    "stage": stage_by_key[key],
                    "quick": quick.get(key),
                    "authoritative": authoritative[key],
                }
            )
    return {
        "matches": not differences,
        "differences": differences,
        "quick": quick,
        "authoritative": authoritative,
    }


def _cache_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    ledger = result.get("execution", {}).get("cache_ledger", [])
    stages: dict[str, dict[str, int]] = {}
    for item in ledger:
        stage = str(item.get("stage", "unknown"))
        entry = stages.setdefault(stage, {"reuse": 0, "recompute": 0})
        action = str(item.get("action", "recompute"))
        entry[action] = entry.get(action, 0) + 1
    return {key: stages[key] for key in sorted(stages)}


def report_payload(build: BuildResult) -> dict[str, Any]:
    payload = build.as_report()
    payload["report"]["cache_stage_summary"] = _cache_summary(payload)
    payload["report"]["quick_vs_authoritative"] = compare_quick_view(payload)
    payload["report"]["generated_at_unix_ns"] = time.time_ns()
    payload["report"]["root"] = str(build.root)
    payload["report"]["render_points_consumed"] = payload["stats"]["total"][
        "render_points_consumed"
    ]
    payload["report"]["physical_render_points_recomputed"] = payload.get(
        "execution", {}
    ).get("physical_render_points_recomputed", 0)
    payload["report"]["dedupe_removed"] = payload["stats"]["total"]["removed_duplicates"]
    payload["report"]["per_template_sample_counts"] = {
        template_id: values["input_samples"]
        for template_id, values in payload["stats"]["per_template"].items()
    }
    return payload


def _markdown(payload: Mapping[str, Any]) -> str:
    report = payload["report"]
    stats = payload["stats"]
    total = stats["total"]
    lines = [
        "# Edge Ledger Construction Report",
        "",
        "## Summary",
        "",
        f"- Global version: `{payload['global_version']}`",
        f"- Config hash: `{payload['config_hash']}`",
        f"- Canonical digest: `{report['canonical_digest']}`",
        f"- End-to-end: {report['runtime_ms']} ms",
        f"- Snapshot generation: {report['snapshot_generation']}",
        f"- Recovery action: `{report['recovery']['action']}`",
        f"- Input samples: {total['input_samples']}",
        f"- Rendered samples: {total['rendered_samples']}",
        f"- Kept samples: {total['kept_samples']}",
        f"- Dedupe removed: {total['removed_duplicates']}",
        f"- Clusters: {total['clusters']}",
        f"- Render points consumed: {total['render_points_consumed']}",
        f"- Physical render points recomputed: {report.get('physical_render_points_recomputed', 0)}",
        f"- Pending samples: {total['pending_samples']}",
        f"- Quota-stopped samples: {total['quota_stopped_samples']}",
        f"- Quick view matches ledger: {report['quick_vs_authoritative']['matches']}",
        "",
        "## Per-template counts",
        "",
        "| Template | Input | Rendered | Kept | Removed | Failed | Pending | Quota stopped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for template_id, values in stats["per_template"].items():
        lines.append(
            "| {template} | {input_samples} | {rendered_samples} | {kept_samples} | "
            "{removed_duplicates} | {failed_samples} | {pending_samples} | {quota_stopped_samples} |".format(
                template=template_id, **values
            )
        )
    lines.extend(
        [
            "",
            "## Business codes",
            "",
            "| Code | Count |",
            "|---|---:|",
        ]
    )
    for code, count in stats["code_counts"].items():
        lines.append(f"| {code} | {count} |")
    if not stats["code_counts"]:
        lines.append("| (none) | 0 |")
    lines.extend(
        [
            "",
            "## Injection hits",
            "",
            "| Injection | Target | Hit | Stages | Codes |",
            "|---|---|---|---|---|",
        ]
    )
    for item in payload.get("injection_hits", []):
        lines.append(
            f"| {item['injection_id']} | {item['target_kind']}:{item['target_id']} | "
            f"{item['hit']} | {', '.join(item['observed_stages'])} | "
            f"{', '.join(item['observed_codes'])} |"
        )
    lines.extend(
        [
            "",
            "## Cache stage summary",
            "",
            "| Stage | Reuse | Recompute |",
            "|---|---:|---:|",
        ]
    )
    for stage, counts in report["cache_stage_summary"].items():
        lines.append(f"| {stage} | {counts.get('reuse', 0)} | {counts.get('recompute', 0)} |")
    lines.extend(
        [
            "",
            "## Sample ledger",
            "",
            "| Sample | Shard | Template revision | Score | Residual | Cluster | Dedupe | Render digest |",
            "|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for sample in payload["samples"]:
        template_ref = f"{sample['template_id']}@{sample['template_revision']}"
        lines.append(
            f"| {sample['sample_id']} | {sample['source_shard']} | {template_ref} | "
            f"{sample.get('quality_score', '')} | {sample.get('quality_residual', '')} | "
            f"{sample.get('cluster_id', '')} | {sample.get('dedupe_status', sample['status'])} | "
            f"`{sample.get('render_digest', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Pending queue",
            "",
            "| Sample | Shard | Code | Sort key |",
            "|---|---|---|---|",
        ]
    )
    for item in payload.get("pending_queue", []):
        lines.append(
            f"| {item['sample_id']} | {item['source_shard']} | {item['code']} | `{item['sort_key']}` |"
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "| Stage | Code | Target | Injection | Message |",
            "|---|---|---|---|---|",
        ]
    )
    for failure in payload["failures"]:
        target = f"{failure['target_kind']}:{failure['target_id']}"
        lines.append(
            f"| {failure['stage']} | {failure['code']} | {target} | "
            f"{failure.get('injection_id') or ''} | {failure['message']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    build: BuildResult,
    *,
    reports_dir: Path | None = None,
    ledger_dir: Path | None = None,
) -> dict[str, Path]:
    root = build.root
    reports_dir = reports_dir or (root / "reports")
    ledger_dir = ledger_dir or (root / "ledger")
    payload = report_payload(build)
    json_path = reports_dir / "latest.json"
    md_path = reports_dir / "latest.md"
    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _atomic_write(md_path, _markdown(payload) + "\n")
    ledger_path = ledger_dir / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "recorded_at_unix_ns": payload["report"]["generated_at_unix_ns"],
        "global_version": payload["global_version"],
        "config_hash": payload["config_hash"],
        "template_chain_fingerprint": payload["template_chain_fingerprint"],
        "canonical_digest": payload["report"]["canonical_digest"],
        "snapshot_generation": payload["report"]["snapshot_generation"],
        "runtime_ms": payload["report"]["runtime_ms"],
        "recovery": payload["report"]["recovery"],
        "stats_total": payload["stats"]["total"],
        "code_counts": payload["stats"]["code_counts"],
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"json": json_path, "markdown": md_path, "ledger": ledger_path}
