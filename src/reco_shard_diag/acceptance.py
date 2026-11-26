from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import fingerprint
from .fixtures import build_demo_config
from .models import ProjectConfig
from .node import NodeRuntime
from .persistence import SnapshotStore, termination_classification
from .state import materialize


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _run_cli(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reco_shard_diag.cli", *args],
        text=True,
        capture_output=True,
        env=_cli_env(),
        timeout=timeout,
    )


def _canonical_view(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": result.get("phase"),
        "metadata": result.get("metadata"),
        "per_shard": {
            shard_id: {
                "ready": value.get("ready"),
                "aggregate": value.get("aggregate"),
                "receipt": value.get("receipt"),
                "watermark": value.get("watermark"),
                "cache_keys": value.get("cache_keys"),
            }
            for shard_id, value in sorted(result.get("per_shard", {}).items())
        },
        "ready_wait_history": result.get("ready_wait_history"),
        "ready_wait_queue": result.get("ready_wait_queue"),
        "errors": result.get("errors"),
        "error_counts": result.get("error_counts"),
        "budget": result.get("budget"),
        "batch_plan": result.get("batch_plan"),
        "completed_cursor": result.get("completed_cursor"),
        "reuse_ledger": result.get("reuse_ledger"),
        "invalidation_ledger": result.get("invalidation_ledger"),
        "reuse_plan": result.get("reuse_plan"),
        "invalidation_plan": result.get("invalidation_plan"),
    }


def _load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicate_delivery_probe(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = ProjectConfig.from_raw(raw)
    state = materialize(cfg)
    node = state.effective_node(sorted(state.nodes)[0])
    spec = type(node)(
        node_id=node.node_id,
        host=node.host,
        port=node.port,
        capacity_slots=max(2, node.capacity_slots),
        lease_limit=max(2, node.lease_limit),
    )
    runtime = NodeRuntime(spec=spec)
    first = runtime.acquire({"shard_id": "duplicate-probe", "ttl_ms": 1000})
    second = runtime.acquire({"shard_id": "duplicate-probe", "ttl_ms": 1000})
    common = {
        "shard_id": "duplicate-probe",
        "batch_id": "duplicate-probe",
        "quota_version": 1,
        "width": 16,
        "dimensions": 2,
        "samples": [
            {"group": "g0", "values": [2, 3]},
            {"group": "g0", "values": [4, 5]},
        ],
    }
    first_response = runtime.aggregate(
        {
            **common,
            "request_id": "dup-request-1",
            "lease_id": first["lease_id"],
            "expected_generation": first["lease_generation"],
        }
    )
    second_response = runtime.aggregate(
        {
            **common,
            "request_id": "dup-request-2",
            "lease_id": second["lease_id"],
            "expected_generation": second["lease_generation"],
        }
    )
    runtime.release(
        {"lease_id": first["lease_id"], "expected_generation": first["lease_generation"]}
    )
    runtime.release(
        {"lease_id": second["lease_id"], "expected_generation": second["lease_generation"]}
    )
    return {
        "first_ok": bool(first_response.get("ok")),
        "second_ok": bool(second_response.get("ok")),
        "second_duplicate": bool(second_response.get("duplicate_delivery")),
        "supersedes_request_id": second_response.get("supersedes_request_id"),
        "overwrite_relation_recorded": second_response.get("supersedes_request_id")
        == "dup-request-1",
    }


def _event_prefix(raw: dict[str, Any], count: int) -> dict[str, Any]:
    result = json.loads(json.dumps(raw))
    result["events"] = result.get("events", [])[:count]
    return result


def _selective_reuse_probe(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    baseline = _event_prefix(raw, 0)
    selected = _event_prefix(raw, 1)
    baseline_path = root / "select-baseline.json"
    selected_path = root / "select-final.json"
    _write_json(baseline_path, baseline)
    _write_json(selected_path, selected)
    wd = root / "selective-work"
    out1 = root / "selective-baseline-artifacts"
    out2 = root / "selective-final-artifacts"
    first = _run_cli(
        [
            "run",
            "--workdir",
            str(wd),
            "--config",
            str(baseline_path),
            "--fresh",
            "--outdir",
            str(out1),
        ]
    )
    second = _run_cli(
        [
            "run",
            "--workdir",
            str(wd),
            "--config",
            str(selected_path),
            "--outdir",
            str(out2),
        ]
    )
    if first.returncode != 0 or second.returncode != 0:
        return {
            "ok": False,
            "baseline_returncode": first.returncode,
            "second_returncode": second.returncode,
            "baseline_stderr": first.stderr,
            "second_stderr": second.stderr,
        }
    result = _load_run(out2 / "run.json")
    reuse_plan = result.get("reuse_plan", [])
    invalidation_plan = result.get("invalidation_plan", [])
    cache_hits = result.get("cache_hit_ledger", [])
    reused_shards = sorted(entry["shard_id"] for entry in reuse_plan)
    invalidated_shards = sorted(
        entry["shard_id"] for entry in invalidation_plan
    )
    reused_stages = sorted(
        {entry["stage"] for entry in cache_hits if entry["shard_id"] == "shard-e"}
    )
    return {
        "ok": True,
        "reused_shards": reused_shards,
        "invalidated_shards": invalidated_shards,
        "reused_stages_for_shard_e": reused_stages,
        "persisted_hit_count": len(cache_hits),
        "only_shard_e_reused": reused_shards == ["shard-e"],
        "downstream_invalidated": invalidated_shards
        == ["shard-a", "shard-b", "shard-c", "shard-d"],
        "all_stages_hit": reused_stages
        == ["aggregate", "readiness", "receipt", "watermark"],
    }


def _report_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [
        "# 验收报告",
        "",
        f"- 生成时间（UTC）: `{report['generated_at_utc']}`",
        f"- Python: `{report['environment']['python']}`",
        f"- NumPy: `{report['environment']['numpy']}`",
        f"- 平台: `{report['environment']['platform']}`",
        f"- 回环绑定: `{report['environment']['loopback_bind']}`",
        "",
        "## 检查项",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- `{name}`: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 关键结论",
            "",
            f"- 全量/增量逐字段一致: `{report['comparisons']['incremental_equals_full']}`",
            f"- 二次全量逐字段一致: `{report['comparisons']['rerun_equals_full']}`",
            f"- 强杀恢复逐字段一致: `{report['comparisons']['resume_equals_full']}`",
            f"- 强杀快照分类: `{report['crash_recovery']['termination_kind']}`",
            f"- 注入命中: `{', '.join(report['diagnostics']['injection_hits'])}`",
            f"- 业务码计数: `{json.dumps(report['diagnostics']['error_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- 水位结论: `{json.dumps(report['diagnostics']['watermarks'], ensure_ascii=False, sort_keys=True)}`",
            f"- 诊断报告指纹: `{report['diagnostics']['report_fingerprint']}`",
            "",
            "详细节点账本见 `node-ledger.json`，全量结论见 `full-run.json`，诊断报告见 `diagnostic-report.json`。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_acceptance(workdir: Path, outdir: Path) -> int:
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(tempfile.mkdtemp(prefix=f"accept-{stamp}-", dir=workdir))
    raw = build_demo_config(include_events=True)
    config_path = root / "config.json"
    _write_json(config_path, raw)
    baseline_path = root / "baseline.json"
    _write_json(baseline_path, _event_prefix(raw, 3))

    full_wd = root / "full-work"
    full_out = root / "full-artifacts"
    full_cmd = _run_cli(
        [
            "run",
            "--workdir",
            str(full_wd),
            "--config",
            str(config_path),
            "--fresh",
            "--outdir",
            str(full_out),
        ]
    )
    if full_cmd.returncode != 0:
        print(full_cmd.stdout)
        print(full_cmd.stderr, file=sys.stderr)
        return full_cmd.returncode
    full = _load_run(full_out / "run.json")
    diagnostics = _load_run(full_out / "diagnostic-report.json")
    node_ledger = _load_run(full_out / "node-ledger.json")

    rerun_wd = root / "rerun-work"
    rerun_out = root / "rerun-artifacts"
    rerun_cmd = _run_cli(
        [
            "run",
            "--workdir",
            str(rerun_wd),
            "--config",
            str(config_path),
            "--fresh",
            "--outdir",
            str(rerun_out),
        ]
    )
    if rerun_cmd.returncode != 0:
        print(rerun_cmd.stdout)
        print(rerun_cmd.stderr, file=sys.stderr)
        return rerun_cmd.returncode
    rerun = _load_run(rerun_out / "run.json")

    incremental_wd = root / "incremental-work"
    baseline_out = root / "incremental-baseline-artifacts"
    baseline_cmd = _run_cli(
        [
            "run",
            "--workdir",
            str(incremental_wd),
            "--config",
            str(baseline_path),
            "--fresh",
            "--outdir",
            str(baseline_out),
        ]
    )
    if baseline_cmd.returncode != 0:
        print(baseline_cmd.stdout)
        print(baseline_cmd.stderr, file=sys.stderr)
        return baseline_cmd.returncode
    incremental_out = root / "incremental-final-artifacts"
    incremental_cmd = _run_cli(
        [
            "run",
            "--workdir",
            str(incremental_wd),
            "--config",
            str(config_path),
            "--outdir",
            str(incremental_out),
        ]
    )
    if incremental_cmd.returncode != 0:
        print(incremental_cmd.stdout)
        print(incremental_cmd.stderr, file=sys.stderr)
        return incremental_cmd.returncode
    incremental = _load_run(incremental_out / "run.json")

    crash_wd = root / "crash-work"
    crash_out = root / "crash-artifacts"
    crash_cmd = _run_cli(
        [
            "run",
            "--workdir",
            str(crash_wd),
            "--config",
            str(config_path),
            "--fresh",
            "--crash-after",
            "2",
            "--outdir",
            str(crash_out),
        ]
    )
    crashed_snapshot: dict[str, Any] | None = None
    try:
        cfg = ProjectConfig.from_raw(raw)
        store = SnapshotStore(
            crash_wd,
            global_version=cfg.global_version,
            base_config_hash=cfg.base_config_hash,
        )
        crashed_snapshot = store.load()
    except Exception as exc:
        crashed_snapshot = {"load_error": str(exc)}
    resume_out = root / "resume-artifacts"
    resume_cmd = _run_cli(
        [
            "resume",
            "--workdir",
            str(crash_wd),
            "--config",
            str(config_path),
            "--outdir",
            str(resume_out),
        ]
    )
    if resume_cmd.returncode != 0:
        print(resume_cmd.stdout)
        print(resume_cmd.stderr, file=sys.stderr)
        return resume_cmd.returncode
    resumed = _load_run(resume_out / "run.json")

    selective = _selective_reuse_probe(root, raw)
    duplicate_probe = _duplicate_delivery_probe(raw)
    full_view = _canonical_view(full)
    incremental_equals = _canonical_view(incremental) == full_view
    rerun_equals = _canonical_view(rerun) == full_view
    resume_equals = _canonical_view(resumed) == full_view
    crash_kind = (
        termination_classification(crashed_snapshot).get("kind")
        if crashed_snapshot is not None
        else "SNAPSHOT_LOAD_FAILED"
    )
    expected_injections = {
        "I_DEPENDENCY_CYCLE",
        "I_QUOTA_PARENT_MISSING",
        "I_SAMPLE_DIMENSION",
        "I_STALE_LEASE",
        "I_EXPIRED_RENEW",
    }
    injection_hits = set(diagnostics.get("injection_hits", []))
    checks = {
        "full_run_complete": full.get("phase") == "COMPLETE"
        and not full.get("pending_queue"),
        "all_authoritative_receipts": all(
            value.get("receipt", {}).get("authoritative")
            for value in full.get("per_shard", {}).values()
        ),
        "incremental_equals_full": incremental_equals,
        "rerun_equals_full": rerun_equals,
        "forced_termination_returncode_137": crash_cmd.returncode == 137,
        "crash_snapshot_was_running": bool(
            isinstance(crashed_snapshot, dict)
            and crashed_snapshot.get("phase") == "RUNNING"
        ),
        "crash_recovery_classified": crash_kind
        == "RECOVERED_AFTER_FORCED_TERMINATION",
        "resume_equals_full": resume_equals,
        "selective_reuse": bool(selective.get("ok"))
        and selective.get("only_shard_e_reused")
        and selective.get("downstream_invalidated")
        and selective.get("all_stages_hit"),
        "duplicate_delivery_overwrite_relation": bool(
            duplicate_probe.get("first_ok")
            and duplicate_probe.get("second_ok")
            and duplicate_probe.get("second_duplicate")
            and duplicate_probe.get("overwrite_relation_recorded")
        ),
        "injections_all_hit": expected_injections <= injection_hits,
        "watermark_warning_observed": "E_QUOTA_SOFT_LIMIT"
        in full.get("error_counts", {}),
        "budget_within_limit": full.get("budget", {}).get("committed_units", 0)
        <= full.get("budget", {}).get("resource_budget", 0),
        "ready_wait_distribution_present": bool(
            diagnostics.get("ready_wait_distribution", {}).get("blocked_shard_count")
        ),
        "fixedpoint_attribution_present": bool(
            diagnostics.get("conclusion_consistency", {}).get("different_shards")
        ),
        "node_ledgers_present": all(
            ledger.get("ok")
            for ledger in node_ledger.values()
        ),
    }
    report = {
        "schema_version": "shard-acceptance-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "loopback_bind": True,
            "subprocess_nodes": True,
        },
        "checks": checks,
        "comparisons": {
            "incremental_equals_full": incremental_equals,
            "rerun_equals_full": rerun_equals,
            "resume_equals_full": resume_equals,
            "canonical_view_fingerprint": fingerprint(full_view),
        },
        "crash_recovery": {
            "crash_returncode": crash_cmd.returncode,
            "termination_kind": crash_kind,
            "completed_cursor_before_resume": (
                crashed_snapshot.get("completed_cursor", [])
                if isinstance(crashed_snapshot, dict)
                else []
            ),
            "resume_snapshot_status": resumed.get("execution", {}).get(
                "snapshot_status", {}
            ),
        },
        "selective_reuse": selective,
        "duplicate_delivery": duplicate_probe,
        "diagnostics": {
            "injection_hits": diagnostics.get("injection_hits", []),
            "error_counts": diagnostics.get("error_counts", {}),
            "ready_wait_distribution": diagnostics.get("ready_wait_distribution", {}),
            "node_lease_generations": diagnostics.get("node_lease_generations", {}),
            "watermarks": {
                shard_id: (value.get("watermark") or {}).get("state")
                for shard_id, value in sorted(
                    full.get("per_shard", {}).items()
                )
            },
            "report_fingerprint": fingerprint(diagnostics),
        },
        "artifacts": {
            "root": str(root),
            "full_run": str(full_out / "run.json"),
            "diagnostic_report": str(full_out / "diagnostic-report.json"),
            "node_ledger": str(full_out / "node-ledger.json"),
            "resume_run": str(resume_out / "run.json"),
        },
    }
    passed = all(checks.values())
    report["passed"] = passed
    _write_json(outdir / "acceptance-report.json", report)
    (outdir / "REPORT.md").write_text(_report_markdown(report), encoding="utf-8")
    for source, name in (
        (full_out / "run.json", "full-run.json"),
        (full_out / "diagnostic-report.json", "diagnostic-report.json"),
        (full_out / "node-ledger.json", "node-ledger.json"),
        (full_out / "summary.md", "summary.md"),
        (full_out / "samples.jsonl", "samples.jsonl"),
        (resume_out / "run.json", "resume-run.json"),
    ):
        if source.exists():
            (outdir / name).write_bytes(source.read_bytes())
    print(json.dumps({"passed": passed, "checks": checks, "outdir": str(outdir)}, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1
