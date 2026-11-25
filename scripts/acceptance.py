from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ner_review.config import RuntimeConfig
from ner_review.coordinator import CommandCoordinator
from ner_review.diagnostics import run_injection_suite
from ner_review.engine import IncrementalEngine
from ner_review.io_utils import fingerprint, write_json
from ner_review.snapshot import SnapshotStore
from ner_review.synth import build_command_scenario, write_demo_bundle
from ner_review.validation import load_package


def build_final_full(engine: IncrementalEngine) -> IncrementalEngine:
    full = IncrementalEngine(
        package=copy.deepcopy(engine.package),
        config=replace(engine.config),
        samples=copy.deepcopy(list(engine.samples.values())),
        lexicon=copy.deepcopy(engine.lexicon_list),
        corpus_failures=copy.deepcopy(engine.corpus_failures),
        lexicon_failures=copy.deepcopy(engine.lexicon_failures),
    )
    full.run_incremental()
    return full


def main() -> int:
    started = time.perf_counter()
    data_dir = ROOT / "examples" / "acceptance-data"
    workspace = ROOT / "state" / "acceptance"
    kill_workspace = ROOT / "state" / "acceptance-kill"
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    write_demo_bundle(data_dir, RuntimeConfig(seed=20261126, corpus_size=10))
    package_result = load_package(data_dir / "model_params.json")
    assert package_result["ok"], package_result["failures"]

    engine = IncrementalEngine.from_bundle(data_dir / "corpus.jsonl", data_dir / "lexicon.json", package_result)
    engine.run_incremental()
    scenario = build_command_scenario()
    command_reports = []
    for command in scenario:
        applied = engine.apply_command(command)
        run = engine.run_incremental()
        command_reports.append(
            {
                "op": command["op"],
                "ok": applied["ok"],
                "affected_samples": applied.get("affected_samples", []),
                "invalidated_count": len(applied.get("invalidations", [])),
                "recomputed_count": len(run["recomputed"]),
                "reused_count": len(run["reused"]),
            }
        )

    incremental_result = engine.authoritative_result()
    full = build_final_full(engine)
    full_result = full.authoritative_result()
    parity = incremental_result == full_result
    incremental_hash = fingerprint(incremental_result)
    full_hash = fingerprint(full_result)

    store = SnapshotStore(workspace)
    snapshot = store.save(engine)
    loaded = store.load()
    snapshot_parity = False
    snapshot_code = None
    if loaded["ok"]:
        loaded["engine"].run_incremental()
        snapshot_parity = loaded["engine"].authoritative_result() == incremental_result
    version_probe = store.load(expected_model_version="incompatible-version")
    snapshot_code = version_probe.get("reason_code")

    gate = threading.Event()
    coordinator = CommandCoordinator(engine, budget_ms=2, work_hook=lambda _engine, _command: gate.wait(2.0))
    stale_response = coordinator.submit(
        {"op": "set_transition_weight", "from": "O", "to": "O", "value": 0.61},
        budget_ms=5,
    )
    stale_ok = stale_response.get("code") == "NER-E3002"
    gate.set()
    fresh_response = coordinator.wait_for_latest(timeout=5.0)
    budget_ok = stale_ok and fresh_response.get("status") == "fresh"
    coordinator.shutdown()

    shutil.rmtree(kill_workspace, ignore_errors=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    killed = subprocess.run(
        [sys.executable, "-m", "ner_review.cli", "kill-target", "--data", str(data_dir), "--workspace", str(kill_workspace), "--mode", "before-save"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    recovered_cli = subprocess.run(
        [sys.executable, "-m", "ner_review.cli", "run", "--data", str(data_dir), "--workspace", str(kill_workspace), "--report", str(report_dir / "acceptance_kill_run.json")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    kill_recovery_ok = killed.returncode == -9 and recovered_cli.returncode == 0
    kill_parity = False
    if recovered_cli.returncode == 0:
        recovered_store = SnapshotStore(kill_workspace).load()
        if recovered_store["ok"]:
            recovered_engine = recovered_store["engine"]
            recovered_full = build_final_full(recovered_engine)
            kill_parity = recovered_engine.authoritative_result() == recovered_full.authoritative_result()

    injection_report = run_injection_suite(
        data_dir / "corpus.jsonl",
        data_dir / "model_params.json",
        data_dir / "injection_manifest.json",
        report_dir / "acceptance-injections",
    )

    report = {
        "ok": all([parity, snapshot_parity, budget_ok, kill_recovery_ok, kill_parity, injection_report["ok"]]),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "sample_count": incremental_result["sample_count"],
        "entity_count": len(incremental_result["entities"]),
        "suppression_count": len(incremental_result["suppressions"]),
        "repair_count": len(incremental_result["repairs"]),
        "type_distribution": incremental_result["type_distribution"],
        "parity": {
            "incremental_vs_blank_full": parity,
            "incremental_fingerprint": incremental_hash,
            "full_fingerprint": full_hash,
            "cache_key_map_equal": incremental_result["cache_key_map"] == full_result["cache_key_map"],
            "live_cache_key_count": len(incremental_result["live_cache_keys"]),
        },
        "commands": command_reports,
        "snapshot": {
            "roundtrip_parity": snapshot_parity,
            "snapshot_id": snapshot["snapshot_id"],
            "version_business_code": snapshot_code,
        },
        "budget": {
            "stale_ok": stale_ok,
            "stale_code": stale_response.get("code"),
            "stale_generation": stale_response.get("stale_generation"),
            "fresh_status": fresh_response.get("status"),
            "budget_ok": budget_ok,
        },
        "sigkill_recovery": {
            "kill_returncode": killed.returncode,
            "recovery_returncode": recovered_cli.returncode,
            "parity": kill_parity,
            "stderr_tail": killed.stderr[-300:],
        },
        "injections": {
            "ok": injection_report["ok"],
            "passed_count": injection_report["passed_count"],
            "injection_count": injection_report["injection_count"],
            "business_code_counts": injection_report["business_code_counts"],
        },
    }
    write_json(report_dir / "acceptance_report.json", report)

    markdown = [
        "# 验收运行结果",
        "",
        f"- 总状态：{'通过' if report['ok'] else '失败'}",
        f"- 端到端耗时：{report['elapsed_ms']} ms",
        f"- 样本/实体/抑制/修复：{report['sample_count']} / {report['entity_count']} / {report['suppression_count']} / {report['repair_count']}",
        f"- 增量与空白全量逐字段一致：{parity}（指纹 `{incremental_hash[:16]}`）",
        f"- 快照往返一致：{snapshot_parity}；版本探测码：`{snapshot_code}`",
        f"- 预算陈旧摘要：{budget_ok}（`{stale_response.get('code')}` → fresh）",
        f"- SIGKILL 后恢复并 parity：{kill_recovery_ok and kill_parity}（kill code {killed.returncode}）",
        f"- 异常注入：{injection_report['passed_count']}/{injection_report['injection_count']}，业务码 {json.dumps(injection_report['business_code_counts'], ensure_ascii=False)}",
        "",
        "## 命令级失效/复用",
        "",
        "| op | affected | invalidated | recomputed | reused |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in command_reports:
        markdown.append(f"| {item['op']} | {len(item['affected_samples'])} | {item['invalidated_count']} | {item['recomputed_count']} | {item['reused_count']} |")
    (report_dir / "acceptance_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
