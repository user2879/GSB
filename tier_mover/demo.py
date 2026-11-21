"""可重复演示：在独立工作区内跑通全部业务场景并汇总结果。

场景顺序固定、种子固定，输出结构化报告，供文档与验收复用。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .applog import StructuredLogger
from .engine import TierEngine
from .errors import TierMoverError
from .faults import (inject_capacity_overflow, inject_truncated_object,
                     inject_version_conflict)
from .ledger import Ledger
from .manifest import load_manifest
from .planner import derive_plan, split_batches
from .recovery import RecoveryService
from .replay import ReplayService
from .workspace import workspace_paths, write_sample_manifests


def _fresh(root: Path, seed: int, hot_capacity: int | None = None):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    paths = workspace_paths(root)
    logger = StructuredLogger(paths["logs"])
    manifest = write_sample_manifests(root, seed=seed, hot_capacity=hot_capacity)
    engine = TierEngine(root, logger=logger)
    engine.prepare_dirs(manifest)
    ledger = Ledger(paths["ledger"])
    ledger.register_spec(manifest, 1_000)
    engine.seed_objects(manifest)
    ledger.initialize_objects(manifest, 1_001)
    engine.reconcile_bytes(manifest, ledger)
    return paths, logger, manifest, engine, ledger


def _error_code(exc: TierMoverError) -> dict:
    return exc.to_dict()


def run_demo(base_dir: str | Path, seed: int = 20261122) -> dict:
    base = Path(base_dir)
    report: dict = {"seed": seed, "scenarios": {}}

    # 场景 1：计划确定性 + 真实搬运 + 批次预算
    root = base / "01_happy_path"
    paths, logger, manifest, engine, ledger = _fresh(root, seed)
    plan_a = derive_plan(manifest, ledger.placement())
    plan_b = derive_plan(manifest, ledger.placement())
    deterministic = plan_a.to_json() == plan_b.to_json()
    saved = engine.plan(ledger, manifest)
    batches = split_batches(saved, manifest.rules.batch_limit)
    conclusion = engine.execute(ledger, manifest, plan=saved)
    engine.reconcile_bytes(manifest, ledger)
    replay = ReplayService(ledger, manifest).run()
    report["scenarios"]["happy_path"] = {
        "plan_deterministic": deterministic,
        "plan_digest": saved.plan_digest,
        "steps": len(saved.steps),
        "batch_limit": manifest.rules.batch_limit,
        "batch_count": len(batches),
        "evict_keys": list(saved.evict_keys),
        "promote_keys": list(saved.promote_keys),
        "retained_keys": list(saved.retained_keys),
        "conclusion": conclusion,
        "replay": replay.to_dict(),
        "ledger_view": ledger.current_view(),
    }
    ledger.close()

    # 场景 2：E_CAPACITY 容量越界注入
    root = base / "02_capacity"
    paths, logger, manifest, engine, ledger = _fresh(root, seed)
    inject_capacity_overflow(root, logger)
    manifest_bad = load_manifest(paths["tiers"], paths["objects"], paths["seed"])
    try:
        engine.plan(ledger, manifest_bad)
        engine.execute(ledger, manifest_bad)
        capacity_result = {"raised": False}
    except TierMoverError as exc:
        capacity_result = {"raised": True, **_error_code(exc)}
    # 账本仍可恢复：原账本视图完好
    capacity_result["recoverable_view"] = ledger.current_view()
    report["scenarios"]["capacity_injection"] = capacity_result
    ledger.close()

    # 场景 3：E_INTEGRITY 截断注入
    root = base / "03_integrity"
    paths, logger, manifest, engine, ledger = _fresh(root, seed)
    victim = inject_truncated_object(root, logger)
    try:
        engine.reconcile_bytes(manifest, ledger)
        integrity_result = {"raised": False}
    except TierMoverError as exc:
        integrity_result = {"raised": True, **_error_code(exc)}
    integrity_result["victim"] = str(victim)
    integrity_result["ledger_reopen_ok"] = bool(Ledger(paths["ledger"]).current_view())
    report["scenarios"]["integrity_injection"] = integrity_result
    ledger.close()

    # 场景 4：E_VERSION 账本版本冲突注入
    root = base / "04_version"
    paths, logger, manifest, engine, ledger = _fresh(root, seed)
    conflict_db = inject_version_conflict(root, logger)
    try:
        Ledger(conflict_db)
        version_result = {"raised": False}
    except TierMoverError as exc:
        version_result = {"raised": True, **_error_code(exc)}
    version_result["original_ledger_ok"] = bool(
        Ledger(paths["ledger"]).current_view())
    report["scenarios"]["version_injection"] = version_result
    ledger.close()

    # 场景 5：强制终止 -> 未知标注 -> rollback 恢复
    root = base / "05_crash_rollback"
    paths, logger, manifest, engine, ledger = _fresh(root, seed)
    crash_key = derive_plan(manifest, ledger.placement()).steps[0].key
    try:
        plan = engine.plan(ledger, manifest)
        engine.execute(ledger, manifest, plan=plan, crash_after_key=crash_key)
    except SystemExit:
        pass
    ledger.close()
    ledger2 = Ledger(paths["ledger"])
    engine2 = TierEngine(root, logger=logger)
    recovery = RecoveryService(engine2, ledger2, manifest, logger)
    inspected = recovery.inspect()
    rollback_report = recovery.recover(mode="rollback")
    replay_after = ReplayService(ledger2, manifest).run()
    report["scenarios"]["crash_rollback"] = {
        "crash_key": crash_key,
        "inspect": inspected.to_dict(),
        "recovery": rollback_report.to_dict(),
        "replay": replay_after.to_dict(),
    }
    ledger2.close()

    # 场景 6：强制终止 -> rebuild 恢复
    root = base / "06_crash_rebuild"
    paths, logger, manifest, engine, ledger = _fresh(root, seed)
    try:
        plan = engine.plan(ledger, manifest)
        engine.execute(ledger, manifest, plan=plan, crash_after_key=crash_key)
    except SystemExit:
        pass
    ledger.close()
    ledger3 = Ledger(paths["ledger"])
    engine3 = TierEngine(root, logger=logger)
    recovery3 = RecoveryService(engine3, ledger3, manifest, logger)
    rebuild_report = recovery3.recover(mode="rebuild")
    report["scenarios"]["crash_rebuild"] = {
        "crash_key": crash_key,
        "recovery": rebuild_report.to_dict(),
        "view": ledger3.current_view(),
    }
    ledger3.close()

    # 场景 7：策略版本演进与 stale 失效传播
    root = base / "07_versioning"
    paths, logger, manifest, engine, ledger = _fresh(root, seed)
    plan_v1 = engine.plan(ledger, manifest)
    doc = json.loads(paths["tiers"].read_text(encoding="utf-8"))
    doc["spec_version"] = 2
    doc["rules"]["batch_limit"] = 3
    paths["tiers"].write_text(json.dumps(doc, indent=2), encoding="utf-8")
    manifest_v2 = load_manifest(paths["tiers"], paths["objects"], paths["seed"])
    new_version = ledger.register_spec(manifest_v2, 2_000)
    stale_fragments = ledger.superseded_plan_fragments()
    plan_v2 = engine.plan(ledger, manifest_v2)
    conclusion_v2 = engine.execute(ledger, manifest_v2, plan=plan_v2)
    report["scenarios"]["version_evolution"] = {
        "v1": ledger.spec_blob(1)["spec_digest"],
        "v2": new_version,
        "superseded_plan_fragments": stale_fragments,
        "v2_conclusion": conclusion_v2,
        "current_view": ledger.current_view(),
        "active_conclusion": ledger.active_conclusion(),
    }
    ledger.close()

    report["all_business_codes"] = {
        "E_CAPACITY": report["scenarios"]["capacity_injection"].get("code"),
        "E_INTEGRITY": report["scenarios"]["integrity_injection"].get("code"),
        "E_VERSION": report["scenarios"]["version_injection"].get("code"),
    }
    out = base / "demo_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    return report
