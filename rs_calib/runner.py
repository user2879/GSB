"""端到端运行编排：增量批次运行 / 空白全量重建，产出确定性 JSON 诊断报告。"""
from __future__ import annotations

import json
import os
from typing import Any

from .edits import EditEngine, load_edit_plan
from .errors import E_SNAPSHOT_STALE, STAGE_REGRESS, StageError
from .fingerprint import content_fingerprint
from .injection import apply_injections, load_injection_config, mark_hits
from .loader import load_calibration_table, load_regression_config, load_tile_pack
from .pipeline import Pipeline
from .store import StateStore

def compare_reports(a: dict, b: dict) -> dict:
    keys = ("calibration_records", "regression_models", "alignment_records",
            "failure_list", "content_fingerprints")
    diffs = {}
    for key in keys:
        if a.get(key) != b.get(key):
            diffs[key] = "DIFFERENT"
    reuse_a = _normalize_reuse(a)
    reuse_b = _normalize_reuse(b)
    if reuse_a != reuse_b:
        diffs["reuse_invalidation_log"] = "DIFFERENT"
    return {"equal": not diffs, "different_fields": sorted(diffs)}


def _normalize_reuse(report: dict) -> list:
    return sorted(
        (r.get("stage"), r.get("tile_id", r.get("model_id")), r.get("band"),
         r.get("key"), r.get("action")) for r in report.get("reuse_invalidation_log", []))


STABLE_TILE_ORDER_KEY = lambda t: (t.tile_id, t.band)  # noqa: E731


def _read_jsonl_pack(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_regression_config(config: dict, table: dict) -> list[dict]:
    from .errors import (E_REG_SOURCE_EMPTY, E_REG_SOURCE_UNKNOWN,
                         STAGE_LOAD, StageError)
    failures = []
    whitelist = config.get("source_whitelist") or []
    if not whitelist:
        failures.append(StageError(E_REG_SOURCE_EMPTY, "回归源传感器白名单为空",
                                   stage=STAGE_LOAD).to_failure())
    for sid in whitelist:
        if sid not in table:
            failures.append(StageError(E_REG_SOURCE_UNKNOWN,
                                       f"回归源传感器 {sid} 不在定标表内",
                                       sensor_id=sid, stage=STAGE_LOAD).to_failure())
    ref = config.get("reference_sensor")
    if ref not in table:
        failures.append(StageError(E_REG_SOURCE_UNKNOWN,
                                   f"参考传感器 {ref} 不在定标表内",
                                   sensor_id=ref, stage=STAGE_LOAD).to_failure())
    return failures


def prepare_inputs(sample_dir: str, work_dir: str, injection_path: str | None):
    """读取固定种子样本，应用注入后落盘到工作目录，返回资源句柄。"""
    os.makedirs(work_dir, exist_ok=True)
    table, table_failures = load_calibration_table(os.path.join(sample_dir, "calibration_table.json"))
    config, config_failures = load_regression_config(
        os.path.join(sample_dir, "regression_config.yaml"), table)
    plan_batches = load_edit_plan(os.path.join(sample_dir, "edit_plan.json"))

    pack_files = sorted(f for f in os.listdir(sample_dir)
                        if f.endswith(".jsonl") and f.startswith("tiles_"))
    raw_packs = {name: _read_jsonl_pack(os.path.join(sample_dir, name)) for name in pack_files}

    injection_hits = []
    if injection_path and os.path.exists(injection_path):
        injections = load_injection_config(injection_path)
        raw_packs, table, config, plan_batches, injection_hits = apply_injections(
            raw_packs, table, config, plan_batches, injections)
        # 注入可能改变配置/定标表，直接对注入后的内存配置重新执行资源校验
        post_failures = _validate_regression_config(config, table)
        for failure in post_failures:
            if failure not in table_failures + config_failures:
                config_failures.append(failure)
    for name, rows in raw_packs.items():
        _write_jsonl(os.path.join(work_dir, name), rows)

    pack_paths = [os.path.join(work_dir, name) for name in sorted(raw_packs)]
    return {
        "table": table,
        "config": config,
        "plan_batches": plan_batches,
        "pack_paths": pack_paths,
        "load_failures": table_failures + config_failures,
        "injection_hits": injection_hits,
    }


def _build_report(store: StateStore, batch_summaries: list[dict], mode: str,
                  injection_hits: list[dict], snapshot_events: list[dict]) -> dict:
    calib = sorted(store.calib.values(), key=lambda r: (r["tile_id"], r["band"]))
    alignments = sorted(store.alignments.values(),
                        key=lambda r: (r["tile_id"], r["band"]))
    models = sorted(store.models.values(), key=lambda m: (m["source_sensor"], m["band"]))
    failures = Pipeline._sort_failures(store.failures)
    for failure in failures:
        failure.pop("_failure_key", None)
    ledger = sorted(store.ledger, key=lambda row: (row["batch_id"], row["sort_key"]))
    reuse = sorted(store.reuse_log, key=lambda row: (
        row.get("stage", ""), row.get("tile_id", row.get("model_id", "")),
        row.get("band", -1), row.get("key", "")))
    hits = mark_hits(injection_hits, store.failures)

    processed = sum(row["pixels"] for row in ledger if row["status"] == "PROCESSED")
    total_grant = sum(row["budget_grant"] for row in batch_summaries)
    return {
        "mode": mode,
        "global_version": store.global_version,
        "config_hash": store.config_hash,
        "table_hash": store.table_hash,
        "calibration_records": calib,
        "regression_models": models,
        "alignment_records": alignments,
        "failure_list": failures,
        "reuse_invalidation_log": reuse,
        "budget_ledger": {
            "entries": ledger,
            "total_budget_grant": total_grant,
            "total_processed_pixels": processed,
            "reconciled": processed <= total_grant,
            "pending_pixels": _pending_pixels(ledger),
        },
        "batch_summaries": batch_summaries,
        "injection_report": hits,
        "snapshot_events": snapshot_events,
        "content_fingerprints": {
            "calib_set": content_fingerprint(calib),
            "model_set": content_fingerprint(models),
            "alignment_set": content_fingerprint(alignments),
            "failure_set": content_fingerprint(failures),
        },
    }


def _pending_pixels(ledger: list[dict]) -> int:
    pending = [row for row in ledger if row["status"] == "PENDING"]
    return len({(row["batch_id"], row["tile_id"], row["band"]) for row in pending})


def run(sample_dir: str, output_dir: str, *, mode: str = "incremental",
        injection_path: str | None = None, resume: bool = False) -> dict:
    work_dir = os.path.join(output_dir, "_work")
    snapshot_dir = os.path.join(output_dir, "snapshot")
    os.makedirs(output_dir, exist_ok=True)
    inputs = prepare_inputs(sample_dir, work_dir, injection_path)
    table, config = inputs["table"], inputs["config"]

    store = StateStore(snapshot_dir, config, table)
    snapshot_events: list[dict] = []
    if resume:
        try:
            resumed = store.try_resume()
            snapshot_events.append({"event": "resume_checked", "resumed": resumed})
        except StageError as exc:
            if exc.code == E_SNAPSHOT_STALE:
                snapshot_events.append({"event": E_SNAPSHOT_STALE, "message": str(exc)})
                store.reset_for_rebuild()
            else:
                raise
    elif os.path.exists(snapshot_dir):
        # 非续跑启动：从空白状态重建，避免账本/产物重复累加
        store.reset_for_rebuild()

    # 初始瓦片（批次 0 的初始包）
    initial_packs = [p for p in inputs["pack_paths"]
                     if os.path.basename(p) == "tiles_batch0.jsonl"]
    tiles, load_failures = load_tile_pack(initial_packs, table)
    for failure in inputs["load_failures"] + load_failures:
        store.failures.append(failure)
    engine = EditEngine(table, config, tiles)
    pipeline = Pipeline(store, config, table)

    commit_interval = max(1, int(config.get("snapshot_commit_interval", 1)))
    batch_summaries = []
    if mode == "full":
        # 全量：先施加全部编辑，再从空白按最终集合一次性运行（批次仅作为预算轮次）
        all_edit_failures = []
        for batch in inputs["plan_batches"]:
            all_edit_failures.extend(engine.apply_batch(batch))
        for failure in all_edit_failures:
            pipeline._record_failure(failure)
        final_tiles = engine.tile_list()
        grant = int(config["pixel_budget"])
        round_id = 0
        while True:
            round_id += 1
            store.config = config
            store.table = table
            summary = pipeline.run_batch(round_id, final_tiles)
            batch_summaries.append(summary)
            store.commit(STAGE_REGRESS if not pipeline.pending else "calib",
                         summary["completed_tiles"])
            if not pipeline.pending:
                break
            pipeline.total_grant += grant
    else:
        for batch in inputs["plan_batches"]:
            edit_failures = engine.apply_batch(batch)
            for failure in edit_failures:
                pipeline._record_failure(failure)
            current_tiles = engine.tile_list()
            store.config = config
            store.table = table
            summary = pipeline.run_batch(int(batch["batch_id"]), current_tiles)
            batch_summaries.append(summary)
            if int(batch["batch_id"]) % commit_interval == 0 or not pipeline.pending:
                store.commit(STAGE_REGRESS if not pipeline.pending else "calib",
                             summary["completed_tiles"])
                snapshot_events.append({
                    "event": "commit", "batch_id": int(batch["batch_id"]),
                    "global_version": store.global_version,
                })
        # 所有编辑批次应用后，继续消耗后续预算轮次直至待办清空
        drain_round = int(batch["batch_id"])
        while pipeline.pending:
            drain_round += 1
            current_tiles = engine.tile_list()
            store.config = config
            store.table = table
            summary = pipeline.run_batch(drain_round, current_tiles)
            batch_summaries.append(summary)
            store.commit(STAGE_REGRESS if not pipeline.pending else "calib",
                         summary["completed_tiles"])
            snapshot_events.append({
                "event": "drain_commit", "round_id": drain_round,
                "global_version": store.global_version,
            })
        engine.mark_completed(engine.tile_list())

    report = _build_report(store, batch_summaries, mode,
                           inputs["injection_hits"], snapshot_events)
    report_path = os.path.join(output_dir, f"diagnostic_report_{mode}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return report
