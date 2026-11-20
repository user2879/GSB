"""编排层: prepare / run / verify / inject / acceptance."""

import json
import os
import shutil

from . import records as records_mod
from .canon import canonical_json
from .consistency import compare_views
from .errors import E_CHAIN, E_INTEGRITY, E_UPGRADE, LedgerError
from .ledger import Ledger
from .logging_util import StructuredLogger
from .normalize import normalize_record, order_views, view_digest
from .pipeline import Pipeline
from .snapshot import (list_snapshots, load_latest, replay_snapshot,
                       verify_snapshot, write_snapshot)
from .upgrade import upgrade

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "samples", "config.json")


def _paths(ws):
    rt = os.path.join(ws, "runtime")
    return {
        "samples": os.path.join(ws, "samples"),
        "runtime": rt,
        "backup": os.path.join(rt, "backup"),
        "snapshots": os.path.join(rt, "snapshots"),
        "logs": os.path.join(rt, "logs"),
        "checkpoint": os.path.join(rt, "checkpoint.json"),
        "trail": os.path.join(rt, "upgrade_trail.json"),
        "report": os.path.join(rt, "run_report.json"),
        "ledger_mixed": os.path.join(rt, "ledger_mixed.db"),
        "ledger_r1": os.path.join(rt, "ledger_r1.db"),
        "ledger_r2": os.path.join(rt, "ledger_r2.db"),
        "ledger_r3": os.path.join(rt, "ledger_r3.db"),
    }


def ensure_config(ws):
    dst = os.path.join(ws, "samples", "config.json")
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(DEFAULT_CONFIG, dst)
    with open(dst, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_ledger(path, records):
    if os.path.exists(path):
        os.remove(path)
    ledger = Ledger(path)
    for rec in records:
        ledger.insert(rec["request_id"], rec["fmt"], rec)
    ledger.close()


def _mixed_formats(pipeline):
    """按到达序循环分配 r1/r2/r3; 含被覆盖引用的请求强制为 r3."""
    views = pipeline.ordered_views()
    cycle = ["r1", "r2", "r3"]
    fmts = {}
    for i, view in enumerate(views):
        fmts[view["request_id"]] = cycle[i % 3]
    for view in views:
        if view["superseded_refs"]:
            fmts[view["request_id"]] = "r3"
    return fmts


def cmd_prepare(ws):
    config = ensure_config(ws)
    pipeline = Pipeline(config)
    p = _paths(ws)
    if os.path.exists(p["runtime"]):
        shutil.rmtree(p["runtime"])
    for key in ("backup", "snapshots", "logs"):
        os.makedirs(p[key], exist_ok=True)

    views = pipeline.ordered_views()
    samples = {}
    for fmt in ("r1", "r2", "r3"):
        samples[fmt] = [records_mod.encode(fmt, v) for v in views]
        _write_json(os.path.join(p["samples"], f"records_{fmt}.json"),
                    {"records": samples[fmt]})
    fmts = _mixed_formats(pipeline)
    mixed = [records_mod.encode(fmts[v["request_id"]], v) for v in views]
    _write_json(os.path.join(p["samples"], "records_mixed.json"), {"records": mixed})
    corrupt = dict(samples["r3"][0])
    corrupt["output_digest"] = "0" * 64
    _write_json(os.path.join(p["samples"], "records_corrupt.json"),
                {"records": [corrupt]})

    _build_ledger(p["ledger_mixed"], mixed)
    for fmt in ("r1", "r2", "r3"):
        _build_ledger(p[f"ledger_{fmt}"], samples[fmt])
    summary = {
        "records": len(views),
        "formats_in_mixed_ledger": sorted(set(fmts.values())),
        "input_digest": pipeline.input_digest,
        "param_digest": pipeline.param_digest,
        "backend_fingerprint": pipeline.backend_fingerprint,
    }
    print(canonical_json(summary))
    return summary


def _normalize_ledger(ledger, pipeline):
    views, metas = [], {}
    for record in ledger.decode_all(pipeline.supported_slots):
        view, meta = normalize_record(record, pipeline)
        views.append(view)
        metas[view["request_id"]] = meta
    return views, metas


def _load_trail(path):
    if os.path.exists(path):
        return _read_json(path)
    return []


def cmd_run(ws, max_upgrades=None):
    config = ensure_config(ws)
    pipeline = Pipeline(config)
    p = _paths(ws)
    logger = StructuredLogger(os.path.join(p["logs"], "ledger.jsonl"))
    try:
        ledger = Ledger(p["ledger_mixed"])
        views, metas = _normalize_ledger(ledger, pipeline)
        mixed_digest = view_digest(views)
        per_format = []
        for fmt in ("r1", "r2", "r3"):
            for record in ledger.decode_all(pipeline.supported_slots, fmt=fmt):
                per_format.append(normalize_record(record, pipeline)[0])
        if view_digest(per_format) != mixed_digest:
            raise LedgerError(E_INTEGRITY,
                              "mixed-order read and per-format read diverge",
                              stage="read")

        trail_new, events = upgrade(ledger, pipeline, p["backup"],
                                    p["checkpoint"], logger,
                                    max_upgrades=max_upgrades)
        trail = _load_trail(p["trail"]) + trail_new
        _write_json(p["trail"], trail)

        consistency = _run_consistency(ws, pipeline, p, logger)

        final_views, _ = _normalize_ledger(ledger, pipeline)
        cursor = len(_load_trail(p["trail"]))
        snap_path, snap = write_snapshot(p["snapshots"], ledger, final_views,
                                         trail, cursor)
        report = {
            "status": "ok",
            "record_count": len(final_views),
            "view_digest": view_digest(final_views),
            "records": [
                {
                    "request_id": v["request_id"],
                    "fmt": metas[v["request_id"]]["fmt"],
                    "arrival_seq": v["arrival_seq"],
                    "batch_slot": v["batch_slot"],
                    "group_rank": v["group_rank"],
                    "filled": metas[v["request_id"]]["filled"],
                }
                for v in order_views(final_views)
            ],
            "upgrade_trail": trail,
            "upgrade_events": events,
            "consistency": consistency,
            "snapshot": {"path": snap_path, "view_digest": snap["view_digest"]},
            "snapshots": list_snapshots(p["snapshots"]),
        }
        _write_json(p["report"], report)
        ledger.close()
        print(canonical_json({"status": "ok", "records": len(final_views),
                              "view_digest": report["view_digest"],
                              "upgraded": len(trail_new),
                              "events": events,
                              "consistency": consistency}))
        return 0
    except LedgerError as exc:
        logger.event(exc.code, stage=exc.stage or "run", record_id=exc.record_id,
                     field=exc.field, message=exc.message, **exc.extra)
        print(canonical_json({"status": "error", **exc.to_dict()}))
        return 2


def _run_consistency(ws, pipeline, p, logger):
    direct = Ledger(p["ledger_r3"])
    direct_views, _ = _normalize_ledger(direct, pipeline)
    direct_map = {v["request_id"]: v for v in direct_views}
    result = {}
    for label in ("r1", "r2"):
        led = Ledger(p[f"ledger_{label}"])
        upgrade(led, pipeline,
                os.path.join(p["runtime"], f"backup_{label}"),
                os.path.join(p["runtime"], f"checkpoint_{label}.json"),
                logger)
        views, _ = _normalize_ledger(led, pipeline)
        cand_map = {v["request_id"]: v for v in views}
        digest, count = compare_views(direct_map, cand_map, label)
        result[label] = {"digest": digest, "count": count}
        led.close()
    direct.close()
    return {"status": "ok", "digest": result["r1"]["digest"],
            "count": result["r1"]["count"],
            "paths": {"r1->r3": result["r1"], "r2->r3": result["r2"]}}


def cmd_verify(ws):
    config = ensure_config(ws)
    pipeline = Pipeline(config)
    p = _paths(ws)
    logger = StructuredLogger(os.path.join(p["logs"], "ledger.jsonl"))
    try:
        ledger = Ledger(p["ledger_mixed"])
        snap_path, snap = load_latest(p["snapshots"])
        verify_snapshot(snap, ledger)
        count = replay_snapshot(snap, pipeline)
        ledger.close()
        print(canonical_json({"status": "ok", "snapshot": snap_path,
                              "replayed": count}))
        return 0
    except LedgerError as exc:
        logger.event(exc.code, stage=exc.stage or "verify",
                     record_id=exc.record_id, field=exc.field,
                     message=exc.message, **exc.extra)
        print(canonical_json({"status": "error", **exc.to_dict()}))
        return 2


def _rebuild_mixed_ledger(ws):
    p = _paths(ws)
    mixed = _read_json(os.path.join(p["samples"], "records_mixed.json"))["records"]
    _build_ledger(p["ledger_mixed"], mixed)


def cmd_inject(ws, kind):
    """注入异常: 仅改变输入样本文件或参数文件."""
    cmd_prepare(ws)
    p = _paths(ws)
    sample_path = os.path.join(p["samples"], "records_mixed.json")
    data = _read_json(sample_path)
    if kind == "digest":
        rec = data["records"][0]
        d = rec["output_digest"]
        rec["output_digest"] = ("0" if d[0] != "0" else "1") + d[1:]
        _write_json(sample_path, data)
        _rebuild_mixed_ledger(ws)
        print(canonical_json({"injected": "digest-corruption",
                              "record_id": rec["request_id"],
                              "expect": E_INTEGRITY}))
    elif kind == "chain":
        target = next(r for r in data["records"]
                      if r["fmt"] == "r3" and r["superseded_refs"])
        dropped = target["superseded_refs"].pop()
        _write_json(sample_path, data)
        _rebuild_mixed_ledger(ws)
        print(canonical_json({"injected": "missing-superseded-ref",
                              "record_id": target["request_id"],
                              "dropped_ref": dropped,
                              "expect": E_CHAIN}))
    elif kind == "checkpoint":
        ledger = Ledger(p["ledger_mixed"])
        pending = [r["request_id"] for r in ledger.read_all() if r["fmt"] != "r3"]
        ledger.close()
        _write_json(p["checkpoint"],
                    {"processed": pending[:2], "cursor": 2})
        print(canonical_json({"injected": "checkpoint-mismatch",
                              "claimed_processed": pending[:2],
                              "expect": E_UPGRADE}))
    else:
        raise ValueError(f"unknown injection kind {kind!r}")
    return 0


def cmd_acceptance(ws):
    """完整验收序列: 混存归一、升级一致、快照重放、三类注入."""
    results = []

    def step(name, fn, expect_code=0):
        code = fn()
        ok = code == expect_code
        results.append({"step": name, "ok": ok, "exit_code": code})
        return code

    step("prepare", lambda: 0 if cmd_prepare(ws) else 1)
    step("run", lambda: cmd_run(ws))
    step("verify", lambda: cmd_verify(ws))
    step("inject-digest", lambda: cmd_inject(ws, "digest"))
    step("run-expect-E_INTEGRITY", lambda: cmd_run(ws), expect_code=2)
    step("inject-chain", lambda: cmd_inject(ws, "chain"))
    step("run-expect-E_CHAIN", lambda: cmd_run(ws), expect_code=2)
    step("inject-checkpoint", lambda: cmd_inject(ws, "checkpoint"))
    step("run-expect-E_UPGRADE-event", lambda: cmd_run(ws))
    step("final-prepare", lambda: 0 if cmd_prepare(ws) else 1)
    step("final-run", lambda: cmd_run(ws))
    step("final-verify", lambda: cmd_verify(ws))
    ok = all(r["ok"] for r in results)
    print(canonical_json({"acceptance": "PASS" if ok else "FAIL",
                          "steps": results}))
    return 0 if ok else 1
