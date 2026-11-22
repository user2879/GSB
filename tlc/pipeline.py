"""单批运行编排：加载校验 -> 基座替换 -> 分阶段计算 -> 快照提交 -> JSON 报告。

阶段划分：
  A_forward  共享前向与贡献追踪（对存量任务计算当前基座下的贡献份额）
  B_decide   失效边界判定（贡献偏移 vs 阈值，禁止引用即失效）
  C_train    适配器并发训练（预算不足保留待办）
  D_shares   全任务最终贡献份额与闭合校验
  E_eval     验证集评估（待办任务不写入验证汇总）
"""

import json
import os
import time

from . import adapter as adapter_mod
from .encoder import contribution_shares, normalized_shares
from .errors import BizError, E_CLOSURE_VIOLATION, E_RETRAIN_BUDGET_PENDING
from .hashing import sha256_hex
from .invalidation import decide
from .loaders import (
    apply_replacements,
    fingerprint_features,
    load_config,
    load_contributions,
    load_encoder,
    load_features,
    load_replacement_plan,
    load_tasks,
    validate_task_samples,
)
from .model import Snapshot
from .scheduler import plan_training, stable_key, train_all
from .state import Staging, compute_run_key, load_snapshot


class CrashError(BaseException):
    """测试钩子：模拟强制终止。"""


def _crash_hook(stage):
    if os.environ.get("TLC_CRASH_AFTER_STAGE") == stage:
        if os.environ.get("TLC_CRASH_MODE") == "exit":
            os._exit(1)
        raise CrashError(f"simulated crash after stage {stage}")


def _stage_delay():
    ms = int(os.environ.get("TLC_STAGE_DELAY_MS", "0"))
    if ms:
        time.sleep(ms / 1000.0)


def _batch_dir(data_dir, batch_no):
    return os.path.join(data_dir, "batches", f"batch_{batch_no}")


def _opt(path):
    return path if os.path.isfile(path) else None


def load_run_inputs(data_dir, batch_no, prev: Snapshot):
    """加载并校验本批全部输入，返回命名元组式字典。"""
    bdir = _batch_dir(data_dir, batch_no)
    cfg = load_config(
        os.path.join(data_dir, "config.json"),
        overrides=None
        if not _opt(os.path.join(bdir, "config.json"))
        else __import__("json").load(open(os.path.join(bdir, "config.json"))),
    )
    tasks = load_tasks(_opt(os.path.join(bdir, "tasks.json")) or os.path.join(data_dir, "tasks.json"))
    bank = load_features(
        _opt(os.path.join(bdir, "features.jsonl")) or os.path.join(data_dir, "features.jsonl")
    )
    features_fp = fingerprint_features(bank)
    validate_task_samples(tasks, bank)

    encoder = prev.encoder if prev.encoder is not None else load_encoder(
        os.path.join(data_dir, "encoder.json")
    )
    contributions = load_contributions(
        os.path.join(data_dir, "contributions.json"), encoder, tasks
    )
    if batch_no >= 1:
        plan = load_replacement_plan(os.path.join(data_dir, "replacement_plan.json"))
        if batch_no > len(plan):
            raise BizError("E_PLAN_INVALID", f"替换计划不包含第 {batch_no} 批")
        encoder = apply_replacements(encoder, plan[batch_no - 1], contributions)
    for task in tasks:
        for layer in task.target_layers:
            if layer not in encoder.layer_index():
                raise BizError(
                    "E_TASK_INVALID", f"任务 {task.name} 目标层 {layer} 不在编码器中"
                )
    return {
        "cfg": cfg,
        "tasks": tasks,
        "bank": bank,
        "features_fp": features_fp,
        "encoder": encoder,
        "contributions": contributions,
    }


def build_manifest(cfg, encoder, tasks, features_fp, batch_no, prev: Snapshot):
    manifest = {
        "run_index": batch_no,
        "config": cfg.to_dict(),
        "config_hash": sha256_hex(cfg.to_dict()),
        "base": encoder.to_dict(),
        "base_fingerprint": sha256_hex(encoder.to_dict()),
        "base_layers": [
            {"name": l.name, "version": l.version, "checksum": l.checksum}
            for l in encoder.layers
        ],
        "tasks_fingerprint": sha256_hex([t.to_dict() for t in tasks]),
        "features_fingerprint": features_fp,
        "prev_run_key": prev.manifest["run_key"] if prev.manifest else None,
    }
    return manifest


def _task_xy(task, bank):
    import numpy as np

    ids = sorted(task.train_ids)
    X = np.asarray([bank.samples[s]["features"] for s in ids], dtype=np.float64)
    pos = {c: i for i, c in enumerate(task.classes)}
    Y = np.asarray([pos[bank.samples[s]["labels"][task.name]] for s in ids], dtype=np.int64)
    return X, Y


def run_batch(data_dir, batch_no, state_dir):
    """执行一批运行，返回报告字典（可能抛出 BizError / CrashError）。"""
    prev = load_snapshot(state_dir)
    inp = load_run_inputs(data_dir, batch_no, prev)
    cfg, tasks, bank = inp["cfg"], inp["tasks"], inp["bank"]
    encoder, contributions = inp["encoder"], inp["contributions"]
    features_fp = inp["features_fp"]

    manifest = build_manifest(cfg, encoder, tasks, features_fp, batch_no, prev)
    run_key = compute_run_key(manifest)
    staging = Staging(state_dir, run_key, manifest)

    task_map = {t.name: t for t in tasks}
    ordered_tasks = sorted(tasks, key=stable_key)
    prev_pending = [n for n in prev.pending if n in task_map]

    # ---- 阶段 A：共享前向与贡献追踪（存量且内容未变任务） ----
    _stage_delay()
    if staging.stage_done("A_forward"):
        stage_a = staging.load_stage("A_forward")
    else:
        stage_a = {}
        for task in ordered_tasks:
            art = prev.adapters.get(task.name)
            if art is None:
                continue
            if art["content_key"] != adapter_mod.content_key(task, cfg, features_fp):
                continue
            X, Y = _task_xy(task, bank)
            shares, loss = contribution_shares(
                encoder, task, art["params"], X, Y, contributions,
                lambda p, Zm, Ym: adapter_mod.loss_and_dz(task, p, Zm, Ym),
            )
            stage_a[task.name] = {"shares": shares, "loss": loss}
        staging.save_stage("A_forward", stage_a)
    _crash_hook("A_forward")

    # ---- 阶段 B：失效边界判定 ----
    _stage_delay()
    if staging.stage_done("B_decide"):
        stage_b = staging.load_stage("B_decide")
    else:
        stage_b = {}
        for task in ordered_tasks:
            art = prev.adapters.get(task.name)
            if task.name in prev_pending:
                # 上批待办任务：始终进入训练候选，仍计算贡献偏移诊断
                detail = None
                if art is not None and task.name in stage_a:
                    detail = _shift_detail(prev.last_shares.get(task.name, {}),
                                           stage_a[task.name]["shares"], cfg)
                stage_b[task.name] = {"decision": "retrain",
                                      "reason": "pending_retry", "detail": detail}
                continue
            current = stage_a.get(task.name, {}).get("shares")
            decision, detail = decide(
                task, art, prev.last_shares.get(task.name, {}),
                current if current is not None else {},
                cfg, features_fp,
            )
            stage_b[task.name] = {"decision": decision, "detail": detail}
        staging.save_stage("B_decide", stage_b)
    _crash_hook("B_decide")

    # ---- 阶段 C：适配器训练（预算 + 并发） ----
    _stage_delay()
    if staging.stage_done("C_train"):
        stage_c = staging.load_stage("C_train")
    else:
        candidates = []
        for name in sorted(prev_pending, key=lambda n: task_map[n].sort_key()):
            candidates.append((task_map[name], "pending_retry"))
        for task in ordered_tasks:
            if task.name in prev_pending:
                continue
            d = stage_b[task.name]["decision"]
            if d in ("new", "retrain"):
                candidates.append((task, d))
        to_train, pending_entries = plan_training(candidates, cfg)
        trained = train_all(to_train, encoder, bank, cfg, contributions, features_fp)
        stage_c = {"trained": trained, "pending": pending_entries}
        staging.save_stage("C_train", stage_c)
    _crash_hook("C_train")

    # ---- 阶段 D：最终贡献份额与闭合校验 ----
    _stage_delay()
    if staging.stage_done("D_shares"):
        stage_d = staging.load_stage("D_shares")
    else:
        stage_d = {}
        for task in ordered_tasks:
            name = task.name
            if name in stage_c["trained"]:
                art = stage_c["trained"][name]
                shares, loss = art["shares"], art["shares_loss"]
            elif name in stage_a:
                shares, loss = stage_a[name]["shares"], stage_a[name]["loss"]
            else:
                continue
            total = 0.0
            for layer in sorted(shares):
                total += shares[layer]
            rel = abs(total - loss) / max(abs(loss), 1e-12)
            stage_d[name] = {"shares": shares, "loss": loss, "closure_rel_err": rel}
        staging.save_stage("D_shares", stage_d)
    _crash_hook("D_shares")

    # ---- 阶段 E：验证评估（待办任务除外） ----
    _stage_delay()
    if staging.stage_done("E_eval"):
        stage_e = staging.load_stage("E_eval")
    else:
        pending_names = {e["task"] for e in stage_c["pending"]}
        stage_e = {}
        for task in ordered_tasks:
            name = task.name
            if name in pending_names:
                continue
            art = stage_c["trained"].get(name) or prev.adapters.get(name)
            if art is None:
                continue
            stage_e[name] = adapter_mod.evaluate(task, encoder, bank, art["params"])
        staging.save_stage("E_eval", stage_e)
    _crash_hook("E_eval")

    # ---- 组装新快照并整体替换提交 ----
    pending_names = sorted(
        (e["task"] for e in stage_c["pending"]),
        key=lambda n: task_map[n].sort_key(),
    )
    adapters, last_shares = {}, {}
    for task in ordered_tasks:
        name = task.name
        art = stage_c["trained"].get(name) or prev.adapters.get(name)
        if art is not None:
            adapters[name] = art
        if name in stage_d:
            last_shares[name] = stage_d[name]["shares"]
    snapshot_data = {
        "adapters": adapters,
        "last_shares": last_shares,
        "pending": pending_names,
    }
    staging.commit(snapshot_data)

    return _assemble_report(
        batch_no, run_key, manifest, ordered_tasks,
        prev, stage_a, stage_b, stage_c, stage_d, stage_e, cfg,
    )


def _shift_detail(prev_shares, current_shares, cfg):
    before = normalized_shares(prev_shares)
    after = normalized_shares(current_shares)
    compared, max_shift = [], 0.0
    for layer in sorted(before):
        shift = abs(after.get(layer, 0.0) - before[layer])
        compared.append({"layer": layer, "previous": before[layer],
                         "current": after.get(layer, 0.0), "shift": shift})
        max_shift = max(max_shift, shift)
    return {"reason": "base_shift", "compared": compared, "max_shift": max_shift,
            "threshold": cfg.contribution_shift_threshold}


def _assemble_report(batch_no, run_key, manifest, ordered_tasks, prev,
                     stage_a, stage_b, stage_c, stage_d, stage_e, cfg):
    codes = set()
    task_entries = []
    summary = {"new": [], "retrained": [], "reused": [], "pending": [], "removed": []}
    pending_names = {e["task"] for e in stage_c["pending"]}
    pending_reasons = {e["task"]: e for e in stage_c["pending"]}

    for task in ordered_tasks:
        name = task.name
        info = stage_b[name]
        decision = info["decision"]
        trained_art = stage_c["trained"].get(name)
        art = trained_art or prev.adapters.get(name)
        if name in pending_names:
            decision_out = "pending"
            codes.add(E_RETRAIN_BUDGET_PENDING)
            summary["pending"].append(name)
        elif trained_art is not None:
            decision_out = "new" if decision == "new" else "retrain"
            summary["new" if decision == "new" else "retrained"].append(name)
        else:
            decision_out = decision
            summary["reused" if decision == "reuse" else decision].append(name)

        entry = {
            "task": name,
            "sort_key": [task.name, list(task.target_layers)],
            "decision": decision_out,
            "invalidation": info.get("detail"),
            "adapter": None,
            "train": None,
            "metrics": stage_e.get(name),
            "contribution_shares": stage_d.get(name, {}).get("shares"),
            "closure": None,
        }
        if name in pending_names:
            entry["pending"] = pending_reasons[name]
        if art is not None:
            entry["adapter"] = {
                "type": art["adapter"]["type"],
                "rank": art["adapter"]["rank"],
                "scale": art["adapter"]["scale"],
                "param_checksum": art["param_checksum"],
                "base_layer_versions": art["base_layer_versions"],
            }
        if trained_art is not None:
            entry["train"] = {
                "epoch_losses": trained_art["train_losses"],
                "cost": trained_art["train_cost"],
                "budget": task.retrain_budget,
            }
        if name in stage_d:
            rec = stage_d[name]
            entry["closure"] = {
                "loss": rec["loss"],
                "share_sum": sum(rec["shares"][k] for k in sorted(rec["shares"])),
                "rel_err": rec["closure_rel_err"],
            }
            if rec["closure_rel_err"] > cfg.closure_rel_tol:
                codes.add(E_CLOSURE_VIOLATION)
        task_entries.append(entry)

    current_names = {t.name for t in ordered_tasks}
    summary["removed"] = sorted(
        (n for n in prev.adapters if n not in current_names), key=str
    )

    return {
        "batch": batch_no,
        "run_key": run_key,
        "config": manifest["config"],
        "config_hash": manifest["config_hash"],
        "base_layers": manifest["base_layers"],
        "base_fingerprint": manifest["base_fingerprint"],
        "tasks_fingerprint": manifest["tasks_fingerprint"],
        "features_fingerprint": manifest["features_fingerprint"],
        "tasks": task_entries,
        "summary": summary,
        "codes": sorted(codes),
    }


def rebuild(data_dir, batches, state_dir):
    """从空白状态按批次序列全量重放（每批全量计算所有阶段）。"""
    report = None
    for k in range(0, batches + 1):
        report = run_batch(data_dir, k, state_dir)
    return report
