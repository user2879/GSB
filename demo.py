"""Headless acceptance demo: runs every business requirement against the local
deterministic samples and writes artifacts/run_record.md."""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shardbench.engine import WorkbenchEngine
from shardbench.errors import BizError
from shardbench.leases import LeaseManager

WORKDIR = "demo_workspace"
REPORT = os.path.join("artifacts", "run_record.md")
CFG_V1 = "samples/sharding_config.json"
CFG_V2 = "samples/sharding_config_v2.json"

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def section(title):
    say()
    say(f"## {title}")
    say()


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs("artifacts", exist_ok=True)
    eng = WorkbenchEngine(WORKDIR, global_limit=3, per_db_limit=1,
                          lease_ttl_ms=1_500)

    say("# 分库分表路由工作台 - 验收运行记录")
    say()
    say(f"- 运行时间(本地): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"- 工作目录: {WORKDIR}/ (SQLite 账本 + 快照 + 结构化日志)")
    say("- 样本: samples/sharding_config.json, samples/predicates.json (固定本地样本)")

    section("1. 配置加载与强校验")
    eng.load_config(CFG_V1)
    v = eng.versions.current()
    say(f"- 配置加载成功: rule_v={v.rule_version}, ring_v={v.ring_version}, "
        f"dep_graph_v={v.dep_graph_version}, budget_v={v.budget_version}")
    say(f"- 路由单元: {sorted(eng.config.units)}")
    bad = os.path.join(WORKDIR, "corrupted.json")
    shutil.copy(CFG_V1, bad)
    err = eng.inject_config_corruption(bad, mode="invalid_json")
    say(f"- 异常注入(配置损坏): code={err.code}, field={err.field}, "
        f"message={err.message} -> 整体拒绝, 未进入探查")

    section("2. 谓词规范化与路由计划 (三种算法)")
    preds = json.load(open("samples/predicates.json", encoding="utf-8"))["predicates"]
    plans = {}
    for p in preds:
        plan = eng.compile(p["unit"], p["where"])
        plans[p["id"]] = plan
        targets = ", ".join(f"db{t['db']}.t{t['table']}" for t in plan.targets)
        say(f"- {p['id']} [{p['unit']}] -> {targets}")
        say(f"    digest={plan.plan_digest[:40]}... rule_v={plan.rule_version} ring_v={plan.ring_version}")
        say(f"    reason[0]={plan.reasons[0]}")
    p_dup = eng.compile("orders_by_user",
                        {"and": [{"eq": ["status", " PAID "]},
                                 {"in": ["user_id", [1001, 42, 7, 7]]}]})
    say(f"- 语义等价谓词(大小写/空白/IN顺序/AND顺序不同) digest 一致: "
        f"{p_dup.plan_digest == plans['p_user_in'].plan_digest}")

    section("3. 并发租约: 签发 / 续约 / 版本冲突 / 真实过期")
    lease = eng.lease_manager.acquire(db=0)
    say(f"- 签发: {lease.lease_id} db=0 granted={lease.granted_ms}ms "
        f"expires={lease.expires_ms}ms version={lease.version}")
    eng.lease_manager.renew(lease.lease_id, expected_version=1)
    say(f"- 续约成功: version -> {lease.version}")
    eng.inject_lease_version_conflict(lease.lease_id)
    try:
        eng.lease_manager.renew(lease.lease_id, expected_version=lease.version - 1)
    except BizError as exc:
        say(f"- 异常注入(租约版本冲突): code={exc.code}, detail={exc.detail}, 租约未延长")
    short = LeaseManager(ttl_ms=40)
    l2 = short.acquire(db=1)
    time.sleep(0.08)  # 真实单调时钟等待到期
    reclaimed = short.reap()
    say(f"- 真实过期: {l2.lease_id} state={l2.state}, 回收依据={l2.reclaim_reason}, "
        f"reaped={len(reclaimed)}")

    section("4. 双重配额并行探查调度")
    batch_id, records = eng.probe(list(plans.values()))
    stats = eng.scheduler.stats()
    say(f"- 批次 batch_id={batch_id}, 探查任务={len(records)} 全部落账")
    say(f"- 全局并发上限=3, 实际峰值={stats['max_observed_global']}; "
        f"每库配额=1, 实际峰值={stats['max_observed_per_db']}")
    waits = [r.wait_ms for r in records]
    say(f"- 等待窗口(单调时钟ms): min={min(waits)} max={max(waits)}; "
        f"每任务租约占用与耗时已写入账本")
    summary = eng.ledger.summary()
    say(f"- 账本汇总(仅当前版本): {summary}")

    section("5. 规则演进 -> 账本整体替换")
    eng.load_config(CFG_V2)
    say(f"- 版本单调递增: rule_v={eng.versions.current().rule_version}")
    say(f"- 路由单元差异: {eng.last_diff}")
    say(f"- 失效传播(稳定拓扑序): {eng.last_invalidation['invalidated']}")
    for step in eng.last_invalidation["steps"]:
        say(f"    {step['from']} -> {step['to']}: {step['reason']}")
    say(f"- 旧版本账本行已整体标记陈旧, 当前视图任务数={eng.ledger.summary()['tasks']}")
    say(f"- 计划缓存已整体清空: {eng.plan_cache == {}}")

    section("6. 仅失效子集重算, 未受影响单元逐字节不变")
    before_amount = eng.compile("orders_by_amount", {"eq": ["amount", 500]}).plan_digest
    result = eng.invalidate(
        ["orders_by_user"],
        {"orders_by_user": [{"eq": ["user_id", 42]}],
         "order_user_join": [{"eq": ["user_id", 42]}]})
    after_amount = eng.compile("orders_by_amount", {"eq": ["amount", 500]}).plan_digest
    say(f"- 失效集合: {result['invalidated']}")
    say(f"- 重算单元: {sorted(result['recomputed'])}")
    say(f"- 未受影响单元 orders_by_amount 摘要逐字节一致: {before_amount == after_amount}")
    say(f"- 引擎校验 unaffected_byte_identical={result['unaffected_byte_identical']}")
    plan_v2 = eng.compile("orders_by_user", {"eq": ["user_id", 42]})
    eng.probe([plan_v2])
    say(f"- 新版本探查落账, 当前视图任务数={eng.ledger.summary()['tasks']} "
        f"(全部为 rule_v=2)")

    section("7. 快照恢复与强制终止")
    eng.save_snapshot()
    say(f"- 快照已保存: {eng.snapshot_path()} (含版本/租约/缓存摘要/账本游标+内容摘要)")
    held = eng.lease_manager.acquire(db=1)  # 模拟崩溃时未归还的租约
    eng.ledger.record_lease(held.as_dict(), 2, 1)
    eng.save_snapshot()
    eng2 = WorkbenchEngine(WORKDIR, global_limit=3, per_db_limit=1,
                           lease_ttl_ms=1_500)
    eng2.load_config(CFG_V2)
    rec = eng2.recover()
    say(f"- 重启恢复: 仅依据已提交批次, cursor={rec['recovered_cursor']}")
    say(f"- 未提交租约标记为 unknown 并要求重跑: {rec['unknown_leases']}")
    doc = json.loads(open(eng2.snapshot_path(), encoding="utf-8").read())
    doc["payload"]["ledger_cursor"] = 999
    open(eng2.snapshot_path(), "w", encoding="utf-8").write(json.dumps(doc))
    try:
        eng2.recover()
    except BizError as exc:
        say(f"- 快照摘要被篡改: code={exc.code}, field={exc.field} -> 整体替换")

    section("8. 结构化日志")
    say(f"- 日志文件: {eng.logger.path}")
    with open(eng.logger.path, encoding="utf-8") as fh:
        events = [json.loads(l)["event"] for l in fh if l.strip()]
    say(f"- 事件序列: {events}")

    eng.shutdown()
    eng2.shutdown()
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n报告已写入 {REPORT}")


if __name__ == "__main__":
    main()
