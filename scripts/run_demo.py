#!/usr/bin/env python3
"""端到端演示：增量下发、失效传播、逐字节稳定、回执收敛、崩溃恢复与重放。

用法：python scripts/run_demo.py [--workdir demo_workspace]
顺序执行即可复现全部结论；退出码非零表示验证失败。
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfgpipe.pipeline import Pipeline, free_port
from cfgpipe.replay import replay_verify

ITEMS = [
    {"key": "alpha", "value": 1.5, "version": 1, "expire_ms": 5000,
     "derived_from": None, "derived_coeff": None},
    {"key": "beta", "value": 0, "version": 2, "expire_ms": 5000,
     "derived_from": "alpha", "derived_coeff": 2.5},
    {"key": "gamma", "value": 3.25, "version": 3, "expire_ms": 5000,
     "derived_from": None, "derived_coeff": None},
    {"key": "delta", "value": 0, "version": 4, "expire_ms": 5000,
     "derived_from": "beta", "derived_coeff": 0.5},
    {"key": "epsilon", "value": 7, "version": 5, "expire_ms": 5000,
     "derived_from": None, "derived_coeff": None},
]

NODE_DEFS = [
    {"id": "n1", "subscribes": ["alpha", "beta", "delta"], "seed": 11,
     "sample_size": 400},
    {"id": "n2", "subscribes": ["beta", "delta", "gamma"], "seed": 22,
     "sample_size": 300},
    {"id": "n3", "subscribes": ["gamma", "epsilon"], "seed": 33,
     "sample_size": 200},
]

CHANGESETS = [
    {"version": 10, "changes": [{"key": "alpha", "value": 1.75}]},
    {"version": 11, "changes": [
        {"key": "epsilon", "value": 8.5},
        {"key": "gamma", "subscribers_add": ["n1"]},
    ]},
]

SESSION = {"scale_exp": 3, "groups": 8, "global_concurrency": 2,
           "node_concurrency": 1, "dispatch_timeout_ms": 800,
           "backoff_ms": [50, 100, 200]}


def init_workspace(workdir):
    os.makedirs(os.path.join(workdir, "changesets"), exist_ok=True)
    nodes = []
    for nd in NODE_DEFS:
        nodes.append({"id": nd["id"], "host": "127.0.0.1",
                      "port": free_port(), "subscribes": nd["subscribes"],
                      "seed": nd["seed"], "sample_size": nd["sample_size"]})
    decl = {"session": SESSION, "items": ITEMS, "nodes": nodes}
    with open(os.path.join(workdir, "declaration.json"), "w",
              encoding="utf-8") as fh:
        json.dump(decl, fh, indent=2)
    for idx, cs in enumerate(CHANGESETS, start=1):
        with open(os.path.join(workdir, "changesets",
                               "changeset_%04d.json" % idx), "w",
                  encoding="utf-8") as fh:
            json.dump(cs, fh, indent=2)
    return decl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="demo_workspace")
    args = parser.parse_args()
    workdir = os.path.abspath(args.workdir)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))

    print("== phase 0: init workspace ==")
    init_workspace(workdir)
    print("workspace: %s" % workdir)

    print("== phase 1: initial + changeset 1 (incremental dispatch) ==")
    pipe = Pipeline(workdir)
    pipe.load()
    pipe.open_ledger()
    pipe.start_nodes()
    try:
        report1 = pipe.run(upto=1)
    except Exception:
        pipe.stop_nodes()
        raise
    stab1 = report1["byte_stability"][0]["unchanged_items"]
    check("cs1 未受影响项逐字节不变",
          all(e["identical"] for e in stab1) and
          {e["key"] for e in stab1} == {"gamma", "epsilon"},
          json.dumps(stab1))
    inv1 = report1["batches"][1]["invalidation"]
    check("cs1 失效集合 = {alpha,beta,delta} + {n1,n2}",
          inv1["items"] == ["alpha", "beta", "delta"]
          and inv1["nodes"] == ["n1", "n2"],
          "items=%s nodes=%s" % (inv1["items"], inv1["nodes"]))
    check("cs1 仅下发受影响键",
          pipe.dispatch_log[1]["upsert"] == {"n1": ["alpha", "beta", "delta"],
                                             "n2": ["beta", "delta"]},
          json.dumps(pipe.dispatch_log[1]["upsert"]))

    print("== phase 2: simulate force-kill mid-batch (no exit report) ==")
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-m", "cfgpipe.crash_batch", "--workdir", workdir,
         "--batch-id", "batch_0002", "--nodes", "n1,n3"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, capture_output=True)
    check("崩溃进程非零退出且无退出报告", proc.returncode != 0,
          "returncode=%d" % proc.returncode)

    print("== phase 3: recovery + resume remaining changeset ==")
    pipe2 = Pipeline(workdir)
    pipe2.load()
    pipe2.open_ledger()
    report2, info = pipe2.resume()
    check("未提交批次节点标记为 unknown", info.unknown_nodes == ["n1", "n3"],
          "unknown=%s" % info.unknown_nodes)
    check("恢复后续跑 batch_0002 已提交",
          any(b["batch_id"] == "batch_0002" for b in report2["batches"]),
          "")

    print("== phase 4: replay verification ==")
    try:
        replay_verify(pipe2.ledger, pipe2.decl, pipe2.changesets)
        check("重放校验 (E_REPLAY 未触发)", True)
    except Exception as exc:  # noqa: BLE001
        check("重放校验 (E_REPLAY 未触发)", False, str(exc))

    print("== phase 5: full rebuild comparison ==")
    rebuild = pipe2.full_rebuild_compare()
    check("增量结果与从零全量重建逐项一致", rebuild["item_diffs"] == [],
          "diffs=%s" % rebuild["item_diffs"])
    last = pipe2.ledger.last_receipts()
    digest_match = all(last[nid]["digest"] == dg
                       for nid, dg in rebuild["node_digests"].items())
    check("账本最终回执摘要与重建期望一致", digest_match, "")

    stab2 = report2["byte_stability"][0]["unchanged_items"]
    check("cs2 未受影响项逐字节不变（含仅订阅变化的 gamma）",
          all(e["identical"] for e in stab2) and
          {e["key"] for e in stab2} == {"alpha", "beta", "delta", "gamma"},
          json.dumps(stab2))
    receipts2 = report2["batches"][0]["receipts"] if report2["batches"] else {}
    check("cs2 未受影响节点沿用上一轮回执",
          receipts2.get("n2") == "carried", json.dumps(receipts2))

    pipe.stop_nodes()  # 节点子进程由 phase 1 的 Pipeline 启动并持有
    pipe2.ledger.close()
    pipe.ledger.close()

    summary = {
        "workdir": workdir,
        "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
        "all_passed": all(o for _, o, _ in checks),
    }
    with open(os.path.join(workdir, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("== summary: %s ==" % ("ALL PASSED" if summary["all_passed"]
                                   else "FAILURES PRESENT"))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
