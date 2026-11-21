"""强制终止 -> 账本恢复 -> 重放校验 的可复现演示。

流程：
  1. 启动节点并发布 v1（完整提交 compile/task_outcome/receipt 事件）。
  2. 强制 kill 全部节点子进程并直接关闭账本（模拟平台被强制终止，无退出报告）。
  3. 新平台进程打开同一账本，仅依据已提交事件 recover，逐项重放比对。
  4. 重启节点，继续发布 v2，验证恢复后双路径仍逐字节一致。
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configbench.manifest import load_all
from configbench.platform import Platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    registry, manifest, samples = load_all(
        os.path.join(ROOT, "samples/registry.json"),
        os.path.join(ROOT, "samples/manifest.json"),
        os.path.join(ROOT, "samples/samples.json"))
    ledger = os.path.join(tempfile.mkdtemp(prefix="configbench-crash-"), "crash.db")

    p1 = Platform(registry, manifest, samples, ledger)
    p1.start_nodes()
    p1.publish_version(1)
    print("[1] v1 已发布并完整入帐")
    for node_id in sorted(p1.node_processes):
        p1.node_processes[node_id].kill_hard()
    p1.ledger.close()
    print("[2] 节点被强制 kill，平台账本直接关闭（无退出报告）")

    p2 = Platform(registry, manifest, samples, ledger)
    rec = p2.recover()
    print("[3] 恢复结论:", json.dumps(
        {k: rec[k] for k in ("format_version", "ledger_replaced",
                             "unknown_tasks", "restored_nodes")},
        ensure_ascii=False))
    assert rec["unknown_tasks"] == []
    assert all(m.get("match") for m in rec["replay_mismatches"])
    print("    重放逐项匹配:", len(rec["replay_mismatches"]), "个已完成任务")

    p2.start_nodes()
    pub2 = p2.publish_version(2)
    print("[4] 恢复后发布 v2 成功:", pub2["receipts"])
    cmp2 = p2.compare_paths(2)
    print("    双路径一致性:", cmp2["match"], "校验节点:", cmp2["checked"])
    p2.close()
    print("OK: 强制终止恢复与重放校验通过")


if __name__ == "__main__":
    main()
