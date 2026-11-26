# 实际验收记录

运行环境：WSL2 Ubuntu，Python `3.11.0rc1`，NumPy `1.26.4` 固定依赖；无容器、无真实集群、无网络访问依赖。

## 自动测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

实际结果：

```text
Ran 31 tests in 8.777s

OK
```

覆盖项包括：`E_SPEC` 字段路径/行号、v1 宽度与格式区分、未知版本/缺字段/越界、`E_PORT`、真实套接字 `E_TIMEOUT`、缺字段 `E_REPLY`、节点枚举逆序确定性、迁移强制终止后 checkpoint 续做、只读备份、增量与全量一致、未受影响 Pod 回执不变、账本整体替换差异、摘要注入 `E_SNAPSHOT`、强制终止恢复语义和独立核对 `E_VERIFY` 首差异定位。

## 一键验收

```bash
.venv/bin/python examples/run_acceptance.py \
  --workdir . \
  --summary-out artifacts/acceptance-result.json
```

结果摘要：

```json
{
  "fault_injection": {
    "ledger_digest": "E_SNAPSHOT",
    "record_corruption": "E_FORMAT",
    "v1_width": "E_WIDTH"
  },
  "incremental": {
    "affected_nodes": ["node-a", "node-c", "node-d"],
    "affected_pods": ["pod-0001", "pod-0003", "pod-0004"],
    "consistent_with_full": true,
    "unaffected_pods_byte_identical": true,
    "scheduled_v2_rescored": ["pod-0001", "pod-0003", "pod-0004"],
    "ledger_snapshot": {
      "code": "E_SNAPSHOT",
      "format_version_set": ["3"],
      "mismatch": true,
      "reason": "content_digest_mismatch+ledger_entries_diverge_from_snapshot_digest",
      "replacement": {
        "added": [],
        "deleted": [],
        "replaced": ["pod-0001", "pod-0003", "pod-0004"]
      }
    }
  },
  "records": {
    "conclusion_hashes_equal": true,
    "mixed_format_versions": [1, 2, 3],
    "post_migration_format_versions": [3],
    "record_count": 3
  },
  "recovery": {
    "committed": ["pod-0001", "pod-0002", "pod-0003", "pod-0004"],
    "cursor": 4,
    "exit_report_available": false,
    "recovery_basis": "committed_only",
    "requires_rerun": false,
    "unknown": []
  },
  "scheduling_v1": {
    "bindings": {
      "pod-0001": "node-a",
      "pod-0002": "node-b",
      "pod-0003": "node-c",
      "pod-0004": "node-c"
    }
  },
  "spec": {
    "nodes": 4,
    "parameter_version": 1,
    "pods": 4,
    "scale_factor": 1000
  },
  "verification": {
    "mode": "readonly-ledger+full-independent-recompute",
    "ok": true,
    "verified_pods": ["pod-0001", "pod-0002", "pod-0003", "pod-0004"]
  }
}
```

关键核对：

- 迁移前版本集合为 `[1,2,3]`，迁移后为 `[3]`；3 条记录的规范结论 SHA-256 全部一致。
- 参数版本 2 只重算 `pod-0001/0003/0004`；`pod-0002` 的稳定结论字节和回执原样复用。
- 增量路径与从零全量路径逐 Pod 逐字段一致。
- 账本归档差异只列出被替代的 3 个 Pod，`pod-0002` 不进入替代清单。
- 独立核对以只读模式读取账本并完成 4 个 Pod 的从零重算。
- v1 宽度、记录损坏、摘要异常分别返回 `E_WIDTH`、`E_FORMAT`、`E_SNAPSHOT`，并写入含单调毫秒的 JSONL 日志。
- 强制终止恢复输出始终声明 `exit_report_available: false`；只承诺已提交行可恢复，未提交行为 unknown 并需要重跑。
