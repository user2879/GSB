# 实际验证结果

验证时间：2026-11-27（Asia/Shanghai）。

环境：

- Python `3.11.0rc1`（满足 >=3.10）
- NumPy `1.26.4`
- pytest `8.2.0`
- 回环地址：`127.0.0.1`

## 执行命令

```bash
python3 -m topic_ledger.cli generate
python3 -m topic_ledger.cli validate
python3 -m pytest -ra --tb=short
python3 -m topic_ledger.cli demo
```

原始输出：

- `artifacts/pytest-output.txt`
- `artifacts/pytest-full.log`
- `artifacts/verification-run.json`
- `artifacts/verification-summary.md`

## 测试结果

```text
...........................................                              [100%]
43 passed in 15.71s
```

测试覆盖：

- 声明、文档、推断配置和命令流的首个非法字段；
- 真实子进程 JSONL Worker、重复投递摘要闸和拒收批次号；
- 租约代次不匹配不前进、真实 TTL 回收和在途槽释放；
- 采样在途时回收租约，验证迟到回执被代次围栏拒绝且不会计入账本；
- 权威来源排序、覆盖账本、逐主题闭合、`E_LEDGER_MISMATCH` 冻结；
- 预算耗尽、待办顺序、下一轮优先和节点租约上限；
- 文档级精确失效、未受影响分片持久化复用；
- 快照写入、原子提交和同配置恢复；
- 交互预算超时返回陈旧摘要；
- 12 组固定种子随机命令序列在并发度 1 与 4 下的逐字段重放一致；
- 真实子进程集群的命令流重放一致。

## 一键验收结果

`python3 -m topic_ledger.cli demo` 退出码为 `0`，机器可读结果中的总判定：

```json
{
  "overall_equal": true,
  "ledger_all_closed": true,
  "command_replay_equal": true,
  "command_replay_difference_count": 0,
  "final_full_rebuild_equal": true,
  "final_full_rebuild_difference_count": 0
}
```

命令流 canonical report 的左右 SHA-256 完全相同：

```text
887115d98920846159d10142252015d037677ef62ae1ce40ecd219f5ddad5970
887115d98920846159d10142252015d037677ef62ae1ce40ecd219f5ddad5970
```

## 租约回收

真实单调时钟 TTL 探针结果：

```json
{
  "shard_id": "demo-expiring-shard",
  "generation": 1,
  "expired_count": 1,
  "code": "E_LEASE_EXPIRED",
  "slot_released": true,
  "global_held_after_recovery": 0
}
```

这里实际 sleep `45ms` 仅用于越过 `20ms` TTL；配额由恢复逻辑调用幂等 `SlotTicket.release()` 真实释放。

## 重复投递拒收

```json
{
  "rejected_code": "E_DUPLICATE_DELIVERY",
  "rejected_batch_id": "demo-duplicate-rejected",
  "overwrite_ok": true
}
```

缺失 `prior_receipt_digest` 的第二次投递被整条拒绝；携带首次回执 SHA-256 后，第三次投递成功覆盖。

## 共享词项账本闭合

演示配置在命令流末尾将主题数调整为 5。全局分布共 60 条（12 个词项 × 5 个主题），
产生 12 条词项权威裁决和 31 条被覆盖来源记录。逐主题闭合如下：

| topic | distribution sum | authoritative source sum | closed |
|---:|---:|---:|---|
| 0 | 10 | 10 | true |
| 1 | 13 | 13 | true |
| 2 | 7 | 7 | true |
| 3 | 14 | 14 | true |
| 4 | 4 | 4 | true |

## 预算与缓存

- 预算设为 `1` 时，4 个分片均以 `E_BUDGET_PENDING` 保留，业务码计数为 4。
- 替换 `doc-000` 后，第二次运行仅失效 `shard-00`，`shard-01`、`shard-02`、`shard-03` 命中持久化缓存，复用数为 3。
- 增量命令流与空白重放差异数为 0；增量最终集合与空白全量重建的语义字段差异数也为 0。

## 复现说明

演示会在启动前程序化清理 `runtime/demo/` 与 `snapshots/demo/`。固定端口为
`127.0.0.1:48101`、`48102`、`48103`；演示启动前会探测这些端口是否空闲。
pytest 中的子进程用例自动申请空闲回环端口，避免与固定演示端口冲突。
