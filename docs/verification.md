# 实际验证记录

验证日期：2026-11-20（Asia/Shanghai）

环境：

```text
Python 3.11.0rc1
numpy 2.4.6
pytest 9.2.0
```

## pytest

执行：

```bash
python -m pytest -q
```

结果：

```text
45 passed
```

覆盖范围：

- 配置边界与字段定位；
- 确定性账本与重复生成文件字节一致；
- v1/v2/v3 严格解码与错误字段路径；
- v1、v2、v3 归一后逐字节一致；
- 整体替换、`E_RESTORE`、`E_DIVERGE` 首差异定位；
- 原子升级、只读备份、检查点续做；
- 重写窗口 `unknown` 与显式重试；
- 检查点内容不符 `E_UPGRADE`；
- 父链缺失 `E_CHAIN` 与摘要损坏 `E_INTEGRITY` 不破坏已有清单；
- SQLite 快照保存、匹配恢复、失效快照只读替换、`E_REPLAY`；
- 三种 CLI 注入的非零退出码与结构化日志；
- JSONL 会话循环。

## 完整演示

执行：

```bash
python -m ckptmeta demo --workspace .interop-demo
```

关键结果：

```text
target_step: 20
target_view_digest: f87e7405244ff5c4e86df676112c326d5136040c0bc5cdddcccb3c8904c845b2
v1_filled_fields:
  manifest_id
  parent_manifest
  partition_mode
  rng_state
  shards[*].data_cursor
  shards[*].momentum_digest
v2_filled_fields:
  manifest_id
  parent_manifest
  partition_mode
  shards[*].data_cursor
byte_identical: true
forced_exit_code: 137
interrupted_run_exit_report: unavailable
resume_status: resumed_complete
resume_skipped_done: [step-00000010.v1.json]
resume_processed: [step-00000020.v2.json]
snapshot_replay_reproduced: true
all_checks_passed: true
```

同一工作区重复执行会清理并重建演示子目录，视图摘要保持不变。

## 异常注入

三个命令均在独立工作区执行，输入文件由命令修改，实现代码不变。

```bash
python -m ckptmeta demo --workspace .interop-demo-integrity --inject-integrity
```

```text
exit: 12
business_code: E_INTEGRITY
path: shards[2].parameter_digest
shard_index: 2
```

```bash
python -m ckptmeta demo --workspace .interop-demo-chain --inject-chain
```

```text
exit: 13
business_code: E_CHAIN
path: parent_manifest.manifest_id
parent_manifest_id: step-missing
```

```bash
python -m ckptmeta demo --workspace .interop-demo-upgrade --inject-upgrade
```

```text
exit: 14
business_code: E_UPGRADE
path: plan_digest
```

日志行示例（字段顺序按规范 JSON 排序）：

```json
{"business_code":"E_INTEGRITY","details":{"field":"parameter_digest","path":"shards[2].parameter_digest"},"level":"ERROR","manifest_path":".../historical_v1/step-00000020.v1.json","message":"parameter digest mismatch for shard 2","operation":"inject_integrity","shard_index":2,"timestamp":"2026-11-20T00:47:54.316+00:00"}
```

三条日志均含 `business_code`、`manifest_path`、`shard_index`、`timestamp`。`E_CHAIN` 与 `E_UPGRADE` 为清单级错误，`shard_index` 为 `null`。

## 退出码

| 业务码 | 实测退出码 |
| --- | ---: |
| `E_INTEGRITY` | 12 |
| `E_CHAIN` | 13 |
| `E_UPGRADE` | 14 |

强制终止使用 `os._exit(137)`，重启后的升级报告记录 `previous_run_report: "unavailable"`，不声称可获得被终止进程的退出报告。
