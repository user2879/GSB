# 实际验证结果

执行时间：2026-11-23（Asia/Shanghai）。环境为 WSL2 Linux、Python 3.11.0rc1、NumPy 2.2.6、PyYAML 6.0.2。完整原始产物位于 `results/acceptance/`，机器可读摘要为 `results/acceptance/summary.json`。

## 一键结果

```bash
./scripts/run_acceptance.sh
```

```text
status: passed
unit tests: 28 passed
incremental vs final full: equal
parallel vs serial: equal (including execution sequence)
resume vs incremental: equal
crash exit: 137
resume source: generation 2
quota: partial with 8 pending entities
injections: 4/4 isolated and mapped
```

## 输入校验

| 场景 | 退出码 | 业务码 |
| --- | --- | --- |
| 默认夹具 | 0 | 无 |
| 重复行标识 | 1 | `E_ROW_INVALID` |
| `NaN` 数值 | 1 | `E_ROW_INVALID` |
| 时间戳倒序 | 1 | `E_ROW_INVALID` |
| 特征 DAG 环 | 1 | `E_FEATURE_DAG_INVALID` |
| 缺失上游特征 | 1 | `E_FEATURE_DAG_INVALID` |
| 缺失原始字段 | 1 | `E_FEATURE_DAG_INVALID` |
| 配额为 0 | 1 | `E_CONFIG_INVALID` |
| 租约 TTL 为 0 | 1 | `E_CONFIG_INVALID` |
| 非法计划操作 | 1 | `E_PLAN_INVALID` |

## 租约真实过期

`lease-demo --ttl-ms 60` 实测：

```text
status: reclaimed
elapsed_ms: 61.509
conflict_observed: true
events: lease_granted, lease_conflict, lease_reclaimed, lease_granted
reclaim.basis: monotonic_deadline_reached
reclaim.clock_source: monotonic_ns
reclaim.previous_owner: worker-A
requester: worker-B
```

冲突事件读数早于 `expires_ns`，回收读数不早于 `expires_ns`。接管方是另一个工作单元 `worker-B`。

## 增量与全量一致

默认计划包含 7 批，最终 9 个实体、259 个特征点、63 个实体汇总、51 条跳过明细。规范化比较哈希：

```text
incremental: ac76e6f82f43fe10a20007f96eff2fc0b990694e3369178e50ae07de86439064
final full:  ac76e6f82f43fe10a20007f96eff2fc0b990694e3369178e50ae07de86439064
```

比较字段：逐实体特征值、汇总、跳过明细、待办队列、失败清单、逐实体租约版本。无差异。

计划实际覆盖：样本追加、单行修正、窗口调整、先追加后回滚到第 2 批、回滚后重新追加、最终再修正并调整窗口。

## 并发一致与上限

```text
parallel max_workers=4, max_observed_concurrency=4
serial   max_workers=1, max_observed_concurrency=1
execution_equal: true
```

`--execution` 比较了实体分配顺序、`task_sort_key`、每批点数、累计点数和完成批次。4 线程没有突破配置上限。

## 配额耗尽

配置 `batch_point_quota: 60` 时：

```text
status: partial
consumed_points_by_batch: 42, 35, 42, 35, 42, 35, 42
pending: E002, E003, E004, E005, E006, E007, E008, E009
failure code: E_BATCH_QUOTA_EXHAUSTED
```

每批累计点数不超过 60，未处理实体保持稳定 ID 顺序。

## 崩溃与断点续跑

第 3 批窗口计算后、提交前调用 `os._exit(137)`：

```text
crash exit: 137
CURRENT generation: 2
uncommitted temp generations: 1
resume requested: true
resumed: true
resumed_from_generation: 2
resume final status: completed
resume vs incremental: equal
```

未发布临时代没有 `CURRENT` 指针，恢复未读取其半成品。续跑重新执行第 3 批及后续批次。

## 注入隔离

| 注入 ID | 类型 | 实体 | 观测码 | 命中数 | 待办 |
| --- | --- | --- | --- | --- | --- |
| `INJ-001-NONFINITE` | 非有限值 | E003 | `E_ROW_INVALID` | 1 | 空 |
| `INJ-002-TIMESTAMP-REVERSAL` | 时间戳倒序 | E004 | `E_ROW_INVALID` | 1 | 空 |
| `INJ-003-LEASE-TTL-ZERO` | 租约 TTL=0 | E005 | `E_CONFIG_INVALID` | 1 | 空 |
| `INJ-004-CACHE-BUDGET-ZERO` | 缓存预算<1 块 | E006 | `E_CONFIG_INVALID` | 1 | 空 |

每个隔离运行中其余 7 个实体完成，失败实体不再保留待办，避免将永久输入故障误报为配额耗尽。

## 快照拒绝场景

测试覆盖并在以下情况输出 `E_SNAPSHOT_STALE`：

- 配置哈希改变；
- 清单租约版本被篡改；
- `state.json` 内容被追加一个字节但未更新哈希。

清单还校验阶段文件哈希、COMMITTED 标记、全局版本、计划哈希、基础输入哈希、数据集/特征指纹、完成批次/实体和待办队列。

## 复现命令

```bash
.venv/bin/python scripts/generate_fixtures.py
.venv/bin/python -m unittest discover -s tests -t . -v
./scripts/run_acceptance.sh
cat results/acceptance/summary.json
```

注：租约耗时依赖机器调度，但断言使用实际单调时钟到期点，不依赖固定睡眠值；64 位 WSL 实测约 61.5 ms（TTL 60 ms）。
