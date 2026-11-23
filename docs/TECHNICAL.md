# kvdiag 技术说明

## 总体结构

| 模块 | 职责 | 对应业务要求 |
| --- | --- | --- |
| `kvdiag/spec.py` | 键空间声明与样本的加载、强校验 | 1 |
| `kvdiag/index.py` | 前缀/区间索引的确定性构建 | 2 |
| `kvdiag/snapshots.py` | 快照链创建、完整性校验与回放 | 3 |
| `kvdiag/leases.py` | 租约签发、乐观版本续约、真实单调时钟过期回收 | 4 |
| `kvdiag/converge.py` | 索引位点与租约过期位点收敛判定 | 5 |
| `kvdiag/scheduler.py` | 全局并发上限 + 每分片配额的线程池调度 | 6 |
| `kvdiag/ledger.py` / `kvdiag/recovery.py` | SQLite 账本、批次事务、版本识别整体替换恢复 | 7 |
| `kvdiag/inject.py` / `kvdiag/cli.py` | 三类异常注入与结构化日志 | 8 |
| `kvdiag/ui.py` | Tkinter 诊断界面（Windows 本地） | 界面交付 |

## 关键设计

### 1. 声明校验（E_SPEC）
`load_spec` 按声明顺序校验：分片标识唯一；`high_byte` 模式下区间按 lo
排序后首尾相接（`cur.lo == prev.hi + 1`）且不重叠；`hash_bucket` 模式下
桶号恰好覆盖 `0..buckets-1`。样本键必须落在分片区间/桶内；索引规则引用
的列必须在每个样本行中存在且类型匹配（前缀列字符串、区间列整数）；
`ttl_ms` 为正整数、`renew_interval_ms < ttl_ms`、`max_inflight_renewals`
为非负整数（布尔值被显式拒绝）。任一违规以 `E_SPEC` 抛出并携带首个非法
分片标识与字段名；校验全部通过前不落任何数据，因此不存在部分载入。

### 2. 确定性索引
索引条目 = `{shard_id, rule, key_digest, position, index_version}`。
前缀索引按 `(value[:prefix_len], key)` 排序，区间索引按
`(score // width, key)` 排序后赋位点；位点只依赖显式排序，与字典/哈希
遍历顺序无关。`canon.canonical_json`（排序键、紧凑分隔符、UTF-8）保证
逐字节可比较；`tests/test_index.py` 用 5 种随机行序验证条目集合逐字节
一致。键摘要在写入前经 `ensure_key_compatible` 校验与分片区间相容。

### 3. 快照链
每批次为每个分片写一个快照行：`snapshot_id, parent_id, shard_id,
index_pos_lo, index_pos_hi, lease_expiry_pos, content_digest`，按分片标识
排序链接，批间链头相接。`check_chain` 校验父引用存在且无环；
`replay_from_snapshot` 沿链收集分片状态，链上缺失声明分片时以
`E_CHAIN` 报告断链快照与缺失分片标识。`replay_verify` 将链回放结果与
不经过快照、直接从样本与已提交租约状态重建的结果做逐字节比较。

### 4. 租约
续约携带期望版本号做乐观校验，不符抛 `E_LEASE`（含期望/实际版本）。
过期判定使用 `time.monotonic()` 毫秒真实计时；过期回收把分片过期位点
推进到稳定值（每回收一个租约位点 +1）。每分片在途续约数受
`max_inflight_renewals` 配额约束，超限申请排队并记录等待开始/结束的
单调时钟毫秒。

### 5. 收敛判定
对每个分片比较回放索引位点（低水位 `index_pos_lo`）与当前租约过期位点，
`|diff| <= tolerance` 判定收敛并输出判定依据字符串；超差以 `E_DIVERGE`
报告分片标识、两侧位点与差值。判定按排序后的分片标识迭代且只依赖位点，
因此结论与分片遍历顺序、续约处理顺序无关（有序列打乱测试佐证）。

### 6. 并行调度
`Scheduler` 将任务真实投射到 `ThreadPoolExecutor`：全局运行数不超过
`max_workers`，每分片运行数受信号量配额约束；排队任务记录入队到实际
启动的单调毫秒等待窗口。测试用共享计数器验证运行峰值不超限。

### 7. 账本与恢复
每个分片批次在同一 SQLite 事务中写入索引位点、租约状态、收敛结论与
进度游标，并同步刷新 `meta.content_digest`（业务表规范化转储的
SHA-256）与整数 `format_version`。恢复时比较会话快照的版本与摘要：
不一致则以 `E_SNAPSHOT` 报告并**整体替换**索引条目、租约集合与结论
（`DELETE` + 批量 `INSERT`，绝不增量合并），差异项按稳定排序列出
（`表:键:only-in-session/only-on-disk/content-differs`）。强制终止
（`--crash-before-batch N`，`os._exit(3)`）后只有已提交批次可见；
缺少已提交游标的分片被标记 `unknown` 并可用 `--rerun-unknown` 重跑，
快照链从既有链头继续。租约 ID 含批次号，重跑不会与已提交租约冲突。

### 8. 异常注入与结构化日志
`kvdiag inject` 只改写输入文件：样本文件（键移出区间）、续约计划
（期望版本倒退）、会话快照（摘要置零），分别触发 `E_SPEC`、`E_LEASE`、
`E_SNAPSHOT`。结构化日志为 JSON 行，含 `ts_mono_ms`（单调时钟）、
`shard_id`、`code` 与事件细节。

## 位点语义

- `index_pos_hi`：分片键数（下一个可用位点）。
- `index_pos_lo`：索引回收前沿 = 已过期并释放的租约位点数。
- `lease_expiry_pos`：已过期租约数。
- 过期回收同时推进两侧，因此稳态下 `diff = 0`，容差内收敛。

## 实际验证记录（本工作区，Python 3.11）

- `python -m pytest tests -q` → **35 passed**（约 7s）。
- 端到端：`run` 提交 2 个批次（各 2 分片），`replay_ok: true`；
  `recover` 干净重开 `replaced: false`。
- 异常注入：`sample-corrupt` → 退出码 1、`E_SPEC`（shard-1, rows.key）；
  `lease-version-regression` → 退出码 1、`E_LEASE`（expected 0 / actual 1）；
  `ledger-digest-mismatch` → `replaced: true` 并写 `E_SNAPSHOT` 日志。
- 强制终止：`--crash-before-batch 2` 退出码 3；恢复报告
  `unknown_shards: [shard-2, shard-3]`，`--rerun-unknown` 重跑后
  `replay_ok: true`。
- Tkinter 界面在本 Linux 容器缺少 libtk 无法启动（目标为 Windows 本地，
  Python 官方安装包自带 tkinter）；`kvdiag/ui.py` 延迟导入 tkinter，
  包其余功能不依赖图形环境。
