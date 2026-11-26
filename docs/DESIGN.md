# GSB 事件总线订阅消费链路 — 技术说明

## 1. 范围与运行方式

纯本地、无外部网络的确定性流水线。每个订阅者是独立 Python 子进程，在
`127.0.0.1` 的声明端口上提供真实 TCP 服务；总线经真实套接字逐行下发批次并
接收三格式回执，所有持久状态落在单个 SQLite 账本中。

```bash
python3 -m venv .venv            # 本环境用 virtualenv 创建（ensurepip 缺失）
source .venv/bin/activate
pip install numpy==1.26.4
export PYTHONPATH=src
python -m gsb.cli gen --out samples/demo --events 48
python -m gsb.cli run --spec samples/demo/spec.json \
    --events samples/demo/events.ndjson --run-dir artifacts/runs/demo-final
./scripts/demo.sh                # 一键全流程（含崩溃恢复）
```

## 2. 模块映射（对应八项业务要求）

| 要求 | 模块 | 关键不变量 |
| --- | --- | --- |
| 1 强校验 | `gsb/spec.py` | 首个非法条目以 `E_SPEC` 报告，携带 `entry` 与 `field` |
| 2 三格式编解码/归一 | `gsb/receipts.py` | 未知版本/宽度不足 `E_FORMAT`；依据不可追溯 `E_BACKFILL` |
| 3 确定性切批/批标识 | `gsb/batching.py`、`gsb/digest.py` | 纯函数 + 规范 JSON，与派发/字典遍历顺序无关 |
| 4 子进程+真实 TCP | `gsb/processor.py`、`gsb/client.py` | 真实握手/读取超时；`E_PORT`/`E_TIMEOUT` |
| 5 租约 | `gsb/lease.py`、`gsb/clock.py` | 单调时钟真实到期；版本号乐观续约，冲突 `E_LEASE` 不延长 |
| 6 三重约束调度 | `gsb/scheduler.py` | 全局租约数、每订阅者在途事件、每通道速率；真实线程池 |
| 7 采纳裁决/归因 | `gsb/adoption.py` | 口径版本→格式降序→归并序号；三段耗时占比与主导段 |
| 8 账本/快照/恢复 | `gsb/ledger.py` | 同事务原子提交；摘要不符整体替换；恢复重放定位首个差异 |

## 3. 三种回执格式

- `x1`：53 字符定宽行 = 32 位十六进制批标识 + 21 位带符号零填充定点摘要，
  无分隔符。字段不足直接 `E_FORMAT`。
- `x2`：`|` 分隔，新增订阅者、完成毫秒与显式格式版本 `2`。
- `x3`：再新增批内事件序号集合、被覆盖重复投递引用与口径版本 `v1`。

归一在 `receipts.normalize` 中完成。`x1` 缺少的订阅者、完成毫秒、事件序号
集合、覆盖引用、口径版本全部从账本 `dispatches` 中**已登记**的派发记录补齐；
输出携带 `source_format`、`backfilled_fields` 与 `backfill_basis`
（形如 `ledger:dispatch:<id>:<digest 前缀>`）。三格式归一后的采纳核心
（`_adoption_core`）逐字节一致，由测试逐批断言。

关键时序：处理器对同一批次只产生一个 `completion_ms`，总线在归一前调用
`ledger.mark_completion` 落盘，因此 `x1` 补齐的完成时刻与 `x2/x3` 显式携带的
值相同，这是逐字节一致的前提。

## 4. 确定性批标识与聚合

批标识来自显式规范文档（排序键、紧凑分隔、无空白）的 SHA-256 前 32 位：

```json
{"kind":"gsb.batch_id.v1","channel_id":"...","topic":"...",
 "batch_index":0,"subscriber_id":"..."}
```

`plan_batches` 是纯函数：通道按标识排序、订阅者按标识排序、事件保持流序，
不依赖任何 dict 哈希或派发顺序。NumPy 聚合全部为 `int64` 整数运算，权重由
`(种子, 事件序号)` 的 SHA-256 派生，与批次/到达顺序无关；摘要再与批标识、
事件序号集合、口径版本共同绑定为 `summary_digest`。

## 5. 租约、续约与背压的耦合

- 租约字段：`lease_id/subscriber_id/batch_id/granted_at_ms/expires_at_ms/version`。
- 排队批次在通过准入前不签发租约，因此不占用任何在途额度。
- 准入后才 `grant`，TTL 取订阅者预算的固定比例；处理时长跨过续约阈值时，
  总线凭当前版本号向子进程发起乐观续约，版本不符返回 `E_LEASE` 且不延长。
- 到期判定只读 `time.monotonic_ns()`，等待全部由真实 `sleep` 产生，没有
  人工时钟或套接字关闭。
- 调度器在真实 `ThreadPoolExecutor` 上运行；全局约束按活动租约（批次）数、
  每订阅者按在途事件数、每通道按 1000ms 滑动窗口速率。每轮变化写入
  `trajectory`，记录全局租约占用、排队长度、积压事件数。

## 6. 采纳裁决与三段归因

排序键（高者优先）：口径版本序号 → 格式版本（x3>x2>x1）→ 归并序号 →
规范文本。被否决候选保留在账本 `receipts` 并记录理由。候选定点耗时差落在
声明 `error_margin_ms` 内时，结论标注 `error_margin_applied=true`，仍由格式
降序这一显式决胜规则选定。端到端耗时拆为排队 `queue_ms`、格式补齐
`backfill_ms`、实际处理 `processing_ms`，给出占比与主导段。

## 7. 账本原子性、快照替换与崩溃恢复

`dispatches/receipts/adoptions/lease_states` 的批次写入在单一事务
（`with self.conn`）内完成，每条记录带整数格式版本与内容摘要。重新打开时
`meta.schema_version` 中的会话摘要若不一致，直接 DROP 全部业务表并重建
（整体替换，绝不增量合并）。

强制终止通过 `os._exit(137)` 实现，刻意不生成退出报告。恢复只信任已提交
行：未出现在 `adoptions` 的批次标记为 `unknown` 并要求重跑；`diff_replay`
按计划批次顺序重放，返回首个不一致的 `batch_id` 与 `field_path`，并写入
`replay_findings`。

## 8. 已验证结果（本地样本，48 事件 / 3 订阅者 / 2 通道）

- 24 个单元/集成测试全部通过（`python -m unittest discover -s tests`），
  并在 `-W error::ResourceWarning` 下无资源泄漏。
- 22 个批次、66 条三格式回执；每批三格式归一核心唯一（逐字节一致）。
- 轨迹最大排队 22 批、最大积压 96 事件、全局租约峰值 3；22 个租约均发生
  至少一次续约；陈旧版本续约被 `E_LEASE` 拒绝。
- 崩溃在第 3 批后：恢复报告 3 批已提交、19 批 unknown、`rerun_required=true`，
  并定位首个未知批标识。
