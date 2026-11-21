# StreamAgg 技术设计

## 1. 组件边界

本组件只处理本地确定性输入。事件由 JSON 声明中的整数种子生成，或由调用方直接提供带 `arrival_no` 的事件。账本为本地 SQLite，快照与日志目录位于工作区内。核心计算不访问网络，不使用 Docker。

主要目录：

- `streamagg/spec.py`：JSON 强校验与规范化。
- `streamagg/generator.py`、`streamagg/prng.py`：跨 Python 版本稳定的 SplitMix64 确定性生成器。
- `streamagg/engine.py`：水位、窗口定稿、预算裁决、定点聚合、误差归因、版本替换。
- `streamagg/ledger.py`：SQLite 事务、哈希链、陈旧标记、查询。
- `streamagg/snapshot.py`：原子快照、内容摘要校验、整体替换恢复。
- `streamagg/replay.py`：按账本游标重放及首差异定位。
- `streamagg/diagnostics.py`：只读 Tkinter 诊断界面。
- `streamagg/scenarios.py`、`streamagg/demo.py`、`streamagg/cli.py`：三场景演示与异常注入。

## 2. JSON 声明

必填字段：

```json
{
  "format_version": 1,
  "sources": [{
    "source_id": "main-src",
    "seed": 20261122,
    "event_count": 420,
    "out_of_order_ratio_bp": 2500
  }],
  "rules": [{
    "rule_id": "main-rule",
    "width_ms": 1000,
    "lateness_ms": 700,
    "partition_keys": ["p0", "p1"],
    "aggregation_functions": ["COUNT", "SUM", "MEAN"]
  }],
  "session": {
    "fixed_scale": 1000,
    "lateness_budget_count": 2,
    "lateness_budget_amount": 5000000
  }
}
```

可选生成器参数：`partition_keys`、`base_time_ms`、`time_step_ms`、`measure_min`、`measure_max`、`reorder_delay_events`。可选会话参数：`dominant_threshold_bp`（默认 6000）、`order_backward_limit_ms`（默认 86400000）。规则可选携带 `rule_version`；初始样本通常省略并按版本 1 处理。

校验失败整单拒绝：先校验 sources，再校验 rules，最后校验 session。`E_SPEC` 携带首个违规字段路径；非法 `fixed_scale` 使用 `E_GRID`。不会返回部分载入对象。

## 3. 确定性生成与到达顺序归一

每个源使用独立 SplitMix64 序列：

1. 生成单调事件时间、分区键与整数度量。
2. 选取 `floor(event_count * ratio_bp / 10000)` 个事件，在 `1..reorder_delay_events` 内分配确定性延迟。
3. 按 `(release_position, source_seq)` 得到源内到达序列。
4. 多源按源顺序轮转合并，并写入全局唯一 `arrival_no`。
5. 源摘要为规范化事件字段的 SHA-256，重复生成逐字节一致。

满足“任意到达顺序同一结果”的业务入口是批处理 `Engine.run_events()`：推进水位前按 `(arrival_no, event_id)` 归一化，因此外部打乱列表不改变定稿、丢弃、聚合与归因。低层 `Engine.ingest()` 假定调用方已按规范到达序投递。

## 4. 窗口与水位语义

- 窗口为 epoch 对齐的半开区间 `[start, end)`，`start = floor(t / width) * width`。
- 每分区水位为已见最大事件时间。第一个事件的水位前后值均记为事件时间。
- 处理事件前读取 `watermark_before`。`t < watermark_before` 判为迟到。
- 事件处理完后更新 `watermark_after = max(watermark_before, t)`。
- 已知非空窗口在 `window_end <= watermark_after - lateness_ms` 时定稿。
- 窗口定稿后，再来的事件不能改变其统计，只登记拒绝裁决。

裁决优先级固定且不依赖遍历顺序：

1. 若快照恢复将该窗口标记为 `UNKNOWN`：`WINDOW_STATE_UNKNOWN`。
2. 若窗口已定稿，或逻辑上已越过关闭线：`WINDOW_ALREADY_FINALIZED`。
3. 非迟到：`ACCEPT / ON_TIME`。
4. 迟到且预算条数不足：`LATE_BUDGET_COUNT_EXHAUSTED`。
5. 迟到且加入后绝对量超过预算：`LATE_BUDGET_AMOUNT_EXCEEDED`。
6. 迟到且两个预算均满足：`ACCEPT / LATE_WITHIN_BUDGET`。

预算公式为：

```text
used_count + 1 <= budget_count
used_abs_amount + abs(q) <= budget_amount
```

等号表示恰好耗尽并接纳，下一事件才拒绝。每个事件、每条适用规则都写一条裁决证据，含到达序号、窗口边界、预算前后值、累计迟到绝对量和原因。

## 5. 整数定点与饱和

设会话缩放因子为 `S`，整数度量为 `m_i`：

```text
q_i = clamp_i64(m_i * S)
e_i = abs(m_i * S - q_i)
sum_raw = sum(q_i)
sum_fixed = clamp_i64(sum_raw)
mean_fixed = clamp_i64(trunc_toward_zero(sum_fixed / count))
```

口径固定为：

- 均值除法向零截断。
- 单值、求和、均值均按有符号 64 位饱和。
- 误差界展示值同样按有符号 64 位饱和；账本中的 `sum_exact` 以十进制文本保存，避免 SQLite 64 位整数列溢出。
- 聚合以整数保存，不使用浮点。
- 定稿值只写入账本一次，后续迟到事件不会修改。

## 6. 误差界与归因

记：

- `Q = Σe_i + abs(sum_raw - sum_fixed)`：量化截断与求和溢出贡献。
- `L = Σ accepted_late abs(q_i)`：迟到接纳暴露量。
- `B = Σ boundary_event abs(q_i)`，其中 `t_i mod width == 0`：边界归属贡献。

输出：

```text
single_value_quantization = max(e_i)
sum_bound = Q + L + B
mean_bound = ceil(sum_bound / count) + 1 + mean_saturation_error
```

均值界中的 `+1` 覆盖向零截断的余数。整数度量乘以整数缩放因子通常没有量化误差；若发生 64 位饱和，残差进入 `Q`。迟到与边界段是账本口径下的保守暴露量，便于定位主导风险。

各段占比以 basis point 输出，默认阈值 `6000bp`（60%）。达到阈值时输出 `QUANTIZATION`、`LATE_ADMISSION` 或 `WINDOW_BOUNDARY`，否则为 `NONE`；平票按 `QUANTIZATION > LATE_ADMISSION > WINDOW_BOUNDARY` 的固定优先级。

## 7. 不可区分分组

两个窗口均值差满足以下条件时视为误差影响下不可区分：

```text
abs(mean_a - mean_b) <= mean_bound_a + mean_bound_b
```

每对证据写入 `ambiguity_pairs`，`error_affected=1`。稳定分组使用并查集形成连通分量；组内按：

```text
(partition_key, window_start_ms, window_end_ms, rule_id, rule_version)
```

稳定排序。组间按组内最小均值与最小稳定键排序。结果与事件到达顺序、分区遍历顺序无关。

## 8. 规则/缩放因子演进

`Engine.evolve_spec()` 强制下一版本号不小于 `current_version + 1`。替换时：

1. 记录旧视图中窗口、丢弃裁决和分区。
2. 将旧窗口、迟到裁决、误差归因和歧义对全部标记 `stale=1`。
3. 新建 `view_id`、`view_manifest` 和快照可见的规则/源记录。
4. 清空内存窗口状态，不增量合并旧结论。
5. 返回稳定去重并排序的受影响窗口和分区标识。

当前统计查询默认只读取 `stale=0` 的活动视图。

## 9. SQLite 账本

主要表：

- `watermark_trace`：逐事件水位前值、后值、迟到标记与本次定稿数。
- `window_state`：`PENDING`、`COMMITTED`、`UNKNOWN` 状态及定点统计。
- `late_decisions`：每事件每规则的接纳/丢弃、预算与原因。
- `error_attribution`：三段误差、上下界、主导段与占比。
- `ambiguity_pairs`：不可区分对与稳定决胜键。
- `ledger_batches`：窗口定稿批次及链式摘要。
- `staleness_events`：版本替换、受影响窗口与陈旧摘要。
- `snapshots`：快照版本、游标、文件与账本摘要。

所有关键记录包含整数 `format_version` 和 SHA-256 内容摘要。每个窗口定稿在单个 `BEGIN IMMEDIATE` 事务中提交窗口状态、误差归因、歧义对和批次链摘要。批次摘要定义为：

```text
batch_digest = SHA256(previous_digest || canonical(payload_digest))
```

## 10. 快照、整体恢复与重放

快照以临时文件加 `os.replace` 原子写入，包含 manifest 和完整状态。加载时：

1. 校验 JSON、整数格式版本、视图 ID、游标和状态内容摘要。
2. 自动读取当前 SQLite 会话清单，比较规则版本、规格摘要和账本内容摘要。
3. 校验账本批次链：首批次前驱、每批 `content_digest`、运行末尾摘要和视图摘要必须闭合。
4. 不一致时标记需要整体替换，不增量合并。
5. 从快照重建 `SessionSpec`、水位、定稿窗口、裁决证据和规则版本。
6. 对未提交 `PENDING` 窗口写入 `UNKNOWN`；后续事件不能修改这些未知窗口。

演示中的强制终止使用真实子进程：子进程完成快照与账本提交后直接 `os._exit(17)`，跳过 Python 清理；父场景重新打开 SQLite/WAL 并恢复。

重放校验按账本规则版本、缩放因子、规格摘要和事件游标重新推进，比较：

- 规则版本、缩放因子、规格摘要。
- 每分区最终水位。
- 定稿窗口集合与计数/求和/均值、迟到接纳字段。
- 窗口误差归因全部字段。
- 每事件接纳/丢弃裁决、预算前后值与累计量。
- 歧义对集合。

差异按 `(partition_key, window_start_ms, window_end_ms, field)` 稳定排序，首个差异以 `E_REPLAY` 返回。

## 11. 业务码与退出码

| 业务码 | 进程退出码 | 含义 |
|---|---:|---|
| `OK` | 0 | 成功 |
| `E_SPEC` | 2 | 声明或字段校验失败 |
| `E_ORDER` | 3 | 事件时间大幅倒退越过顺序保护线 |
| `E_GRID` | 4 | 定点缩放因子非法 |
| `E_SNAPSHOT` | 5 | 快照摘要/版本不符、账本批次链不符或恢复失败 |
| `E_REPLAY` | 6 | 重放与账本首差异 |
| `E_INTERNAL` | 70 | 未预期内部异常 |

每次错误写入 JSONL，含序号、UTC 时间、级别、业务码、事件名、消息与上下文。

## 12. 诊断界面

`run_ui.py` 或 `streamagg.cli ui` 以 `mode=ro` 和 `PRAGMA query_only=ON` 打开 SQLite。界面展示：

- 分区水位曲线。
- 窗口定稿数量与丢弃数量。
- 定点求和/均值、误差界和主导段。
- 每窗口迟到预算余量与最后裁决。
- 规则版本、视图摘要和账本批次链。

刷新只执行 SELECT，不写账本。
