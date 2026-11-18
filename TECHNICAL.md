# 技术说明

## 模块边界

| 模块 | 职责 |
| --- | --- |
| `quota_app/codec.py` | v1/v2/v3 识别、字段校验、归一化与原子编码 |
| `quota_app/config_workspace.py` | 目录扫描、逐条升级、备份、检查点与中断恢复 |
| `quota_app/derive.py` | 派生图解析、链回溯、成环/缺失引用校验、冲突仲裁 |
| `quota_app/engine.py` | 整数令牌桶与容量配额判定 |
| `quota_app/events.py` | NDJSON 解析、序号/时间戳顺序校验 |
| `quota_app/ledger.py` | SQLite 快照、批次事务、陈旧替换、恢复与重放校验 |
| `quota_app/preview.py` | 主线程快速预览、权威重放、`E_PREVIEW`/`E_LATENCY` |
| `quota_app/gui.py` | Tkinter 界面和后台线程调度 |
| `quota_app/cli.py` | 本地命令行与异常注入入口 |

核心模块不导入 Tkinter。`gui.py` 在导入时安全降级，只有在 `launch_gui()` 中才创建窗口。

## 格式校验与归一

版本识别顺序是 `format_version`、`version`、`schema_version`。值是整数且必须为 1、2、3；其他情况立即返回 `E_FORMAT`。文档随后整体校验：

1. `revision` 为正整数，缺省为 1。
2. `volumes` 是非空数组；卷标识非空且唯一。
3. 按 v1/v2/v3 的要求逐项检查字段存在性和整数类型。
4. 负数值返回 `E_FORMAT`，字段路径采用 `$.volumes[序号].字段`。
5. 任一项失败即抛出，未完成的文档不会写入目录或 SQLite。

`ConfigDocument` 始终保存为当前 v3 语义，同时保留 `original_version`，编码时写 `upgraded_from`，以便界面显示例如 `v3 (源 v1)`。

## 升级与检查点

配置文件按文件名稳定排序。每个待升级条目：

1. 读取原始字节并在内存中完整校验、归一为 v3。
2. 把原始字节原子写入 `backups/<name>.v<version>.bak`，再设置只读属性。
3. 原子重写原文件为 v3。
4. 以 `original_sha256`、`upgraded_sha256`、备份相对路径、完成时间更新 `upgrade_checkpoint.json`；检查点本身同样通过临时文件 + `os.replace` 原子替换。

如果进程在第 3 步和第 4 步之间被强制终止，下一次打开会发现“文件已是 v3、检查点尚未完成、同 stem 的旧版本备份存在”，于是只补写检查点，不重复升级。检查点已完成且文件摘要一致时报告 `already_completed` 并跳过。

检查点缺失或损坏时，所有低版本条目标记为 `blocked`，返回 `E_UPGRADE`；已存在的 v3 文档仍可读取，但不会把低版本条目当作可以猜测状态继续写。不可读的旧条目记录 `E_UPGRADE` 后继续处理其他条目。

测试注入环境变量 `QUOTA_UPGRADE_KILL_AFTER=1` 会在第一个条目重写后执行 `os._exit(86)`，用于验证真实中断。

## 派生图

派生关系为 `child -> upstream_id`。解析使用带 `visiting/done` 状态的深度优先遍历：

- 引用不在卷表中：`E_DERIVE`，详情给出 `reference_chain`。
- 遇到 `visiting` 节点：`E_DERIVE`，详情 `cycle` 是闭环序列，例如 `["a","b","a"]`。
- 派生值：`upstream_effective_capacity * factor + offset`。
- 显式容量与派生容量冲突时执行固定规则：`effective = min(explicit, derived)`。该规则是两侧上限的交集，属于保守仲裁。结果记录 `capacity_basis`、`arbitration`（`explicit`/`derived`/`equal`）和完整 `derivation_chain`。

系数和偏移必须是非负整数。容量和所有相关数值仍受格式层负数检查约束。

## 定点判定

令牌单位是配置维度的 1/1000。设 `rate` 为该维度每秒上限，`ts` 为整数毫秒：

```text
token_capacity_scaled = burst_capacity * 1000
refill_scaled         = (ts - last_ts_ms) * rate
token_balance_scaled  = min(token_capacity_scaled, old + refill_scaled)
cost_scaled           = cost_units * 1000
```

当维度为 `operations` 时 `cost_units=operation_count`；为 `bytes` 时 `cost_units=write_bytes`。容量判断为：

```text
projected = cumulative_allowed_write_bytes + write_bytes
projected <= capacity_bytes  -> 容量通过
```

显式规则：

- 先做容量判断，再做令牌判断；同时失败返回 `OVER_QUOTA`。
- 容量等于上限可通过；令牌等于成本可通过。
- 拒绝事件不增加 `cumulative_allowed_write_bytes`，也不扣令牌。
- 时间戳先推进令牌补充，再判定；配额拒绝时也保留已补充令牌。
- 格式层只接受整数毫秒，不存在小数截断。1/1000 单位下 `delta_ms * rate` 是精确整数；如果未来接入更细时间源，方向规定为向下取整。
- `ops_per_sec == burst_capacity == 0` 表示不启用限流，用于 v1 归一语义；`rate=0, burst>0` 表示只有初始突发、无补充。

因为所有状态量、比较和累加都在整数域完成，模型值与存储值一致，单事件与聚合量化误差上界均为 0。这个上界逐条写入 `judgments.single_error_upper/aggregate_error_upper`，策略也写入 SQLite `meta`。

## SQLite 账本

主要表：

- `config_versions`：不可变配置快照、摘要、格式来源、`stale`。
- `upgrade_checkpoints`：升级检查点的账本副本。
- `event_batches`：`pending/committed/unknown/superseded` 批次状态。
- `event_inputs`：已提交事件原文，支持脱离外部 NDJSON 的权威重放。
- `judgments`：历史判定，含误差界与 `stale`。
- `current_judgments`、`current_usage`：当前修订的物化视图。
- `progress_cursor`：当前已提交事件索引与最后序号。
- `config_replacements`：整体替换时的受影响卷和稳定排序的变动字段。
- `structured_events`：关键业务错误。

## 批次事务与恢复

每批处理分两个提交点：

1. 先用独立事务插入 `event_batches.status='pending'` 并提交。若进程稍后被强杀，该行会留下。
2. 计算完成后在同一 `BEGIN IMMEDIATE` 事务中写入：
   - `event_inputs`
   - `judgments`
   - `current_judgments`
   - 整体替换后的 `current_usage`
   - `progress_cursor`
   - 批次状态改为 `committed`

崩溃时第二段事务整体回滚，不会留下部分判定、用量或游标。重启时 `recover_incomplete()` 把遗留 `pending` 改为 `unknown`，返回 `exit_report_available=false` 和 `rerun_required=true`；新批次从最后一个已提交游标继续。旧 `unknown` 批次保留审计记录，但不参与当前视图。

恢复后 `verify_replay()` 从账本中的 `event_inputs` 顺序重放，逐项比较判定、累计写入、令牌、补充时间、误差界等字段，并核对 `current_usage` 的累计字节、令牌余额、补充时间和放行/拒绝计数。第一处差异返回 `E_REPLAY`，详情含卷、事件序号和差异字段；游标大于 0 但用量快照不完整时也会拒绝继续追加。

## 配置整体替换

配置修订必须严格递增。新修订激活时在单一事务中：

1. 把所有历史 `judgments` 和 `config_versions` 标记 `stale=1`。
2. 清空 `current_judgments`、`current_usage`、进度游标。
3. 写入 `config_replacements`，受影响卷和字段列表均按卷标识排序。
4. 更新 `active_state` 与当前修订。

随后事件流从开头重放到新修订。因此当前视图不是旧结果的增量合并，而是整表替换后的新账本；旧数据只作为不可见的历史审计保留。

## 预览与延迟

界面输入事件触发 180ms 防抖。主线程只即时解析候选配置、计算派生摘要并渲染状态；快速预览本身在后台线程按候选配置重放，权威重算则在另一轮后台独立重放账本事件输入，主线程始终继续响应。

每个预览记录 `elapsed_ms = floor(monotonic_ns 差值 / 1,000,000)`。当 `elapsed_ms > budget_ms` 时写 `E_LATENCY`，包含显式预算和工作量，不抛出异常，也不阻止后续编辑。快速预览与权威结果逐事件比较；差异写 `E_PREVIEW`，含 `volume_id`、`seq` 和 `difference_fields`。

## 稳定渲染

卷按 `volume_id` 排序，配置版本按 `revision` 排序，判定按 `seq` 排序，结构化错误按写入顺序排序。派生链、受影响卷和变动字段也均稳定排序，便于人工复核和差异比对。
