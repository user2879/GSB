# 设计与一致性说明

## 1. 阶段与输入指纹

引擎按以下阶段构建：

1. **解析**：读取配置、请求流和每个回执文件；逐行归一化 r1/r2/r3。
2. **归属归一**：按 `(batch_id, worker_id)` 聚合，消解重复投递，选出每个请求的唯一权威回执。
3. **资源计算**：按批次、工作进程维度、耗时和量化策略计算整数定点指标。
4. **账本汇总**：形成请求条目、批次、失效/复用清单、失败清单、诊断和降级记录。

输入全局版本是以下内容的规范化 SHA-256：

```text
parser_schema_version
config_hash
requests_hash
每个回执文件路径与内容指纹
fault_config_hash
render_version
```

命令游标不进入账本内容，只进入快照清单。这样同一最终事实集合无论经过多少命令，账本都不变；但命令游标变化会阻止错误续跑并重新提交清单。

## 2. 显式缺失与三版本归一

内部模型用 `MISSING(field)` 哨兵表示缺失，同时记录排序后的 `missing_fields`。序列化时转为：

```json
{"__missing__":"duration_ms"}
```

r1 天然缺少 r2/r3 字段；r2 缺少 r3 字段。解析成功不等于字段存在，缺失仍会显式保留。r1 若通过来源元数据解析工作进程和参数版本，仅用于归属上下文，不会把来源值伪装成回执字段。

## 3. 双向转换与旁路

`r3 -> r1` 时，所有目标版本不支持的现有字段写入 sidecar：

```json
{
  "receipt_key": "b-alpha-001/r-001",
  "source_version": "r3",
  "target_version": "r1",
  "preserved_fields": {
    "duration_ms": 14,
    "param_version": 4,
    "worker_id": "w-alpha",
    "rank": 0,
    "covered_refs": [],
    "param_digest": "sha256:..."
  },
  "lost_fields": []
}
```

重新升级时逐字段恢复。sidecar 缺失或损坏时字段仍为显式缺失，并在 `lost_fields` 中列出。账本通过 `--render-version r1` 先降级、写旁路、再升回原版本，从而记录 `degradations` 而不改变权威数值。

`r1 -> r3` 无旁路时，缺失的参数版本、耗时、工作进程、名次、覆盖引用和参数摘要全部以哨兵补齐，完整性标签为 `explicit-missing` 或 `partial-restore`。

## 4. 请求归属裁决

先过滤以下记录并形成失败清单：

- 工作进程字段与来源上下文冲突；
- 批次内参数版本冲突；
- 重复 `(batch_id, request_id)`；
- r3 批次内重复 `rank`；
- 完全未知的参数版本。

再由配置 `merge_rule.entries` 决定参与归属的 `(worker_id, param_version)` 组合。每个请求的候选回执形成覆盖图：

1. 若只有一条候选，依据为 `single_delivery`。
2. 若有覆盖边且只有一个未被覆盖的根，依据为 `coverage_ref`。
3. 若有多个根，按整数 `duration_ms` 升序（缺失最后），再按参数版本降序、批次标识升序；依据为 `coverage_ref_then_duration_ms_asc`。
4. 若无覆盖边，按参数版本降序、批次标识升序；依据为 `param_version_desc_batch_id_asc`。
5. 若覆盖图成环，使用无覆盖规则回退并报 `E_COVERAGE_CYCLE`。

每条账本条目保留所有投递、`covers`、`covered_by` 和覆盖边。落选投递不会从账本消失。

## 5. 参数版本失效边界

批次的活动参数版本来自配置；回执解析出的版本来自字段或来源默认值。判定：

```text
param_version == active_param_version  -> 复用边界
param_version != active_param_version  -> 失效边界，必须重算
param_version 无法解析                 -> 失效边界
```

因此某工作进程版本切换时，只有其旧版本批次和这些批次的请求归属进入重算；其他工作进程批次保持复用。账本同时输出：

- `batches_invalidated` / `batches_reused` / `batches_excluded`
- `invalidated_request_ids`（权威条目失效）
- `invalidated_scope_request_ids`（失效批次中的全部请求引用）
- `reused_request_ids`
- `dependency_chain`（工作进程 → 批次）

资源缓存以内容键寻址；失效批次不允许命中缓存，未失效批次可命中。缓存命中状态不写入账本，因此不会让增量与全量结果出现差异。

## 6. 整数定点资源公式

模型的每请求乘加量：

```text
ops_per_request = 8*m*h + 4*m*m + 2*h
raw_ops = request_count * ops_per_request
```

位宽 `b`：

```text
quantized_step  = 2^max(0, 32-b)
precision_factor = 2^max(0, b-8)
fixed_macs = floor(raw_ops * quant_scale * precision_factor / quantized_step)
             * quantized_step
retained_decimal_digits = max(1, (b+3)//4)
utilization_scale = 10^retained_decimal_digits
capacity = duration_ms * throughput_ops_per_ms * quant_scale * precision_factor
utilization_units = floor(fixed_macs * utilization_scale / capacity)
```

全程使用 Python 整数，不经过浮点。位宽变化会同时改变定点精度因子、整数 McA、利用率整数和保留位数。耗时缺失或为零时标记 `unavailable` / `zero_duration`，不伪造利用率。

## 7. 确定性排序

- 请求账本：`request_id` 字典序。
- 批次：`(batch_id, worker_id)`。
- 投递链：批次、工作进程、来源路径、来源行。
- 失败：路径、行号、业务码、明细码、条目、消息。
- 诊断：`injection_id`。
- 降级：回执键、源版本、目标版本。

所有哈希使用 `sort_keys=True`、紧凑分隔符和 UTF-8 的规范化 JSON。输出中不含生成时间、PID、耗时等非事实字段。

## 8. 快照与恢复

每次运行先在 `.state/staging/<随机目录>/` 写完整阶段产物：

```text
parse.json
attribution.json
resources.json
ledger.json
conversion_sidecar.json
manifest.json
```

全部完成后把临时目录整体 `os.replace` 到 `.state/runs/run-...`，最后原子替换 `.state/CURRENT.json` 指针。`CURRENT` 指向的目录和账本版本一致才可续跑。

清单记录全局版本、配置哈希、请求哈希、回执文件指纹、故障配置哈希、命令游标、各阶段产物哈希和账本版本。启动状态分为 `cold`、`resume_available`、`input_changed`、`config_changed`、`cursor_changed`、`corrupt`。

## 9. 快速与权威视图

权威视图按固定账本顺序补齐所有条目。快速视图输出同一顺序的已完成前缀，并持续报告：

- `completed / total`
- `invalidated_completed / invalidated_total`
- 当前归属摘要

预算耗尽时状态为 `partial`，业务码为 `E_RESPONSE_BUDGET_EXCEEDED`。共同完成范围由 `compare_views` 逐层比较，差异以 `E_LEDGER_DIVERGED` 报出，`context.fields` 给出具体字段路径。

## 10. 故障注入隔离

故障配置不修改源文件，而是在解析和归属阶段按 `(path, line)` 变换临时对象。每项有独立 `injection_id` 和预期业务码：

| 类型 | 命中位置 | 业务码 |
| --- | --- | --- |
| `truncate_line` | 解析 | `E_RECEIPT_MALFORMED` |
| `non_finite_duration` | 解析 | `E_RECEIPT_MALFORMED` |
| `unknown_param_version` | 版本校验 | `E_UNKNOWN_PARAM_VERSION` |
| `duplicate_rank` | 批次归属 | `E_DUPLICATE_RANK` |

诊断报告汇总 `hit`、`hit_count`、`observed_codes`、`observed_details` 和消息；未命中可被显式识别。

## 11. 已知边界

- 批成员由回执中的 `batch_id` 决定；请求流和 `batch_size` 用于生成/重生成回执。未重新生成时，调整 `batch_size` 不会凭空改变磁盘上的批次事实。
- r1 不携带工作进程时，根因归属只能依赖配置中的来源元数据；若来源也未提供，则该批次无法被归并规则选中。
- 快速视图先尝试精确快照续跑；冷启动时预算若已在最小配置/请求读取后耗尽，直接返回基于请求总数的 `0/N` 部分摘要，不再进入完整构建。构建消耗的时间会从后续视图装配预算中扣除。预算极小时允许返回 0 条前缀，不伪造未完成结果。
