# groupcalib：逐列组分位标定、饱和截断与增量一致性流水线

`groupcalib` 从零实现一个面向表格特征的逐列组定点量化组件。每个列组独立完成分位标定、缩放因子/零点推导、定点重缩放、饱和截断、反量化、逐元素误差归因和分层敏感性分析，并通过阶段快照与内容指纹在批次编辑后只重算受影响列组。

实现不依赖外部账号、网络服务、GPU 或 Docker。运行期只使用 Python 3.11 标准库；已安装的 NumPy 2.x 可以直接共存，但本项目不调用 NumPy API，因此也没有 NumPy 版本相关的浮点行为差异。

## 快速开始

```bat
python -m pip install "numpy>=2,<3"
python generate_inputs.py --out inputs
python -m groupcalib.pipeline --config inputs/config.json --table inputs/features.jsonl --plan inputs/plan.json --out out
```

成功时进程退出码为 `0`，终端输出一行确定性 JSON 摘要。校验失败时以非零退出码结束，并在 `out` 下写入 `diagnostics.json`、`run_report.json`、`failures.jsonl` 等诊断文件。

可选参数：

- `--through-batch BATCH_ID`：只应用编辑计划中截至该批次（含）的动作，用于逐批快照和增量验收。
- `--full-rebuild`：忽略已有阶段缓存，仍使用同一套确定性代码全量重建。

## 工作区结构

```text
generate_inputs.py          固定种子 20261123 的样本生成器
inputs/config.json          3 个列组、8 列、4/6/8 bit 候选、3 bit 小数位宽
inputs/features.jsonl       24 行确定性浮点特征表，含缺失和尾部异常值
inputs/plan.json            5 批修正、追加、删除、行重分组、列重划分
groupcalib/
  config.py                 配置校验
  table.py                  JSONL 校验、表指纹、分组指纹
  plan.py                   编辑计划加载和事务式动作应用
  quantile.py               分位、缩放因子、零点
  quantize.py               定点重缩放、饱和截断、归因
  sensitivity.py            组/位宽分层结论
  keys.py                   唯一缓存键来源
  scheduler.py              稳定顺序预算调度与待办快照
  cache.py                  不可变阶段快照、原子指针、陈旧恢复
  pipeline.py               编排、线程池、CLI、产物写出
tests/test_pipeline.py      11 个端到端与单元验收测试
VALIDATION.md               本次实际运行结果
```

重新生成样本不会访问网络：

```bat
python generate_inputs.py --out inputs
```

## 配置格式

`inputs/config.json` 展示了完整可运行配置。主要字段如下：

| 字段 | 说明 |
| --- | --- |
| `schema_version` | 当前固定为 `1`。 |
| `table_name` | 表名，非空。 |
| `tensor_dimensions` | 张量维度数组，例如 `[24, 8]`。 |
| `column_axis` | 列轴，必须在 `tensor_dimensions` 下标范围内；若该维不为 `null`，还须等于表的特征列数。 |
| `row_id_field` / `group_id_field` | 行标识与组标识字段名，必须不同。 |
| `missing_value_markers` | 缺失标记数组，可包含 `null` 和字符串。 |
| `column_groups` | 组标识到非空列数组的映射。列必须只出现一次，且覆盖表内全部特征列。 |
| `quantile_permille.lower` / `upper` | 千分比分位，要求 `0 <= lower < upper <= 1000`。 |
| `scheme` | `symmetric` 或 `asymmetric`。 |
| `quantization.bit_width_candidates` | 位宽候选，必须是 `{2,3,4,5,6,8,10,12,16}` 的非空无重复子集。 |
| `quantization.fractional_bits` | 定点缩放因子量化的小数位宽，要求小于最小候选位宽。 |
| `quantization.gear` | 0 基候选下标；活动位宽为 `bit_width_candidates[gear]`。 |
| `clipping_strategy` | 当前固定支持 `saturate`。 |
| `tolerance.absolute` / `relative` | 绝对/相对误差容差，非负。 |
| `tolerance.comparison` | `either` 表示绝对或相对任一超差；`both` 表示两者同时超差。 |
| `tolerance.outlier_ratio_limit` | 组/位宽分层超差比例上限，范围 `[0,1]`。 |
| `tolerance.tie_epsilon` | 归因并列判定的附加绝对阈值。 |
| `tolerance.zero_relative_branch` | 固定为 `absolute_only`；零元素不使用相对误差。 |
| `thread_pool_size` | 线程池最大工作线程数，`1..64`，实际提交的并发任务不会超过它。 |
| `element_budget_per_batch` | 每个编辑批次可消耗的单批标定元素预算，正整数。 |
| `interpolation` | 固定为 `linear`；实现为 type-7/线性位置插值。 |
| `tie_break` | 固定为 `lower`；相邻并列值回落到低位序统计量。 |

配置结构错误使用 `E_CONFIG_INVALID`。表内列集合不满足划分、列轴维度不匹配等也归为配置错误；失败时不会提交新的阶段快照。

## 编辑计划格式

编辑计划根对象包含 `schema_version` 和 `batches`。每个批次有唯一的 `batch_id` 与 `actions` 数组。每个动作是事务式的：校验失败的动作不会部分生效。

支持的动作为：

```json
{"op": "update", "row_id": "r0003", "values": {"score": 7.75}, "group_id": "g_sensor"}
{"op": "append", "row": {"row_id": "r0025", "group_id": "g_sensor", "temp": 41.5}}
{"op": "delete", "row_id": "r0011"}
{"op": "regroup", "row_id": "r0005", "group_id": "g_ratio"}
{"op": "repartition_columns", "column_groups": {"g_numeric": ["age", "income"]}}
```

- `update.group_id` 和 `regroup` 可触发行级重新分组；旧组和新组都会失效。
- `repartition_columns` 必须重新覆盖全部表列、每组非空、所有行当前使用的组标识仍被声明；所有受影响的旧组和新组都会失效。
- 追加行未提供的列按缺失处理；不能追加表结构之外的新列。
- 未知组使用 `E_GROUP_UNKNOWN`，其他非法行/动作使用 `E_DATA_INVALID`，计划结构错误使用 `E_CONFIG_INVALID`。

## 计算语义

### 分位与缩放因子

对组内每列的非缺失值升序排序。分位位置固定为：

```text
h = (n - 1) * permille / 1000
lower = floor(h), upper = ceil(h)
value = ordered[lower] + (h - lower) * (ordered[upper] - ordered[lower])
```

若上下位置值相等，直接采用低位序值。插值方法、并列策略会写入标定产物元数据。

固定缩放栅格为 `2^-fractional_bits`。先由分位构造 `scale_raw`，再执行：

```text
scale_units = max(1, round_half_away(scale_raw * 2^fractional_bits))
scale = scale_units / 2^fractional_bits
```

对称方案：

```text
qmin = -2^(w-1), qmax = 2^(w-1)-1
scale_raw = max(|q_low|, |q_high|) / qmax
zero_point = 0
```

非对称方案：

```text
qmin = 0, qmax = 2^w-1
scale_raw = (q_high - q_low) / (qmax - qmin)
zero_point_unclipped = round_half_away(qmin - q_low / scale)
zero_point = clip(zero_point_unclipped, qmin, qmax)
```

定点中间值和反量化值均为真实数值计算：

```text
q_unclipped = round_half_away(x / scale) + zero_point
q_value = clip(q_unclipped, qmin, qmax)
dequantized = (q_value - zero_point) * scale
```

`q_value != q_unclipped` 时记录 `clipping_direction` 为 `low` 或 `high`，同时保留 `q_unclipped`。宽度、gear、小数位宽都会改变缩放因子、定点整数和反量化值。

### 误差归因

每个非缺失元素输出四个归因判据：

- `quantile_discretization`：以全量观测最小值/最大值同法标定为参考，参考反量化与所选分位反量化之差。
- `scale_quantization`：原始 scale 与固定小数栅格 scale 引起的反量化差异，加零点裁剪差值。
- `saturation_truncation`：未截断反量化与截断后反量化之差。
- `dequant_rounding`：原值到未截断反量化的差值。

主导来源取判据最大值；差值在 `tie_epsilon` 内视为并列。固定优先级为：

```text
saturation_truncation > quantile_discretization > scale_quantization > dequant_rounding
```

产物包含 `criterion_values`、`dominant_error_source`、`source_tie`、`tied_sources`、`source_priority` 和 `attribution_trace`，可逐元素追溯所选/参考 scale、零点、定点整数及截断结果。

非零元素相对误差为 `abs_error / abs(x)`。精确零元素强制 `relative_error = null`、`zero_value_branch = true`，只走绝对误差分支。容差通过条件严格使用 `>`，等于容差视为通过。

### 分层敏感性

每组在每个位宽上输出一层，统计：

- 最大相对误差，以及零分支最大绝对误差；
- 截断元素数/比例；
- 超差元素数/比例；
- 总体与超差元素中的主导来源计数和份额；
- 固定格式的 `conclusion` 与 `layer_status`。

当 `outlier_ratio > outlier_ratio_limit` 时，额外写入 `threshold_exceedances.jsonl`，键为 `group_id + bit_width`。

## 缓存、失效与快照

四个阶段固定为：

```text
quantile_calibration -> fixed_rescale -> error_attribution -> sensitivity
```

每个组/位宽的阶段缓存键由以下内容派生：

```text
阶段名 + 组内容指纹 + 参数指纹 + bit_width + fractional_bits + 上游缓存键
```

内容指纹按行标识排序，并覆盖该组当前列集合；参数指纹覆盖分位、方案、插值、并列策略、小数位宽、截断、容差等阶段参数。阶段产物记录 `row_ids`、列集合、所有阶段缓存键和参数指纹。

失效规则：

- 单行修正、追加、删除、行重分组：只使旧组/新组及其四个阶段失效。
- 列重划分：使列集合发生变化的组及所有受重划分影响的组重算。
- 未受影响组直接命中阶段缓存；`reuse_recompute.jsonl` 记录实际命中/重算。
- `canonical_reuse_plan.jsonl` 是同计划语义下的规范失效清单，不依赖磁盘初始缓存；同一计划从空白、分批或部分快照继续运行均逐字节一致。

阶段快照先写临时目录，再落到不可变版本目录，最后原子替换 `CURRENT` 指针。`manifest.json` 记录全局版本、配置哈希、列组划分哈希、表哈希、计划哈希、已完成组/组位宽、预算、以及四个阶段产物版本。

启动恢复：

- `global_version` 或配置哈希不一致：输出 `E_SNAPSHOT_STALE`，忽略全部旧阶段并重建。
- 单阶段 `CURRENT`、清单版本或内容哈希不一致：只废弃对应阶段及下游阶段，输出带 `stage` 的 `E_SNAPSHOT_STALE`，从该阶段重建。
- 恢复事件不导致失败退出；若随后数据/配置有效，最终产物仍可与全量重建逐字节一致。

## 预算调度

任务按 `(group_id, column, bit_width)` 稳定排序，每次扣减当前队首任务。单个任务可跨预算批次拆分，但绝不会跳过队首：

- 每个预算批次写入 `batch_ledger.jsonl`，包含扣减序列、已完成任务、待办快照和待办剩余元素数。
- 有待办时写入 `E_CALIB_BUDGET_PENDING` 业务事件；下一预算批次优先继续队首待办。
- 已完成组结果不会被后续预算周期改写。
- 默认 CLI 自动推进预算周期直到队列清空。若外部限制周期数，仍有待办的组/位宽不会进入敏感性统计，也不会得到默认缩放因子。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 全局版本、哈希、完成组、预算、阶段版本。 |
| `diagnostics.json` / `run_report.json` | 校验码、事件、预算、并发、复用、阈值等运行报告。 |
| `failures.jsonl` | `E_CONFIG_INVALID`、`E_DATA_INVALID`、`E_GROUP_UNKNOWN` 失败清单，确定性排序。 |
| `business_events.jsonl` | `E_CALIB_BUDGET_PENDING`、`E_SNAPSHOT_STALE` 等非致命业务事件。 |
| `calibration.jsonl` | 每个组/位宽/列的 quantile、scale、zero point 和元数据。 |
| `quantized.jsonl` | 每个组/位宽/行/列的原始值、定点值、截断方向、反量化值。缺失单元也保留记录。 |
| `attribution.jsonl` | 非缺失元素的绝对/相对误差、超差、四源判据、主导来源和追溯值。 |
| `sensitivity.jsonl` | 每组/位宽分层指标、来源份额和结论。 |
| `groups.jsonl` | 组/位宽的行集合、列集合、参数指纹和阶段缓存键。 |
| `threshold_exceedances.jsonl` | 超过配置超差比例上限的组/位宽组合。 |
| `reuse_recompute.jsonl` | 本次实际缓存命中/重算记录。 |
| `canonical_reuse_plan.jsonl` | 只依赖计划语义的规范复用/重算清单。 |
| `batch_ledger.jsonl` | 预算扣减序列和待办快照。 |
| `batch_history.jsonl` | 每个编辑批次的受影响组、计算组数和预算摘要。 |
| `effective_table.jsonl` | 应用全部选定编辑后的表，按行标识排序。 |
| `effective_groups.json` | 最终列组划分。 |
| `stages/<stage>/CURRENT` | 当前阶段不可变版本指针。 |
| `stages/<stage>/<version>/*.json` | 每组/位宽的阶段产物快照。 |

## 退出码与业务码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功；可能包含已恢复的陈旧快照或已清偿的预算待办。 |
| `2` | 存在 `E_CONFIG_INVALID`，或计划结构不合法。 |
| `3` | 存在 `E_DATA_INVALID` / `E_GROUP_UNKNOWN`；有效行仍可产出结果。 |
| 其他非零 | 运行期异常。 |

配置无效等致命错误会先清理该 `out` 下旧的成功产物，再写诊断，避免旧结果冒充本次结果。数据行级错误只排除失败行；其他有效行继续进入阶段产物。

## 确定性边界

- 所有 JSON 使用固定键排序和紧凑 JSONL 编码，不写入时间戳、随机数或线程完成顺序。
- 所有循环按组标识、列名、位宽、行标识排序；线程结果收集后再次排序。
- 相同输入、配置、计划和相同初始缓存状态下，产物逐字节一致。
- 同一计划从空白全量、分批继续或快照恢复，规范数据产物、规范复用清单、预算账本和失败清单顺序一致。
- `reuse_recompute.jsonl` 是物理缓存操作记录；若从不同初始缓存状态启动，命中/重算状态会真实不同，这是缓存复用语义本身决定的。需要跨初始状态比较时使用 `canonical_reuse_plan.jsonl`。

实际验证命令、计数和 SHA-256 见 [VALIDATION.md](VALIDATION.md)。
