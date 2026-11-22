在空白工作区构建一个面向逐列量化标定与饱和截断的算子诊断工作台，核心业务问题不是矩阵乘累加链，而是每个张量按列做分位标定、零点推导、饱和截断与反量化后的误差归因：逐列标定使得同一张量不同列拥有不同缩放因子与零点，饱和截断又会让部分元素落在可表示区间之外，误差既来自分位插值规则的离散选择，也来自截断方向与反量化舍入的叠加。输入为一份逐列标定配置 JSON（张量形状与列轴、分位下限与上限千分比、标定方案为对称或非对称、量化位宽候选、定点小数位宽与量化档位、饱和截断策略、容差、分层列组、线程池并发数、单批标定元素预算、快照提交间隔批次）与一份固定种子的样本描述，运行环境为 Windows 本地，输出可复核的 JSON 诊断报告。

任务环境与准备：
确认 Windows 本地可运行 Python 3.10 及以上解释器。
安装 NumPy 依赖，仅用于 float32 与 float64 数组运算及固定种子样本生成。
在空白工作区新建项目目录，准备固定种子的逐列标定配置 JSON 与样本描述文件，不依赖网络或外部数据。
确保工作区可写，用于输出临时目录、整体替换的快照产物与 JSON 诊断报告。

业务要求：
1. 实现配置与样本加载校验层：读取逐列标定配置与固定种子样本描述，校验张量形状维度为正整数、列轴在维度范围内、分位下限千分比小于上限千分比且均在 0 至 1000 之间、量化位宽落在允许枚举内、定点小数位宽为正、饱和策略在枚举内、容差为正有限数、并发数为正整数；对不合法配置分别输出 E_CONFIG_INVALID 与具体出错字段并写入失败清单，其余合法张量继续处理。
2. 实现逐列分位标定与缩放因子、零点推导：对每列在其非缺失元素上以升序排序后按分位千分比取分位值，插值规则与并列值决断规则固定并写入产物元数据；对称方案取绝对值最大分位构造缩放因子，非对称方案按最小值与最大值区间构造缩放因子与零点，缩放因子须用 float64 计算后按定点小数位宽量化并记录量化前后差值；同一列在相同分位参数下多次运行必须逐字段一致。
3. 实现逐列定点重缩放与饱和截断内核：把浮点张量按列缩放因子与零点转为定点整数，超出量化位宽允许区间的元素按配置饱和策略处理并逐元素记录截断方向与截断前定点值；定点小数位宽与量化位宽切换必须真实改变中间整数与反量化结果，不得仅改显示；零缩放因子列须按配置退化规则处理并输出 E_SCALE_ZERO 记录列标识。
4. 实现反量化与逐列误差归因：反量化后逐元素计算绝对误差与相对误差，按配置容差判超差，相对误差在真值为零时须改用绝对误差单独判定并标注该分支；对每个超差元素区分其误差主要来自分位离散、缩放量化、饱和截断还是反量化舍入，按固定归因优先级输出主导来源与四项份额，份额之和与总误差之差须在容差内。
5. 实现列组分层敏感性分析：按配置的列组标号把列分层，统计各组最大相对误差、饱和截断元素比例、超差比例与主导误差来源份额并给出每层结论；当某组在某量化位宽下超差比例高于配置上限时记录组标识与位宽组合，分层结论须随分位参数与位宽变化而变化，不得输出与参数无关的固定结论。
6. 实现基于内容指纹的列级精确失效与重算范围判定：为分位标定、缩放因子与零点、定点重缩放与反量化误差各阶段产物计算由列内容指纹、标定参数指纹与位宽档位派生的缓存键，并记录每个列结果引用的样本元素集合；当替换样本、修改分位参数、切换量化位宽或饱和策略时仅使受影响列及其下游失效并纳入重算集合，未失效列必须直接复用持久化产物且不得重算。
7. 保证任意编辑序列后增量运行与从空白全量重建逐字段一致：施加包含样本替换、分位参数调整、量化位宽与定点小数位宽切换、饱和策略变更与列组重新划分的编辑序列，每次增量运行得到的逐列缩放因子、零点、逐元素定点值、反量化值、误差归因、分层表、复用与重算清单及失败清单顺序，必须与按相同最终配置和样本一次性全量运行的结果完全相同。
8. 实现单批标定元素预算约束下的并发调度、阶段快照整体替换与诊断报告输出：使用线程池并发执行逐列标定与重缩放，最大并发数取自配置且不得超过，每个列任务携带由列标识与阶段名派生的稳定排序键，最终列顺序、逐元素结果与失败清单须与单线程按列标识顺序执行一致；调度器按列标识稳定顺序消耗单批标定元素预算，超预算列保留待办并以 E_CALIB_BUDGET_PENDING 标记；各阶段产物先写临时目录再整体替换提交，清单记录全局版本号、配置哈希、已完成列列表与产物版本，启动时以版本号判定是否续跑；每次完整运行输出含超差摘要、归因份额、分层表、注入命中编号与各阶段耗时的 JSON 报告。

执行约束：不使用 Docker 或其他容器环境；使用本地确定性样本验证，不访问未授权目标。按上述任务意图交付相应代码、测试或技术说明，记录实际验证结果。

验收：
用本地固定种子样本验证逐列标定、饱和截断与误差归因，并与 float64 参考路径逐列比对，检查增量与全量结果一致。

---

# 实现与运行说明

## 快速开始（Windows PowerShell）

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt

py -m qcolumn_diag run `
  --config examples\config.json `
  --sample examples\sample.json `
  --state-dir runtime `
  --report reports\diagnostic_report.json
```

再次运行同一配置和样本时，已完成的列直接从持久化快照复用。空白全量重建与报告比较：

```powershell
py -m qcolumn_diag run --config examples\config.json --sample examples\sample.json `
  --state-dir runtime_full --report reports\diagnostic_report_full.json --full-rebuild

py -m qcolumn_diag compare `
  --left reports\diagnostic_report.json `
  --right reports\diagnostic_report_full.json `
  --output reports\compare_incremental_full.json
```

测试：`py -m pytest -q`。项目不使用 Docker、网络数据或外部样本；NumPy 只用于 float32/float64 数组、固定种子生成和 float64 参考校验。

## 配置语义

配置支持顶层单张量，也支持 `tensors` 数组。每个 `tensors[]` 条目独立校验；某个张量非法时只跳过该张量，其余合法张量继续处理。

```json
{
  "schema_version": "1.0",
  "config_id": "demo",
  "interpolation": "linear",
  "tie_break": "stable_ascending_first_occurrence",
  "scale_quantization": "round_half_away_from_zero",
  "allowed_bit_widths": [4, 6, 8],
  "max_workers": 4,
  "calibration_element_budget": 96,
  "snapshot_commit_interval_batches": 2,
  "tensors": [
    {
      "tensor_id": "activation_tile",
      "shape": [10, 12],
      "column_axis": 1,
      "quantile_lower_permille": 50,
      "quantile_upper_permille": 950,
      "scheme": "asymmetric",
      "bit_width_candidates": [4, 8],
      "active_bit_width": 8,
      "fractional_bits": 6,
      "saturation_strategy": "saturate",
      "zero_scale_policy": "clamp_to_min_scale",
      "tolerance": 0.03,
      "group_over_tolerance_limit": 0.3,
      "column_groups": [
        {"group_id": "left", "columns": [0, 1, 2, 3, 4, 5]},
        {"group_id": "right", "columns": [6, 7, 8, 9, 10, 11]}
      ]
    }
  ]
}
```

- `shape` 必须全为正整数，`column_axis` 必须在维度范围内。
- `quantile_lower_permille < quantile_upper_permille`，且二者都在 `[0, 1000]`。
- 每个候选位宽必须属于 `allowed_bit_widths`；`active_bit_width` 必须同时位于候选中。实际码值档位数为 `2 ** bit_width`。
- `fractional_bits` 控制缩放因子的定点分辨率 `2^-fractional_bits`，不是显示参数。
- `saturation_strategy` 支持 `saturate`、`wrap`、`reject`；`zero_scale_policy` 支持 `clamp_to_min_scale`、`reject`。
- `tolerance` 是正有限数。非零真值按相对误差判超差；真值为零改用绝对误差，并记录 `absolute_at_zero` 分支。
- `max_workers`、`calibration_element_budget`、`snapshot_commit_interval_batches` 必须是正整数。

非法字段产生 `E_CONFIG_INVALID` 并携带具体字段路径；样本错误产生 `E_SAMPLE_INVALID`。

## 固定种子样本

```json
{
  "schema_version": "1.0",
  "sample_id": "fixed-seed-v1",
  "default_dtype": "float64",
  "tensors": {
    "activation_tile": {
      "dtype": "float64",
      "shape": [10, 12],
      "generator": {"kind": "normal", "seed": 20261123, "loc": 0.25, "scale": 1.35},
      "column_offsets": [-2.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 3.0, -3.0, 0.0, 1.5, -1.5],
      "column_scales": [0.6, 0.8, 1.0, 1.2, 1.4, 0.9, 1.1, 1.6, 1.8, 1.0, 1.3, 0.7],
      "constant_columns": {"9": 0.0},
      "missing_values": [[1, 2], [4, 5]],
      "injections": [
        {"id": "INJ-001", "row": 2, "column": 0, "operation": "set", "value": 5.5}
      ]
    }
  }
}
```

- 生成器支持 `normal`、`uniform`，使用 `numpy.random.PCG64(seed)`。
- `dtype` 仅支持 `float32`、`float64`；计算和校验路径统一提升为 float64。
- 缺失元素为 NaN。索引支持完整坐标，也支持二维简写 `row`/`column`。
- 注入操作支持 `set`、`add`、`multiply`，报告记录实际命中编号。

## 数值路径

### 分位、缩放因子和零点

每列只在非缺失元素上稳定升序排序，固定使用线性插值：

```text
pos = q * (n - 1)
q_value = v[floor(pos)] + (pos - floor(pos)) * (v[ceil(pos)] - v[floor(pos)])
```

- 对称：`ideal_scale = max(abs(q_low), abs(q_high)) / ((2^(b-1)) - 1)`，零点为 0。
- 非对称：`ideal_scale = (q_high - q_low) / (2^b - 1)`。
- 缩放因子用 float64 计算，再四舍五入到 `2^-fractional_bits` 网格；量化前后绝对差、相对差和网格单位写入报告。
- 非对称零点为 `code_min - q_low / quantized_scale`，按 half-away-from-zero 取整并夹到码值区间。
- 零缩放列记录 `E_SCALE_ZERO`；`clamp_to_min_scale` 退化到最小正网格，`reject` 保留零因子诊断结果。

### 定点重缩放、饱和和反量化

```text
exact_code = value / quantized_scale + zero_point
preclip_code = round_half_away(exact_code)
quantized_code = strategy(preclip_code, code_min, code_max)
dequantized = (quantized_code - zero_point) * quantized_scale
```

每个元素保留 `exact_code`、`preclip_code`、`quantized_code`、截断方向、是否截断、反量化值、绝对误差、相对误差、容差分支和归因份额。`saturate` 夹到边界，`wrap` 环绕，`reject` 记录拒绝饱和失败。

### 误差归因

四项因果量为：

1. `quantile_discretization`：观测 min/max 全范围反事实重建与分位范围重建之差。
2. `scale_quantization`：理想 float64 scale/zero-point 与定点网格 scale/zero-point 的反事实差。
3. `saturation_clipping`：截断前后码值差乘 scale。
4. `dequant_rounding`：整数码舍入前后差乘 scale。

固定优先级：`saturation_clipping > dequant_rounding > scale_quantization > quantile_discretization`。四个原始量按元素归一化到该元素绝对误差，最大原始分量决定主导来源；逐元素记录份额和、总误差和差值。总体摘要提供 `attribution_share_error_max`、`attribution_share_violation_count`、`attribution_share_sum_within_tolerance`。

### 分层敏感性

每个组在每个候选位宽上统计覆盖率、最大有效相对误差、饱和比例、超差比例、四项归因份额和主导来源。零真值列以绝对误差参与最大有效相对误差，并记录 `zero_reference_count`。超差比例严格高于 `group_over_tolerance_limit` 时输出组×位宽告警；结论文本包含分位参数、位宽、定点位宽和实测指标。

## 增量缓存、预算与快照

每列分为三个阶段：

1. `quantile_calibration`
2. `scale_zero_point`
3. `requantize_dequantize_errors`

缓存键由阶段名、列内容指纹、样本元素引用指纹、标定参数和位宽档位派生。样本替换只失效内容变化的列；分位参数变化失效标定及其下游；定点位宽变化失效缩放及下游；饱和策略、容差、活动位宽只失效重缩放与误差阶段；列组重划分不触发重算。每个列结果保存 `sample_element_refs`，记录引用的样本元素和注入编号。

调度器按列标识稳定排序，以非缺失元素数消耗单批预算。列成本超过整个批次预算时保留待办并记录 `E_CALIB_BUDGET_PENDING`。其余列按预算切成稳定批次，线程池峰值不超过 `max_workers`。任务键为 `column_id|calibration+requantization`；最终列、元素和失败清单按稳定键排序。

快照提交流程：

1. 在 `runtime/tmp` 写完整 `snapshot.json` 并 `fsync`。
2. 将临时目录重命名为不可变代次目录 `runtime/snapshots/vNNNNNNNN`。
3. 原子替换 `runtime/CURRENT` 指针。
4. 保留最近几代。

清单保存 `run_version`、配置哈希、样本哈希、完成列、待办列、产物版本和失败清单。内容版本一致且完整时，下次启动全部复用。

## 报告与一致性边界

`reports/diagnostic_report.json` 包含逐列标定参数、逐元素码值/反量化/误差/归因、候选位宽参数、分层表、注入命中、失败清单和各阶段耗时。`equivalence.deterministic_result_hash` 覆盖配置、样本元数据、逐列结果、分层、失败和汇总。

数值与诊断域在“任意编辑序列后的增量运行”与“相同最终配置/样本的空白全量重建”之间逐字段一致，并通过测试中的确定性哈希校验。缓存复用/重算清单、快照代次、并发峰值和耗时属于执行历史，增量与空白全量按定义不同，因此不纳入确定性哈希；`compare` 命令比较结果域。未失效列直接复用和空白全量结果等价分别有独立测试。

## 实际验证结果

- `python -m pytest -q`：`24 passed`。
- 示例：2 个张量、20 列、184 个元素，其中 7 个缺失。
- float64 参考：20/20 列通过；分位最大差 `4.44e-16`；码值不一致 0；反量化与误差最大差 0。
- 逐元素归因份额最大和误差 `1.11e-16`，份额校验失败数 0。
- 零缩放列 `activation_tile:c000009` 在 4-bit、8-bit 下分别输出 `E_SCALE_ZERO`。
- 4-bit 与 8-bit 分层结论不同，示例产生 7 个组×位宽超差告警。
- 首次运行、同版本复用运行、空白全量重建的确定性哈希均为 `5dcd6e55a5904216b3d9a984b7cd253c72c7d6594199db202557762f1129a0b4`。
- `compare` 对复用/全量均返回 `equal: true`，详见 `reports/compare_*.json`。

实际产物：`reports/diagnostic_report.json`、`reports/diagnostic_report_full.json`、`reports/validation_summary.json`、`reports/compare_reuse.json`、`reports/compare_incremental_full.json`。
