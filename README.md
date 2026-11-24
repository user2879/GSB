# quantcalib

面向视觉表征与检索模型部署前的本地确定性工具：逐层逐组分位标定、定点重量化、蒸馏校正、三版本回执归一、增量失效与预算调度。

首版不依赖 GPU、网络、数据库或容器。所有数值计算、缓存键、样本和报告均可重复生成；同一输入重复运行时，标定阶段产物逐字节一致。

## 快速验收

环境为 Python 3.11/3.12，依赖安装：

```powershell
python -m pip install --upgrade pip
python -m pip install "numpy>=1.26,<2" "PyYAML>=6,<7"
```

在工作区根目录执行：

```powershell
python -m unittest discover -s tests -v
python -m quantcalib demo --workdir artifacts/demo
```

`demo` 会生成固定种子样本，并依次验证基线、编辑增量、空白全量重建、二次增量、四类故障注入和 q3 回执降级/还原。结果写入：

- `artifacts/demo/reports/baseline.json`
- `artifacts/demo/reports/incremental.json`
- `artifacts/demo/reports/full_rebuild.json`
- `artifacts/demo/reports/incremental_second.json`
- `artifacts/demo/reports/fault_injection.json`
- `artifacts/demo/receipts_roundtrip/roundtrip_summary.json`
- `artifacts/demo/demo_summary.json`

单次运行入口：

```powershell
python -m quantcalib run `
  --weights samples/weights.jsonl `
  --calibration samples/calibration.json `
  --config samples/config.yaml `
  --receipts samples/receipts/q1.jsonl samples/receipts/q2.jsonl samples/receipts/q3.jsonl `
  --state artifacts/manual-state `
  --report artifacts/manual-report.json `
  --print-summary
```

增量命令流与独立故障注入：

```powershell
python -m quantcalib run --weights samples/weights.jsonl --calibration samples/calibration.json --config samples/config.yaml --receipts samples/receipts/q1.jsonl samples/receipts/q2.jsonl samples/receipts/q3.jsonl --commands samples/commands.jsonl --state artifacts/manual-state --report artifacts/manual-incremental.json

python -m quantcalib run --weights samples/weights.jsonl --calibration samples/calibration.json --config samples/config.yaml --receipts samples/receipts/q1.jsonl samples/receipts/q2.jsonl samples/receipts/q3.jsonl --faults samples/faults.yaml --state artifacts/fault-state --report artifacts/fault-report.json
```

校验失败时报告仍会生成，`run` 返回 2；预算待办导致的部分完成返回 0，报告 `status` 为 `partial`。
未显式给出 `--report` 时，默认写入 `<state>/latest-report.json`。

回执互转：

```powershell
python -m quantcalib convert-receipts --input samples/receipts/q3.jsonl --to q1 --output artifacts/q3.as.q1.jsonl --sidecar artifacts/q3.sidecar.json --report artifacts/q3-to-q1.json
python -m quantcalib convert-receipts --input artifacts/q3.as.q1.jsonl --to q3 --output artifacts/q1.as.q3.jsonl --sidecar artifacts/q3.sidecar.json --report artifacts/q1-to-q3.json
```

## 输入契约

### 权重包 JSONL

每行一个层对象。主要字段：

```json
{"layer_name":"encoder.block","shape":[4,3],"weights":[...12 个数...],"content_version":1,"group_granularity":4}
```

- `layer_name` 必须唯一且非空。
- `shape` 维度必须为正整数，乘积必须等于扁平权重元素数；也接受按 shape 嵌套的权重数组。
- 权重必须为有限数。
- `content_version` 必须为非负整数，并按 JSONL 行序单调不减。
- 层标记与配置中的有效分组粒度都必须整除元素数。
- 字段别名：`name/layer`、`values/data`、`version`、`group_size/grouping`。

### 校准样本 JSON

```json
{"input_dim":4,"output_dim":3,"samples":[{"id":"cal-000","input":[...],"reference_output":[...]}]}
```

输入与参考输出维度必须分别等于 `input_dim`/`output_dim`，向量元素必须有限且样本 id 唯一。

由于输入不包含显式模型图，工具用层名、层版本、组号、全局元素下标和输出下标派生稳定的伪随机投影基，将权重变化映射为校准集上的逐输出响应。参考输出保留在报告中作为原始模型审计锚点。该定义完全确定，便于逐层误差归因和缓存复用。

### 配置 YAML

```yaml
quantization:
  scheme: asymmetric             # asymmetric | symmetric
  bit_width: 8                   # 2..16
  group_granularity: 4
  percentile_lower_permille: 100
  percentile_upper_permille: 900
  fractional_bits: 8             # 0..24，进入真实运算与角色指纹
  quantization_levels: 1         # >0，进入真实缩放步长
  scale_effective_digits: 15
layers:
  encoder.block:
    bit_width: 8
    fractional_bits: 8
distillation:
  temperature: 2.0
  steps: 3
  soft_label_weight: 0.5
  tolerance: 0.02
scheduling:
  thread_pool_workers: 3
  global_calibration_point_budget: 64
  per_layer_calibration_quota: 24
  batch_quantization_point_budget: 8
  snapshot_commit_interval_batches: 2
```

非法分位区间产生 `E_CONFIG_INVALID`，位宽越界产生 `E_QUANT_BOUNDS`。非默认调度字段也支持 `workers`、`global_budget`、`layer_quota`、`batch_budget` 等别名。

### 编辑命令流 JSONL

```json
{"op":"replace_layer","layer":"retrieval.head","shape":[2,4],"weights":[...],"content_version":2,"group_granularity":4}
{"op":"patch_config","set":{"layers":{"encoder.block":{"bit_width":6}}}}
{"op":"delete_layer","layer":"optional.layer"}
{"op":"noop"}
```

命令按行顺序应用，替换后仍校验版本单调性、形状和分组整除性。最终数据集指纹不包含命令历史，因此不同命令路径只要得到同一数据，结果一致。

### 量化回执 JSONL

- q1：`layer_name`、`compression_permille`。
- q2：增加 `parameter_version`、`elapsed_ms`。
- q3：增加 `group_scale_digest`、`supersedes`。

加载后会保留 `source_version`、来源文件/行号与 `missing_fields`。未知层产生 `E_RECEIPT_ORPHAN` 并进入失败清单。

### 独立故障注入 YAML

```yaml
injections:
  - {id: F01_NONFINITE, kind: nonfinite_weight, layer: embedding.proj}
  - {id: F02_GROUP_INDIVISIBLE, kind: group_indivisible, layer: encoder.block, value: 5}
  - {id: F03_BIT_WIDTH, kind: bit_width_out_of_range, layer: retrieval.head, value: 32}
  - {id: F04_CALIBRATION_DIM, kind: calibration_dimension_mismatch, sample_index: 0}
```

支持的 kind 还包括 `invalid_quantile_bounds` 与 `orphan_receipt`。命中项编号、对应业务码和文字说明写入报告 `fault_injections`。

## 输出与验证

`run` 的 JSON 报告主要包含：

- 输入指纹与全局内容版本；
- 提交前版本号/配置哈希比较结果；
- 逐层/逐组标定与定点摘要；
- 饱和截断方向及截断前定点值；
- 逐输出元素校正前后误差、零真值绝对误差分支与四类归因；
- q1/q2/q3 归一账本与权威条目裁决依据；
- 全局、单层、单批预算消耗序列与待办；
- 逐层逐阶段缓存命中、失效边界；
- 快照阶段版本、清单字段和业务码计数；
- `result_digest`：用于比较增量运行与空白全量重建的核心结果摘要。

详细算法、失效键、快照事务与本次实测结果见 `TECHNICAL.md`。
