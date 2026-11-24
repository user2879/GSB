# LLM 权重量化与蒸馏校正管线

本项目从本地确定性样本出发，完成逐层分位标定、分组定点重量化、饱和截断、反量化误差归因、校准集软标签蒸馏校正、层级缓存失效、点数预算调度、阶段化快照恢复和异常注入隔离。实现只依赖 Python 标准库与 NumPy；PyYAML 可用时优先使用，否则内置的 YAML 子集解析器也能读取本项目生成的 JSON 兼容 YAML。

## 快速开始

环境要求：Python 3.11+，命令行为 `python` 和 `pip`。

```bash
pip install numpy pytest
python -m quant_pipeline generate-samples \
  --sample-dir samples --config-dir configs
python -m quant_pipeline run \
  --sample-dir samples --output-dir artifacts/run
python -m pytest
```

也可以不安装包，直接运行根目录脚本：

```bash
python generate_samples.py --sample-dir samples --config-dir configs
python run_pipeline.py --sample-dir samples --output-dir artifacts/run
```

全量/增量一致性验证：

```bash
python -m quant_pipeline verify \
  --sample-dir samples \
  --output-dir artifacts/verify_incremental \
  --fresh-dir artifacts/verify_fresh
```

切换量化档位与定点小数位宽：

```bash
python -m quant_pipeline run --sample-dir samples --output-dir artifacts/q6f4 \
  --profile q6 --fractional-bits 4
```

运行六类异常注入：

```bash
python -m quant_pipeline run --sample-dir samples \
  --output-dir artifacts/injection_run \
  --injections samples/injections_all.yaml
```

每次运行在输出目录写出 `report.json`、`manifest.json` 和 `stages/` 快照。CLI 退出码：`0` 全部成功，`2` 存在失败/无效输入，`3` 存在预算或中断待办。

## 目录与输入格式

```text
samples/
  weights.jsonl       # 固定种子逐层权重包
  calibration.json    # 校准输入与参考输出
  config.yaml         # 量化、蒸馏、运行时、容差配置
  injections.yaml     # 默认空注入配置
  injections_all.yaml # 六类注入演示
configs/              # 独立配置副本
quant_pipeline/       # 实现包
tests/                # pytest 验收测试
```

### 权重包 JSONL

每行一个 JSON 对象：

```json
{"layer":"block0.attn.k","shape":[6,8],"weights":[0.1,-0.2],"version":1,"group_size":4}
```

逐行校验：JSON 可解析、层名唯一且非空、`shape` 每维为正整数、权重元素数等于形状乘积、权重全为有限数、`version` 非负且跨行单调不减、`group_size` 为正整数且整除元素数。嵌套权重数组也支持，但其实际形状必须等于声明形状。

### 校准集 JSON

支持简单的共享向量，也支持逐层向量。推荐格式：

```json
{
  "version": 1,
  "samples": [
    {
      "id": "s0000",
      "inputs": {"block0.attn.k": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
      "reference_outputs": {"block0.attn.k": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    }
  ]
}
```

层输入维度为该层 `shape[-1]`，输出维度为除最后一维外的元素数（一维层为元素数）。维度、有限性和样本 ID 唯一性均校验。报告为每层记录其引用的完整校准样本集合。

### 量化与蒸馏配置 YAML

生成样本中的规范形如：

```yaml
{
  "quantization": {
    "active_profile": "q8",
    "profiles": [
      {"name": "q6", "bits": 6, "fractional_bits": 6},
      {"name": "q8", "bits": 8, "fractional_bits": 8},
      {"name": "q10", "bits": 10, "fractional_bits": 10}
    ],
    "allowed_bit_widths": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "scheme": "symmetric",
    "saturation": "saturate",
    "rounding": "nearest_even",
    "group_size": 4,
    "min_group_size": 3,
    "percentile_low": 0.05,
    "percentile_high": 0.95
  },
  "distillation": {"temperature": 1.5, "soft_label_weight": 0.35},
  "runtime": {
    "max_workers": 4,
    "point_budget": 1000,
    "layer_point_quota": 1000,
    "snapshot_interval": 2
  },
  "tolerances": {"absolute": 0.015, "relative": 0.12},
  "layers": {
    "block0.attn.k": {"group_size": 4},
    "block0.attn.q": {"group_size": 2}
  }
}
```

支持的声明值：

- 量化方案：`symmetric`、`asymmetric`（别名 `sym`、`asym`）。
- 饱和策略：`saturate`（别名 `clip`、`clamp`）或 `error`。`error` 会在发生截断时使该层量化阶段失败。
- 舍入模式：`nearest_even`、`nearest_away`、`floor`、`ceil`、`truncate`。
- 分位插值固定为 `linear`；并列规则固定为 `lower_index`，并写入阶段产物元数据。
- 量化档位按声明顺序位宽严格递增；所有目标位宽必须在 `allowed_bit_widths` 枚举中。
- 定点小数位宽为正整数，安全上限为 30，以避免定点边界在 float64/int64 转换时溢出。

CLI 的 `--profile` 会强制所有层使用指定档位；`--fractional-bits` 会强制所有层使用指定小数位宽。两者都会真实改变定点整数与反量化值。

### 异常注入配置

```json
{
  "version": 1,
  "injections": [
    {"id": "INJ-001", "type": "non_finite_weight", "layer": "block0.attn.k"},
    {"id": "INJ-002", "type": "group_indivisible", "layer": "block0.attn.q"},
    {"id": "INJ-003", "type": "bit_width_out_of_range", "layer": "block0.mlp.up"},
    {"id": "INJ-004", "type": "calibration_dimension_mismatch", "layer": "block1.attn.k"},
    {"id": "INJ-005", "type": "zero_temperature"},
    {"id": "INJ-006", "type": "duplicate_layer_name", "layer": "block1.mlp.down"}
  ]
}
```

注入作用于真实的内存输入后再走正常校验路径。报告中的 `injections` 项包含 `injection_id`、`applied`、`hit`、`business_code` 和消息，用于证明故障命中。

## 计算语义

### 分位标定

- 每组权重按升序稳定排序，分位位置为 `(n - 1) * p`，相邻秩线性插值。
- 对称方案：`qmax = 2^(bits-1)-1`，`scale = max(|q_low|, |q_high|) / qmax`，零点为 0。
- 非对称方案：`scale = (q_high - q_low) / (qmax - qmin)`，零点由 `qmin - q_low/scale` 舍入并裁剪到整数范围。
- 初始组按配置粒度连续切分；少于 `min_group_size` 的组合并到前一组（首组则合并后一组），组记录包含 `source_group_ids` 与 `merge_basis`。
- 每组记录组 ID、元素区间、元素数、分位值、缩放因子、零点及退化标记。

### 定点重量化

设连续缩放值 `z = x/scale + zero_point`，定点乘数 `M = 2^fractional_bits`：

1. `raw_fixed = round(z * M)`，舍入模式来自配置。
2. 有符号定点范围是 `[qmin*M, qmax*M]`。
3. `saturate` 裁剪越界值，并按元素索引升序记录方向、裁剪前/后的定点值和组 ID；`error` 直接使该层失败。
4. 反量化值严格使用裁剪后的整数：`(fixed/M - zero_point) * scale`。

因此位宽改变边界与缩放因子，定点小数位宽改变 `M` 和真实中间整数；两者都不是显示层转换。

### 误差归因

每个元素计算绝对误差；真值为 0 时相对误差记为 `null`，仅用绝对容差，并标注 `absolute_for_zero_truth`。非零真值在绝对或相对容差任一超限时判为超差。

归因贡献为四类：`percentile_discretization`、`scaling_quantization`、`saturation_clipping`、`dequantization_rounding`。主来源取最大贡献；精确并列时固定按以下优先级：

```text
saturation_clipping > percentile_discretization > scaling_quantization > dequantization_rounding
```

### 蒸馏校正

对每个校准样本计算原始浮点输出和反量化输出。软标签为参考输出与量化输出分别除以温度后的 softmax。校正公式：

```text
corrected = quantized + soft_label_weight * temperature
            * (softmax(reference/temperature) - softmax(quantized/temperature))
```

报告保留校正量、校正前后相对原始输出的偏差、相对参考输出的误差和逐样本汇总。

## 缓存、调度与快照

每个阶段缓存键由阶段名、层名、层内容指纹、阶段参数指纹、量化档位、定点小数位宽、该层引用的校准样本子集指纹及上游阶段键派生；全局校准集哈希另用于快照身份与审计。阶段包括标定、量化、误差分析、蒸馏和解析校验。

- 权重/版本、组粒度、位宽、方案、舍入或蒸馏参数变化时，只重算键不匹配的层。
- 某层变化会强制重算该层本身及其稳定层序中的下游蒸馏校正；上游未失效层直接复用缓存，不调用计算函数。
- 缓存账本按层名排序，分别列出每阶段 `reused`、`recomputed`、`failed`、`skipped`。
- 调度器按层名稳定顺序预留单批量化点数预算。层的量化点成本等于其元素数；仅误差/蒸馏缓存失效时为 0 点。
- 单层成本超过 `layer_point_quota` 记录 `E_POINT_QUOTA_EXCEEDED`；批次预算不足记录 `E_POINT_BUDGET_PENDING` 并保留待办队列。
- 层任务最多使用配置线程数，任务排序键为 `层名|calibration>quantization>error_analysis>distillation`；最终报告和账本均按稳定顺序规范化。

### 阶段提交

阶段目录写入同盘临时目录后整体 rename 替换，不会逐文件覆盖。清单记录：

- `global_version`
- 模型配置哈希、权重包指纹、校准集哈希、量化档位
- 已完成层、累计已消耗点数、本批已消耗点数
- 各阶段版本、层级阶段键、待办、失败与注入结果

提交间隔由 `snapshot_interval` 控制。收到 `SIGINT`/`SIGTERM` 后不再启动新层，等待在途层完成并写出 `interrupted` 清单。进程被强制终止时，最近一次清单保留 `running` 状态。下次启动：

- 未完成快照且权重/校准/模型配置一致：加载阶段缓存续跑。
- 未完成快照但身份不一致：报告 `E_SNAPSHOT_STALE` 并全量重建。
- 已完成快照：允许按缓存键做正常增量编辑。

调度参数 `max_workers`、`point_budget`、`layer_point_quota`、`snapshot_interval` 不影响模型语义，因此不参与未完成快照的 stale 判定，但写入完整配置审计哈希与运行报告。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_CONFIG_INVALID` | 配置字段、枚举、位宽、温度、软标签权重、运行时参数或层配置无效 |
| `E_WEIGHT_PACK_INVALID` | 权重包 JSON、层名重复、形状/有限数/版本/组粒度无效 |
| `E_CALIBRATION_INVALID` | 校准样本结构、维度或有限性无效 |
| `E_LAYER_CALIBRATION` | 单层标定运行异常 |
| `E_LAYER_QUANTIZATION` | 单层量化或误差分析运行异常 |
| `E_LAYER_DISTILLATION` | 单层蒸馏校正运行异常 |
| `E_SATURATION_EXCEEDED` | `saturation=error` 时发生越界 |
| `E_POINT_BUDGET_PENDING` | 批次点数预算不足，层进入待办 |
| `E_POINT_QUOTA_EXCEEDED` | 单层点数超过配额 |
| `E_SNAPSHOT_STALE` | 未完成快照身份与当前输入不符，已全量重建 |
| `E_SNAPSHOT_CORRUPT` | 快照清单损坏，已全量重建 |
| `E_INJECTION_INVALID` | 注入配置无效或目标层不存在 |
| `E_RUN_INTERRUPTED` | 优雅退出导致未启动层进入待办 |

单层在标定、量化/误差或蒸馏阶段抛异常时，失败项记录层名、阶段、业务码与异常类型；其余层继续执行，失败层不进入成功误差统计。

## 报告结构

`report.json` 的主要字段：

- `status`：`completed`、`partial`、`failed`
- `business_codes`：本次失败、待办和恢复事件涉及的业务码去重排序
- `quantization_gear`：当前档位、全局小数位宽及逐层有效位宽
- `layers[]`：有效配置、校准样本集合、分位组、定点整数摘要、裁剪记录、误差统计/归因、蒸馏逐样本结果、阶段缓存键
- `cache_ledger`：逐阶段稳定排序的复用/重算/失败/跳过清单
- `schedule`：稳定任务排序键、预算成本、缓存命中与执行状态
- `pending`：预算或中断待办
- `failures`：校验/运行失败清单
- `injections`：注入命中证明与业务码映射
- `snapshot`：恢复模式、最终快照状态、阶段版本
- `result_fingerprint`：排除缓存账本/运行调度元数据后的业务结果指纹，用于增量与全量逐字段等价比对

## 实际验证记录

验证日期：2026-11-24。环境：Python 3.11.0rc1、NumPy 1.26.4、pytest 8.2.0。

```text
$ python -m pytest -q
37 passed in 10.40s

$ python -m quant_pipeline verify ...
first = second_incremental = fresh_full
match = true

$ python -m quant_pipeline run --profile q6 --fractional-bits 4 ...
status = completed
first layer fixed ints = [0, 496, -1, -26, 496, -333]
first layer dequantized prefix = [0.0, 6.374999999999997, -0.012852822580645155]

注入运行：
INJ-001 -> E_WEIGHT_PACK_INVALID
INJ-002 -> E_WEIGHT_PACK_INVALID
INJ-003 -> E_CONFIG_INVALID
INJ-004 -> E_CALIBRATION_INVALID
INJ-005 -> E_CONFIG_INVALID
INJ-006 -> E_WEIGHT_PACK_INVALID
六项 hit 均为 true，未注入层继续完成量化诊断。
```

机器可读摘要见 `artifacts/validation_summary.json`。
