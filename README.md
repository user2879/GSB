# Softmax / LayerNorm 格式互操作与版本兼容实验环境

本项目在纯本地 Python 环境中实现一个可重复运行的定点算子实验台，覆盖：

- v1 定宽文本权重与 v2 JSON 权重的双向转换、回读与损失旁路；
- 数值稳定的整数定点行向 Softmax 与 LayerNorm；
- 基于内容指纹的行级增量复用与精确失效；
- 单批元素预算、单层配额和线程池并发下的串行一致调度；
- 阶段化原子快照、版本比较提交与独立异常注入；
- “编辑命令序列增量运行”与“空白全量重建”的逐字段验收。

所有样本均由固定种子在本地生成，不访问网络，不使用 Docker 或外部服务。唯一第三方依赖为 `numpy==1.26.4`，运行代码本身使用 Python 整数完成定点计算。

## 快速开始（Windows PowerShell）

```powershell
python --version
python -m pip install numpy==1.26.4
python -c "import numpy; print(numpy.__version__)"

# 样本已随项目提供；如需重新生成：
python tools/build_samples.py

# 从空白运行 8 条命令，逐步增量提交快照
python -m opexlab replay --reset

# 双向格式转换
python -m opexlab convert --to v2 --output runtime/conversions/weights_v2_from_v1.jsonl
python -m opexlab convert --to v1 --output runtime/conversions/weights_v1_from_v2.txt

# 五类异常注入（每项独立运行）
python -m opexlab inject

# 完整验收：14 项检查
python -m opexlab verify

# 单元测试：9 项
python -m unittest discover -s tests -v
```

也可以直接执行 `tools/run_demo.ps1` 完成以上主流程。

常用单步命令：

```powershell
python -m opexlab run --mode incremental
python -m opexlab run --mode full
python -m opexlab apply --command '{"type":"set_quant_level","value":2}'
python -m opexlab manifest
python -m opexlab reset
```

## 目录结构

```text
opexlab/                 核心实现
  config.py              配置与有效状态
  loaders.py             激活包、v1、v2 逐行加载校验
  conversion.py          v1/v2 双向转换与旁路
  fixed.py               整数定点原语与指数近似
  operators.py           Softmax / LayerNorm
  fingerprints.py        阶段内容键与指纹
  scheduler.py           预算/配额规划与线程池执行
  engine.py              端到端阶段执行、缓存与失败隔离
  snapshots.py           原子版本化快照
  injections.py          异常注入隔离运行器
  verification.py        本地验收套件
  cli.py                 `python -m opexlab` 入口
samples/                 确定性输入样本
tests/                   标准库 unittest 测试
tools/                   样本生成与演示脚本
docs/DESIGN.md           详细设计说明
runtime/                 运行状态、快照、报告（运行后生成）
VERIFICATION.md          最近一次实际验证记录
```

## 输入格式

### 激活张量包 `samples/activations.jsonl`

每个 JSONL 行是一层：

```json
{"layer":"layer_alpha","seed":101,"rows":1,"dim":6,
 "activation":[[0.1,0.2,0.3,0.4,0.5,0.6]],
 "layernorm":{"scale":[[1.2,1.2,1.2,1.2,1.2,1.2]],
              "bias":[[-0.03,-0.03,-0.03,-0.03,-0.03,-0.03]]}}
```

加载器校验：

- 层名唯一且非空；
- `rows`、`dim` 为正整数；
- 激活矩阵形状与声明一致；
- 激活与参考 LayerNorm 参数均为有限数；
- 单个坏行记入失败清单，其他行继续处理。

### v1 定宽权重 `samples/weights_v1.txt`

默认字段总宽 36，按顺序切片：

| 字段 | 宽度 | 含义 |
|---|---:|---|
| `layer` | 12 | 左对齐层名 |
| `row` | 5 | 行序号，从 0 开始 |
| `scale_permille` | 8 | 千分比整数缩放因子 |
| `bias_code` | 8 | 千分比整数偏置码 |
| `quant_bits` | 3 | 量化位宽，码表为 8/16/32 |

示例：

```text
layer_alpha     0    1200     -30 16
```

行短于配置总宽输出 `E_FIELD_TRUNCATED`；字段数字非法输出 `E_PARSE`；位宽不在码表输出 `E_WIDTH_OUT_OF_RANGE`。

### v2 JSON 权重 `samples/weights_v2.jsonl`

每个 JSONL 行是一条权重：

```json
{"layer":"layer_alpha","row":0,"scale":1.2,"bias":-0.03,
 "scheme":"layernorm_affine","precision":"int16"}
```

`scheme` 必须在 `allowed_norm_schemes` 中；`precision` 必须在量化码表中。未知方案输出 `E_SCHEME_UNKNOWN`。

### 配置 `samples/config.json`

关键字段：

```json
{
  "exchange_version": "v1",
  "fixed_point": {
    "fractional_bits": 12,
    "quant_level": 1,
    "allowed_quant_levels": [0, 1, 2, 3]
  },
  "layernorm": {"variance_floor": 0.000001},
  "limits": {
    "max_workers": 4,
    "batch_element_budget": 1000,
    "per_layer_element_quota": 1000
  },
  "allowed_norm_schemes": ["layernorm_affine", "layernorm_scale_only"],
  "scheme_bias_policy": {
    "layernorm_affine": true,
    "layernorm_scale_only": false
  }
}
```

`paths` 均按项目根目录解析，也可被 `--project-root` 覆盖。

### 命令流 `samples/commands.jsonl`

支持：

- `set_exchange_version`
- `set_fractional_bits`
- `set_quant_level`
- `set_max_workers`
- `set_batch_element_budget`
- `set_per_layer_element_quota`
- `replace_activation_row`
- `replace_layer_params`

每条命令递增全局版本号，稳定后立即执行增量并提交快照。

## 定点语义

设 `Q = 2^fractional_bits`，量化档位为 `L`，量化步长为 `S = 2^L`。

- 浮点输入先转为 `round_half_away(x * Q)`，再按 `S` 做整数步长量化；
- 加减乘除均使用整数，乘除采用半数远离零的确定性舍入；
- 量化档位同时改变输入/参数/中间量步长和指数多项式阶数，因此会真实改变定点整数结果；
- Softmax 先减行最大值，再用 `k*ln2 + r` 范围归约与整数泰勒多项式近似 `exp(-r)`，最后右移 `k` 位；
- 指数近似名称和多项式阶数写入产物元数据；
- Softmax 输出使用整数归一化，并修正到严格满足 `sum(output_fixed) == Q`；
- LayerNorm 的均值、方差、标准差、缩放和偏置全部以整数计算，统计量随产物保留；
- 方差低于配置下限时使用下限方差并标记 `variance_floor_applied`，不产生非有限值。

Softmax 最大并列时选择最小列序号；分母为零时使用均匀分布并把余数放在第 0 列。两类规则均写入元数据。

## 版本转换与旁路

v1 → v2：

- `scale = scale_permille / 1000`
- `bias = bias_code / 1000`
- `quant_bits` 通过码表映射为 `int8/int16/int32`
- 生成 `precision_tag = permille_exact_to_float`

v2 → v1：

- 使用半数远离零规则还原千分比和整数偏置；
- 旁路记录缩放/偏置截断残差、精度标签、被丢弃的命名方案和转换哈希；
- v2 命名方案不能写入 v1 定宽行，但可在旁路中查回；
- 旁路哈希参与下游参数指纹与 LayerNorm 输出缓存键。

## 缓存键与失效粒度

四类可复用产物分别有内容键：

- 版本转换产物：源权重 + 方向 + 交换版本 + 定点位宽 + 量化档位；
- Softmax 行：激活行指纹 + 版本 + 位宽 + 档位 + 指数近似；
- LayerNorm 统计量：激活行指纹 + 版本 + 位宽 + 档位 + 方差下限；
- LayerNorm 输出：统计量键/内容哈希 + 参数指纹 + 方案 + 偏置策略 + 版本 + 位宽 + 档位。

每行记录所引用的激活行、参数集合、上游内容哈希和阶段内容哈希。缓存命中前会重新校验内容哈希。

实际失效范围：

- 替换单行激活：仅该行 Softmax、LayerNorm 统计量、LayerNorm 输出重算；
- 替换单层参数：仅该层转换产物与 LayerNorm 输出重算，Softmax 和统计量复用；
- 切换版本、定点位宽或量化档位：所有相关行失效；
- 调整并发数：业务产物不变。

`cache_plan` 是“相对上一逻辑状态的键稳定性清单”。增量模式真实复用缓存；全量模式真实重算所有产物，但会根据命令历史重建上一逻辑状态的键集合，因此与增量模式输出相同的行级失效/复用清单。

## 调度与预算

- 计划固定按层名字典序、行序号升序生成，与线程完成顺序无关；
- 每个激活行对 Softmax + LayerNorm 合计消耗 `dim` 个元素预算；
- 批次预算不足单行时输出 `E_BUDGET_TOO_SMALL`；预算在批次中耗尽输出 `E_BUDGET_EXHAUSTED`；
- 单层累计元素将超过配额时，该层剩余行进入待办队列并输出 `E_LAYER_QUOTA`，后续层继续；
- 线程池上限严格取配置的 `max_workers`；
- 结果、预算扣减序列、待办队列和失败清单从计划结构生成，不依赖线程调度顺序。

## 快照与故障隔离

阶段为 `load_validation`、`version_conversion`、`softmax`、`layernorm`、`summary`。每次运行：

1. 在 `runtime/tmp` 创建临时快照目录；
2. 写入五个阶段 JSON；
3. 整体重命名到 `runtime/snapshots/rXXXX-sYYYY`；
4. 最后原子替换 `runtime/manifest.json`。

清单记录全局版本、配置哈希、交换版本指纹、已完成层、已消耗预算和五个阶段的内容版本。提交前比较状态与阶段版本；无变化时不重写。

## 异常注入

`samples/injections.json` 中的每项独立运行，彼此不共享运行目录：

| 编号 | 故障 | 业务码 |
|---|---|---|
| `INJ-001` | 截断定宽权重行 | `E_FIELD_TRUNCATED` |
| `INJ-002` | 非有限激活值 | `E_NONFINITE` |
| `INJ-003` | 方案枚举越界 | `E_SCHEME_UNKNOWN` |
| `INJ-004` | 量化位宽越界 | `E_WIDTH_OUT_OF_RANGE` |
| `INJ-005` | 预算小于单行元素数 | `E_BUDGET_TOO_SMALL` |

注入编号写入失败记录和 `runtime/reports/injection_report.json`。单行任一阶段异常记录层名、行序号、阶段和业务码，其余行继续。

## 验证结果

最近一次本地执行：

- 环境：Python `3.11.0rc1`，NumPy `1.26.4`；
- 完整验收：`14/14` 通过；
- 单元测试：`9/9` 通过；
- v1/v2 互转：11 行回读可还原字段一致，损失字段均可在旁路查回；
- 量化档位 0 与 2 的 Softmax 整数向量不同：`[332,115,1070,1187,505,887]` vs `[332,116,1072,1184,504,888]`；
- 单行激活替换仅重算目标行的三个下游阶段；
- 单层参数替换仅重算该层转换与该层 LayerNorm 输出，统计量复用；
- 增量最终状态与空白全量重建的业务字段、缓存键和失效/复用清单一致；
- 五类注入全部命中预期业务码。

详细记录见 `VERIFICATION.md`、`runtime/reports/verification.json` 和 `runtime/reports/injection_report.json`。

## 主要设计取舍

- 行元素预算对同一行的 Softmax 与 LayerNorm 合计只扣一次 `dim`，便于把“单行”作为原子调度单位。
- LayerNorm 方差是围绕定点舍入后的均值计算的总体方差，保证整数执行完全确定。
- v1 不承载命名方案，升级时使用配置的 `v1_upgrade_scheme`；原方案只能通过 v2→v1 旁路追踪。
- `execution_metrics` 中的实际并发峰值不参与业务字段或缓存清单比较。
