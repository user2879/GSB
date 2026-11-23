# 长上下文位置编码外推诊断工作台

这是一个本地、确定性、无 GPU/外部服务依赖的诊断工作台，用于定位旋转类位置编码（RoPE）在长上下文外推区间中由定点量化、累加、减最大值归一化造成的数值精度损失和逐级误差传播。输入是配置 JSON、注意力样本 JSONL 和命令流 JSONL；输出是阶段快照、JSON 诊断报告、可视化摘要、SVG 与自包含 HTML 仪表盘。

## 快速开始

要求 Python 3.11 与 NumPy 1.26.4。Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m pe_workbench validate `
  --config samples/position_encoding.json `
  --samples samples/attention_samples.jsonl `
  --commands samples/commands.jsonl

python -m pe_workbench run `
  --config samples/position_encoding.json `
  --samples samples/attention_samples.jsonl `
  --commands samples/commands.jsonl `
  --state-dir state --reports-dir reports --clean
```

Linux/macOS 使用相同参数并把续行符改成 `\` 即可。`--clean` 会清空 `state/` 和 `reports/`，适合重复实验。主样本由 NumPy PCG64 种子 `20261124` 生成，可用脚本从零重建：

```bash
python tools/generate_samples.py
```

## 目录

```text
pe_workbench/
  config.py          配置模型、枚举、逐字段校验
  validation.py      样本/命令流加载校验
  encoding.py        RoPE、线性内插、NTK 感知缩放
  quantization.py    定点量化、升序点积、确定性 softmax
  error_tracking.py  float64 参考路径与逐级偏差归因
  cache_keys.py      内容指纹与阶段缓存键
  engine.py          稳定调度、预算、增量执行、阶段快照
  snapshot.py        staging 目录 + CURRENT 指针原子提交
  sensitivity.py     方案/精度/量化/位宽/数据规模敏感性矩阵
  reporting.py       JSON、SVG、HTML 报告
  executor.py        命令流执行器
  __main__.py        CLI
samples/             固定种子配置、主样本、预算样本、异常样本、命令流
tests/               15 个确定性单元/集成测试
state/               当前阶段快照与运行 trace
reports/             诊断报告、可视化摘要、SVG、HTML 仪表盘
tools/generate_samples.py
```

## 输入协议

### 配置 JSON

`samples/position_encoding.json` 包含：

- `scheme`: `rope`、`linear`、`ntk`，也接受中文别名。
- `base_frequency`、`train_context_length`、`extrapolation_target_length`、`dimension`、`scale_factor`。
- `precision_level` 与 `precision_levels`：`fp16`、`fp32`、`fp64`。
- `quantization_level` 与 `quantization_levels`：`fine`、`balanced`、`coarse`。
- `fixed_decimal_bits` 与 `fixed_decimal_widths`：定点小数位宽及允许枚举。
- `dimension_block_size`：点积按维度升序分块记录的块大小。
- `tolerances`：`position_vector`、`dot_score`、`attention_weight`，必须为正有限数。
- `tolerance_candidates`：敏感性诊断候选容差。
- `scheme_candidates`：方案敏感性候选。
- `thread_pool_size`、`point_budget`、`sample_quota`、`snapshot_interval`。

### 样本 JSONL

每行一个对象：

```json
{"sample_id":"s0001","query_position":66,"key_positions":[4,3,2],"query_vector":[...],"key_vectors":[[...],[...],[...]]}
```

校验包括：唯一样本 ID、非负整数位置、有限特征数、向量长度等于维度、键位置与键向量数量一致。坏样本产生带 `sample_id`、`field`、业务码的失败项，其他样本继续处理。

### 命令流 JSONL

支持命令：

- `{"command":"run","params":{}}`，可覆盖 `point_budget`、`sample_quota`、`snapshot_interval`。
- `{"command":"set","patch":{"scheme":"linear"}}`。
- `{"command":"sensitivity","params":{"scheme_candidates":[...],"precision_levels":[...],"quantization_levels":[...],"fixed_decimal_widths":[...],"data_scales":[0.25,0.5,1.0]}}`。
- `{"command":"report"}`、`{"command":"status"}`。

坏命令进入失败清单，不会阻止其他合法命令加载。CLI 退出码：`0` 完成、`1` 配置致命错误、`2` 存在校验/运行问题、`3` 仍有预算待办。

## 数值实现

### 三种位置编码

设位置为 `p`，维度为 `d`，基频为 `b`，缩放因子为 `s`。

- RoPE：`θ_i = p · b^(-2i/d)`，对相邻维度对 `(2i, 2i+1)` 做旋转。
- 线性内插：位置压缩为 `p' = p/s`，其余同 RoPE。
- NTK 感知缩放：位置不变，基频变为 `b' = b · s^(d/(d-2))`，再生成旋转角。

三者都走 `encode_vector/encode_pair` 同一接口并返回同样结构。产物元数据记录方案名、基频、缩放因子、精度和有效基频。测试验证同输入可逐元素复现，且三者数值确实不同。

### 定点域与注意力打分

量化档位改变有效小数位宽和裁剪范围：

| 档位 | 位宽偏移 | 裁剪绝对值 |
|---|---:|---:|
| `fine` | 0 | 8.0 |
| `balanced` | -2 | 4.0 |
| `coarse` | -4 | 2.0 |

`有效位宽 = max(2, fixed_decimal_bits + offset)`，转换使用 nearest-even，超出范围会裁剪并记录 `clipped_count`。点积严格按维度 0 到 d-1 累加，同时记录每个维度的 `product/cumulative_sum`，以及每个块的中间累计值。打分使用：

```text
raw_dot / 2^(2b) / sqrt(d)
```

以 Decimal、half-even 确定性舍入到 12 位小数。注意力权重使用减最大值的稳定 softmax，并记录最大值索引、分母和权重。

### 分层统计

相对距离为 `abs(query_position - key_position)`：

- `distance < train_context_length`：`in_training`。
- 其他：`extrapolation`。

每个距离同时输出两个层。若某层计数为零，输出 `count=0, empty_layer=true, mean/min/max=null`，而不是填入 0。

### 逐级误差追踪

参考路径始终使用 float64。对每个外推键依次比较：

1. `position_encoding`：量化位置向量相对 float64 参考向量的最大绝对偏差。
2. `dot_score`：定点打分相对 float64 打分的偏差。
3. `attention_weight`：减最大值归一化后注意力权重偏差。

按上述顺序找到首个超过对应容差的环节，记录 `first_exceeded_stage` 和该环节偏差。报告同时保留三阶段完整误差、分母、最大索引与超差清单。

敏感性分析会跨精度/量化档位对翻转样本做反事实归因，并将主导来源区分为：

- `position_encoding_angle`：位置/旋转角变化主导。
- `quantization_rounding`：定点舍入/打分偏差主导。
- `normalization_denominator`：减最大值后的最大值索引或归一化分母变化主导。

每个来源都带位置向量、打分、权重、分母等证据字段；报告保存全量计数、逐样本归因汇总和代表性案例，避免只报一个总偏差。

## 缓存、失效与调度

阶段缓存键均由内容指纹派生：

- 位置编码：样本内容指纹 + 方案公式版本 + 与方案真正相关的基频/维度/缩放因子 + 精度档位。
- 打分：位置编码键 + 量化档位 + 定点位宽 + 维度块大小 + 舍入规则。
- 误差追踪：打分键 + 区域掩码 + 各环节容差 + float64 参考精度。

因此：

- 量化档位或定点位宽变化只失效打分和误差追踪，位置编码复用。
- RoPE 下修改缩放因子不会无谓失效；线性内插/NTK 下会失效。
- 训练长度变化只失效区域掩码实际变化的样本的误差追踪；位置编码和打分复用。
- 方案、基频、维度或精度变化使相关样本位置编码及全部下游失效。
- 线程数、预算、快照间隔、候选清单变化不改变数值阶段产物。

调度器按样本 ID 排序分配预算，线程池并发数不超过配置值。单样本成本为 `1 + len(key_positions)`；超过单样本配额或全局预算耗尽时中止该样本并写 `E_POINT_BUDGET_PENDING`，保留待办和最小待处理样本 ID 作为续算游标。快照与每个样本/阶段任务都带由样本 ID、方案、阶段派生的稳定排序键。

## 快照与报告

一次提交先把 loading、position_encoding、scoring、error_tracking 和 manifest 写入 `state/.staging/`，随后整体改名到不可变快照目录，最后原子替换 `state/CURRENT` 指针。manifest 记录：

- 全局兼容版本号与递增 `snapshot_version`。
- 配置哈希、完整配置、样本内容哈希。
- 已完成方案和样本、待办样本、续算游标、命令游标。
- 累计已消耗预算与本批已消耗点数。
- 各阶段条目数与原子提交方式。

启动时若快照兼容版本不符则重建；版本兼容且配置哈希相同进入 `exact_resume`；版本兼容但哈希变化进入 `incremental`，并按阶段键精确复用。

每次运行写出：

- `reports/diagnostic_report.json`：元数据、逐样本诊断、分层统计、阶段误差分布、超差/边界样本、敏感性排序、翻转归因。
- `reports/visual_summary.json`：可被其他前端消费的图表数据。
- `reports/summary.svg`：独立矢量摘要。
- `reports/dashboard.html`：自包含诊断仪表盘。
- `state/run_trace.json`：本次实际复用/重算、调度、预算、待办和快照信息。
- `reports/acceptance_summary.json`：本次工作区实际验收指标归档（会随 `--clean` 重建）。

报告不含墙钟时间或线程时序，排序全部固定。为保证“增量结果与冷重建逐字段一致”，`diagnostic_report.json` 中的 `reuse_recompute` 是规范化的冷重建工作清单；实际本次复用/重算清单属于运行遥测，放在 `state/run_trace.json`，因为增量执行与空白重建的实际复用次数按定义不可能相同，但最终诊断字段与报告 ID 必须相同。

## 本次实际验证

环境：Python `3.11.0rc1`，NumPy `1.26.4`。代码只依赖标准库与 NumPy，路径和 `os.replace` 用法兼容 Windows。

### 自动化测试

```text
python -m unittest discover -s tests -v
Ran 15 tests
OK
```

覆盖三种方案可复现且数值不同、定点升序累加、空层 null、逐级首超环节、量化仅下游失效、全局预算与单样本配额待办、续算游标、精确快照恢复、版本/配置哈希恢复判定、原子快照布局、增量与冷重建报告逐字段一致、四类报告产物。

### 主样本运行

```text
48 个有效样本，0 校验失败，0 待办
最终方案 ntk / NTK 感知缩放，precision=fp32，quantization=fine，fixed_decimal_bits=6
135 个敏感性变体，报告约 1.0 MB
43 个含外推键样本首次超差环节均为 position_encoding
8608 个精度/量化决策翻转：position_encoding_angle=7820，
  quantization_rounding=116，normalization_denominator=672
报告 ID：fac5eb3c439002e9c51869f25206e785ebf59dc3223f697e3a2daf6e334a4cd4
```

同一命令流在已有状态上再次启动后为 `exact_resume`，最终运行 `consumed_points=0`，`pending=0`，重新生成的报告 ID 仍为 `fac5eb3c439002e9c51869f25206e785ebf59dc3223f697e3a2daf6e334a4cd4`。

### 预算/恢复样本

`samples/budget_commands.jsonl` 的实测命令日志显示：

- 7 点全局预算：完成 1 个样本，6 个待办，游标 `s0002`。
- 再运行 7 点：完成 2 个样本，5 个待办，游标 `s0003`。
- 单样本配额 3：已有缓存不重算，成本超过配额的样本保持待办。
- 恢复配额 32、预算 500：7 个样本全部完成，本批消耗 29 点。
- 快照 manifest 记录累计 `consumed_budget=38`，阶段条目各 7 个，`CURRENT` 原子指向唯一快照。

### 异常输入

异常样本共 8 行，合法继续处理 2 行；失败清单包含重复 ID、负查询位置、负键位置、非有限特征、向量长度不符和 JSON 解析错误。异常命令流中仅 1 条合法命令保留，坏 JSON、未知命令、非法补丁和非法预算参数进入失败清单且不会被执行。

## 业务码摘录

- 配置：`E_CONFIG_DIMENSION_EVEN`、`E_CONFIG_LENGTH_ORDER`、`E_CONFIG_BASE_FREQUENCY`、`E_CONFIG_QUANTIZATION_LEVEL`、`E_CONFIG_FIXED_WIDTH_NOT_ALLOWED`、`E_CONFIG_TOLERANCE_VALUE`。
- 样本：`E_SAMPLE_ID_DUPLICATE`、`E_SAMPLE_QUERY_POSITION`、`E_SAMPLE_KEY_POSITION`、`E_SAMPLE_FEATURE_LENGTH`、`E_SAMPLE_QUERY_VECTOR`、`E_SAMPLE_JSON`。
- 命令：`E_COMMAND_JSON`、`E_COMMAND_NAME`、`E_COMMAND_PATCH`、`E_COMMAND_PARAM`。
- 调度：`E_POINT_BUDGET_PENDING`。
- 快照：`E_SNAPSHOT_POINTER`、`E_SNAPSHOT_MANIFEST`、`E_SNAPSHOT_VERSION`、`E_SNAPSHOT_STAGE`。

## 设计边界

- 量化档位定义为对有效定点小数位宽和裁剪范围的确定性组合，具体映射见上表，不包含随机量化。
- 注意力缩放固定为标准 `1/sqrt(dimension)`；可通过修改代码中的 score 归一化扩展。
- 外推判定按键相对距离，而非仅按绝对位置；这与 RoPE 的相对位置依赖一致。
- 翻转归因给出主导来源和三项反事实证据，不声称统计因果推断。
