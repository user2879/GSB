# qcal：LLM 权重量化标定、互转与轻量蒸馏校正

这是一个仅依赖 Python 3.10+ 与 NumPy 的本地确定性示例组件。它覆盖 v1/v2 标定互操作、分位标定、定点量化、校准集蒸馏、内容指纹缓存、受限预算并行调度，以及可恢复的阶段快照。默认样本由固定种子生成，不下载数据、不使用 GPU 或容器。

## 快速开始

```bash
python -m pip install -r requirements.txt
python scripts/generate_samples.py --out samples --seed 20261121
python -m qcal.cli run --samples samples --output artifacts/full --state artifacts/state_full --fresh
python -m qcal.cli acceptance --samples samples --artifacts artifacts
python -m pytest -q
```

Windows 下使用同样的 `python ...` 命令即可。若 PyYAML 未安装，仓库内自带的 `.yaml` 文件是合法 JSON（YAML 1.2 子集），`load_yaml` 会回退到标准库 JSON 解析；安装 PyYAML 后可直接使用任意常规 YAML。

常用入口：

- `python -m qcal.cli run ...`：运行或恢复四阶段流水线。
- `python -m qcal.cli convert --direction roundtrip ...`：显式执行 v1→v2、v2→v1 与旁路还原。
- `python -m qcal.cli status --state artifacts/state`：查看快照完成阶段与全局版本。
- `python scripts/run_acceptance.py`：生成样本并执行完整验收套件。

## 样本与格式

`scripts/generate_samples.py` 生成 `samples/` 下的确定性输入：

- `weights.jsonl`：每行包含 `layer`、`shape`、`weights` 与 `content_version`。
- `calibration_v1.txt`：定宽文本，字段宽度依次为层名 20、千分比缩放整数 8、零点 8、位宽 4。
- `calibration_v2.json`：浮点 `scale`、`zero_point`、`bit_width` 及命名 `scheme`、`rounding_mode`、`precision_label`、`per_channel`。
- `calibration_samples.json`：6 个输入向量及逐层参考输出。
- `config.yaml`：逐层位宽/分组候选、分位范围、全精度白名单、蒸馏、定点、并发与预算配置。
- `exchange_spec.json`：交换版本与损失字段规范。
- `injections.yaml`：四类故障注入；只通过 CLI/API 参数传入，不修改组件代码。
- `edit_sequence.yaml`：版本切换、位宽/分组/舍入变更、单层权重替换和蒸馏参数变更。

示例 v1 记录：

```text
layer               scale_1e3    zerobits
layer.00                   8      -1   8
```

v1 的缩放因子语义固定为 `scale = scale_1e3 / 1000`；v2 的缩放因子是二进制浮点数，两者不能视为等价。v2 降 v1 时采用 Decimal `ROUND_HALF_UP` 还原千分比，并写入：

- 原始浮点缩放因子的 `repr` 与 IEEE-754 `hex`；
- 截断残差、千分比整数、精度标签；
- 命名舍入模式、量化方案、逐通道标记；
- 可恢复/不可恢复字段及恢复规则。

旁路存在时，反向转换可按 `source_scale_hex` 与保留字段精确重建 v2；旁路缺失返回 `E_SIDECAR_MISSING`，无效旁路返回 `E_SIDECAR_INVALID`。若字段被显式标为不可恢复，恢复记录会保留该标记，不会静默伪造原值。

## 计算流程

1. **解析与统一校验**：两版本按层名唯一合并为统一记录；校验缩放因子为正有限数、零点在 signed `[qmin, qmax]` 内、标定层与权重包层名一致。失败记 `E_CALIB_PARSE`，权重非有限值记 `E_WEIGHT_NONFINITE`，单层失败不阻断其他层。
2. **分位标定**：权重按 `mergesort` 升序排列，使用固定位置公式 `pos=(n-1)p` 与线性插值。对称方案取两个分位绝对值较大者构造 `scale`，零点为 0；非对称方案由分位上下界和 `[qmin, qmax]` 推导比例因子与零点，并裁剪零点。算法、插值、并列规则写入每层产物元数据。
3. **定点重缩放**：把原始缩放因子四舍五入到 `2^-fraction_bits` 网格；若变成零，则显式钳到最小非零步长并记录 `scale_minimum_clamped`。随后按组执行命名舍入、饱和裁剪，持久化整数、反量化值、逐元素裁剪方向和误差。全精度白名单层走精确旁路。
4. **蒸馏校正**：原始/量化矩阵在固定样本上产生输出；对原始输出按温度做 softmax 得到软标签与样本置信度，校正量为 `soft_label_weight × confidence × (teacher-student)`，并输出校正前后最大绝对偏差、MSE 与逐元素校正量。报告顺序固定为 `sample_id` 升序、层名升序。

## 增量账本与失效

每个阶段条目都有稳定缓存键。键由层内容指纹（含 `content_version`）、标定/量化参数指纹、上游阶段键和交换版本指纹组合而成；蒸馏条目另记录所用校准样本 ID 与样本集指纹。`run_report.json` 的 `cache_ledger` 可直接检查键、状态、稳定排序键和样本引用。

典型编辑的精确失效范围：

| 编辑 | 重算范围 |
| --- | --- |
| 切换交换版本 | 所有层的解析、转换、分位、重缩放、蒸馏 |
| 单层位宽或分组粒度 | 该层分位、重缩放、蒸馏 |
| 全局舍入模式 | 所有层重缩放、蒸馏 |
| 单层权重替换 | 该层解析、转换、分位、重缩放、蒸馏 |
| 蒸馏温度/软标签权重 | 所有层蒸馏 |

同一配置再次运行时应显示全部阶段 `recomputed` 为空。验收脚本还会对编辑序列的每个状态分别执行“增量状态”与“全新快照状态”，要求 `result_digest` 完全相同。

## 调度与快照

- 层任务按层名字典序获得预算；预算只消耗真正需要重算的量化点数。某层超出剩余预算时，该层及后续待算层保留为 `E_QUANT_BUDGET_PENDING`。
- 线程池只并行独立层任务；结果在收集后按 `<layer>:<stage>` 稳定键排序，因此 1 线程和任意线程数的报告顺序、误差表顺序和失败清单一致。
- 阶段顺序为 `parse_validation`、`quantile_calibration`、`fixed_point_rescale`、`distillation_correction`。每阶段先把完整 entries/manifest 写入临时快照目录，再用目录 rename 发布，最后原子替换 `current.json` 指针。
- manifest 记录全局版本号、配置哈希、交换版本指纹、旁路哈希、已完成层和改变阶段。已发布快照不再修改；重复运行创建新版本快照。`--interrupt-after quantile_calibration` 会以退出码 75 中止，后续进程可从已完成解析/分位阶段恢复。

## 业务码

| 代码 | 含义 |
| --- | --- |
| `E_CALIB_PARSE` | 标定格式、唯一性、缩放因子、零点或层名一致性校验失败 |
| `E_WEIGHT_NONFINITE` | 权重包含 NaN/Inf |
| `E_WEIGHT_SCHEMA` | 权重 JSONL、形状或数组结构错误 |
| `E_SIDECAR_MISSING` | 反向转换所需旁路缺失 |
| `E_SIDECAR_INVALID` | 旁路 JSON/结构无效 |
| `E_QUANT_BUDGET_PENDING` | 单批点数预算不足，层保留待算 |
| `E_QUANT_CLIP` | 使用 `saturation=error` 时发生越界 |
| `E_QUANTILE_RANGE` | 分位范围或方案无法构造有效参数 |
| `E_ZERO_SCALE` | 推导缩放因子为零或非法 |
| `E_SAMPLE_PARSE` | 校准样本结构、ID、有限性或输入维度错误 |
| `E_INTERNAL` | 未分类单层异常，报告仍会隔离并继续其他层 |

## 实际验证结果

本地环境：Python 3.11.0rc1、NumPy 1.26.4、PyYAML 6.0.2。最终命令结果：

```text
python -m pytest -q
15 passed
```

`python scripts/run_acceptance.py --samples samples --artifacts artifacts` 的结果保存在 `artifacts/acceptance_report.json`。本轮关键结果：

- 全量运行：0 失败、0 待算、24 条蒸馏记录；`layer.00` 全精度误差为 0。
- 定点误差：`layer.01` 裁剪率 `3/64`，`layer.02` 裁剪率 `1/48`；误差摘要含位宽、小数位、舍入模式。
- 双版本转换：v2→v1→v2 逐字段恢复；样例残差分别为 `-0.000436478`、`+0.000177156`、`-0.000291413`、`-0.000471396`（浮点显示可能多出二进制尾差）。
- 缺失旁路：状态 `sidecar_missing`，业务码 `E_SIDECAR_MISSING`。
- 编辑序列：7 个状态（基线 + 6 次编辑）增量与全量摘要全部一致；最终同状态重跑五阶段 `recomputed` 全为空。
- 预算：预算 96 时仅完成 `layer.00`、`layer.01`，`layer.02/03` 待算；取消预算恢复后摘要与全量一致。
- 强制中断：退出码 75；恢复快照只含解析/分位，续跑后摘要与全量一致。
- 注入：`INJ-001..004` 全部命中，分别映射 `E_WEIGHT_NONFINITE`、`E_CALIB_PARSE`、`E_CALIB_PARSE`、`E_SIDECAR_MISSING`；仅 `layer.00` 完成量化。

最终全量结果摘要：`0db12a3143fc38f5e28984449b788cd032f0539bb6e45b1b8f99b3bc5ab56fcc`。

## 目录

```text
qcal/               组件实现与 CLI
scripts/            确定性样本生成与验收入口
tests/              单元/集成/端到端测试
samples/            固定种子生成输入
artifacts/          运行报告、验收报告与快照证据
```

该实现是本地可集成领域组件与完整示例，不包含真实 LLM 推理后端；校准样本输出使用逐层矩阵乘法，便于在无外部数据条件下验证量化与校正语义。
