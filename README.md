# Deterministic Feature Crossing and Reduction Pipeline

本项目从空白工作区实现了一条可审计的离线表格特征工程链路：

`确定性样本 -> 类别列组合交叉编码 -> 标准化 -> PCA / 截断随机投影 / 方差筛选 -> 单双精度指标与诊断`

实现只依赖 Python 3.10+ 与 NumPy。YAML 使用项目内置的轻量解析器，不依赖 PyYAML，也不访问外部数据或网络。

## Quick Start

```bash
python3 -m pip install -r requirements.txt

# 1. 固定种子生成一次样本，并记录离线校验摘要
python3 main.py prepare --config configs/pipeline.yaml

# 2. 后续所有运行复用已持久化样本
python3 main.py run --config configs/pipeline.yaml

# 3. 可选：原子替换单列数据，再运行增量链路
python3 main.py replace-column \
  --config configs/pipeline.yaml \
  --name region \
  --values-file path/to/region_values.yaml
python3 main.py run --config configs/pipeline.yaml

# 4. 配置失败码检查
python3 main.py validate --config configs/pipeline.yaml
```

主报告写入 `reports/diagnostic_report.json`。`prepare` 将样本写入 `data/sample.json`，离线摘要写入 `data/sample_manifest.json`。所有相对路径都以 CLI 的 `--root`（默认当前目录）为基准；`--root` 是全局参数，应放在子命令之前。

## Layout

```text
configs/pipeline.yaml          管线参数、路径、精度与并发配置
configs/columns.yaml           固定种子与列描述
feature_pipeline/config.py     配置加载、业务码与失败聚合
feature_pipeline/sample.py     样本生成、完整性校验、单列替换
feature_pipeline/cross.py      确定性列组合枚举与哈希编码
feature_pipeline/standardize.py 精度感知标准化
feature_pipeline/reduce.py     三路降维与定点量化
feature_pipeline/metrics.py    下游指标与边界归因
feature_pipeline/snapshot.py   内容寻址阶段快照与恢复
feature_pipeline/pipeline.py   四阶段编排、缓存账本与报告
main.py / feature_pipeline/cli.py 命令行入口
tests/                         14 项 unittest
scripts/acceptance.py          完整验收脚本
VERIFICATION.md                最近一次实测结论
reports/acceptance/            验收生成的 JSON 报告
```

## Deterministic Sample

`configs/columns.yaml` 固定 `seed`、`rows` 和每列类型。每个类别列按列名派生独立 RNG，按 `values` 生成取值；数值列在 `[min, max]` 内生成。缺失值确定性注入并填写：

- 类别列缺失值填写 `__MISSING__`，再按稳定标量键生成类别编码。
- 数值列缺失值填写该列观测均值。
- 样本摘要由完整规范 JSON 的 SHA-256 计算，不包含时间戳。
- `prepare` 在样本和清单都有效时直接复用；`--force` 重建后校验摘要仍不变。
- `run` 不自动重造样本；缺失、摘要不符或列描述变化会输出 `E_SAMPLE`。

当前样例离线摘要：

```text
5b44fb245a90cd93e31480791e9f13f95c1423c7b424285877921fa257c053fa
```

单列替换保存在 `data/overrides/<column>.json`，内容先写临时文件再原子替换。替换列会重新做缺失填充与类别编码；默认标签 `target` 在样本生成时固定，不因特征列替换而重算。

## Pipeline And Cache Graph

每阶段都有稳定排序键、内容指纹、缓存键和独立快照。缓存键均为上游内容指纹、参数指纹和精度档位的 SHA-256 派生值。

| Stage | 排序键示例 | 缓存键上游 | 失效范围 |
|---|---|---|---|
| `cross_encoding` | `cross_encoding/region+device/tier=single` | 组合所引用列的独立数据指纹 + 交叉参数 | 仅该组合及该精度档位 |
| `standardization` | `standardization/x__region__device/tier=double` | 对应交叉产物内容指纹 + 标准化参数 | 对应特征 |
| `dimensionality_reduction` | `dimensionality_reduction/pca/tier=double` | 全部入模标准特征内容指纹 + 该路径参数 | 该路径及下游指标 |
| `metrics` | `metrics/pca/tier=single` | 路径产物内容指纹 + 标签指纹 + 指标参数 | 该路径指标 |

关键行为：

- 交叉组合按排序后的白名单列枚举；任务按稳定键排序，不依赖完成顺序。
- 单列替换只改变引用该列的交叉组合键。未被引用的组合、其标准化特征、路径参数未变的其它降维路径直接以硬链接复用旧快照文件。
- 调整 PCA 保留维度只使 PCA 路径及其指标失效；随机投影和方差筛选产物直接复用。
- 随机投影种子由“随机投影路径配置 + 入模特征键”的路径级配置指纹派生，因此修改 PCA 参数不会无意改变随机投影矩阵。
- 降维账本明确记录 `referenced_cross_features`；路径报告也包含 `included_cross_feature_keys`。
- 冷启动与热启动的物理 `reused/recomputed` 动作必然不同，但键、产品顺序、依赖账本、阶段版本和业务结果可逐字段比较。

## Precision And Quantization

单精度 `float32` 与双精度 `float64` 作为同阶段任务并行执行。固定点量化在投影输出上执行，档位和 `fractional_bits` 来自配置，例如：

```yaml
precision:
  tiers: [single, double]
  stage_tolerance: 1.0e-7
  quantization:
    tier: coarse
    profiles:
      fine: {enabled: true, fractional_bits: 12}
      coarse: {enabled: true, fractional_bits: 5}
```

诊断分别比较四个环节：

1. `cross_encoding`：交叉编码逐特征向量差值。
2. `standardization`：标准化逐特征差值。
3. `projection`：每条路径分别比较 `raw` 与 `quantized` 投影。
4. `final_metrics`：每个指标标量及逐行决策向量。

`first_exceeded_stage` / `first_exceeded_item` 按上述固定顺序标注首次超容差的位置。PCA 主成分按绝对值最大分量的符号归一，且同一投影矩阵在单双精度间保持相同数值身份。

## Boundary Attribution

对每条路径逐行比较单双精度最终决策。仅当某行判定不一致时写入 `boundary_cases`：

- 若未量化投影已经导致两档位决策不一致，归因为 `dimensional_approximation`。
- 若未量化决策一致、量化后不一致，归因为 `quantization`。
- 同时记录量化是否改变本档位决策、近似是否改变原始决策、两种分数与分数差。

阈值比较使用 double 精度执行，使 float32 投影分数的舍入边界可被审计。本次验收分别构造并通过了 2 个量化翻转样本与 2 个降维近似翻转样本。

## Snapshots And Recovery

四个阶段分别为 `cross_encoding`、`standardization`、`dimensionality_reduction`、`metrics`。每次提交：

1. 先判断旧阶段的每个产品缓存键和文件 SHA-256。
2. 在配置的临时目录构建完整新阶段目录；未失效文件硬链接复用。
3. 内容寻址版本目录以重命名方式一次性发布。
4. 原子替换阶段 `current` 符号链接。
5. 最后原子写全局 `cache/manifest.json`，记录全局版本、配置哈希、样本/数据指纹、精度档位指纹和四阶段版本。

恢复时会清理遗留 `.stage-build-*`，若 `current` 缺失或损坏，则优先使用全局清单记录版本，否则扫描有效版本目录恢复。旧版本不会被当前提交覆盖，因此中断后可回退。

## Concurrency

`runtime.max_workers` 控制唯一任务池。交叉、标准化、降维和指标任务都在同一配置上限内执行。每个任务携带稳定键；返回后按键重新排序，因此串行与任意并发的列表顺序一致。报告中的 `max_observed_workers` 不超过配置值。

## Failure Codes

配置校验不会因单列失败而短路，最终按稳定顺序进入 `failures` 清单。

| Code | Meaning |
|---|---|
| `E_CONFIG` | 文档、路径、字段结构或数值无效 |
| `E_UNKNOWN_COLUMN` | 白名单或替换数据引用未知列 |
| `E_TYPE_MISMATCH` | 列类型、类别值、数值值或白名单类型不符 |
| `E_CROSS_ORDER` | 交叉阶数非正或超过可用类别列数 |
| `E_CARDINALITY_BUDGET` | 基数预算或哈希桶数非正 |
| `E_QUANTIZATION_PROFILE` | 量化档位、profile 或小数位宽非法 |
| `E_PRECISION_TIER` | 精度档位不是 single/double 两档 |
| `E_REDUCTION_DIM` | 降维维度非正 |
| `E_SAMPLE` | 样本缺失、过期或摘要不符 |
| `E_DATA` | 样本/替换列结构或长度错误 |
| `E_TARGET` | 配置标签列不可用 |

失败时仍输出 JSON 报告，包含已加载列、已有阶段版本、失败清单和恢复动作；任务失败不会中断对其余配置项的校验。

## Diagnostic Report

`reports/diagnostic_report.json` 包含：

- 业务配置、配置哈希、样本摘要、有效数据指纹、全局版本。
- 单双精度指纹、四阶段版本、版本变化与快照替换状态。
- 完整交叉特征编码、唯一取值数、降级动作与哈希桶明细。
- 三路降维的保留维度、输出特征、解释方差/投影矩阵摘要/选中特征和完整降维数值。
- 四环节逐项误差、首次超差位置、边界及归因。
- 复用账本、失效账本、依赖账本、缓存摘要。
- 失败清单与恢复动作。
- `semantic_fingerprint`：排除物理缓存动作和运行环境后的业务结果指纹。

## Verification

```bash
python3 -m unittest discover -v
python3 scripts/acceptance.py
```

本次实际结果：

- 单元测试：`14/14 OK`。
- 验收检查：`13/13 passed`，详见 `reports/acceptance/acceptance.json` 与 `VERIFICATION.md`。
- 冷启动 36 个产品；温启动 36 个全部直接复用。
- 替换 `region` 后复用 12 个产物，仅失效引用该列的 24 个交叉/下游产品；未引用该列的 `channel+device`、`channel+tier`、`device+tier` 直接复用。
- 仅调整 PCA 维度时复用 32 个、失效 4 个，且失效路径仅为 PCA。
- 增量最终结果与空白全量重建：语义指纹、全局版本、阶段版本、三路输出、逐项误差、边界清单、失败清单、依赖账本全部一致。
- 串行与 4 线程：语义、阶段版本、逐项误差一致；观测并发分别为 1 与 4。
- 单双精度首次超差环节：`standardization`；随后 `projection` 也超差。
- 量化边界与降维近似边界各命中 2 个样本，并正确分类来源。
- 中断快照恢复动作：`remove_abandoned_staging`、`restore_current_pointer`，恢复后语义与阶段版本不变。

## Notes

- 只使用本地固定种子数据，没有外部数据集或网络目标。
- `numpy` 是唯一运行时第三方依赖。
- YAML 解析器只覆盖本项目和样例所需子集；完整 JSON 配置同样可被读取。
- `stage_tolerance` 是业务审计参数，默认样例故意设为 `1e-7` 以展示 float32/float64 分级超差；容差可随时调整。
