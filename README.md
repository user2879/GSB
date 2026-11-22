# 表格线性模型正则化实验台

这是一个仅依赖 Python、NumPy 与 PyYAML 的本地确定性实验台，用于在合成表格数据上训练 L2 岭回归、L1 套索与弹性网络，并支持：

- 固定种子合成数据、训练集缺失填充、类别列 one-hot 编码与数值标准化；
- 固定批量顺序的确定性批量梯度下降，逐轮损失、系数摘要与验证指标；
- v1 定宽文本 / v2 JSON 双版本模型互转与降级 sidecar；
- 基于内容指纹的模型级增量复用，只重训真正受影响的模型；
- 全局训练轮次预算、单模型配额、稳定顺序的线程池并发与异常隔离；
- 版本化原子快照、SIGINT 优雅退出、SIGKILL 后的持久化检查点续跑；
- 复用/重算、失败、待办、阶段版本与耗时的机器可读审计报告。

验收结果见 [`reports/verification_report.md`](reports/verification_report.md)，机器可读结果见 [`reports/acceptance_results.json`](reports/acceptance_results.json)。

## 快速开始

```bash
python --version                 # 需要 3.10+
pip install -r requirements.txt

# 校验配置、生成数据并打印指纹
python -m tabular_lab validate --config configs/experiment.yaml

# 首次运行
python -m tabular_lab run \
  --config configs/experiment.yaml \
  --state-dir .lab-state \
  --report reports/last_run.json

# 相同配置再次运行：应命中全部训练与转换缓存
python -m tabular_lab run --config configs/experiment.yaml --state-dir .lab-state

# 与旧快照做严格兼容续跑；配置或数据变化返回 E_SNAPSHOT_STALE
python -m tabular_lab run --resume --config configs/experiment.yaml --state-dir .lab-state

# 空白重建（保留旧代快照，但本次不使用其缓存）
python -m tabular_lab run --rebuild --config configs/experiment.yaml --state-dir .lab-state
```

Windows 下 CLI 会主动把标准输出/错误流重配为 UTF-8；所有命令返回整数退出码。

## 项目结构

```text
tabular_lab/
  config.py        YAML 配置、预算、并发与故障注入校验
  data.py          合成数据、缺失填充、编码、切分与内容指纹
  trainer.py       三种正则化的确定性批量梯度下降
  model_io.py      v1/v2、统一中间表示、sidecar 与往返比较
  cache.py         内容寻址训练/转换缓存键
  scheduler.py     稳定预算分配与线程池执行
  snapshot.py      原子快照、代际指针、live checkpoint 与预算账本
  runner.py        增量失效、并发任务、审计、导出与提交编排
  cli.py           run / validate / convert / inspect
configs/           本地 YAML 配置与数据描述
samples/           v1 定宽样例、v2 JSON 样例及降级 sidecar
tests/             24 个 unittest 场景
scripts/           一键验收报告脚本
reports/           实际运行报告与验收导出
```

运行时状态目录由 `--state-dir` 指定，默认为 `.lab-state`：

```text
state/
  CURRENT                              原子指针，指向完整快照代际
  snapshots/g00000001-xxxx/
    manifest.json                      版本、配置/数据哈希、阶段版本、模型引用
    cache/train/...                    内容寻址训练产物
    cache/conversion/...               内容寻址版本转换产物
    stages/{load,standardize,train,convert}/manifest.json
    models/<model>.json                完整模型结果、身份与转换产物
    exports/models.v1.txt|models.v2.json
    sidecars/*.downgrade.json           v1 降级旁路元数据
    reports/run_report.json
  live/train_<key>.json               批间隔原子检查点
  live/ledger.json                    持久化预算账本
  last_run_report.json                最近一次 CLI 报告
```

## 配置

最小配置：

```yaml
name: demo
seed: 20261123
exchange_version: v2          # v1 或 v2
data: data.yaml               # 或内联 mapping
conversion_tolerance: 0.001

training:
  epochs: 55
  learning_rate: 0.055
  batch_size: 16
  tolerance: 1.0e-12
  l1_zero_subgradient: 0.0    # 只允许 -1, 0, 1；零处固定次梯度决断
  snapshot_every_batches: 12

budget:
  global_epochs: 500
  per_model_epochs: 120

runtime:
  max_concurrency: 3
  concurrency_limit: 8        # 硬上限 16

snapshot:
  strict_resume: false
  expected_version: null      # 可选，严格校验全局版本号
  keep_generations: 4

models:
  - name: ridge_core
    scheme: l2                # l2/ridge
    penalty: 0.12
    features: [income, tenure, region]
  - name: lasso_compact
    scheme: l1                # l1/lasso
    penalty: 0.025
  - name: elastic_balanced
    scheme: elastic_net       # elastic_net/elasticnet/enet
    penalty: 0.04
    mix: 0.55                 # 必须位于 [0, 1]
```

白名单项既可写原始列名，也可写编码后的特征名，例如 `region` 会展开为 `region=south`、`region=west` 等；也可直接写 `region=south`。白名单顺序、模型声明顺序会被规范化，不改变缓存键、模型结果顺序或语义摘要。未知列返回 `E_CONFIG_INVALID`。

## 数据生成

`configs/data.yaml` 描述数值列、类别列、冗余列、缺失率和二值目标。数据文件中的 `seed` 优先于实验配置顶层 `seed`。生成器只使用按 `seed + 逻辑名称 + 列 variant` 派生的局部随机数生成器，不读取或修改 NumPy 全局随机状态。

- 行级潜变量由独立固定种子生成；
- 每个特征列拥有独立子种子，替换单列的 `variant` 不改变其他列、切分或目标；
- 冗余列显式引用源列并记录为独立列指纹；
- 目标由潜变量和独立噪声生成，因此替换单个特征列不隐式改变目标；
- 训练/验证索引由独立固定种子生成；
- 数值列缺失值使用训练集均值填充，类别列使用训练集众数填充；
- 类别列按声明顺序做 one-hot，并默认丢弃第一个类别；
- 数值特征在训练集上做 `(x-mean)/std`，标准差接近零时使用 1；类别 one-hot 不做缩放。

模型系数与截距导出的是**标准化特征空间**中的系数。标准化均值/尺度保存在训练产物与 `standardize` 阶段快照中。

## 训练目标与确定性

记 `r = Xw + b - y`，`n` 为训练样本数。数据项为 `0.5 * mean(r^2)`，显式正则为：

```text
L2:           J = 0.5/n * ||r||^2 + lambda/2 * ||w||^2
L1:           J = 0.5/n * ||r||^2 + lambda * ||w||_1
Elastic Net:  J = 0.5/n * ||r||^2
                  + lambda * ((1-mix)/2 * ||w||^2 + mix * ||w||_1)
```

梯度：

```text
L2:           g_w = X^T r / n + lambda * w
L1:           g_w = X^T r / n + lambda * sign_fixed(w)
Elastic Net:  g_w = X^T r / n
                  + lambda * ((1-mix) * w + mix * sign_fixed(w))
全部方案:     g_b = mean(r)
```

`sign_fixed(0)` 取配置值，默认固定为 `0`。每个 epoch 使用固定连续批量划分和固定顺序；批量正则梯度乘以 `1/批次数`，使一轮批量梯度的正则贡献与整轮目标一致。初始 `w=0`、`b=mean(y_train)`。学习率固定，不依赖运行时全局状态。

每个 epoch 记录：

- 总损失、数据损失、正则损失；
- L1/L2 范数、非零系数数；
- 全精度系数的 SHA-256 摘要。

同输入重复训练的逐轮轨迹与最终系数逐字段相同；线程池并发仅影响耗时，不影响训练结果。

## 模型交换格式

### v1 定宽文本

每行一个模型-特征系数，字段总宽 50：

| 字段 | 宽度 | 内容 |
|---|---:|---|
| model name | 20 | 左对齐 |
| feature index | 4 | 十进制整数 |
| coefficient milli | 12 | 千分比整数，四舍五入 |
| penalty code | 2 | 十位=方案，个位=收敛状态 |
| intercept milli | 12 | 千分比整数 |

惩罚码：

```text
十位：1=l2, 2=l1, 3=elastic_net
个位：1=converged, 2=max_epochs, 3=budget_exhausted, 4=interrupted
例：11=L2/converged, 22=L1/max_epochs, 31=elastic_net/converged
```

解析器也接受旧式 `1/2/3` 并解释为对应方案 + `converged`。

### v2 JSON

v2 保存浮点系数、命名方案、命名收敛状态与 `precision_label`，并附带惩罚参数、混合比例、特征序号/名称和截距。样例见 [`samples/models.v2.json`](samples/models.v2.json)。

### 降级旁路

v2 → v1 使用十进制 `ROUND_HALF_UP` 还原千分比。sidecar 记录：

- 原始 float64 系数/截距与千分比值；
- 每个系数的截断残差和最大绝对残差；
- 原 `precision_label`；
- v1 无法表达的命名方案、收敛状态、惩罚系数、混合比例和特征名。

v1 加载器默认会自动查找同目录的 `sidecars/`，也可显式传入 sidecar 目录。重新加载后，v1 码表恢复命名方案/状态，sidecar 恢复弹性网络 mix、penalty、特征名和原始精度标签；无 sidecar 时才生成 `v1_milli_exact_to_float64`。v2→v1→v2 的系数/截距最大偏差不超过 `0.001`（实际样例为 `0.0005`）。

训练运行写 v1 时也会计算该最大截断残差；若超过配置 `conversion_tolerance`，该模型的转换阶段失败并进入失败清单，而不是静默产出超差模型。

## 缓存与精确失效

每个候选模型的稳定键为：

```text
scheme | penalty_fingerprint | model_name
```

训练缓存键由以下内容的规范化 SHA-256 派生：

- 选中特征的白名单/顺序规范化结果；
- 训练与验证数据切片内容指纹；
- 标准化统计量与缺失填充统计指纹；
- 样本集合与目标切片指纹；
- scheme、penalty、mix、学习率、批量、轮次、次梯度决断；
- 调度器分配的单模型配额、实验种子、该模型故障注入配置。

版本转换缓存键由以下内容派生：

- 训练产物内容指纹；
- 交换版本号 `v1/v2`；
- 完整特征 schema 指纹与特征序号；
- 该模型的转换阶段故障配置。
- 配置的转换容差。

交换版本只会改变版本相关产物：切换 v1/v2 时训练产物直接复用，转换产物按新版本键重算。学到的系数本身与序列化版本无关，这样既满足“不重训真正未受影响的模型”，又可审计 `training_reused` 与 `conversion_recomputed`。

常见编辑的失效面：

| 编辑 | 训练失效 | 转换失效 |
|---|---|---|
| 某方案 penalty/mix | 该方案模型 | 该模型及后续旧转换键 |
| 特征白名单变化 | 白名单选择内容变化的模型 | 对应模型 |
| 单列数据替换 | 白名单包含该列的模型 | 对应模型 |
| 目标/切分变化 | 全部模型 | 全部模型 |
| 交换版本切换 | 无 | 全部模型 |
| 白名单/模型声明顺序变化 | 无 | 无 |

每次运行输出排序稳定的：

- `training_reused` / `training_retrained`；
- `conversion_reused` / `conversion_recomputed`；
- `reuse_list` / `recompute_list`；
- 逐模型失效原因；
- 失败清单与待办清单。

`semantic_digest` 只覆盖最终数据、阶段版本、模型系数、损失轨迹、验证指标、失败清单、待办和导出内容，不覆盖运行耗时、缓存命中动作或快照代际号。任意增量编辑序列后的 `semantic_digest` 与相同最终配置的空白全量运行一致；实际验收还逐字节比较了最终 v1 导出与 sidecar。

## 预算与并发

预算分配先按稳定键排序，再依次为每个候选模型预留 `min(per_model_epochs, remaining_global)`。预留额度在本次运行中不因其他模型提前收敛而回收，从而保证分配不受线程完成顺序影响。该预留值进入训练缓存键。

- 模型到达配额仍未收敛：`E_MODEL_BUDGET_EXHAUSTED`，不进入验证汇总；
- 全局额度为零、模型未开始：`E_TRAIN_BUDGET_EXHAUSTED`，保留在待办清单；
- 每批按 `snapshot_every_batches` 写原子 live checkpoint，快照记录已消耗 epoch；
- 线程池并发数来自 `runtime.max_concurrency`，不能超过 `concurrency_limit`，后者硬上限为 16；
- 所有结果、失败与清单在收集后按稳定键排序，因此串行与并发语义一致。

## 快照、续跑与退出

一次运行在完整 staging 代际目录中构建所有阶段产物，最后：

1. 原子写入 `manifest.json`；
2. 将 staging 目录原子改名为最终代际目录；
3. 原子替换 `state/CURRENT` 指针。

旧代快照不会被半成品覆盖；新代提交前的崩溃只留下可忽略目录。每个阶段都有内容派生的 `stage_versions`：`load`、`standardize`、`train`、`convert`。

训练过程中每批间隔写入独立、完整的 live checkpoint（含系数、截距、已完成 epoch、下一批位置、损失轨迹）。SIGINT/SIGTERM 设置停止标志，在批/epoch 安全点持久化当前状态，提交快照后退出 130。SIGKILL 后，下一次运行依据最后已提交代际与原子 live checkpoint 继续；续跑轨迹与全量训练逐字段一致。

恢复判定：

- 普通运行：允许配置/数据编辑后做增量复用；
- `--resume` 或 `snapshot.strict_resume: true`：配置哈希与数据指纹必须与清单一致，否则 `E_SNAPSHOT_STALE`；
- 严格恢复时可用 `snapshot.expected_version` 额外要求清单全局版本号匹配。

## 业务错误码与退出码

| 业务码 | 含义 |
|---|---|
| `E_CONFIG_INVALID` | 配置/数据/未知列/结构无效 |
| `E_SCHEME_UNKNOWN` | 正则化方案或整型惩罚码未知 |
| `E_MIX_RANGE` | mix 不在 `[0,1]` |
| `E_SCALE_RANGE` | 千分比超出 `±999,999,999` |
| `E_FIELD_TRUNCATED` | v1 字段宽度不足/长度截断 |
| `E_MODEL_BUDGET_EXHAUSTED` | 单模型配额耗尽 |
| `E_TRAIN_BUDGET_EXHAUSTED` | 全局预算耗尽，模型未开始 |
| `E_SNAPSHOT_STALE` | 严格续跑与清单配置/数据/版本不兼容 |
| `E_MODEL_EXCEPTION` | 标准化、梯度或转换阶段异常 |

退出码：`0` 全部成功；`2` 配置/格式错误；`3` 快照过期；`4` 运行完成但存在模型失败/待办；`1` 未预期内部错误；`130` 被可处理信号中断。

## 双版本样例

```bash
python -m tabular_lab inspect --input samples/models.v2.json
python -m tabular_lab convert \
  --input samples/models.v2.json \
  --to v1 \
  --output /tmp/models.v1.txt \
  --sidecar-dir /tmp/sidecars
python -m tabular_lab convert \
  --input /tmp/models.v1.txt \
  --to v2 \
  --output /tmp/models.v2.json \
  --sidecar-input /tmp/sidecars
```

## 验证

```bash
python -m unittest discover -s tests -v
python scripts/run_acceptance.py
```

本次实际运行：

- Python 3.11.0rc1、NumPy 1.26.4、PyYAML 6.0.2；
- 24 个自动化测试全部通过；
- 三方案训练、缓存全命中、惩罚/版本/单列/白名单增量失效、增量与全量语义及导出字节一致；
- v1/v2 往返最大偏差 `0.0005`，稀疏特征序号的实际 v1 导出可正确回读；
- 预算错误顺序、串行/并发一致性、转换故障隔离；
- SIGINT 退出 130 并提交中断快照；
- SIGKILL 后由 live checkpoint 续跑，结果与全量一致；
- 配置变更严格续跑返回 `E_SNAPSHOT_STALE`。

原始任务说明保存在 [`docs/TASK.md`](docs/TASK.md)。
