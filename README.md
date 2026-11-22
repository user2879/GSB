# 确定性增量领域数据管线

本项目在本地、离线、固定种子合成数据上实现一条完整管线：

`样本校验 -> 特征处理 -> 模型训练 -> 定点推理 -> 结果汇总 -> 阶段快照`

核心不是“每次全量重跑”，而是基于内容指纹的精确阶段复用。阶段产物是可复用的持久化快照；样本、配置或上游产物变化时，只有内容键真正变化的阶段及其下游会失效。并发线程调度、预算截断、失败收集和输出排序均经过稳定化处理。

## 快速开始

环境要求：Python 3.10+、NumPy。PyYAML 可选；未安装时自动使用内置 YAML 子集解析器。

```bat
python --version
python -c "import numpy; print(numpy.__version__)"

python app.py demo --workspace demo_workspace --reset
python -m pytest -q
python verify_acceptance.py
```

Windows 也可直接执行 `run_demo.bat` 或 `run_demo.ps1`。

常用命令：

```bat
python app.py generate --workspace demo_workspace --reset
python app.py run --workspace demo_workspace --plan demo_workspace\plans\edits.json --verify-full
python app.py versions --workspace demo_workspace
python app.py restore --workspace demo_workspace --version 2
python app.py compare --left demo_workspace --right full_rebuild_workspace
```

`run` 在已有快照时默认续跑。不要把编辑后的“最终配置”强行当作续跑配置；若传入 `--config`，其语义哈希必须与当前快照清单一致，否则输出 `E_SNAPSHOT_STALE`。

## 目录与产物

- `src/domain_pipeline/`：实现代码。
- `app.py`：免安装 CLI 入口。
- `tests/test_pipeline.py`：11 个自动化测试，覆盖校验、确定性、并发、预算、失效、快照和全量对拍。
- `verify_acceptance.py`：执行完整验收并生成 `acceptance/acceptance_report.json`。
- `demo_workspace/`：固定种子输入、4 批编辑序列、阶段快照和 JSON 报告。
- `full_rebuild_workspace/`：由增量最终状态重新生成的冷启动全量管线。
- `acceptance/`：预算串并行探针、量化探针和验收汇总。

工作区布局：

```text
samples/samples.jsonl
configs/pipeline.yaml
plans/edits.json
snapshots/CURRENT
snapshots/versions/v000001-<content-id>/
  manifest.json
  samples.jsonl
  config.yaml
  state.json
  ledger.json
  stages/features.json
  stages/training.json
  stages/inference.json
  stages/summary.json
reports/latest.json
reports/run-v000004.json
```

## 输入格式

### 样本 JSONL

```json
{"id":"S0001","age":37.2,"income":88.4,"region":"east","channel":"web","label":1,"split":"train"}
```

- `numeric_columns` 的值必须是有限数值；`null` 表示缺失。
- `categorical_columns` 的缺失值用 `null` 表示；非缺失值必须在声明域内。
- 标签只能为整数 `0` 或 `1`。
- `split` 只能为 `train` 或 `validation`。
- 重复 ID、非法 JSON、非有限数、域外类别、非法标签分别进入失败清单，其他行继续加载。

### 管线 YAML

完整字段见 `demo_workspace/configs/pipeline.yaml`。关键字段：

- `features.bins`：数值列等频分箱数，必须为正整数。
- `training.candidates`：初始化尺度、初始偏置和 L2 候选；`selected_candidate` 按索引或 ID 选择。
- `training.batch_size`、`learning_rate`、`epochs`：确定性小批量梯度下降参数。
- `quantization.fractional_bits`：定点分母为 `2^fractional_bits`。
- `quantization.scale_level`：量化步长，内部整数为 `round(raw_p * denominator / step) * step`。
- `concurrency.max_workers`：线程池上限；超过 `max_workers_limit` 时钳制并记录 `E_CONFIG_CONCURRENCY_LIMIT`。
- `concurrency.batch_budget`：单批样本计算预算。
- `snapshot.commit_interval_batches`：每多少个完整批次提交一次不可变快照。

### 编辑计划

```json
{
  "batches": [
    {
      "batch_id": "b001",
      "operations": [
        {"type": "append_samples", "samples": []},
        {"type": "correct_sample", "sample_id": "S0007", "patch": {"age": 41.5}},
        {"type": "update_feature_params", "patch": {"bins": 6}},
        {"type": "update_model_hyperparams", "patch": {"learning_rate": 0.08}},
        {"type": "update_quantization", "patch": {"fractional_bits": 10, "scale_level": 8}},
        {"type": "update_config", "patch": {"concurrency.max_workers": 2}}
      ]
    }
  ]
}
```

批次 ID 会持久化；续跑时已应用批次不会重复追加或修正。预算耗尽时，当前批次保留为已应用状态，待办样本 ID 和特征片段缓存进入快照，下一次运行继续。

## 确定性设计

### 内容键与依赖

| 阶段 | 缓存键来源 | 直接上游 |
| --- | --- | --- |
| `features` | 特征输入指纹 + 特征参数 + 实现版本 | 有效样本的特征字段 |
| `training` | 训练样本特征向量 + 标签 + 有效模型参数 | `features` 的训练子集 |
| `inference` | 模型键 + 验证集内容 + 量化参数 | `training` 与验证特征 |
| `summary` | 推理键 + 汇总/量化参数 | `inference` |

每个阶段账本记录缓存键、上游键、样本 ID 集合、状态和原因。标签修正不改变特征键，因此 `features` 直接复用；只改变量化档位时 `features`、`training` 复用，仅 `inference`、`summary` 失效。新增仅验证集样本不会改变训练数据集指纹，训练产物可复用。

### 特征

- 数值列按样本 ID 稳定排序后的非缺失值做等频切分；缺失值映射到独立箱，并保留 `missing: true`。
- 类别列按声明域顺序生成确定性编码，缺失值使用独立编码。
- 每个数值列和类别列都保存由列内容、参数、边界/编码表派生的 SHA-256 内容指纹。
- 单行特征片段另以“行内容 + 分箱边界/编码表 + 参数”为键缓存；边界不变时，追加或修正其他行不会重算未受影响片段。

### 训练与推理

- 初始权重由候选参数和特征维度确定，不使用全局随机状态。
- 每个小批次内，逐样本梯度在线程池中并行计算；结果按稳定键排序后以固定顺序累加，再更新参数。
- 损失轨迹每轮按稳定样本顺序累加。
- 推理概率先转成定点整数；预测标签、准确率和对数损失均由量化整数计算，不是显示层格式化。
- 推理和汇总按样本 ID 升序输出。

### 并发与预算

- 任务排序键形如 `feature:S0001`、`train:e0000:b0000:sS0001:gradient`、`inference:S0001`。
- 预算在提交线程任务前按稳定键串行扣减，因此线程完成顺序不会改变预算序列或待办队列。
- 预算不足项保留稳定排序，并生成 `E_BATCH_BUDGET_PENDING`。
- 预算单位是特征阶段的一个样本行计算；缓存命中成本为 0。训练梯度与推理行的并发由同一 `max_workers` 限制，并在上游阶段完整后运行。
- `max_workers=1` 与 `max_workers=4` 的最终产物、预算序列和失败顺序一致。

## 快照与恢复

- 每个版本是不可变目录，先写完整暂存目录，再原子替换 `CURRENT` 指针；不逐文件覆盖线上快照。
- 提交前比较内容身份；无变化时直接复用当前版本。
- 清单包含全局版本、配置哈希、有效数据指纹、原始数据指纹、各阶段版本、各阶段文件指纹、待办队列和应用批次。
- 加载时校验引擎版本、配置哈希和样本原始指纹；不一致输出 `E_SNAPSHOT_STALE`。
- `restore` 原子切换 `CURRENT`，可恢复到任一历史完整快照。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_JSON_PARSE` / `E_SAMPLE_OBJECT` | 样本行不是合法 JSON 对象 |
| `E_SAMPLE_ID_INVALID` / `E_SAMPLE_ID_DUPLICATE` | 样本 ID 非法或重复 |
| `E_NUM_NONFINITE` | 数值列不是有限数值 |
| `E_CATEGORY_DOMAIN` | 类别值不在声明域内 |
| `E_LABEL_BINARY` | 标签不是 0/1 |
| `E_SPLIT_VALUE` | 划分不是 train/validation |
| `E_CONFIG_BINS` | 分箱数不是正整数 |
| `E_CONFIG_CONCURRENCY_LIMIT` | 并发数超过上限，已钳制 |
| `E_CONFIG_BUDGET` | 单批预算不是正数 |
| `E_SNAPSHOT_STALE` | 版本、配置或快照内容不一致 |
| `E_BATCH_BUDGET_PENDING` | 单批预算不足，任务进入待办 |
| `E_EDIT_*` | 编辑计划/操作/路径非法 |

失败清单按 `scope`、行号、路径、样本 ID、批次和操作序号的稳定顺序输出。

## 全量对拍口径

每个批次完成后，`--verify-full` 会从当前最终样本与配置创建全新冷启动管线，并比较：

- 分箱边界、缺失箱、类别编码和全部特征行；
- 每轮损失、最终权重、偏置和批次顺序；
- 定点整数、逐样本预测、准确率和对数损失；
- 阶段缓存键、上游内容键和样本集合；
- 配置/样本状态失败清单。

冷启动全量运行的运行时失效/复用标记必然是“全部失效、无复用”，而增量运行记录真实缓存决策；因此前者与后者的运行决策清单分别核验，不把“冷启动全失效”伪装成“增量复用”。逐字段一致性约束落在产物内容、依赖账本与状态失败清单上。这也是本项目对“增量复用”和“全量重建一致”两个目标的明确边界。

## 实际验证结果

本机环境：Python `3.11.0rc1`、NumPy `1.26.4`、Linux/WSL 工作区；实现仅使用标准库和本地文件，无网络访问。

自动化测试：

```text
...........                                                              [100%]
11 passed
```

`python verify_acceptance.py` 生成 `acceptance/acceptance_report.json`，实际结果：

- 4 批编辑全部完成；每批 `full_rebuild.equal = true`。
- 增量最终快照与冷启动全量重建的 4 个阶段版本完全相同：
  - `features = 88396672...7909`
  - `training = 5d164d79...c116`
  - `inference = 9dac14c5...5406`
  - `summary = 6a0bac54...16b1`
- 第 4 批只修正标签并改量化参数：`features` 复用（36/36 行片段复用），训练、推理、汇总失效。
- 预算串行/并行探针均在 3 次运行后完成，均出现 `E_BATCH_BUDGET_PENDING`，预算序列、待办队列和最终产物完全相等。
- 快照恢复：成功从版本 1 原子切换并恢复至最新版本 4。
- 陈旧检测：版本不匹配和配置哈希不匹配均返回 `E_SNAPSHOT_STALE`。
- 量化真实性：同一样本内部整数由 `2468` 变为 `616`，汇总对数损失由 `0.6791168765290896` 变为 `0.6801305060422056`。

完整明细见：

- `acceptance/acceptance_report.json`
- `demo_workspace/reports/latest.json`
- `demo_workspace/snapshots/versions/*/manifest.json`

## Windows 说明

- 代码路径使用 `pathlib`，原子替换使用同目录临时文件与 `os.replace`。
- 快照目录整体切换通过 `CURRENT` 指针完成，避免 Windows 上替换非空目录的限制。
- 不需要 Docker、数据库、网络或外部服务。
