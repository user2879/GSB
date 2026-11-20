# 增量树集成诊断工作台

本工作区实现了一个本地、确定性、可恢复的表格树集成工作台。它读取 YAML 实验配置、JSON 合成数据描述和增量编辑计划，完成：

- 确定性合成数据、缺失值填充、类别编码与训练/验证切分；
- 数值列等频分箱与内容指纹；
- 逐棵树的行/列采样、最大增益分裂、残差叶值与结构指纹；
- 集成打分、验证准确率/对数损失、逐轮损失与逐树/逐特征贡献；
- 基于内容指纹的精确阶段失效与产物复用审计；
- 线程池并发、单树异常隔离；
- 原子快照、版本冲突识别、SIGTERM 可处理退出与 SIGKILL 后恢复；
- 增量运行与空白全量重建的逐字段等价性验证。

实现只依赖 Python 3.10+、numpy 和 PyYAML，不使用 Docker，不访问外部数据。

## 快速开始

```bash
python3 -m pip install -r requirements.txt

# 依次执行初始训练与四个编辑批次，所有批次共享同一增量状态目录
python3 -m tree_workbench run \
  --config samples/experiment.yaml \
  --data-spec samples/data_spec.json \
  --edits samples/edit_plan.json \
  --state-dir work/state \
  --tmp-dir work/tmp \
  --report-dir work/reports

# 每个批次都与同输入、同有效配置的空白全量重建比较；
# 额外比较并发数 1 与 4 的结果。
python3 -m tree_workbench verify \
  --config samples/experiment.yaml \
  --data-spec samples/data_spec.json \
  --edits samples/edit_plan.json \
  --work-dir work \
  --parallel-workers 4

# 自动测试
python3 -m unittest discover -s tests -v
```

`run` 每批次生成 `report-NN-name.json`，同时写入 `latest.json`。`verify` 生成
`verification-<id>/verification.json`，其中包含每个场景的差异路径、增量/全量报告路径、
缓存审计和并发一致性结果。

## 目录

```text
samples/                 YAML 配置、合成数据描述、增量编辑计划
tree_workbench/
  config.py              配置/数据描述/编辑计划业务校验
  data.py                确定性数据生成、编辑、填充、编码、切分
  binning.py             等频分箱与内容指纹
  tree.py                单树训练、树遍历与贡献拆分
  ensemble.py            集成打分、指标、逐轮损失、贡献汇总
  snapshot.py            原子快照与版本检查
  engine.py              增量调度、缓存审计、线程池、恢复
  verify.py              增量/全量逐字段对照
  cli.py                 命令行入口
tests/test_workbench.py  13 个端到端与边界测试
VALIDATION.md            本机实际验证记录
artifacts/verification.json  最终等价性验证结果
```

## 输入

`samples/experiment.yaml` 包含：

- 数据种子、特征顺序与目标列；
- 树数量候选、当前树数量、最大深度、学习率；
- 行采样比例、列采样比例、叶子最小样本数、分箱数；
- 训练集比例、线程池并发数、快照提交间隔、精度档位与模型类型。

`samples/data_spec.json` 描述数值列、类别列、缺失率、冗余复制列和二分类目标生成参数。
`samples/edit_plan.json` 的每一批可包含：

- `append_rows: {count, seed}`：追加确定性合成行；
- `corrections: [{row_id, column, value}]`：修正单行特征；同一批次先追加后修正；
- `tree_count` 或 `tree_count_delta`：只允许扩容；
- `max_depth`、`learning_rate`、`bin_count`。

所有输入生成和训练均使用显式派生的固定种子。同一输入下，单线程和多线程得到完全相同的阶段产物。

## 模型与确定性设计

每棵树使用固定的基础概率 `p0`（训练集正类比例），拟合冻结伪残差
`y_i - p0`。因此树结构只依赖数据、分箱和树级参数，不依赖学习率或总树数：

- 学习率变化只需要重算集成与贡献；
- 树数量增加只需要训练新增树；
- 每棵树可以安全地并发训练，结果按树序号稳定排序后汇总；
- `gradient_boosting` 模式逐树累加 `learning_rate * leaf_value`；
- `random_forest` 模式使用 `learning_rate / applied_tree_count` 的平均权重。

这是为增量依赖判定显式选择的“冻结梯度提升”语义。逐轮损失仍按固定树序累加和学习率计算，
并通过稳定 sigmoid 得到概率。若要改成依赖上一轮预测的传统串行 GBDT，会使学习率编辑和树级并发
同时失效，这与本任务要求的精确失效边界冲突。

树的每个节点按以下显式顺序计算增益：

1. `parent_sse = sum((r - parent_mean)^2)`；
2. 对候选分箱阈值分别计算 `left_sse`、`right_sse`；
3. `gain = parent_sse - left_sse - right_sse`；
4. 最大值并列时按 `(feature_index, threshold_bin)` 升序打破；
5. 叶值取落入样本残差的算术平均，并受 `min_samples_leaf` 约束。

## 内容指纹与失效矩阵

缓存键由上游内容指纹与参数指纹派生：

- `binning`：数据指纹、特征类型、分箱数、种子、精度；
- `tree:i`：数据指纹、分箱产物指纹、树序号、深度、采样比例、最小叶样本、模型类型、注入故障；
- `ensemble`：树产物指纹、数据指纹、学习率、模型类型、基础得分；
- `contributions`：集成产物指纹与树产物指纹。

| 编辑 | 分箱 | 已存在树 | 新增树 | 集成 | 贡献 |
| --- | --- | --- | --- | --- | --- |
| 无变化 | 复用 | 复用 | 无 | 复用 | 复用 |
| 追加样本/单行修正 | 重算 | 重算 | 重算 | 重算 | 重算 |
| 树数量增加 | 复用 | 复用 | 重算 | 重算 | 重算 |
| 深度调整 | 复用 | 重算 | 重算 | 重算 | 重算 |
| 学习率调整 | 复用 | 复用 | 无 | 重算 | 重算 |
| 分箱数调整 | 重算 | 重算 | 重算 | 重算 | 重算 |

报告中的 `cache_audit.reused`、`cache_audit.recomputed`、`cache_audit.deferred` 和逐树原因可直接审计。

## 快照与退出语义

状态目录结构：

```text
state/
  CURRENT
  snapshots/snapshot-000001/
    manifest.json
    stages/binning.json
    stages/trees/tree-000000.json
    stages/ensemble.json
    stages/contributions.json
```

提交过程先在 `.tmp-snapshot-*` 中完整写入并同步所有文件，再将整个目录原子重命名，最后原子替换
`CURRENT`。`manifest.json` 记录：

- `global_version`；
- 配置哈希；
- 数据指纹；
- 已完成树列表；
- 各阶段键、内容指纹与快照版本。

提交前在文件锁内再次读取 `CURRENT`；若版本与加载时不同，返回业务码
`SNAPSHOT_VERSION_CHANGED`。启动时只加载带 `complete: true` 且内容指纹匹配的完整快照；损坏、
未完成或被强制终止时遗留的临时目录会被忽略。

SIGTERM/SIGINT 只设置停止标志：不再提交新树，等待当前线程池任务完成，提交已完成的树并返回退出码
0。SIGKILL 后重启时，只能从最后一次完整替换的 `CURRENT` 快照继续。

## 异常隔离与报告

每个树任务用 `(tree_index, stage)` 标识。抽样、分裂或叶计算失败时，记录
`tree_index`、`stage`、`business_code`、`exception_type` 和消息；该树标记为失败，不进入集成、
损失累加或贡献汇总，其余树继续执行。

报告字段包括：

- `timings_ms`：加载、数据、分箱、训练、集成、贡献、快照提交和总耗时；
- `cache_audit`：复用、重算、延迟任务及逐阶段原因；
- `failures`：按阶段、树序号、业务码、字段稳定排序；
- `recovery`：加载点、提交点、已完成树；
- `integration`：验证指标、逐轮损失、有效树、失败树、集成/贡献指纹。

## 业务码

| 业务码 | 触发条件 |
| --- | --- |
| `CFG_UNKNOWN_COLUMN` | 配置特征或修正列不在数据描述中 |
| `CFG_TYPE_MISMATCH` | 列类型、值类型或数据描述参数不匹配 |
| `CFG_TARGET_MISSING` | 目标列未配置或不在数据描述中 |
| `CFG_TREE_COUNT_INVALID` | 树数量非正或候选列表非法 |
| `CFG_MAX_DEPTH_INVALID` | 最大深度非正 |
| `CFG_LEARNING_RATE_INVALID` | 学习率为负或非数值 |
| `CFG_BIN_COUNT_INVALID` | 分箱数为零或非正 |
| `CFG_CONCURRENCY_INVALID` | 线程池并发数非正 |
| `CFG_TREE_COUNT_DECREASE` | 编辑试图减少树数量 |
| `TREE_SAMPLING_FAILURE` | 单树抽样阶段异常 |
| `TREE_SPLIT_FAILURE` | 单树分裂阶段异常 |
| `TREE_LEAF_FAILURE` | 单树叶值阶段异常 |
| `SNAPSHOT_VERSION_CHANGED` | 提交前检测到 `CURRENT` 版本变化 |

配置错误会全部进入失败清单，不会静默回退到默认值。

## 验证

实际执行结果记录在 `VALIDATION.md`，完整机器可读结果在 `artifacts/verification.json`。
