# 表格聚类与密度估计专业工作台

本项目在本地确定性合成表格数据上完成特征预处理、KMeans / DBSCAN / 高斯混合聚类、轮廓系数与 Calinski-Harabasz 指数、密度对数似然评估、双跑可复现性验证、独立故障注入和阶段化快照恢复。项目不依赖外部数据集或网络服务。

## 快速开始

环境使用 Python 3.11。本机验证版本为 NumPy 2.4.6、scikit-learn 1.9.1、PyYAML 5.4.1。

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pip install -e .
```

运行干净基线：

```bash
python3.11 -m tcw \
  --config config/experiment.yaml \
  --no-injections \
  --snapshot-dir artifacts/snapshots-clean \
  --report artifacts/runs/clean_report.json
```

运行独立故障注入样例：

```bash
python3.11 -m tcw \
  --config config/experiment.yaml \
  --injections config/injections.yaml \
  --snapshot-dir artifacts/snapshots \
  --report artifacts/runs/diagnostic_report.json
```

恢复同一配置、同一注入配置和同一数据哈希对应的快照：

```bash
python3.11 -m tcw \
  --config config/experiment.yaml \
  --injections config/injections.yaml \
  --snapshot-dir artifacts/snapshots \
  --report artifacts/runs/diagnostic_report_resume.json \
  --resume
```

验证版本不匹配时会拒绝续跑：

```bash
python3.11 -m tcw \
  --config config/experiment.yaml \
  --injections config/injections.yaml \
  --snapshot-dir artifacts/snapshots \
  --report artifacts/runs/mismatch_report.json \
  --snapshot-version 2 --resume
# stderr: E_CONFIG_MISMATCH ...
# 退出码: 2
```

测试：

```bash
python3.11 -m unittest discover -s tests -v
```

## 目录

- `src/tcw/config.py`：YAML 配置、候选超参数、逐项校验与合法项继续执行。
- `src/tcw/data.py`：固定种子合成表格、缺失值训练段填充、类别映射与 one-hot 编码。
- `src/tcw/preprocessing.py`：标准化 / 分位数变换、固定累加顺序统计量、内容指纹。
- `src/tcw/models.py`：手写确定性 KMeans、DBSCAN 和固定初始化 GMM。
- `src/tcw/evaluation.py`：三项指标、密度对数似然、双跑差值、加权排名与稳定并列规则。
- `src/tcw/injection.py`：独立注入配置、真实阶段校验探针、唯一注入编号与业务码映射。
- `src/tcw/snapshot.py`：临时目录整体提交、manifest、阶段版本与恢复校验。
- `src/tcw/pipeline.py`：线程池候选执行、检查点、阶段恢复、诊断报告原子落盘。
- `config/`：实验、数据描述与注入样例。
- `tests/test_workbench.py`：六组端到端及边界测试。
- `artifacts/runs/`：实际运行报告。

## 配置协议

`config/experiment.yaml` 的必要结构：

- `experiment.seed`：运行种子；非法时回退到数据描述种子并记录 `E_CONFIG_INVALID`。
- `target.task`：目标结构；缺失或非聚类任务会记录 `E_CONFIG_INVALID`，其余合法项继续。
- `data.description`：相对配置文件目录或绝对路径的数据描述。
- `preprocessing.method`：`standardize` 或 `quantile`。
- `preprocessing.fit_fraction`：仅前 `ceil(rows * fit_fraction)` 行参与填充和变换统计量拟合。
- `preprocessing.include_columns` / `drop_columns`：未知列记录配置失败并从合法特征中移除。
- `preprocessing.expected_types`：列类型校验；类型不符记录配置失败并移除该列。
- `clustering.methods`：`kmeans`、`dbscan`、`gaussian_mixture`。
- `clustering.candidates`：按方法给出超参数列表；也可使用带 `method` 的列表形式。
- `evaluation.metrics`：`silhouette`、`calinski_harabasz`、`log_likelihood`。
- `evaluation.primary` / `weights`：默认使用归一化加权 `composite`；并列按方法名、超参数字典序、稳定键排序。
- `execution.concurrency`：线程池上限；非正数回退为 1 并记录失败。
- `execution.snapshot_commit_interval`：每若干稳定候选组提交一次聚类检查点。
- `execution.interaction_latency_budget_ms`：预处理、聚类、评估阶段预算检查。
- `execution.snapshot_version`：全局快照版本号。

数据描述支持数值列、类别列、高基数类别列与冗余列。生成、缺失掩码和冗余噪声都使用固定 `PCG64` 随机序列；类别映射按字典序稳定生成。

## 确定性规则

- 候选稳定键为 `method:sha256(canonical_params)[:16]`；配置、检查点和指标表按该键排序，失败清单按失败码、阶段、注入编号、候选键和消息组成固定键排序。
- 每个候选在独立任务中训练两次；报告记录标签是否相等、逐样本分配差异数、各指标绝对差和 GMM 责任度最大绝对差。
- KMeans 使用确定性的最远点初始质心、固定迭代上限和行号稳定同分决策。
- DBSCAN 以样本行号顺序遍历，邻域按行号排序，使用稳定 BFS 扩簇；噪声标签保留为 `-1`。
- GMM 使用固定 KMeans 初始化、`n_init=1`、固定收敛阈值和由种子与稳定键派生的 `random_state`；硬分配由软责任度按行号稳定决断。
- GMM 的密度指标直接使用拟合模型的 `score_samples` 均值；KMeans / DBSCAN 使用硬分配簇的对角高斯混合密度。
- 空簇和全噪声候选不会被丢弃，均记录诊断并进入排名。
- 报告通过同目录临时文件、`fsync` 和 `os.replace` 原子写入。业务字段在并发度和恢复前后保持一致；实际耗时字段会随机器负载变化。

## 故障注入

注入配置与实验配置独立。每个注入项必须有唯一 `id`。样例使用以下真实路径：

| 故障 | 阶段 | 业务码 |
| --- | --- | --- |
| `non_finite_feature` | `data` / `preprocess` / `evaluation` | `E_INPUT_NON_FINITE` |
| `missing_category_mapping` | `preprocess` | `E_CATEGORY_UNMAPPED` |
| `hyperparameter_out_of_range` | `clustering` | `E_HYPERPARAM_OUT_OF_RANGE` |
| `negative_neighborhood_radius` | `clustering` | `E_DBSCAN_RADIUS_INVALID` |

注入探针复制当前真实阶段输入并调用相同校验/编码/训练参数检查路径，不污染干净基线。命中清单和失败清单均按稳定键排序；相同输入、配置和注入配置重复运行时逐字段一致。

## 快照协议

manifest 记录：

- 全局 `snapshot_version`
- `config_hash`、`injection_hash`、`data_hash`
- `preprocess`、`clustering`、`evaluation` 三阶段产物版本和内容哈希
- 聚类提交间隔生成的检查点版本

每阶段先写入同父目录的 `.tmp-*` 目录，再整体重命名为 `units/<stage>/<version>`；manifest 本身同样原子替换。恢复时版本、配置哈希、注入哈希或数据哈希任一不符均输出 `E_CONFIG_MISMATCH` 并拒绝续跑；阶段内容哈希不符输出 `E_SNAPSHOT_CORRUPT`。

## 退出码与输出

- `0`：干净运行，无业务失败。
- `1`：流水线完成，但存在配置、注入或可复现性失败；失败项已进入报告。
- `2`：致命配置/快照/内部错误，拒绝继续执行。

所有业务错误均以 `E_` 前缀写入 JSON 报告的 `failure_list` 和 stderr。诊断报告包含候选排名、三项指标、逐样本硬分配、GMM 软责任度、空簇/噪声诊断、注入命中、失败清单、配置与注入哈希、快照版本、阶段耗时和延迟预算检查。

## 实际验收结果

详见 `RESULTS.md`。完整样本产生 12 个候选；5 个注入全部命中；快照恢复后的排名、指标表、逐样本分配、注入命中、失败清单和哈希与首跑一致；自动测试 6/6 通过。
