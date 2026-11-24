# Deterministic Clustering & Density Workbench

这是一个不依赖容器、网络数据或特定硬件的本地工作台，用于对合成表格数据执行可审计的聚类、密度估计、交互式调参和增量诊断。实现只依赖 Python 3.11/3.12、`numpy==1.26.4` 与 `PyYAML==6.0.1`。

核心保证：**任意命令序列结束后的最终状态，与按该最终状态从空白缓存一次性全量重建的结果逐字段一致**。比较字段包括逐样本簇归属、逐候选指标、候选排名、预算账本和失败清单顺序。

## Quick Start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 pyyaml==6.0.1 pytest==8.2.0
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m cluster_workbench `
  --config samples\config.yaml `
  --data samples\data.yaml `
  --commands samples\commands.yaml `
  --injections samples\injections.yaml `
  --workspace artifacts\run `
  --report artifacts\diagnostic_report.json `
  --responses artifacts\responses.json
```

Linux/macOS 使用相同参数即可：

```bash
python -m pytest
python -m cluster_workbench --config samples/config.yaml --data samples/data.yaml \
  --commands samples/commands.yaml --injections samples/injections.yaml \
  --workspace artifacts/run --report artifacts/diagnostic_report.json \
  --responses artifacts/responses.json
```

命令成功完成且增量/全量等价检查通过时进程返回 0。诊断报告中的 `equivalence_check.equal` 为 `true` 表示最终一致性验收通过；注入故障会进入 `failures`，但不会让整批中断或使进程误报失败。

## Workspace Layout

```text
cluster_workbench/
  config.py       配置规范化、业务码校验、候选校验
  data.py         固定种子合成数据、列指纹、类别编码、缺失表示
  preprocess.py   训练分区统计、标准化/分位数变换、内容指纹
  algorithms.py   确定性 KMeans、DBSCAN、Gaussian Mixture
  metrics.py      轮廓系数、CH、密度对数似然、稳定排名
  scheduler.py    预算预留、单候选配额、全局待办与账本
  cache.py        内容寻址缓存（JSON + NPY，原子写）
  pipeline.py     四阶段流水线、线程池、精确复用、诊断报告
  snapshot.py     版本化快照、临时目录写盘、CURRENT 指针原子替换
  workbench.py    后台流水线、交互超时语义、分区查看、恢复
  cli.py          端到端命令与空白缓存重建校验
samples/          本地配置、数据描述、命令流与注入清单
tests/            18 项确定性/预算/缓存/交互/快照回归测试
artifacts/        最终诊断报告、响应日志和可恢复快照
```

## Configuration Model

`samples/config.yaml` 声明：

- 预处理方法：`standardize` 或 `quantile`，训练比例和缺失填充策略；
- 方法默认值：KMeans/DBSCAN/GMM 的迭代上限、收敛阈值、协方差正则；
- 评估指标：轮廓系数、Calinski-Harabasz、密度对数似然；
- 调度：全局迭代预算、单候选迭代配额、线程池并发数、交互延迟预算、快照命令间隔；
- 初始候选及可选注入。

`samples/data.yaml` 使用 `size_levels` 和 `size_level` 控制样本量档位。数值列支持 `normal`、`uniform`、`integer`、`mixture` 与基于前序列的冗余列；类别列支持显式类别、权重、高基数和缺失率。每列独立派生固定 RNG，避免依赖 Python 随机哈希或全局随机状态。

## Determinism Rules

- 所有指纹使用排序后的规范 JSON、SHA-256 和 NumPy 连续数组字节哈希。
- 预处理统计只使用候选训练分区；累加按训练索引升序执行，并随产物记录 `finite_accumulation_order`。
- 缺失值使用训练分区观测值的中位数填充。类别按声明顺序编码；未知类别保留为缺失并产生告警。
- 分位数变换显式记录并列规则 `average_rank`（也可配置 `lower_rank`/`upper_rank`），相同输入必得相同输出。
- KMeans 使用字典序最小样本作首质心，再以固定最远点规则初始化；距离、归属和中止平局均按升序索引确定。
- DBSCAN 使用固定距离矩阵、升序邻居、升序外层遍历与 FIFO 扩展。
- GMM 使用相同的最远点均值初始化、固定 EM 顺序、固定收敛判据，并对协方差奇点应用配置正则；不抛异常。
- 排名先按可比性与指标值降序，完全并列时按方法名、规范超参 JSON、候选 ID 决断。

## Preprocessing & Cache Identity

每个候选引用自己的预处理产物。预处理缓存键由以下内容派生：

```text
selected column description fingerprints
+ selected column content fingerprints
+ preprocessing parameters
+ deterministic training indices
+ implementation schema
```

产物内容指纹还包含实际矩阵字节、统计量、缺失填充和分位数元数据。聚类键包含预处理内容指纹、方法、超参数、运行参数与迭代分配；评估键包含聚类内容指纹、预处理内容指纹与评估配置；诊断汇总键包含所有候选评估键、排名、预算和失败/告警清单。

因此：

- 修改某候选超参数：预处理复用，该候选聚类/评估重算；
- 修改预处理方式：所有候选预处理及其下游失效；
- 替换单列：仅引用该列的候选及其下游失效；
- 修改评估配置：聚类复用，仅评估重算；
- 未失效候选重算点数为 0，并出现在 `recompute_accounting.candidates_fully_reused`。

## Budget Semantics

调度器先按候选稳定键（候选 ID）升序预分配预算，再在线程池中执行。账本记录每个候选的 `requested`、`reserved`、`consumed`、`remaining_before_reservation`、`remaining_after_reservation` 和最终 `outcome`。

- 单候选请求超过 `per_candidate_iteration_quota` 时以配额执行；未收敛则中止并记 `E_CAND_BUDGET_EXHAUSTED`；
- 全局预算不足以完整预留时，候选进入待办队列并记 `E_GLOBAL_BUDGET_PENDING`；
- 线程池最大并发数严格取 `scheduler.concurrency`；
- 缓存命中的候选仍保留其逻辑迭代消费，用于保证增量与全量预算账本一致，但不会执行算法。

## Interaction Semantics

每条命令在后台单流水线中执行，候选阶段使用配置并发度的线程池。交互调用最多等待配置的延迟预算；预算内完成则返回当前结果，否则立即返回最近一次已完成摘要，并附带：

- `E_RESULT_STALE` 告警、`stale: true`；
- `result_command_seq`：旧结果对应的命令序号；
- 当前任务进度、受影响候选、已完成候选数及当前最优候选。

`view` 命令支持 `candidates`、`labels`、`data`、`failures` 分区，并用 `offset/limit` 切片。

## Snapshots & Recovery

快照先完整写入临时目录并校验文件哈希，然后重命名版本目录，最后原子替换 `CURRENT` 指针。默认每 `snapshot_commit_interval` 条命令请求一次快照，`close()` 保证最终状态快照完成。恢复时会验证指针、清单版本、文件哈希；不兼容版本返回 `E_SNAPSHOT_VERSION_MISMATCH`，损坏指针会尝试回退到最近的完整版本。

## Fault Injection

`samples/injections.yaml` 演示四类隔离：

- `nonfinite_feature`：指定列/行注入 `inf`，仅引用该列的候选预处理失败；
- `invalid_hyperparameter`：把目标候选簇数改为 0，候选单独进入失败清单；
- `missing_column_description`：移除列描述，引用该列的候选报 `E_COLUMN_DESCRIPTION_MISSING`；
- `slow_candidate`：仅用于验证交互超时旧摘要语义。

## Verification

```bash
python -m pytest
```

结果：`18 passed`。测试覆盖固定种子数据、训练分区缺失填充、量化并列规则、三种算法重复一致性、协方差奇点、空簇/全噪声、排名平局、预算配额、全局待办、并发上限、单列失效、超时摘要、增量/全量等价和快照版本检测。

最终端到端验收由 CLI 自动完成，并在诊断报告写入：

```json
{
  "equivalence_check": {
    "checked": true,
    "equal": true,
    "incremental_fingerprint": "...",
    "full_rebuild_fingerprint": "..."
  }
}
```

实际验收记录见 `docs/ACCEPTANCE.md`，设计细节见 `docs/DESIGN.md`。
