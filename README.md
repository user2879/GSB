# 本地确定性异常检测诊断工作台

这是一个只依赖 NumPy 的本地交互式诊断工作台，用固定种子生成合成表格数据，支持三种手工实现的异常检测算法、内容指纹增量复用、线程池并发、交互延迟预算、原子阶段快照、故障注入和 JSON 诊断报告。

核心保证：任意交互序列结束后，增量执行结果与从空白缓存按最终配置一次性串行全量执行的结果，在逐样本分数、异常标记、贡献分解和失败清单上逐字段一致。

## 1. 环境与运行

要求 Python 3.10+ 和 NumPy 1.26.4。YAML 由内置的无依赖解析器读取，不需要 PyYAML，也不使用任何第三方机器学习库。

```bash
python -m pip install -r requirements.txt
python -m unittest -v tests.test_workbench
python examples/validation_demo.py
```

Linux / WSL / macOS 启动交互工作台：

```bash
python -m adw --config config.yaml
```

Windows 可直接双击或在 `cmd` 中运行：

```bat
run_workbench.bat
```

带独立注入配置启动：

```bash
python -m adw --config config.yaml --injection injection.yaml
```

运行固定的交互序列：

```bash
python -m adw --config config.yaml --script examples/interaction_sequence.txt
```

工作台输出 JSON Lines，便于脚本逐条解析。交互提示只写到 stderr；stdout 始终是 JSON。

## 2. 目录与输入文件

```text
config.yaml                   主配置
data_description.yaml         固定种子数据描述
injection.yaml                独立故障注入配置
adw/                          工作台实现
  config.py                   加载、归一化、严格校验
  data.py                     确定性合成数据
  preprocess.py               缺失填充、类别编码、缩放
  algorithms.py               三种 NumPy 手工算法
  pipeline.py                 三级缓存、并发流水线、稳定排序
  engine.py                   代际后台调度、预算响应、报告
  storage.py                  内容寻址缓存、原子快照
  injection.py                注入隔离与业务码映射
  commands.py                 交互命令解析
  cli.py                      CLI
examples/
  interaction_sequence.txt    算法切换/调参/预处理/阈值修改示例
  validation_demo.py          一键验收演示
tests/test_workbench.py       12 项自动化测试
```

### 主配置关键字段

- `data.feature_columns`：有序特征列，必须与数据描述中的列完全对应。
- `preprocessing`：`mode`、数值/类别缺失填充方式、类别列和候选值。
- `algorithms`：三个固定算法名、`enabled`、`params`、`candidates`、`weight`。
- `threshold`：最终合成异常分数阈值，范围 `[0, 1]`，判定采用 `>=`。
- `interaction.latency_budget_ms`：每条配置命令的响应预算。
- `interaction.concurrency`：线程池最大并发数。
- `interaction.snapshot_interval`：每完成多少个算法分数产物提交一次评分阶段快照。
- `storage`：缓存、快照、报告路径；相对路径以配置文件所在目录为基准。

### 数据描述

`dataset.columns` 中每列声明 `numeric` 或 `categorical`。数值列支持 `normal` 和 `redundant` 两种分布；类别列声明稳定类别列表。`dataset.target` 必须包含 `field` 与 `anomaly_ratio`，缺失会被记录为 `E_CONFIG_INVALID`。

生成器使用 SplitMix64 和独立列种子，不读取 `numpy.random` 全局状态。缺失值、异常行、类别和冗余噪声均来自固定种子。当前样例为 1400 行、98 个正类异常行、1.8% 缺失单元格；目标标签只用于数据描述和诊断，不参与无监督打分。

### 注入配置

`injection.yaml` 与主配置完全分离。支持：

- `non_finite_feature`：在指定样本的预处理阶段注入 `inf/nan` 并做有限性检查。
- `hyperparameter_out_of_bounds`：对指定算法/样本施加越界超参数。
- `threshold_boundary`：把指定样本的判定边界设为该样本的精确异常分数。
- `data_size_limit`：数据规模超过注入上限时记录 `E_DATA_LIMIT`，仍继续诊断其余故障。

每个注入项必须有唯一 `id`，命中状态、业务码和消息写入报告。单样本异常只影响该样本；其他样本继续执行。

## 3. 交互命令

```text
algo enable NAME
algo disable NAME
param set ALGORITHM KEY VALUE
preprocess set mode VALUE
preprocess set numeric_fill median|mean|zero
preprocess set categorical_fill mode|constant
threshold set VALUE
budget set MILLISECONDS
concurrency set N
snapshot set N
run
status
result
settle
report
recover
compare full
quit
```

- `param set` 和 `preprocess set` 只接受配置中列出的候选值；非法值立即返回 `E_CONFIG_INVALID`，旧配置保留。
- 普通配置命令只返回摘要、受影响算法和代际信息，避免在预算路径上序列化整份结果。
- `settle` 明确阻塞到最新代际完成；它用于调试和确定性验收，不属于预算受限命令。
- `result` 返回最近一次已完成结果，`settle` 等待并返回最新完整结果。
- `compare full` 等待当前增量运行完成后，从空缓存执行串行全量运行，并输出逐字段差异。
- `quit` / EOF 会先刷新最新代际，再写最终报告。

## 4. 算法与确定性

三个算法均使用 NumPy 数组，但所有随机性、归约顺序和排序规则显式固定。

### 稳健马氏距离 `robust_mahalanobis`

1. 每列取中位数 `median(x)` 与稳健尺度：`1.4826 * median(|x - median(x)|)`；退化时依次回退到 IQR 尺度和 1.0。
2. 标准化后按固定列顺序计算协方差。
3. 对角线加配置的固定正则系数。
4. 使用显式 Cholesky、前向代入和回代求二次型，不使用平台相关的随机分解。
5. 分数为标准化向量的马氏距离平方。

### 直方图邻域密度 `histogram_density`

1. 每个维度按全局最小/最大值等宽分箱，固定为 `bins` 个箱。
2. 对每个样本，统计所有维度箱号差都小于等于 `neighborhood` 的样本数。
3. 整数计数通过分块布尔掩码计算，避免浮点归约误差；排序密度分层按固定顺序完成。
4. 原始分数为 `log((N + 0.5) / (density + 0.5))`，密度越低分数越高。

### 随机划分深度 `random_partition_depth`

1. 种子由算法内容键（数据指纹、算法参数、预处理）派生，使用 SplitMix64 而非 `numpy.random`。
2. 每棵树使用固定 Fisher-Yates 部分洗牌抽取 `sample_size` 个样本。
3. 随机选择非恒定特征和切分点，递归构造至 `max_depth` 或叶节点。
4. 按经典路径长度公式计算 `2^(-average_path / c(n))`。
5. 每棵树独立种子，树内和树间顺序固定。

### 合成与排序

- 每个算法原始分数先按配置权重标准化到当前数据范围内的 `[0, 1]`；全相同范围固定为 0。
- 最终异常分数是贡献的稳定加权平均，使用 `math.fsum` 后统一量化到 12 位小数。
- `>= threshold` 判定异常。
- 算法原始排序、最终排序均为分数降序，同分按样本 ID 字典序升序。
- 贡献分解顺序固定为 `histogram_density`、`random_partition_depth`、`robust_mahalanobis`，与输入配置顺序和线程调度无关。
- 结果同时提供按样本 ID 排序的逐样本记录、按分数排序的 `ranking`、`anomalies`、`algorithm_rankings` 和扁平 `score_table`。

## 5. 内容指纹与精确增量失效

缓存键是内容寻址的 SHA-256：

```text
preprocess_key = H(data_fingerprint, preprocess_fingerprint, prep-version)
algorithm_key  = H(data_fingerprint, preprocess_key, algorithm_fingerprint, algorithm-version)
decision_key   = H(data_fingerprint, preprocess_key, all algorithm_keys + weights, threshold, decision-version)
```

指纹覆盖数据内容、特征顺序、预处理参数、算法参数与版本、权重和量化后的阈值。

| 修改 | 预处理 | 算法分数 | 最终判定 |
|---|---|---|---|
| 只改阈值 | 复用 | 全部复用 | 失效重算 |
| 改某算法参数 | 复用 | 仅该算法失效 | 失效重算 |
| 切换启用算法 | 复用 | 新启用算法计算，其余复用 | 失效重算 |
| 改预处理 | 失效重算 | 全部失效重算 | 失效重算 |
| 数据变化 | 失效重算 | 全部失效重算 | 失效重算 |

未命中算法不会进入线程池任务；测试直接断言重算算法清单和缓存命中/失效计数。旧内容键不会被原地修改，而是保留为可复用历史版本。

## 6. 延迟预算语义

`WorkbenchEngine` 使用单个后台调度线程和代际号：

1. 配置命令先在主线程完成校验，再更新目标代际并唤醒后台。
2. 命令最多等待预算的 80%，预留剩余时间给 GIL 回收、摘要构造和输出。
3. 若最新代际未完成，立即返回 `E_BUDGET_PENDING`，并带回最近一次已完成摘要、当前阶段进度、受影响算法、请求/完成代际。
4. 每条响应记录 `elapsed_ms`、`budget_ms`、`over_budget`、`pending`。
5. 后到命令会形成新代际；旧代际完成后，后台继续推进到最新代际。最终 `settle`/`quit` 会刷新。

摘要始终包含 `completed_samples`、`anomaly_count`、`affected_algorithms`。预算不足时不会阻塞等待最新结果。

## 7. 并发与稳定键

评分阶段把每个算法按样本分块，任务稳定键为 `(算法名, 首样本 ID, 起始下标)`；判定阶段同样按样本 ID 分块。线程池最大并发度取自配置。

任务结果先按稳定键归并，再按样本 ID 重建整张表。线程完成顺序不影响：

- 逐样本分数；
- 异常样本顺序；
- 分数表顺序；
- 算法贡献分解顺序；
- 失败清单顺序。

测试覆盖 `concurrency=1`、`concurrency=4` 与强制串行全量执行三者完全相等。

## 8. 阶段快照、退出与恢复

阶段为 `preprocess`、`score`、`decision`。每个快照包含：

- `stage`；
- `config_hash`；
- `data_fingerprint`；
- `completed_samples`；
- `artifact_version`；
- payload 校验和、提交序号和时间。

写入流程为：同目录临时文件 → `flush` → `fsync` → `os.replace` 原子替换。评分阶段按 `snapshot_interval` 提交已完成算法集合。

- 收到 `SIGINT` / `SIGTERM` / Windows `SIGBREAK` 时只设置停止标志；流水线在当前阶段完成后写快照，CLI 以整数 `0` 退出。
- 强制终止后，重启只扫描完整 JSON 快照；`.tmp` 文件、校验和错误、结构缺失均忽略或列入 rejected，并输出 `valid_snapshots`、`ignored_temp_files` 和恢复点信息。
- 完整快照可把预处理器、算法分数和判定产物重新灌入内容缓存。

## 9. 异常码与失败清单

配置错误统一使用：

```text
E_CONFIG_INVALID
```

覆盖未知列、列类型不符、目标结构缺失、异常比例越界、并发数非正、延迟预算非正、未知算法、候选值外参数等。错误按阶段、样本 ID、算法、业务码、注入 ID、异常类型、消息的稳定顺序排序，不会静默跳过。

运行时业务码：

```text
E_BUDGET_PENDING
E_SAMPLE_STAGE
E_NON_FINITE_FEATURE
E_PARAM_INVALID
E_THRESHOLD_BOUNDARY
E_DATA_LIMIT
E_SNAPSHOT_INVALID
```

单样本阶段异常记录 `sample_id`、`stage`、`exception_type`、算法和注入 ID，随后继续其他样本。

## 10. 诊断报告

每次后台完整运行写 `storage.report_path`。带注入配置运行时另写 `injection_report_path`。报告包含：

- 数据指纹、配置哈希、代际和恢复点；
- 三个算法的原始/标准化分数及最终异常分数分布；
- 缓存命中、失效、损坏、写入计数及按阶段细分；
- 重用与重算算法清单；
- 注入项命中状态与业务码映射；
- 失败清单；
- 每条交互命令的响应耗时、预算、是否 pending/超预算；
- 最近一次 `compare full` 一致性结果。

## 11. 实际验证结果

本机环境：Python 3.11.0rc1、NumPy 1.26.4、WSL2 Linux。全部 12 项测试通过，耗时约 16.7 秒：

```text
Ran 12 tests in 16.7s
OK
```

`python examples/validation_demo.py` 的实际结果摘要：

- 合成数据：1400 样本，98 个正类异常行；三种算法均输出 1400 个有限分数。
- 串行全量运行：约 543 ms，默认配置阈值下标记 2 个异常；诊断目标不是调优召回率，阈值可交互修改。
- 同配置第二次运行：约 14 ms，缓存 `5 hits / 0 miss`，算法重算清单为空。
- 只修改直方图 `bins`：只重算 `histogram_density`，另两个算法复用，最终结果与全量运行逐字段相等。
- 只修改阈值：三个算法全部复用，仅判定失效，最终结果与全量运行逐字段相等。
- 算法切换 + 参数 + 预处理 + 阈值组合交互序列：最终结果与全量运行逐字段相等。
- 并发 1、并发 4、串行全量：三者结果完全相等。
- 35 ms 预算下，未完成的最新代际在约 30.01 ms 返回 `E_BUDGET_PENDING`，`over_budget=false`。
- 注入：4/4 命中；命中码为 `E_DATA_LIMIT`、`E_NON_FINITE_FEATURE`、`E_PARAM_INVALID`、`E_THRESHOLD_BOUNDARY`；故障样本为 `S000007`、`S000011`、`S000019`，其余样本继续完成。
- 快照：三个阶段均生成有效快照，重启恢复 5 个缓存产物；人为留下的 `.tmp` 文件被明确忽略。
- 信号：运行中或停在交互输入提示时收到 `SIGINT`/`SIGTERM`，均完成当前阶段快照后以退出码 0 结束。

示例命令序列中 `compare full` 输出：

```json
{"status":"ok","equal":true,"differences":[]}
```

## 12. Windows 说明

实现只使用 `pathlib`、`os.replace`、线程、标准信号和本地文件；没有 POSIX-only 依赖、容器、网络或账号依赖。`run_workbench.bat` 和 `run_tests.bat` 提供 Windows 本地入口。目录 `fsync` 在 Windows 若不支持会安全跳过，文件级 `fsync` 与原子替换仍保留。

## 13. 设计边界

- 目标是确定性诊断和一致性验证，不是追求最高检测精度；阈值和候选参数应围绕本地诊断目的调整。
- 输出分数统一量化到 12 位小数，排序与 PRNG 完全固定；同一平台和 NumPy 版本逐字节稳定。跨平台数学库极端末位差异会被 12 位量化吸收，但未在 Windows 真机上做逐位硬件矩阵测试。
- 内容缓存保留历史键而不主动删除，便于回退配置；磁盘增长需由使用者按需清理。
- `settle`、`compare full`、`report` 是显式诊断/收尾命令，允许阻塞；普通配置命令遵守预算语义。
