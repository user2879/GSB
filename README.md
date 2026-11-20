# Local Scorecard Lab

一个完全本地、确定性的风控评分卡建模实验应用，覆盖：

- 固定种子的合成表格数据生成：数值列、类别列、缺失、尾异常、高基数类别、二值目标。
- 数值列卡方相邻合并 + 最小箱占比约束，类别列按频次/目标率聚合。
- 带零频加性平滑的 WOE/IV 计算，逐箱记录平滑标记和 IV 贡献。
- 数值列单调上升、单调下降或无序约束，按可配置冲突裁决优先级逐轮合并。
- 内容指纹缓存、线程池并发、稳定排序和单列异常隔离。
- 阶段化临时目录写入、原子阶段提交、原子清单与进度记录、强制终止后的精确恢复。
- 交互式编辑后增量运行，并用 JSON 报告输出分箱、WOE、单调修正、评估、缓存和恢复信息。

应用不读取外部数据、不访问网络服务、不使用 Docker 或其他容器环境。

## 环境与安装

要求 Python 3.10 或以上。Windows 本地可直接使用 `python`：

```powershell
python --version
python -m pip install -r requirements.txt
```

依赖只有 `numpy` 和 `PyYAML`。也可以安装为本地命令行工具：

```powershell
python -m pip install -e .
scorecard-lab --version
```

不安装包时，以下命令均可直接运行：

```powershell
python -m scorecard_app --help
```

## 快速开始

工作区已提供 `configs/experiment.yaml` 与 `configs/synthetic.yaml`。执行一次完整实验：

```powershell
python -m scorecard_app run `
  --config configs/experiment.yaml `
  --data configs/synthetic.yaml `
  --state-dir .scorecard-state `
  --report .scorecard-state/last_report.json
```

重复执行同一命令会从已提交阶段和内容缓存恢复，不会重算未失效列。发生强制终止后，用完全相同的输入和状态目录再次执行即可续跑：

```powershell
python -m scorecard_app run --config configs/experiment.yaml --data configs/synthetic.yaml
```

有意修改 YAML 配置或数据描述后再运行，需要显式声明这是新版本 revision：

```powershell
python -m scorecard_app run --config configs/experiment.yaml --data configs/synthetic.yaml --revision
```

未加 `--revision` 时，已提交清单中的配置/数据指纹不一致会返回 `E_SNAPSHOT_STALE` 并拒绝恢复旧状态。

## 交互编辑

```powershell
python -m scorecard_app shell --config configs/experiment.yaml --data configs/synthetic.yaml
scorecard> direction age descending
scorecard> max-bins 5
scorecard> min-ratio 0.08 income
scorecard> priority max_gap,left
scorecard> replace-column income 3
scorecard> show age
scorecard> save configs/edited-experiment.yaml configs/edited-data.yaml
scorecard> quit
```

每次编辑会立即执行一次 revision 运行，只重算失效列及其下游评估。命令也可以从文本文件批量输入：

```powershell
python -m scorecard_app shell --config configs/experiment.yaml --data configs/synthetic.yaml --commands commands.txt
```

支持的编辑命令：

- `max-bins <n> [column]`：全局或单列分箱上限。
- `min-ratio <r> [column]`：全局或单列最小箱占比。
- `direction <column> <ascending|descending|none>`：单列单调方向。
- `priority <rule[,rule...]>`：冲突裁决优先级。
- `max-rounds <n>`：单调求解最大合并轮数。
- `replace-column <column> <seed_offset>`：只替换该列的确定性生成种子偏移，目标列和其他列不变。
- `run`、`show [column]`、`report [path]`、`save <experiment.yaml> <data.yaml>`、`help`、`quit`。

## 配置结构

实验配置包含：

- `features`：参与建模的列。可以省略，省略时使用数据描述中的全部列。
- `runtime.max_workers`：线程池最大并发数，必须为正整数。
- `binning.max_bins`、`binning.min_bin_ratio`：全局分箱数上限与最小箱占比。
- `binning.prebins`：数值列升序预分箱的目标数量。
- `binning.smoothing`：零频加性平滑系数，必须为正数。
- `monotonic.max_rounds`：单调冲突最大合并轮数。
- `monotonic.conflict_priority`：支持 `left`、`right`、`max_gap`、`min_count`、`max_count`，按列表顺序裁决。
- `evaluation.folds`、`evaluation.seed`：固定折划分参数。
- `columns.<name>`：可选的单列类型、`max_bins`、`min_bin_ratio`、`monotonic_direction` 覆盖。
- `snapshot.expected_global_version`：可选；非空时必须与当前已提交全局版本一致，否则拒绝恢复。

数据描述包含行数、固定种子、目标二值枚举及每列类型、缺失率、分布参数、基数与 `seed_offset`。

## 算法与业务规则

### 数值分箱

1. 按非缺失值升序构造确定性预分箱。
2. 对相邻箱计算 2x2 Pearson 卡方统计量。
3. 持续合并最低卡方相邻对，直到非缺失箱数满足上限且非缺失箱占比满足最小占比。
4. 卡方相同时按左边界升序裁决，必要时再用原始相邻位置稳定排序。
5. 缺失值始终作为独立箱追加在末尾，并写入 `is_missing: true`。

`max_bins >= 2` 时，为缺失箱保留一个总槽位；因此非缺失箱上限是 `max_bins - 1`。`max_bins = 1` 且存在缺失时，保留一个数值箱和一个独立缺失箱，这是“缺失单独成箱”约束下的唯一可行表达。

### 类别分箱

初始每个类别为一个组。每次选出频次最小的组，将其与目标率最接近的相邻组按目标率排序后合并，直到组数满足上限且组占比满足最小占比。最终输出组列表、组内排序后的类别和完整的 `category_to_group` 映射；缺失类别单独成组。

### WOE 与 IV

对每箱计算：

```text
bad_share = adjusted_bad / adjusted_total_bad
good_share = adjusted_good / adjusted_total_good
WOE = ln(bad_share / good_share)
IV_contribution = (bad_share - good_share) * WOE
```

当某箱坏样本数或好样本数为零时，对该零频计数加 `smoothing`，重新计算调整后的全局总数，并在箱产物中写入 `smoothed: true`。浮点结果统一保留 12 位小数，保证缓存重载后逐字段一致。

### 单调约束

仅检查非缺失数值箱的 WOE 序列。上升约束要求 WOE 非递减，下降约束要求非递增。每轮找出全部冲突相邻对，按 `conflict_priority` 选择一对合并，然后重算全部箱 WOE。修正记录包括轮次、合并前后边界、合并前后 WOE、计数、冲突间隙和命中规则。

达到 `max_rounds` 仍有冲突时，保留当前箱与 WOE 状态，写入 `E_MONOTONIC_UNRESOLVED` 失败项，并将该列排除出评估汇总。

## 增量缓存与失效

缓存键全部由规范 JSON 的 SHA-256 派生：

- 列内容键：列名、类型、列值、目标值、目标列名、正类枚举。
- 分箱参数键：`max_bins`、`min_bin_ratio`、`prebins`。
- 方向/优先级键：单调方向、冲突优先级、最大轮数。
- `binning` 键：列内容键 + 分箱参数键。
- `woe` 键：`binning` 键 + 平滑系数。
- `monotonic` 键：`woe` 键 + 方向/优先级键。
- `evaluation` 键：全部最终列键 + 数据指纹 + 折数、折种子、目标枚举。

因此：

- 只改某列单调方向：该列的 `binning`/基础 `woe` 命中，只重算 `monotonic` 和评估。
- 只改某列最小箱占比：只有该列的分箱、WOE、单调和评估失效。
- 改裁决优先级：所有适用数值列的单调阶段和评估失效，类别列与分箱/WOE 复用。
- 单列数据替换：只有该列内容键及下游失效；目标列和其他列保持逐字段相同。

## 快照、恢复与并发

状态目录结构：

```text
.scorecard-state/
  manifest.json                 # 唯一原子提交点
  progress.json                 # 中断时的已完成列进度（若存在）
  cache/<stage>/<key>.json      # 内容寻址的不可变列产物
  stages/<stage>/vNNNNNNNN/     # 原子发布的阶段目录
  tmp/                          # 未完成临时写入，恢复时忽略
```

每个阶段先写 `tmp/`，完成并 `fsync` 后整体 `rename` 到 `stages/...`，最后原子替换 `manifest.json`。清单包含格式/算法版本、全局版本号、配置/数据指纹、已完成阶段、每阶段产物版本、已完成列、阶段键和失败清单。

恢复规则：

- 格式版本、算法版本、可选期望全局版本、配置指纹或数据指纹不一致：`E_SNAPSHOT_STALE`，拒绝恢复且不修改清单。
- 身份一致：按清单恢复已完成阶段；未完成阶段从原子缓存和进度记录继续。
- 未引用的临时目录、未提交阶段目录和校验失败缓存不会被当作有效状态。
- `SIGKILL` 不保证退出报告；重启只依赖完整阶段、原子列缓存和进度文件。
- 可处理的 `SIGINT`/`SIGTERM` 会让线程池停止接收新列、等待运行中的列完成原子缓存提交，写入进度和 JSON 报告，进程以整数退出码 `0` 退出。

分箱、WOE 和单调阶段使用 `runtime.max_workers` 的线程池。每个任务携带稳定的 `stage:column` 键；最终结果按列名字典序、箱索引和轮次输出，因此线程调度和并发度不会改变产物顺序或数值。单列抛异常时只记录该列的列名、阶段、业务码和异常类型，其余列继续并照常形成评估。

## 报告与业务码

每次运行原子写出 JSON 报告，主要字段包括：

- `input.config_fingerprint`、`input.data_fingerprint`。
- `snapshot.version`、`snapshot.recovery_point_version`、各阶段产物版本。
- `recovery.recovered`、`recovery.resume_version`、恢复阶段和中断阶段。
- 逐列 `bin_count`、`iv`、`monotonic_correction_count`、完整箱表、类别映射和逐阶段缓存状态。
- 固定折 `evaluation.overall` 与 `evaluation.per_fold`，含 AUC、KS、LogLoss、Brier、Accuracy。
- `cache.hits`、`cache.misses`、`invalidated_artifacts`、`invalidated_columns`。
- 稳定排序的 `failures` 清单。

主要业务码：

- `E_CONFIG_INVALID`：未知列、类型不符、目标缺失/非二值、分箱上限为零、并发非正及其他配置错误。
- `E_SNAPSHOT_STALE`：快照身份或版本不一致，拒绝恢复。
- `E_COLUMN_BINNING_FAILED`、`E_COLUMN_WOE_FAILED`、`E_MONOTONIC_FAILED`：单列异常隔离。
- `E_MONOTONIC_UNRESOLVED`：达到最大轮数仍未满足单调约束。
- `E_EVALUATION_FAILED`：评估阶段异常。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：配置错误、二值目标、卡方并列裁决、最小箱占比、类别映射、零频平滑、单调修正与未决状态、增量/全量逐字段一致、单列数据替换、并发度一致性、异常隔离、失败排序、SIGTERM 退出码 0、SIGKILL 后恢复、旧快照拒绝和显式 revision。实际执行结果见 `docs/VALIDATION.md`，设计与不变量见 `docs/DESIGN.md`。

测试/演示专用环境变量：

- `SCORECARD_FAILPOINT=after_column_cache:binning:age` 等：在指定原子提交点硬退出，用于验证恢复。
- `SCORECARD_TEST_TASK_DELAY_MS=200`：给列任务增加固定延迟，用于发送进程信号。

这两个变量不参与配置指纹或业务产物生成，仅用于本地验收。
