# 插补维护工作台

一个完全本地、确定性的表格机器学习数据准备与缺失值插补维护工作台。它以插补配置、原始 JSONL、分组映射和分批编辑计划为输入，按“统计量汇聚 → 插补值生成 → 线性模型训练/评估”逐批执行，并通过多级内容指纹精确判定复用范围。增量执行、预算切片、并行执行与中断恢复后的结果均与从空白全量重建逐字段一致。

## 快速开始（Windows）

本项目目标环境是 Python 3.11。依赖版本固定在 `requirements.txt`，运行期不使用网络、Docker 或外部服务。

```bat
py -3.11 --version
py -3.11 -m pip install --no-index -r requirements.txt
py -3.11 -m imputation_workbench generate --out samples\demo --seed 20261121 --rows 24
py -3.11 -m imputation_workbench run --workspace samples\demo --state samples\demo\work --fresh
py -3.11 -m imputation_workbench run --workspace samples\demo --state samples\demo\work
py -3.11 -m imputation_workbench verify --workspace samples\demo --state samples\demo\work
py -3.11 -m pytest -q
```

也可以直接执行 `run_demo.bat`。Linux/macOS 下把 `py -3.11` 换成 `python3.11` 或 `python3`。

本机实际环境为 Python `3.11.0rc1`、NumPy `1.26.4`、pytest `8.3.3`、PyYAML `6.0.2`。如果本机没有离线 wheel 缓存，使用已安装的上述固定版本即可；程序本身不会访问网络。

## 命令行

| 命令 | 作用 |
| --- | --- |
| `generate` | 按固定种子生成配置、原始 JSONL、分组映射、编辑计划与异常注入清单。 |
| `run` | 执行或恢复增量批次；`--fresh` 从空白开始，`--max-batches` 可在增量对比时只完成前 N 个编辑批次。 |
| `rebuild` | 从空白输入完整重建一个独立状态目录。 |
| `verify` | 将当前增量快照与临时全量重建做语义逐字段比较，写出 `verification.json`。 |
| `compare` | 比较两个快照状态的逐字段结果指纹。 |
| `report` | 输出最新 JSON 报告。 |

常用调度参数：

- `--max-slices 1`：每个进程最多消耗一个插补预算切片，有剩余时返回 `E_PATCH_BUDGET_PENDING`；下一次调用继续同一队列。
- `--cell-budget N`：覆盖本次运行的每切片单元格预算，不改变持久化配置。
- `--workers N`：覆盖本次线程池并发数，不改变状态配置或结果指纹。
- `--no-injections`：关闭示例异常注入，用于干净基线验证。

`snapshot_interval_batches` 控制批次完成快照的提交节奏。预算待办、停止请求、错误和最终提交不受该间隔延迟，始终立刻落盘。
线程数、单批预算和快照间隔属于执行参数；恢复时可在配置中直接调整，不会要求全量重建，语义配置与数据/映射/计划变化仍会拒绝不兼容恢复。

## 示例输入

`samples/demo` 由固定种子程序生成，不引用外部数据集：

- `imputation.yaml`：逐列声明类型、插补方式、最小有效样本数、兜底常量、分组键、定点小数位宽、量化档位、线程数、预算和快照间隔。
- `raw_data.jsonl`：每行含 `row_id`、原始数值/类别值或 `{"missing": true}`。缺失位置由种子确定性控制。
- `group_map.json`：分组键值到分组名的映射，支持后续在计划中新增映射和分组。
- `edit_plan.json`：四个示例批次，覆盖行追加、单行修正、单列插补方式变更、分组映射新增和新增分组行。
- `injections.json`：非有限原始值、未映射分组值、位宽越界和最小有效样本数越界四类真实输入故障。

类别列必须有显式 `domain`。缺失可写成 `null`、字符串 `__MISSING__` 或 `{"missing": true}`；缺失标记与原始值同时出现会被拒绝。数值列只接受有限数。分类目标必须是数值列且取值仅能为 `0`/`1`。

## 产物

状态目录包含：

- `manifest.json`：全局版本、父版本、配置/数据/分组映射/计划哈希、已完成批次、进行中批次、待办数量，以及各阶段产物版本。
- `generations/vNNNNNNNN/`：不可变完整快照代。`stats.json`、`imputations.json`、`model.json`、`state.json`、`report.json` 均在该代目录中。
- `report.json`：最近一次运行的用户可读 JSON 报告。
- `verification.json`：执行 `verify` 后写出的增量/全量比较结果。

报告包括各列统计量、分组引用行和分组键值、参与样本数、插补单元格计数、逐级复用/重算清单、模型损失轨迹、准确率、对数损失、注入命中、业务码、快照版本与各阶段耗时。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_CONFIG_INVALID` | 配置结构、类型兼容、列存在性、分组键、预算/位宽等非法。 |
| `E_DATA_ROW_INVALID` | 单行 JSON、行标识、缺失标记冲突、非有限数值、类别越域等非法。 |
| `E_GROUP_MAP_INVALID` | 分组映射结构非法，或行的分组键值无映射。 |
| `E_INSUFFICIENT_SAMPLES` | 非缺失有效样本数低于列/分组最小样本阈值，使用兜底常量。 |
| `E_PATCH_BUDGET_PENDING` | 当前预算切片耗尽，仍有稳定顺序的待插补队列。 |
| `E_UPSTREAM_STALE` | 持久化模型依赖的统计/插补指纹与当前内容不一致，旧模型结果被拒绝。 |

配置注入在隔离的影子配置上执行真实验证，产生 `E_CONFIG_INVALID` 诊断，但不会阻断其他合法行和列。行级非有限值与未映射分组会让对应行失败并排除在训练集之外，其余行继续。

## 确定性设计摘要

- 定点统计：所有数值统计量保存为 `10^-decimal_places` 整数单位，再按 `quantization_level` 个整数单位为一档进行整数舍入；记录位宽、舍入模式和舍入方向。
- 均值噪声：仅由 `row_id + column` 派生 SHA-256 随机流并做 Box–Muller 变换，不读取或修改全局随机状态。
- 精确失效：列统计引用非缺失行证据；分组中位数额外记录分组行集合和原始分组键值。未变化的列/分组直接复用持久化记录。
- 预算队列：按 `(column, row_id)` 稳定排序。预算只消耗新计算单元格，缓存命中不消耗预算；待办队列跨进程持久化。
- 并发一致：列任务和行任务均按稳定键提交和收集，线程池并发不改变行/列/失败/归因顺序。
- 模型依赖：模型指纹同时包含全部统计指纹和插补单元格指纹；任何上游变化都会拒绝旧模型并重训，无变化时直接复用。
- 原子快照：新代先在 `generations/.staging-*` 完整写入，再改名为不可变代目录；最后以比较并交换的方式原子替换 `manifest.json`。强制终止只会留下不可见 staging 或已完整提交的代。

更详细的失效矩阵、算法和恢复协议见 `docs/DESIGN.md`。

## 验收与测试

`pytest -q` 覆盖 20 个测试，包括：

- 坏行隔离、业务码和继续加载；
- 重复行号只淘汰重复行，不误伤同标识的有效行；
- 固定种子样本逐字节可复现；
- 单个插补单元格异常被隔离，失败行排除且其他行继续；
- 中位数/众数并列字典序、定点与量化；
- 同配置重复运行逐字段一致；
- 增量执行与全量重建一致；
- 单列策略变更只重算该列，分组映射新增只重算新分组；
- 预算切片续跑、待办顺序和与一次性执行一致；
- 待办中断后在配置中放宽预算可直接续跑，无需全量重建；
- 串行/线程池结果一致；
- 没有新增缺失单元格时，仅统计参数变化也会使模型失效并重训；
- 模型上游过期拒绝；
- 完整快照提交后不可捕获强制终止，再恢复并与全量重建一致；
- 可处理停止请求先提交快照再恢复；
- 快照提交间隔按批次生效，同时不延迟预算待办和最终快照；
- 位宽与量化档位分别真实改变插补值和指纹。

示例目录中的 `samples/demo/work/verification.json` 记录实际验证结果；多次本地验收均为 `status: verified`、`equal: true`。
