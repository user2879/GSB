# ML 训练数据管线诊断工作台

面向本地 Windows、无需网络/GPU/Docker 的确定性训练数据管线工作台。项目从 JSON 声明编译执行计划，在整数定点域中逐批执行，把阶段观测、分片、规格指纹、版本与进度游标写入 SQLite，并支持规格演进、强制终止恢复与逐项重放校验。桌面端使用 Python 标准库 Tkinter。

预置演示数据为 2 个数据源、6 列、137 行，整数随机种子 `20261122` 确定性生成；默认按 32 行打包为 5 个分片（最后一个余数片 9 行）。

## 快速开始（Windows）

要求：64 位 CPython 3.10+，推荐使用 python.org 标准发行版（自带 Tcl/Tk）。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 检查 Python / numpy / sqlite3 / Tkinter，并实际打开一次 Tk 窗口
python scripts/check_windows_env.py --require-tk --open-window

# 三条场景一键演示
python -m mlpipe_workbench demo --out-dir demo_output

# 打开诊断界面
python -m mlpipe_workbench ui --db workbench.db --data data/sample.json --pipeline data/pipeline.json
```

也可双击 `check_env.bat`、`run_demo.bat`、`run_workbench.bat`。

> 当前交付环境：Linux/WSL、Python 3.11.0rc1、numpy 1.26.4、SQLite 3.37.2；该容器未安装 `tkinter`，因此这里验证了核心、CLI、SQLite、演示和测试，UI 启动按预期返回 `E_ENV`。Windows 端的 Tk 启动需用上面的环境检查脚本人工确认。当前验证记录见 `docs/VALIDATION.md`。

## 目录

```text
mlpipe_workbench/       核心库、CLI、Tkinter UI、演示脚本
data/sample.json        确定性样本声明（2 源、137 行、空值位图）
data/pipeline.json      默认阶段链与分片声明
tests/                  33 项 unittest 验收
docs/                   架构、规格、验证记录
demo_output/            最近一次演示总结与结构化输出
scripts/                样本生成与环境检查脚本
```

## 管线声明

`data/pipeline.json` 的核心结构：

```json
{
  "format": "mlpipe.pipeline.v1",
  "fixed_scale": 3,
  "sharding": {"target_rows": 32, "remainder_policy": "keep_partial"},
  "stages": [
    {"id": "canonicalize", "type": "canonicalize", "version": 1, "upstream": null, "params": {"saturate": 1}},
    {"id": "fill_missing", "type": "fill_missing", "version": 1, "upstream": "canonicalize", "params": {"fill_fixed": 0}},
    {"id": "normalize_amount", "type": "normalize", "version": 1, "upstream": "fill_missing", "params": {"column": 1}},
    {"id": "clip_amount", "type": "clip_outliers", "version": 1, "upstream": "normalize_amount", "params": {"column": 1, "z_limit_permille": 2200, "max_outlier_permille": 100}},
    {"id": "pack", "type": "shard_pack", "version": 1, "upstream": "clip_amount", "params": {"remainder_policy": 1}}
  ]
}
```

强校验包括：格式、阶段 ID 唯一、上游存在、线性链且无环、类型顺序、参数键与整数边界、列索引、分片策略一致、数据源/列/行数、空值位图长度、列类型与值范围。任一失败都在分配规格版本或写入运行之前整体拒绝，报告 `E_SPEC` 与首个违规位置。

阶段顺序固定为 `canonicalize -> fill_missing -> normalize -> clip_outliers -> shard_pack`。允许多个同类 `normalize` / `clip_outliers` 阶段；必须各有至少一个填充、归一化和截断阶段，且恰好一个规范化入口和一个分片出口。

数据源按声明顺序拼接为全局行序列。某个数据源之外的列在全局表中按空值处理；`fill_missing` 会把它们一并填充为 `fill_fixed`。阶段参数中的 `column` 是拼接后的全局列下标。

## 确定性契约

- 规范 JSON：对象键排序、紧凑分隔符、UTF-8；数据源/列/分片顺序保留，阶段按拓扑链顺序编号。
- 指纹规则 `WBP-FP-1`：
  - `dataset_digest = SHA256("MLPIPE-DATASET1\\0" || canonical_dataset)`
  - `pipeline_digest = SHA256("MLPIPE-PIPELINE1\\0" || canonical_pipeline)`
  - 组合时对 `dataset`、`pipeline` 两个字段采用 4 字节名字长度 + 名字 + 8 字节值长度 + 值的显式帧格式。
- 指纹包含：样本内容摘要、定点缩放、分片参数、每个阶段的 ID/类型/版本/上游/规范整数参数。阶段 JSON 数组顺序、参数键顺序不改变指纹。
- 新指纹第一次进入账本时得到 `max(spec_version) + 1`；同一指纹复用原规格版本。每个阶段批次与分片记录均冗余写入规格版本和指纹。
- 定点：源整数乘 `10**fixed_scale`；源十进制数乘缩放后向零截断；中间值均为有符号 64 位整数，越界按阶段 `saturate=1` 饱和到 `INT64_MIN/MAX`，`saturate=0` 时报 `E_PARAM`。
- 归一化使用非负偏移的整数 floor division；异常值统计使用整数均值、整数方差和 `isqrt`，不使用浮点。
- 归一化会话参数 `--normalize-scale` 取整数 `0..6`，决定归一化输出的 `10**n` 量化尺度；它与管线定点 `fixed_scale` 可不同。该值写入运行/视图账本并用于恢复重放；它不是阶段声明参数，因此不改变规格指纹。
- 每阶段先计算全列统计量，再按批修改行；因此批大小只改变账本中的批次切分，不改变任何阶段输出摘要或最终摘要。

## 分片规则

- `keep_partial`：余数片保留，显示为 `[last_start, row_count)`。
- `drop_partial`：丢弃不足目标行数的尾片。
- 分片指纹由来源阶段 ID/版本、行区间、分片行摘要推导；不包含整表输出摘要，因此其他分片变化不会无谓改变未污染分片。
- 分片记录同时绑定来源阶段版本、阶段输出摘要、规格版本和规格指纹。版本或指纹不匹配即为陈旧，不进入当前分片集合。
- 演进后，若新分片的行区间、来源阶段版本、分片摘要均与旧分片一致，可保留其字节摘要并重绑到新规格；本次演示变更了一行中段金额，只有包含该行的分片 1 被作废，分片 0/2/3/4 保留。

## 演进与整体替换

新规格载入后，按来源数据摘要、定点缩放、分片参数与逐阶段语义指纹计算失效传播：

- 样本或定点缩放变化：全部数据阶段视为受影响；
- 分片参数变化：分片阶段及分片集合受影响；
- 阶段类型/版本/参数/上游变化：该阶段及所有下游受影响；
- 报告稳定列出旧/新规格版本与指纹、配置变化阶段、受影响阶段、旧/新分片数、作废/保留分片与保留索引。

当前结论视图是 SQLite 单例。运行完成时在**同一事务**中更新运行状态、删除旧单例、插入新视图并追加历史，不存在增量合并。旧观测和旧分片保留在历史账本中供审计。

## 强制终止与恢复

每个阶段批次在同一 SQLite 事务中提交：

1. `stage_batches` 行（输入/输出摘要、行区间、规格版本、指纹）；
2. `stage_progress` 游标（已处理行数、批次号、状态）。

只有阶段最终 `COMMITTED` 才被视为可信；中断阶段即使已有若干批次也会被标为 `UNKNOWN`、批次 `trusted=0`，恢复时整阶段以新 attempt 重跑。恢复步骤：

1. 从 `specs` 读取该运行的规格快照，并校验运行指纹与样本摘要；
2. 列出已提交阶段、未知阶段和重跑计划；明确 `exit_report_available=false`；
3. 从样本快照重算已提交前缀，逐阶段、逐批次比较账本；
4. 从首个未知阶段续跑，重新生成全部受影响分片；
5. 完成后再次执行全量重放：阶段输入/输出摘要、行数、每批摘要、每个分片指纹、最终数据摘要逐项比较。

任何差异都会定位到首个阶段和字段并记录 `E_REPLAY`。进程被强制终止时不会伪造退出报告。

## 异常注入与业务码

```powershell
# 样本文件损坏：破坏空值位图长度
python -m mlpipe_workbench run --inject-data-corrupt ...

# 阶段参数越界：把 max_outlier_permille 改为 1001
python -m mlpipe_workbench run --inject-param-bounds ...

# 异常值比例放大：向目标列注入极端值
python -m mlpipe_workbench run --inject-outlier-ratio 300 ...
```

注入只生成或修改输入 JSON 或会话参数，不改源码。

| 业务码 | 含义 | 报告字段 |
|---|---|---|
| `E_SPEC` | 规格/数据声明非法，整体拒绝 | 首个位置、阶段/参数（若适用） |
| `E_DATA` | 注入后的样本文件损坏 | `<dataset>`、位图位置、受影响行数 |
| `E_PARAM` | 注入后的阶段参数越界 | 阶段、参数名、参数值、首个位置 |
| `E_OUTLIER` | 异常值比例超过声明门槛 | 首个触发阶段、门槛值、受影响行数、实际 permille |
| `E_REPLAY` | 重放与账本不一致 | 首个阶段与字段、期望值、实际值 |
| `E_STATE` | 运行状态不允许当前操作 | 账本状态 |
| `E_ENV` | 本地环境（如 Tk）不可用 | 环境位置与提示 |

所有业务错误写入 SQLite `business_errors`/`events`，同时以一行 JSON 追加到 `logs/workbench.jsonl`（键排序、稳定字段）。

## 常用命令

```powershell
python -m mlpipe_workbench validate
python -m mlpipe_workbench run --db workbench.db --run-id cold-1 --batch-size 32
python -m mlpipe_workbench ledger --db workbench.db --run-id cold-1
python -m mlpipe_workbench inspect --db crash.db --run-id crash-1
python -m mlpipe_workbench recover --db crash.db --run-id crash-1
python -m mlpipe_workbench replay --db workbench.db --run-id cold-1
python -m mlpipe_workbench ui
python -m mlpipe_workbench demo
```

运行 `run --crash-after-stage fill_missing` 会在该阶段提交后直接 `os._exit(75)`，用于可重复验证恢复；这是唯一不写退出报告的路径。

## 测试

```powershell
python -m unittest discover -s tests -v
```

覆盖：规格强校验、指纹顺序无关与内容敏感、并发加载一致、规格版本递增、批次边界不变性、定点截断/饱和、异常门槛、余数分片、分片陈旧、演进失效传播、未受影响阶段摘要不变、单例视图替换、三类注入、强制终止恢复、全量重放、阶段/分片账本篡改的 `E_REPLAY`，以及三场景演示顺序。

详细设计见 `docs/ARCHITECTURE.md`，实际运行输出见 `docs/VALIDATION.md` 和 `demo_output/demo_summary.txt`。
