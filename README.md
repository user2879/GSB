# 本地实验追踪与版本管理

这是一个面向 Windows 本地环境的实验追踪应用，同时提供命令行与 Tkinter 工作台。项目不启动端口、不访问网络、不依赖外部服务；记录库、升级检查点、快照数据库、只读备份和运行报告全部位于本地工作区。

## 功能概览

- 严格读取 v1、v2、v3 三种历史记录格式，未知版本、缺字段、类型错误整体拒绝并返回 `E_FORMAT` 与生效 JSON 路径。
- 将 v1/v2/v3 归一为同一内部模型，校验运行唯一性、记录版本单调递增、父运行存在且无环，失败返回 `E_RUN` 和引用链。
- 打开记录库时逐条升级旧记录，每完成一条原子写检查点；支持中断后续做，旧记录写入只读备份。
- 使用 NumPy 按参数和整数种子确定性地生成逐步损失/准确率，支持 `last`、`mean_last_5`、`mean` 聚合口径并计算 SHA-256 指标摘要。
- 同一 `run_id` 提交更高 `run_version` 时整体替换当前记录，旧记录进入只读历史；输出参数变化、指标变化量和聚合版本差异。
- 将完整记录库、检查点位置与内容、格式版本集合、各运行内容摘要和聚合口径版本保存到 SQLite 快照。
- 恢复快照时先比较摘要集合，再整体替换记录库和检查点，不增量合并；随后重新生成指标并与快照摘要逐项重放校验。
- CLI 输出稳定 JSON；Tkinter 工作台展示运行列表、格式来源、指标曲线、聚合口径、替换历史和快照列表。

## Windows 环境准备

要求 Python 3.11 或以上。Windows 官方 Python 安装包默认包含 Tcl/Tk 与 `sqlite3`。

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install numpy
python -c "import tkinter, sqlite3, numpy; print(tkinter.TkVersion, sqlite3.sqlite_version, numpy.__version__)"
```

也可直接执行 `python -m pip install -r requirements.txt`。除 NumPy 外无第三方运行时依赖。

## 快速开始

在项目根目录运行三个确定性场景：

```bat
run_demo.bat
```

等价命令：

```bat
python scripts\demo.py
```

演示写入 `demo_workspace/`，依次输出：

1. 首次导入 `samples/legacy_experiment.json`，将 v1/v2 升级到 v3，并生成检查点与只读备份。
2. 修改 `baseline` 参数后提交 `run_version=2`，整体替换记录并保留 v1 历史。
3. 保存快照、提交 v3 变更、从快照整体恢复，并对三条运行执行重放校验。

CLI 常用命令：

```bat
python -m exp_tracker init --workspace demo_workspace --from samples\legacy_experiment.json --force
python -m exp_tracker status --workspace demo_workspace
python -m exp_tracker list --workspace demo_workspace
python -m exp_tracker show baseline --workspace demo_workspace
python -m exp_tracker curve baseline --workspace demo_workspace
python -m exp_tracker commit --workspace demo_workspace --record replacement.json
python -m exp_tracker snapshot --workspace demo_workspace --label after-v2
python -m exp_tracker snapshots --workspace demo_workspace
python -m exp_tracker restore --workspace demo_workspace --id 1
python -m exp_tracker replay --workspace demo_workspace --id 1
```

启动工作台：

```bat
run_gui.bat
```

也可运行 `python -m exp_tracker gui --workspace demo_workspace`。若工作区尚无记录库，可使用界面中的 `Init from JSON` 选择声明文件。界面中的 `Save Snapshot`、`Restore Snapshot`、`Replay Snapshot` 与 CLI 使用完全相同的服务层。

## 工作区与文件

```text
demo_workspace/
  records.json                 当前记录库与历史记录
  upgrade-checkpoint.json      原子升级检查点
  snapshots.sqlite3            快照数据库
  backups/                     v1/v2 原记录和替换前记录，只读
  reports/runs/                每条运行的格式、聚合口径和指标摘要
  reports/replacements/        整体替换差异报告
  reports/demo-report.json     三场景演示报告
```

关键源码：

- `exp_tracker/codec.py`：三版本严格编解码、归一与 E_FORMAT/E_RUN 校验。
- `exp_tracker/metrics.py`：NumPy 确定性指标、聚合、摘要、首个偏移比较与重放。
- `exp_tracker/repository.py`：记录库写入、逐条升级、备份与检查点恢复。
- `exp_tracker/service.py`：提交替换、快照恢复编排与差异报告。
- `exp_tracker/snapshots.py`：SQLite 快照、内容摘要集合比较、E_REPLAY。
- `exp_tracker/gui.py`：Tkinter 工作台。
- `samples/legacy_experiment.json`：静态确定性三格式样本。

## 记录格式约定

每条记录都包含 `format_version`、`run_id`、`run_version`、`parameters` 和 `final_metrics`。`parameters` 必须含整数 `seed`、正整数 `model_dim`、正数 `learning_rate`，以及对象 `hyperparameters`；`hyperparameters.steps` 是必填正整数。

v1：

```json
{
  "format_version": 1,
  "run_id": "baseline",
  "run_version": 1,
  "parameters": {
    "seed": 11,
    "model_dim": 32,
    "learning_rate": 0.01,
    "hyperparameters": {"dropout": 0.1, "steps": 12}
  },
  "final_metrics": {"loss": 0.85, "accuracy": 0.64}
}
```

v2 在 v1 基础上增加：

```json
{
  "step_metrics": [{"step": 1, "loss": 1.2, "accuracy": 0.53}],
  "timestamp": 1700000011
}
```

v3 在 v2 基础上增加：

```json
{
  "aggregation": {"name": "mean_last_5", "version": "agg-v1"},
  "parent_run_id": "baseline"
}
```

`step_metrics` 必须从 1 连续递增，且逐点与确定性生成器完全一致；`final_metrics` 必须等于所选聚合口径的结果。v1/v2 归一升级时使用 `last` 口径，并在 provenance 中标记 `aggregation_defaulted=true`。v1 的整数时间戳由种子派生，仅用于统一内部结构。

库文件外层结构：

```json
{
  "library_schema": 1,
  "library_id": "local-demo-experiment",
  "runs": [],
  "history": []
}
```

## 确定性与版本语义

- 生成器固定使用 NumPy PCG64 均匀流、float64 基础 IEEE 运算和整数扰动，不使用跨平台可能产生末位差异的 `exp/sin/cos`；相同参数与种子重复生成时逐步数组逐位相等。
- 指标摘要覆盖算法版本、聚合口径、完整逐步序列和最终指标，可用于快照重放比较。
- 升级按 `(run_id, run_version)` 稳定排序，逐条执行；先写活动检查点，再原子替换记录库，最后原子写完成条目。若进程在文件替换与完成标记之间终止，重启时通过活动条目的源/目标摘要判定记录是否已经升级。
- 替换提交必须使用完整 v3 记录，`run_version` 必须严格大于当前版本。当前条目由新对象整体替换，旧对象进入 `history` 并写只读备份，不进行字段级合并。
- 快照恢复固定使用 `mode=whole_replace`；记录库中快照不存在的运行会被移除，快照版本较新的运行会被恢复。差异摘要包含新增、移除和变更运行。
- 重放按快照中的参数和聚合口径重新生成序列并重算摘要。摘要缺失、口径版本不支持、序列或摘要不一致都会返回 `E_REPLAY`，并携带运行 ID、记录版本、口径版本与依据。

## 异常码

| 代码 | 含义 | 关键字段 |
| --- | --- | --- |
| `E_FORMAT` | 未知格式版本、缺字段、类型错误、未知字段、JSON 无效 | `path`、`effective_path`、`reason` |
| `E_RUN` | 运行标识冲突、版本非单调、父引用缺失或成环 | `run_id`、`chain`、`constraint` |
| `E_METRIC` | 确定性序列或终值不一致 | `run_id`、`step`、`field`、`expected`、`actual` |
| `E_REPLAY` | 快照无法按记录参数/口径复现 | `run_id`、`version`、`basis`、`failures` |
| `E_STATE` | 工作区缺失、快照不存在、非法操作状态 | `path`、`hint` 等 |

CLI 业务错误输出到标准输出，退出码为 2；中断升级模拟退出码为 3。

## 测试与验证

```bat
python -m unittest discover -v
python -m compileall -q exp_tracker tests scripts tools
python scripts\demo.py
```

当前交付已执行 33 项自动化测试，覆盖三版本归一、E_FORMAT 原子拒绝、父引用环、指标首个偏移、检查点中断续做、活动检查点恢复、整体替换、只读历史、快照整体恢复、四类异常码和演示输出稳定性。实际环境结果见 `VERIFICATION.md`。

本项目的验证容器没有安装 Tk，因此无法在该环境创建真实窗口；`exp_tracker.gui` 已验证可在无 Tk 时安全导入，窗口代码使用 Windows 官方 Python 自带的 Tkinter。CLI、SQLite、NumPy、文件持久化与全部业务场景均已在本地完成验证。
