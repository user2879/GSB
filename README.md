# 混合精度推理显存预算诊断工作台

这是一个完全本地、可离线运行的 Windows 优先诊断工作台。它以固定整数随机种子的 NumPy 样本和确定性前馈模型为基础，对 `float32` / 定点量化混合网络执行显存块池规划、增量失效、数值误差归因、SQLite 逐层事务提交、不可变快照和强制终止恢复重放。核心计算不需要端口、容器、数据库服务或远程资源。

## 交付内容

- `memdiag/`：领域模型、计划校验、依赖图、增量引擎、块池、量化路径、SQLite 账本、快照、恢复/重放、CLI、Tkinter 界面。
- `data/plan.json`：五层分支合并示例；故意打乱层数组和引用对象顺序，用于验证规范化与稳定拓扑。
- `data/samples.json`：种子 `20261122`、12 条、8 维确定性样本。
- `data/changes_*.json`：精度/缩放、维度、引用三类增量变更。
- `tools/generate_samples.py`：固定种子样本生成器。
- `tests/`：40 项标准库 `unittest` 验收测试。
- `docs/DESIGN.md`：算法、数值、事务与恢复设计。
- `VERIFICATION.md`：本机实际执行结果。
- `start_workbench.bat` / `run_checks.bat`：Windows 双击入口。

## 环境准备

要求 Python 3.10+，并具备 NumPy、Tkinter、SQLite3。Windows 官方 Python 安装包已自带 Tkinter 和 SQLite3；只需安装 NumPy：

```powershell
py -3 -m pip install -r requirements.txt
py -3 -c "import numpy, tkinter, sqlite3; print(numpy.__version__, tkinter.TkVersion, sqlite3.sqlite_version)"
```

Linux 若缺少 Tk：

```bash
sudo apt-get install python3-tk
python3 -m pip install -r requirements.txt
```

本项目无端口占用需求；所有文件读写都发生在工作区或显式传入的账本/快照路径。

## 快速开始

```bash
# 可选：重新生成完全相同的样本
python tools/generate_samples.py --seed 20261122 --count 12 --dim 8 --output data/samples.json

# 严格加载计划与样本
python -m memdiag.cli validate --plan data/plan.json --samples data/samples.json

# 全量诊断；每层结论在独立事务中提交，并自动保存只读快照
python -m memdiag.cli run --plan data/plan.json --samples data/samples.json \
  --db workbench.sqlite3 --snapshots snapshots --session-id demo-full \
  --report artifacts/latest_report.json

# 应用增量修改并持久化
python -m memdiag.cli incremental --plan data/plan.json --samples data/samples.json \
  --changes data/changes_scale.json --db workbench.sqlite3 --snapshots snapshots

# 用全量数值参考核对增量结论
python -m memdiag.cli check-incremental --plan data/plan.json --samples data/samples.json \
  --changes data/changes_scale.json --report artifacts/incremental_check.json

# 启动桌面向导
python -m memdiag.cli gui --plan data/plan.json --samples data/samples.json \
  --db workbench.sqlite3 --snapshots snapshots
```

Windows 可直接运行 `start_workbench.bat`。`run_checks.bat` 会重新生成样本并运行全部测试与增量对照。

## 推理计划 JSON

```json
{
  "format_version": 1,
  "plan_version": 1,
  "seed": 20261122,
  "block_size_kb": 32,
  "memory_budget_mib": 4,
  "max_batch_size": 4,
  "layers": [
    {
      "id": "stem",
      "input_dim": 8,
      "output_dim": 16,
      "param_count": 65536,
      "precision": "float32",
      "scale": 0
    }
  ],
  "consumers": {"stem": []},
  "memory_derived_from": {"stem": []}
}
```

约束与语义：

- `layers` 可为数组或对象；最终按层标识字典序规范化，层标识必须唯一。
- `input_dim`、`output_dim`、`param_count` 为正整数；`precision` 仅支持 `float32`、`fixed`；`scale` 为 `0..6` 的整数。
- `consumers[producer]` 表示生产者输出被哪些层使用，形成“推理结论引用”边。
- `memory_derived_from[layer]` 表示该层显存由哪些层派生，形成额外失效边。
- 两类引用合并为一张有向图；悬空引用、删除后仍被引用和环均以 `E_DEPEND` 拒绝。删除错误列出全部 `referrers`。
- 层间输入采用稳定适配：父层输出按父层标识字典序拼接；不足 `input_dim` 时补零，超出时截断。这样维度变更始终可诊断，同时结果不依赖 JSON 对象遍历顺序。
- 层参数由 `seed + layer_id + 维度` 派生，缩放或精度修改不会改变同一层的浮点参考权重。

## 增量修改 JSON

```json
{
  "layer_updates": {
    "branch_a": {"precision": "float32", "scale": 0}
  },
  "consumer_updates": {"branch_b": []},
  "memory_updates": {"merge": ["branch_a"]},
  "add_layers": [],
  "delete_layers": []
}
```

`layer_updates` 支持 `precision`、`scale`、`input_dim`、`output_dim`、`param_count`。系统取修改前后图的并集求可达下游，再按新计划稳定拓扑序输出失效层。每个轨迹步骤含 `trigger_source`、`via`、`depth`、`reason` 和完整 `path`。

未失效层直接复用以下字节级状态：输出摘要、逐样本摘要、块号集合、分配模式、误差结论与层版本号。全量对照使用“所有层数值重算 + 未失效层块位锚定”的模式，检查范围包括双路径摘要、块序、请求字节、三段误差界和归因结果。

## 数值规则

浮点路径固定为 NumPy `float32` 矩阵乘、加偏置、`tanh`。定点路径：

1. 权重和激活执行 `round_nearest_ties_even(x * 2**scale)`。
2. 量化值饱和到 signed `int32`。
3. 乘积与累加使用 signed `int64`；累加和偏置按 `int64` 上下界饱和。
4. 累加结果除以 `2**scale` 时再次按“最近偶数”舍入，并饱和到 `int32`。
5. 激活经过 `tanh` 后，再按层 `scale` 量化输出。

每批同时保存实际混合路径与 float32 参考路径摘要。样本始终按样本标识排序执行逐样本计算，所以批次大小、样本排列和调度交错不会改变单个样本输出及全局归因。

逐层误差界拆为 `quantization_truncation`、`accumulation_order`、`memory_reordering`，实际最大偏差不会超过合成界。主导段取最大分量；最大偏差相同或误差区间重叠时，排名稳定地按 `max_abs_error` 降序、层标识升序，并标注 `tie_affected`。

## 显存块池

- 预算块数 = `floor(memory_budget_mib * 1024 * 1024 / (block_size_kb * 1024))`。
- 层请求字节 = 权重 + 激活工作区 + 累加/临时区 + 固定开销；再向上取整为块数。
- 分配优先使用最低地址的连续空闲段；无足够连续段时回退到单块分配，并记录 `contiguous` / `single_block`。
- 增量重排只释放失效层块，再按稳定拓扑重新分配；未失效层块号原样保留。
- 每批记录块序号、所属层、请求字节、峰值块数、碎片块数（空闲块数减最大连续空闲块数）。
- 块不足抛出 `E_OOM`，包含 `needed_blocks`、`free_blocks`，并写入账本事件。

## 账本、快照与恢复

SQLite 账本格式版本为整数 `3`，保存当前内容摘要、计划版本、逐层结论、逐批双路径摘要、显存块序、失效轨迹与误差归因。每完成一层，结论、该层批次、进度游标在同一个 `BEGIN IMMEDIATE ... COMMIT` 中提交。

快照目录中的 `snapshot.json` 和 `metadata.json` 都是只读文件，目录也移除写位。快照自校验计划版本、计划摘要和内容摘要。恢复时：

1. 比较快照与当前账本的计划版本、计划摘要、内容摘要。
2. 若不同，先把当前状态保存为新的只读 `pre-restore-backup` 快照。
3. 在一个事务中删除并重建当前层集合、显存状态、批次、失效轨迹与归因，而不是增量合并。
4. 旧快照永不修改。

强制终止后，`recover` 只承认完整提交的账本层批次；未提交层标记为 `unknown`，`exit_report_available=false`。`resume` 先用记录的计划版本、缩放因子和样本重算并校验已完成层，再补跑未知层。`replay` 逐层比较输出摘要、float32 摘要、块序、请求字节、分配模式和误差归因；首个差异以 `E_REPLAY` 报出层标识、字段、记录值、重放值和账本依据。

## Tkinter 诊断界面

界面固定渲染顺序；左侧显示稳定拓扑下的依赖、精度、版本、失效标记、块号与误差界；中部显示块池占位图、逐层误差归因排名和批次双路径结论；右侧显示计划版本、只读快照和账本会话；底部显示完整失效传播轨迹。选择层后可查看引用链、失效路径、块布局和误差分解。按钮支持全量、增量、保存/恢复快照及会话扫描/恢复重放。

## 错误码

- `E_PLAN`：计划或字段非法；含 `first_invalid_layer`。
- `E_DEPEND`：悬空引用、删除保护或依赖环；含 `referrers`。
- `E_OOM`：块池预算不足；含所需块数与空闲块数。
- `E_INCR`：增量与全量对照首个差异；含层标识与字段。
- `E_REPLAY`：恢复重放与账本不一致；含首个层标识、字段和 `basis`。
- `E_SNAPSHOT`：快照缺失、歧义、只读文件被篡改或摘要不符。
- `E_RECOVERY`：存在未知未提交批次，必须先恢复重跑。

完整算法和表结构见 `docs/DESIGN.md`，实测结果见 `VERIFICATION.md`。
