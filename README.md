# 主从复制延迟诊断工作台

面向 Windows 本地环境的确定性主从复制延迟诊断工具。项目仅使用本地文件、SQLite、Python 标准库与 NumPy，不访问网络或外部数据库。

当前交付已完成：

- 主库日志、从库进度、回放批次与逐条真值样本的 SQLite 持久化与约束校验。
- 整数速率/并发度批量回放、单调虚拟时钟纳秒记录、毫秒展示和延迟样本采集。
- 固定窗口、整数定点均值/P95/最大值，以及不可判定标注。
- 采样截断、定点截断、回放速率离散化三段误差上界、占比与主导段。
- 不可变只读快照、整体替换恢复、日志摘要冲突提示与替换字段报告。
- 每批原子提交、真实强制终止后的 UNKNOWN 批次恢复和确定性续跑。
- 历史重放逐样本/逐统计项校验，首个差异定位到提交序号与字段。
- Tkinter 桌面界面、CLI、异常注入与可重复演示脚本。

本次容器内的实际验证结果为 `PASS`：16 项单元/集成测试通过，完整演示、强制终止恢复和 `E_LOG` / `E_PARAM` / `E_SNAPSHOT` / `E_REPLAY` 注入均通过。详见 `docs/ACCEPTANCE.md`。

## 快速开始

Windows 官方 Python 安装包需勾选 `tcl/tk and IDLE` 组件。建议 Python 3.10+、NumPy 1.24+。

```powershell
py -3 -c "import numpy, sqlite3, tkinter; print('environment ok')"
py -3 demo.py
py -3 -m unittest discover -s tests -v
py -3 -m workbench.ui
```

也可以使用 `run_demo.bat`、`run_tests.bat`、`run_ui.bat`。环境自检命令为 `py -3 -m workbench.cli ui-check --open-window`。

本容器没有安装 `_tkinter`，所以 GUI 代码已完成编译和导入检查，但没有在容器内实际弹出窗口；无头核心路径不受影响。

## 演示内容

`demo.py` 会先清空并重建 `artifacts/workbench/`，然后依次执行：

1. 首次回放：导入 `samples/writes.json`，按 `samples/replay_params.json` 回放并估算。
2. 参数变更重算：按 `samples/replay_params_changed.json` 建立独立参数版本，验证样本不混窗。
3. 快照恢复重放：保存当前不可变快照，构造不同状态后整体替换恢复，再执行历史重放校验。
4. 强制终止恢复：子进程写入一个 `RUNNING` 批次标记后执行 `os._exit(137)`；重启后将该批次标为 `UNKNOWN`，从最后一个完整批次续跑，并核对样本与结论逐字节一致。
5. 异常注入：验证 `E_LOG`、`E_PARAM`、`E_SNAPSHOT`，并额外验证快照篡改产生的 `E_REPLAY`。

单独运行注入：

```powershell
py -3 demo.py --inject E_LOG
py -3 demo.py --inject E_PARAM
py -3 demo.py --inject E_SNAPSHOT
py -3 demo.py --inject E_REPLAY
```

## 目录与落盘位置

```text
samples/                         固定确定性输入与摘要清单
workbench/                       核心实现
  db.py                          SQLite schema、约束、原子批次事务
  simulator.py                   批量回放调度
  estimator.py                   整数定点统计与误差归因
  service.py                     回放、快照、恢复、历史重放编排
  ui.py                          Tkinter 界面
  cli.py                         CLI
tests/test_workbench.py          自动化验收
demo.py                          三场景与异常/崩溃演示
artifacts/workbench/             本次生成的 SQLite、日志、报告、快照
docs/TECHNICAL.md                口径、公式与事务设计
docs/ACCEPTANCE.md               本次实际验证记录
```

固定样本摘要位于 `samples/fixtures_manifest.json`。演示启动时会重新计算 SHA-256 并与清单比对。

## 输入格式

主库写入序列：

```json
{
  "records": [
    {"commit_seq": 1, "txn_id": "txn-0001", "commit_ts_ms": 100, "txn_bytes": 512}
  ]
}
```

约束：`commit_seq` 从 1 开始连续唯一；`txn_id` 非空且唯一；`commit_ts_ms >= 0`；`txn_bytes > 0`。整批校验失败时不写库，返回 `E_LOG` 并给出 `error_seq`。

回放参数：

```json
{
  "replay": {
    "model_version": "batch-barrier-v1",
    "rate_rps": 4,
    "concurrency": 2,
    "clock_mode": "virtual_monotonic_ns",
    "clock_base_ms": 1000
  },
  "aggregation": {
    "version": "fixed-window-v1",
    "window_size": 4,
    "max_samples_per_window": 3,
    "min_samples": 3,
    "latency_ceiling_ms": 2000,
    "scale_power": 3,
    "p95_method": "nearest_rank_ceil",
    "truncation_direction": "floor",
    "dominant_threshold_pct": 60
  }
}
```

## 核心口径

### 时钟与真值

为避免不同进程的真实单调时钟导致历史重放不可复现，核心使用声明式**虚拟单调时钟**：整数纳秒、只前进、`clock_base_ms` 为同一会话内主库提交时间与从库完成时间的共同原点。持久化真值是 `replay_samples.replay_completed_ns`，界面同时按毫秒展示。该口径在配置、快照和报告中显式记录；Windows 生产适配可将同一抽象替换为 `time.monotonic_ns()`，但真实硬件时钟不能同时保证跨进程逐字节历史重放。

### 回放模型

`rate_rps` 为聚合记录/秒，`concurrency` 为单批最大记录数。第 b 个批次的全局结束位置为 e，则：

```text
completed_ns = clock_base_ms * 1_000_000 + ceil(e * 1_000_000_000 / rate_rps)
```

同一批次内的每条记录都记录该完成时刻。并发度通过实际完成簇影响样本和速率离散化上界，但聚合器不读取并发参数，只依赖真实样本；因此同一样本集合即使携带不同并发元数据，结论仍逐字节一致。

`delay_ns = replay_completed_ns - commit_ts_ms * 1_000_000`，延迟必须非负。

### 定点统计

缩放因子为 `S = 10^scale_power`。发布值单位为 `1/S ms`，截断方向显式声明为 `floor`。均值由精确纳秒和一次性下取整；P95 使用 `ceil(0.95*n)` 的 nearest-rank 次序统计量；最大值取精确最大样本。所有比例、误差上界与指纹均使用整数或规范 JSON，不依赖浮点舍入。

### 误差上界

每个窗口的均值、P95、最大值分别输出三段误差上界：

1. `sampling_truncation`：窗口保留前 K 个提交序号的样本，未观察尾部使用显式 `latency_ceiling_ms` 推导保守上界。
2. `fixed_point_truncation`：发布值最终一次 `floor` 贡献 1 个定点单位；P95 还包含 nearest-rank 与 Type-7 分位区间之间的取值差上界。
3. `replay_rate_discretization`：使用实际样本中观察到的最大批次完成簇 `K_obs`，上界为 `K_obs * ceil(1e9 / rate_rps)` 纳秒。它随回放并发簇增长，但不依赖不可验证的并发元数据。

总上界为三段之和；占比以向下取整的 ppm 表示；当最大段占比达到显式阈值（默认 60%）时标为该段，否则标为 `MIXED`。同一集合的注入顺序、窗口遍历顺序不影响整数结果和指纹。

### 不可判定

以下任一条件使窗口标为 `UNDECIDABLE`，统计量输出 `null`，绝不零填充：

- 期望提交序号存在缺失样本。
- 样本数低于 `min_samples`。
- 存在重复或越界提交序号。
- 观察延迟超过配置上界。

报告保留期望序号、已观察序号、缺失序号、样本数、阈值和截断策略，便于回溯。

## 快照、恢复与重放

快照包含格式版本、DB schema 版本、日志 SHA-256、回放参数版本、估算口径、进度游标、批次、逐条样本、统计与误差结果、样本序列摘要。快照写入后设为只读，不会被覆盖或原地修改。

恢复采用单个 SQLite 事务整体删除并替换回放运行、批次、样本和进度，不做增量合并，报告列出每个被替换字段的前后状态。若日志内容摘要不符但序号域仍兼容，仍整体恢复，同时将状态标为 `RESTORED_STALE_LOG_MISMATCH` 并要求先处理日志；历史重放会返回 `E_REPLAY` 并说明日志摘要证据。快照格式或估算口径不兼容返回 `E_SNAPSHOT`。

历史重放按快照保存的速率、并发度和口径重新模拟并重新聚合，先逐条比较样本的完成纳秒、延迟纳秒和定点值，再比较统计量、误差段、总上界与主导段。样本差异定位到首个提交序号和字段；纯统计差异定位到字段路径。

## 原子进度与强制终止

每个批次先单独提交 `RUNNING` 标记。完成时在同一事务中写入样本、完成批次标记和进度游标。进程在 `RUNNING` 与完成事务之间被强杀时，只能看到已完整提交的批次；重启扫描后把未完成批次改为 `UNKNOWN`，进度改为 `REQUIRES_RERUN`。工作台明确报告 `exit_report_available: false`，不伪造强杀退出报告。续跑会创建新的批次 attempt，并从最后完整游标继续。

## 业务码

- `E_LOG`：日志连续性、唯一性、时间戳、字节数或进度超前等错误，带 `error_seq`。
- `E_PARAM`：整数速率、并发度、缩放、P95 方法或截断方向非法。
- `E_SNAPSHOT`：快照格式/口径不兼容、样本摘要损坏、恢复目标冲突或状态不可安全替换。
- `E_REPLAY`：日志摘要、参数/口径或历史结果不可复现，报告中给出依据和首个差异。

结构化 JSONL 日志写入 `artifacts/workbench/logs/workbench.jsonl`。

## Tkinter 界面

界面固定按以下顺序组织：

1. 日志与进度：主库序号区间、事务明细、各参数版本游标与批次状态。
2. 延迟样本分布：逐条真值样本、完成毫秒/单调纳秒、延迟与稳定直方图。
3. 定点统计与误差：窗口均值/P95/最大值、可判定依据、三段上界、总上界和主导段。
4. 快照与重放：按文件名稳定排序的只读快照、整体替换报告、历史重放差异。

所有列表按提交序号、窗口号、参数版本或文件名显式排序，避免渲染顺序漂移。

## 测试

```powershell
py -3 -m unittest discover -s tests -v
```

当前 16 项测试覆盖严格日志校验、进度超前、定点值、缺失/低样本不可判定、输入顺序与并发元数据不变性、参数隔离、快照整体替换、只读快照、日志摘要冲突、快照口径错误、样本/统计篡改的首差异定位，以及真实子进程强杀恢复。

更完整的公式、状态机和验收记录见 `docs/TECHNICAL.md` 与 `docs/ACCEPTANCE.md`。
