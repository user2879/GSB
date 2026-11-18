# 技术说明：SQLite 连接池并行调度与资源预算工作台

## 1. 环境与依赖

- Python 3.11+，仅使用标准库（`sqlite3`、`tkinter`、`threading`、`json`、`unittest` 等），无第三方包、无网络依赖。
- 后端为工作区内的本地文件 SQLite（`workspace/sample.sqlite`），样本表为 `inventory` 与 `ops_log`。
- Windows 上 python.org 官方安装包默认自带 `tkinter`。当前无头 Linux 容器缺少 Tk 绑定，GUI 代码在 `import tkinter` 失败时优雅降级，核心能力全部可经 CLI / 单元测试验证（见 `pool_workbench/gui.py` 的 `_TK_AVAILABLE`）。

## 2. 目录结构

```
pool_workbench/
  errors.py          业务错误码（E_CONFIG/E_SCHEMA/E_GETTIMEOUT/E_HOLDTIMEOUT/E_EXEC/E_SNAPSHOT/E_REPLAY）
  structured_log.py  线程安全的 JSONL 结构化日志
  config.py          JSON 加载与强校验、SQL 白名单解析
  sample_db.py       确定性样本库、模式校验、wb_sleep UDF、只读连接、库克隆
  pool.py            连接池：连接生命周期、配额账本、失效重建、池事件
  planner.py         确定性逻辑规划器（FIFO，整数毫秒）
  scheduler.py       虚拟时钟确定性并行调度器（真实连接执行 SQL）
  snapshot.py        快照持久化、版本识别、整体替换差异
  replay.py          重放校验与不可复现报告（在克隆库上执行）
  fault_injection.py 启动参数驱动的异常注入（只改输入文件/参数）
  app.py             会话编排（加载/执行/快照/恢复/重放）
  cli.py             命令行入口
  gui.py             Tkinter 控制台 + 无头视图模型 PoolViewModel
  demo.py            三轮可重复演示
  samples/           确定性样本任务脚本（合法与非法样例）
  tests/             unittest 测试套件
scripts/             样本库构建脚本与 Windows .bat 入口
workspace/           运行期产物（样本库、快照、events.jsonl、demo_summary.json）
```

## 3. 关键设计决策

### 3.1 强校验与整体拒绝（要求 1）
`config.py` 聚合所有字段错误后一次性抛出单个 `E_CONFIG`，`field_paths` 给出精确路径（如 `pool.workload_quotas.sum`、`tasks[3].sql`）。SQL 仅允许对样本表执行 `SELECT/INSERT/UPDATE`，通过关键字白名单 + 表名白名单 + 多语句/危险关键字（`ATTACH/PRAGMA/DROP/DELETE/...`）拦截；写语句要求 `READ_WRITE` 连接。

### 3.2 三重约束的并行调度（要求 2）
- **规划器 `planner.py`**：FIFO 离散事件模拟，整数毫秒推演，输出每个任务的连接序号、排队开始时刻、开始/结束、等待毫秒、预测结局。
- **运行时 `scheduler.py`**：确定性虚拟时钟把任务调度到真实池连接上执行 SQL；连接占用窗口按声明时长建模，形成真实的多连接繁忙、配额争用与排队。
- 连接池在每次 `acquire` 时独立校验池上限与工作负载配额，调度器另设监控采样，任何越限都会记录到 `constraint_violations`（验证结果始终为空）。
- 每条轨迹含 `task_id / slot_seq / queue_enter_ms / start_ms / end_ms / wait_ms / actual_wait_ms / outcome / error_code`，任务与轨迹一一对应，不存在无轨迹任务。

### 3.3 获取超时（要求 3）
规划器证明任务无法在 `acquire_timeout_ms` 内获得资源时预测 `GET_TIMEOUT`；运行时同样以真实获取窗口兜底。获取超时任务 `slot_seq=None`，绝不占用连接，结局码 `E_GETTIMEOUT`。

### 3.4 连接失效重建（要求 3）
执行中连接抛出底层错误时，`ConnectionPool.rebuild_dead` **先从池中移除并关闭死连接，再按原类型补建新连接**，因此活跃/存活连接数不会瞬间超过 `max_connections`。重建次数与原因写入池事件（`connection_failed` / `connection_rebuilt`）。普通语句错误（如 NOT NULL）不会杀死连接，仅记 `E_EXEC`。

### 3.5 持有时长回收策略（要求 4，显式声明）
采用 **“语句边界中断”** 策略：每个任务启动一个 `hold_timeout_ms` 定时器，到期先置位该连接的中断事件并调用 `Connection.interrupt()`；确定性 UDF `wb_sleep` 在其 2ms 步进边界检测中断并抛出 `OperationalError`，调度器据此归类 `E_HOLDTIMEOUT` 并回滚。依据：SQLite 的 `interrupt()` 保证在当前语句的安全边界生效，不会破坏数据库文件或留下半条写入，因此无需强杀线程/进程即可安全回收持有时长。

### 3.6 快照版本识别与整体替换（要求 5）
快照保存参数版本、连接集合与状态、配额用量、排队任务、任务序列、轨迹与结局统计。恢复时比较快照参数版本 **及参数内容**：任一不同即**整体替换**池参数、连接集合与配额状态（先 `close()` 旧池再按快照重建），绝不增量合并；差异稳定列出变化参数项与受影响排队任务（见 `snapshot.compute_version_diff`）。缺失 `param_version` 的快照以 `E_SNAPSHOT` 拒绝。

### 3.7 重放校验与不可复现（要求 6）
重放在样本库的**临时克隆**上按快照参数与任务序列重新调度，逐项比对 `outcome / slot_seq / wait_ms`，并对样本表做前后校验和，确保重放不改样本数据。参数版本或样本数据版本不一致时，以 `E_REPLAY` 报告首个不一致的任务标识、字段、期望值与实际值后立即中止。

### 3.8 异常注入与结构化日志（要求 7）
`fault_injection.py` 只修改输入文件或启动参数：丢表→`E_SCHEMA`、配额和超限→`E_CONFIG`、快照删版本字段→`E_SNAPSHOT`、样本数据版本篡改→`E_REPLAY`。所有事件以 JSONL 写入 `workspace/events.jsonl`。

## 4. 确定性说明
任务耗时通过 SQLite 自定义函数 `wb_sleep` 声明；调度阶段把它替换为立即返回的纯函数，时长完全由虚拟时钟按声明毫秒推进，因此同一份配置多次运行的轨迹逐字节一致，满足快照可复现。真实 SQL（UPDATE/INSERT/SELECT）仍在真实池连接上执行并真实提交/回滚。

## 5. 如何运行

```bash
# 构建样本库
python scripts/build_sample_db.py
# 或
python -m pool_workbench.cli --workspace workspace build-db

# 三轮演示（冷启动 -> 调配额重跑 -> 异常注入）
python -m pool_workbench.demo

# 测试
python -m unittest discover -s pool_workbench/tests -p "test_*.py" -v

# CLI 示例
python -m pool_workbench.cli --workspace workspace run \
  --config pool_workbench/samples/tasks_cold.json --save-snapshot
python -m pool_workbench.cli --workspace workspace replay \
  --snapshot workspace/snapshots/snapshot-v1-001.json

# GUI（桌面环境 / Windows）
python -m pool_workbench.gui
```

## 6. 实际验证结果
- 单元/集成测试：`23 passed`。
- 冷启动 12 个任务结局：成功 9、获取超时 1（t09）、持有超时 1（t08）、执行错误 1（t12，确定性 NOT NULL 违规），约束违反 0。
- 调配额 v2 重跑：成功 10、获取超时 0、持有超时 1、执行错误 1；恢复 v1 快照触发整体替换并列出 3 项参数变化与受影响排队任务 t09；版本漂移重放报 `E_REPLAY`（首个任务 t01，字段 `pool_config.param_version`）。
- 异常注入：`schema_missing_table→E_SCHEMA`、`config_quota_exceed→E_CONFIG`、`snapshot_missing_version→E_SNAPSHOT`、数据版本篡改→`E_REPLAY`。
- 失效重建：注入连接关闭后，死连接先移除后补建，存活连接数始终 ≤ `max_connections`，重建事件齐全。
