# 实际验证记录

验证日期：2026-11-24（Asia/Shanghai）

## 环境

| 项目 | 值 |
| --- | --- |
| 核心运行环境 | Linux WSL 容器外本地文件系统；Python 3.11.0rc1 |
| SQLite | Python 标准库 `sqlite3` |
| Tk 运行时验证 | 临时解包 Tk/Tcl 8.6.12 + Xvfb，未修改项目依赖 |
| 容器/Docker | 未使用 |
| 网络服务/账号/GPU | 未使用 |
| 应用第三方运行时依赖 | 无 |
| 测试依赖 | pytest 8.3.3 |

目标交付环境为 Windows 本地 Python 3.10+。本容器不提供 Windows 桌面，因此 Windows 安装/启动步骤写在 README，实际核心功能在 Linux 同构 Python 标准库环境中验证；代码未使用 POSIX 专用接口。

## 1. 自动化测试

执行：

```bash
python3 -m pytest -q
```

结果：

```text
........................                                                 [100%]
24 passed
```

覆盖点：

- 有效样本的连续 LSN、事务组、崩溃点、目标版本和目标可见性。
- 非法页大小/重复页标识 `E_PAGE`；序号重复/倒退 `E_ORDER`；未知页、摘要不符、事务分组破坏 `E_LOG`。
- 单线程页版本单调性、版本化文件摘要、检查点游标 4/8/12。
- 并行与单槽位重放的页版本、摘要和页文件逐字节一致。
- 全局槽位与每事务配额上限、等待时间记录、页字节预算上限。
- 子进程 `os._exit(91)` 后仅一个完整批次，任务转为 unknown 后恢复并校验。
- 已提交版本不会在恢复时重复重放。
- 快照自摘要、整数格式版本、整状态替换、新增/删除/变化页、磁盘 WAL 不符时替换。
- `E_REPLAY` 的首个不一致 LSN/页/字段及账本批次定位；账本和预算篡改检测。
- 三类 CLI 注入的退出码、业务码和 NDJSON 审计记录。
- GUI 模块在无 Tk 环境下可安全导入，`launch_gui` 延迟加载 Tk。

## 2. 端到端可重复演示

执行：

```bash
python3 demo.py
```

结果：`all_assertions=passed`

实际观察值：

| 指标 | 结果 |
| --- | --- |
| 页声明 | p1/p2/p3，大小 256/512/128 字节 |
| WAL 条目 | 12 |
| 崩溃点 | LSN 10 |
| 持久可见事务 | tx-001、tx-002、tx-003 |
| 重放任务 | 11 |
| 原子批次 | 3：LSN 1-4、5-8、9-12 |
| 自动检查点 | LSN 4、8、12；另有手工检查点 |
| 全局观察并发 | 3，等于 `global_slots=3` 上限 |
| 等预算任务 | 8 个任务出现等待时间 |
| 正常恢复校验 | `ok=true`，12 个日志、12 个 WAL 页版本 |
| 账本链 | `ok=true` |

强制终止阶段：

```text
child_exit_code = 91
ledger_cursor_after_crash = 4
committed_batches = [1-4]
task_states = {committed: 3, partial: 1, queued: 5, running: 2}
last_run.status = running
exit_report_claimed = false
```

恢复阶段：

```text
run_id = run-0002
cursor_start = 4
cursor_end = 12
recovered_orphan_tasks = 8
batch_count = 2
最终 verify_replay.ok = true
最终 ledger_verification.ok = true
```

快照差异演示把主工作区快照恢复到不同的有效工作区：

```json
{
  "restore_mode": "whole_state_replacement",
  "log_changed": true,
  "added": ["p3"],
  "deleted": ["p4"],
  "changed": ["p1", "p2"]
}
```

恢复后的差异工作区独立校验为 `ok=true`。

## 3. 三类异常注入

执行演示中的实际结果：

| 命令模式 | 业务码 | 退出码 | 上下文字段 | `injections.ndjson` |
| --- | --- | ---: | --- | --- |
| `--inject page_size` | `E_PAGE` | 10 | `size_bytes` | 已记录 |
| `--inject order` | `E_ORDER` | 11 | `lsn` | 已记录 |
| `--inject snapshot_digest` | `E_SNAPSHOT` | 13 | `content_digest` | 已记录 |

日志片段（字段顺序按 `sort_keys`）：

```json
{"code":"E_PAGE","context":{"field":"size_bytes","page_id":"p2","page_index":1,"value":300},"injection":"page_size","level":"ERROR"}
{"code":"E_ORDER","context":{"expected_lsn":2,"field":"lsn","first_invalid_lsn":1,"line":2,"lsn":1},"injection":"order","level":"ERROR"}
{"code":"E_SNAPSHOT","context":{"actual":"000...","expected":"<computed>","field":"content_digest"},"injection":"snapshot_digest","level":"ERROR"}
```

## 4. Tkinter 实际启动

本基础 Python 初始缺少 `libtk8.6.so`，先验证了 `--gui` 会结构化返回 `E_GUI`（退出码 17），核心 CLI 不受影响。

随后在 `/tmp` 临时解包 Tk/Tcl 8.6.12 与 Xvfb，不修改系统/项目依赖，使用虚拟显示执行：

```text
display_ok 1600 1000
gui_smoke_ok
```

该冒烟测试真实创建 Tk 窗口、构建七个页签及所有 Treeview/Text/按钮、执行首次只读刷新，并在 1.8 秒后正常销毁。未进入人工视觉验收；界面布局和功能通过控件创建与刷新路径验证。

## 5. 唯一性、可复核性与刷新只读性

- `verify_replay()` 用重新加载的输入做独立顺序基线，不采用调度器输出作为期望。
- 账本每个事件和每个批次可通过 `SHA-256` 哈希链独立重算；`--action export-ledger` 同时导出事件、批次、任务、检查点、页版本和校验结果。
- GUI 刷新路径只执行 `Storage.status/tasks/batches/...` 的只读查询，不调用 orphan 标记、日志写入或任何 SQL 变更。
- 对快照自摘要篡改的测试断言：恢复前后 `ledger_rows()` 完全相同。

## 6. 重复执行方式

```bash
python3 demo.py
python3 -m pytest -q
```

演示会清理并重建两个相对工作区；测试使用 pytest 临时目录。二者均为确定性样本，适合重复执行。

