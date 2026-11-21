# 实际验证记录

验证日期：2026-11-22。所有推理、SQLite、快照和恢复计算均在本地进程内完成，无服务端口，无容器。

## 环境

- 主运行：Python `3.11.0rc1`、NumPy `1.26.4`、SQLite `3.37.2`。
- 最低版本复核：Python `3.10.12`、NumPy `1.26.4`、Tk `8.6`、SQLite `3.37.2`。
- 主 Python 安装缺少系统 `python3-tk`。为实际验证界面，在 `/tmp` 隔离解包 Ubuntu Tk/Tcl/BLT 运行库，使用已有 Python 3.10 和隔离 NumPy wheel；未修改系统包。Windows 官方 Python 自带 Tkinter，直接使用 `start_workbench.bat` 即可。

## 自动化测试

```text
python3 -m unittest discover -s tests -v
Ran 40 tests in 5.104s
OK
```

同一套 40 项测试在 Python 3.10.12 + Tk 8.6 隔离环境再次执行：

```text
Ran 40 tests in 5.374s
OK
```

覆盖：

- `E_PLAN` 首个违规层、重复 ID、合法范围、悬空引用、环、删除引用保护。
- 稳定失效序列、触发源/路径、JSON 顺序无关、未失效层版本/摘要/块号不变。
- 三类增量变更与全量数值 oracle 一致；`E_INCR` 定位首个差异层/字段。
- 连续分配、单块回退、碎片、`E_OOM` 所需/空闲块数及账本事件。
- ties-to-even 量化、int32/int64 饱和、float32 参考对比、批次/样本位置无关、双路径逐批摘要。
- 相等误差界的稳定字典序决胜与 `tie_affected`。
- SQLite 整数格式版本/内容摘要、只读快照、篡改检测、整体替换与备份。
- 真实子进程 `os._exit(137)`、未知层扫描、恢复补跑、`E_REPLAY` 首差异字段。
- CLI 校验、增量对照、全量报告、重放。
- 无 Tk 环境下视图模型的可测试性约束；Tk 环境另行实际渲染见下。

机器可读汇总：`artifacts/verification_summary.json`；完整测试输出：`artifacts/unittest_py311.log`、`artifacts/unittest_py310.log`。

工作区已用示例计划初始化默认账本 `workbench.sqlite3` 和只读快照 `snapshots/v1-ccd246a52fab-cli-full`，可直接用 `report`、`snapshots` 或 Tkinter 界面查看。

## 增量对照实测

```text
python3 -m memdiag.cli check-incremental \
  --plan data/plan.json --samples data/samples.json \
  --changes data/changes_scale.json
```

结果：`invalidated_layers = [branch_a, merge, head]`；`checked_layers` 为全部 5 层；`ok=true`。精度/缩放、维度、引用三份变更文件均返回增量摘要与全量摘要完全相同的 `content_digest`。

## 强制终止恢复实测

```text
python3 -m memdiag.cli run ... --session-id crash-demo --crash-after 2 --no-snapshot
CRASH_EXIT=137
python3 -m memdiag.cli recover --session-id crash-demo
committed_layers = [stem, branch_a]
unknown_layers   = [branch_b, merge, head]
exit_report_available = false
action_required = rerun_uncommitted_layers
RECOVER_EXIT=8
python3 -m memdiag.cli resume --session-id crash-demo
replay_verification.checked_layers = [stem, branch_a]
rerun_layers = [branch_b, merge, head]
status = complete
python3 -m memdiag.cli replay --session-id crash-demo
checked_layers = [stem, branch_a, branch_b, merge, head]
exit_report_available = true
```

恢复前重放被 `E_RECOVERY` 拒绝；恢复后重放通过。未提交层从未被当作可信结论。

## 快照实测

- 快照 `snapshot.json` 与目录写位均为 0；`read_only=true`。
- 从计划 v2 恢复到 v1：`replacement_mode=wholesale`，当前计划版本、层集合、块池、失效轨迹和内容摘要均回到 v1。
- 当前 v2 状态被保存为 `pre-restore-backup` 只读快照。
- 手工修改快照内容但不更新摘要时，加载以 `E_SNAPSHOT` 拒绝。

## Tkinter 渲染实测

使用真实 `:0` 显示，自动构造完整窗口并跑一次事件循环后自动关闭：

```text
GUI_ROOT_OK 1320 860 widgets 41
GUI_TITLE 混合精度推理显存预算诊断工作台
GUI_EVENT_LOOP_OK
GUI_EXIT=0
```

窗口成功加载 `data/plan.json` 与 `data/samples.json`，执行全量诊断、写入临时 SQLite、保存临时只读快照，并渲染层树、块池、误差排名、快照/会话列表与失效轨迹。

## 限制与说明

- 当前主 Python 3.11 的系统 Tk 包缺失，因此默认 `python3 -m memdiag.cli gui` 会返回 `E_DOMAIN` 并提示安装 Tk；核心 CLI、账本、恢复和 Tk 代码均未依赖网络。
- Windows 上 Tkinter 随官方 Python 安装包提供，无需额外系统包。
- 全量 oracle 对未失效层复用历史块位，是有意保持与增量模式相同的“块号历史语义”；全部数值、误差、双路径摘要仍从零重算。
