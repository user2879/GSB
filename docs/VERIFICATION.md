# 实际验证记录

验证日期：2026-11-18  
环境：Linux 验收主机，Python 3.11.0rc1，SQLite 3（Python 标准库）  
目标环境：Windows Python 3.10+（含 Tkinter），仅标准库  
说明：本验收主机缺少 Tkinter，因此 Tkinter 仅完成语法编译、导入降级验证；核心逻辑、
CLI、快照、恢复、重放与三个演示场景均已实际执行。

## 自动化测试

```text
$ python3 -m unittest discover -s tests -v
test_abort_skips_remaining_and_records_code ... ok
test_failure_skip_and_continue_policies ... ok
test_forced_crash_recovery_unknown_tail_resume_and_replay ... ok
test_normal_run_ledger_events_snapshot_restore_and_replay ... ok
test_plan_refresh_replaces_generation_and_marks_old_stale ... ok
test_plan_validation_is_strong_and_atomic ... ok
test_replay_reports_first_inconsistent_stage_and_field ... ok
test_snapshot_digest_mismatch_is_e_snapshot ... ok
test_snapshot_restore_is_whole_replacement_and_old_is_read_only ... ok
test_render_order_is_stable ... ok

Ran 10 tests
OK
```

## 演示脚本

```text
$ python3 demo.py
=== 异常注入 ===
E_PLAN stage=2 path=stages[1].estimated_seconds fault=plan-corrupt
E_CYCLE stage=3 path=stages[2].depends_on[0] fault=cycle
E_SNAPSHOT stage=1 path=content_digest fault=snapshot-digest
=== 场景 1 正常演练 ===
status=completed stages=1:succeeded|2:succeeded|3:succeeded|4:succeeded|5:succeeded restore_whole=True replay=True
=== 场景 2 失败事件注入 ===
status=completed stages=1:succeeded|2:succeeded|3:failed|4:skipped|5:skipped replay=True
=== 场景 3 强制终止后恢复 ===
child_exit=137 unknown=[3] tail_action=隔离并丢弃不完整尾记录 [10]；不计入迁移、不参与状态恢复
resumed=completed stages=1:succeeded|2:succeeded|3:succeeded|4:succeeded|5:succeeded replay=True
ALL_CHECKS_PASSED=true summary=artifacts/demo-data/summary.json
```

## 关键断言

- 正常演练逻辑时钟终值 `12000` ms；5 个阶段均 `succeeded`，每个阶段 2 次账本迁移。
- 失败事件在阶段 3 的 `6000` ms 触发；阶段 3 `failed`，阶段 4/5 `skipped`，跳过业务码
  `E_STAGE`，上游定位为阶段 3。
- 强制终止子进程退出码 `137`；恢复报告未知阶段 `[3]`、已恢复阶段 `[1,2]`、尾部不完整
  记录被隔离丢弃，`exit_report_available=false`。
- 三个场景的快照恢复均为整代替换；恢复后重放逐阶段比较 `status`、`skip_reason`、
  `migration_count`，全部一致。
- 预案损坏为 `E_PLAN`，阶段 2，字段路径 `stages[1].estimated_seconds`。
- 成环为 `E_CYCLE`，阶段 3，字段路径 `stages[2].depends_on[0]`。
- 快照内容摘要不一致为 `E_SNAPSHOT`，阶段 1，字段路径 `content_digest`。

机器可读结果：

- `artifacts/demo-data/summary.json`
- `artifacts/demo-data/logs/drill.jsonl`
- `artifacts/demo-data/session.db`
- `artifacts/demo-data/snapshots/`

## UI 编译与降级

```text
$ python3 -m py_compile emergency_drill/*.py demo.py run_ui.py tests/*.py
(exit 0)

$ python3 run_ui.py
E_UI: 当前 Python 未安装 Tkinter。请使用 Windows 官方 Python 3.10+
（通常自带 Tkinter）运行 python run_ui.py。
(exit 2)

$ python3 check_env.py
{
  "python_3_10_or_newer": true,
  "python_version": "3.11.0rc1",
  "ready": false,
  "sqlite_version": "3.37.2",
  "third_party_dependencies_required": false,
  "tkinter_available": false,
  "tkinter_version": null
}
(exit 2)
```

Windows 上应使用以下命令先确认 Tkinter：

```powershell
py -3.10 -c "import tkinter; print(tkinter.TkVersion)"
```
