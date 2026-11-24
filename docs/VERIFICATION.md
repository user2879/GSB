# 实际验证记录

验证时间：2026-11-25（Asia/Shanghai）。

## 1. 环境

实际执行环境：

- Python `3.11.0rc1`
- SQLite `3.37.2`
- NumPy `1.26.4`
- 操作系统：Linux 沙箱（目标交付环境为 Windows + Python 3.11+）

依赖检查实际结果：

```text
python 3.11.0rc1
sqlite 3.37.2
numpy 1.26.4
tkinter_unavailable ImportError libtk8.6.so: cannot open shared object file
```

当前 Linux 沙箱缺少系统 Tk 动态库，因此未真实弹出桌面窗口；这不是 Python 包代码异常。Windows 官方 Python 安装器默认包含 Tcl/Tk，可运行：

```powershell
python -c "import sqlite3, tkinter, numpy; print('deps ok', numpy.__version__)"
python -m colpage_recovery.cli gui --state-dir state\demo --report artifacts\demo_report.json
```

在当前 Linux 环境中，`colpage_recovery.gui` 已验证可降级导入，并用 Tk/Mock 桩实际构造应用、执行全部页签的 `render()`，9 个 Treeview 均完成渲染。

## 2. 静态编译和单元测试

命令：

```bash
python -m compileall -q src scripts tests
python scripts/run_tests.py
```

实际结果：

```text
Ran 17 tests in 3.768s

OK
```

完整日志：`artifacts/final-tests.log`。

打包入口也已实际验证：

```text
python -m pip install . --no-deps --force-reinstall
Successfully installed colpage-recovery-diag-0.1.0
colpage-diag validate --samples samples
```

覆盖：

- JSON 强校验、重复 LSN/page、正整数行数、区间、事务成组和标记协议
- `E_COMMIT`、`E_RANGE`、`E_SNAPSHOT`、`E_REPLAY`
- 批次大小 `1/2/3/4/8` 与不同槽位/配额下的原始页字节和摘要逐位一致
- 真实线程池中的全局槽位和单事务配额时间点并发上限
- 子进程 `os._exit(137)` 强杀、running -> unknown、重跑和恢复校验
- 检查点 LSN 3 与已提交游标 LSN 4 的重叠回放
- 预算/页集合变化后的 G1 -> G2 单调代际和陈旧标记
- 最后一个物理版本属于未提交事务时，当前视图只取最新可见版本

## 3. 端到端演示

命令：

```bash
python scripts/demo.py
```

实际输出：

```json
{
  "ok": true,
  "report_path": "artifacts/demo_report.json",
  "determinism": "matched",
  "force_kill_returncode": 137,
  "unknown_batches": [
    {
      "batch_id": 2,
      "round_no": 2,
      "attempt_no": 1,
      "start_lsn": 5,
      "end_lsn": 8
    }
  ],
  "recovery_status": "verified",
  "spec_generation": 2,
  "current_pages": 2
}
```

摘要日志：`artifacts/final-demo-summary.log`；完整机器可读报告：`artifacts/demo_report.json`。

报告中的关键恢复事实：

- 强杀后立即查询账本：R1.1 `committed`，R2.1 `running`
- 重启恢复：R2.1 标记为 `unknown`，随后以 R2.2 重跑并提交
- 恢复基准检查点：LSN 3
- 崩溃前已提交游标：LSN 4
- 重叠校验轮次：`committed_checkpoint_overlap` 重放 LSN 4
- 完整校验轮次：`full_recovery_interval` 重放 LSN 4、5、6、7、8
- 已提交事务集合：`T1/T2/T4`
- 不可见版本集合：`(P-A,v3,lsn5)`、`(P-B,v3,lsn6)`
- 规格演进：G1 -> G2，旧代际检查点、批次和视图快照均有陈旧标记

## 4. 异常注入业务码

命令：

```bash
PYTHONPATH=src python -m colpage_recovery.cli diagnose --inject commit-conflict
PYTHONPATH=src python -m colpage_recovery.cli diagnose --inject range-overflow
PYTHONPATH=src python -m colpage_recovery.cli diagnose --inject snapshot-mismatch
```

完整输出：`artifacts/final-injection-codes.log`。

实际业务码和定位：

| 注入 | 退出码 | 业务码 | 首个定位 |
| --- | ---: | --- | --- |
| 提交标记冲突 | 2 | `E_COMMIT` | LSN 2，字段 `commit_flag`，事务 `T1` |
| 页行区间越界 | 2 | `E_RANGE` | LSN 1，字段 `row_range`，页 `P-A` |
| 账本内容摘要不符 | 2 | `E_SNAPSHOT` | LSN 1，页 `P-A`，字段 `content_digest` |

恢复重放自身与不中断参考不一致时由测试 `test_replay_mismatch_locates_first_page_and_field` 验证返回 `E_REPLAY`，并定位到 LSN 1、页 `P-A`、字段 `content_digest`。

## 5. Windows 验收建议顺序

```powershell
python -m pip install numpy==1.26.4
python -c "import sqlite3, tkinter, numpy; print('deps ok', numpy.__version__)"
python .\scripts\run_tests.py
python .\scripts\demo.py
python -m colpage_recovery.cli diagnose --inject commit-conflict
python -m colpage_recovery.cli diagnose --inject range-overflow
python -m colpage_recovery.cli diagnose --inject snapshot-mismatch
$env:PYTHONPATH = "$PWD\src"
python -m colpage_recovery.cli gui --state-dir state\demo --report artifacts\demo_report.json
```
