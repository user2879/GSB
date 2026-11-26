# 实际验证记录

验证时间：2026-11-27（Asia/Shanghai）。

## 自动化测试

命令：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

结果：

```text
Ran 15 tests in 4.277s

OK
```

覆盖：

- 合法/非法 spec、`E_SPEC` 行号；
- `E_SCRIPT` 缺口与 `E_DUPACK` 聚集；
- o1/o2/o3 归一和四类 `E_FORMAT` 定位；
- 定点状态确定性、两次快速重传、一次超时；
- 开放窗口编辑仅影响 `item-001012`/`item-001013`；
- 新事件追加；
- 全局/连接调度上限与真实线程竞争；
- 真实回环子进程 14 事件交换；
- `E_PORT` 与真实 socket `E_TIMEOUT`；
- SIGKILL 后恢复、重放和只读快照备份。

## 一键复现

命令：

```bash
./scripts/reproduce.sh
```

两个独立输出目录 `artifacts/verify-a`、`artifacts/verify-b` 中以下文件逐字节一致：

```text
summary.json
conclusions.json
ledger_core.jsonl
ledger.sqlite
normalized_receipts.json
edit_recompute.json
scheduler_demo.json
snapshot_replacement.json
recovery/recovery_report.json
recovery/recovered_ledger_core.jsonl
```

实际 SHA-256：

```text
summary.json (file with trailing newline)
70ce7634acd6da13cc9590909ee4bdb35762dfca6f45b0c479053aedd564408b

ledger_core.jsonl
ba2558ae4b2b9ff8e878907a9d98469a23cefd5d3feffbd0b4c15404292574e9
```

摘要对象内部的 `content_digest` 为：

```text
e4759e24744a5a7dcc76e32023dcec5c24b5d776bf214a9ec2c803e74565ec44
```

## 崩溃恢复

`artifacts/demo/validation_report.json` 记录：

```json
{
  "forced_kill_returncode": -9,
  "last_committed_batch_before_recovery": 2,
  "unknown_batches_marked_and_rerun": [3],
  "replay_comparison": {
    "matching_item_count": 14,
    "matching_batch_count": 7,
    "first_difference": null
  },
  "core_ledger_byte_match": true,
  "repeat_ledger_byte_match": true
}
```

## Tkinter

实际检查结果：

```text
ModuleNotFoundError: No module named 'tkinter'
```

系统 Python shim 的底层错误为：

```text
ImportError: libtk8.6.so: cannot open shared object file: No such file or directory
```

因此本次未声称完成 GUI 窗口交互验证；CLI、TCP、SQLite 和测试均已实际运行通过。
