# 实际验证记录

验证日期：2026-11-23（Asia/Shanghai）

## 环境

- Python：3.11.0rc1
- SQLite：3.37.2
- 平台：当前 Linux 验证容器
- `sqlite3`、`threading`、`json`：可用
- Tkinter：导入失败，缺少 `libtk8.6.so`；因此无法在本容器证明窗口可视化启动

## 执行的命令

```text
python3 -m compileall -q txdiag tests scripts
python3 -m unittest discover -v
python3 scripts/validate_acceptance.py
```

## 结果

- 编译：通过
- unittest：24 项全部通过
- 验收矩阵：30/30 通过，记录于 `artifacts/validation.json`
- GUI 无头模型：跨重复执行完全一致
- GUI 图形窗口：本容器缺少系统 Tk 动态库，未执行；模块编译与模型验证通过

### 逐字节路径一致性

以下样本均使用 `--path both`，两条路径的 trace 文件逐字节相同：

| 样本 | trace 字节数 | 规范化 SHA-256 |
|---|---:|---|
| `all_isolation` | 20287 | `ad3149f90467947c643ffd4c1945f558fbab324cd0d88e5d25c9426b27b8171c` |
| `deadlock` | 11949 | `df54ce77bfd9e33f472dfca571aa591fa404bf71bd51781e77be688a04d71587` |
| `wait` | 11950 | `67694286ee7037648ca9557d18ea6df624e8173384a6bf2c5864d61dd3fc279c` |
| `large_range` | 35716 | `a5312e46cc444f7c06a2b6f98b24d8c24342ae0842870910f389d71e20a76daf` |

保留的代表性文件：

- `artifacts/acceptance/deadlock.row.trace.json`
- `artifacts/acceptance/deadlock.range.trace.json`
- `artifacts/acceptance/large_range.row.trace.json`
- `artifacts/acceptance/large_range.range.trace.json`

### 隔离级别

`all_isolation` 样本验证结果：

- RC1 两次读取 `acct:A`：`[0, 10]`，后一次看到 W1 已提交版本；
- RR1 两次读取 `acct:A`：`[0, 0]`，第二次标记 `repeatable_snapshot`；
- RU1 在 W1 未提交时读取到版本 1、值 10、状态 `uncommitted`；
- SER1 的读事件记录 `lock_mode=S+X`，并读到本事务未提交写入。

### 等待与死锁

- `wait` 样本产生真实 `lock_conflict` 与 `wait_start/wait_end(granted)`；观测真实等待耗时大于 0 ms。
- `timeout` 样本在 `hold-lock=SLOW:1` 注入下以真实 monotonic 截止时间触发 1 次 `E_TIMEOUT`，事务 TIMEOUT 回滚。
- `deadlock` 样本稳定产出环 `["TX1","TX2"]`，牺牲 `TX2`，依据 `max(start_seq,txid)`；重复执行结果相同。

### 崩溃恢复与重放

- `crash-in-round=2` 进程以退出码 70 被强制终止；
- 账本保留 1 个完整提交轮次，`inflight` 保留第 2 轮运行集合；
- 恢复报告 unknown 事务为 `["HOLDER","WAITER"]`，`exit_report_available=false`；
- 恢复续跑后账本标记 complete；
- 重放 5/5 轮规范化内容一致，状态 `REPLAY_OK`；
- 人工篡改预期轮次时返回 `E_REPLAY`，定位 `HOLDER / lock:A / at_ms`。

### 双路径差异注入

`--path both --inject diverge` 返回退出码 5，`E_DIVERGE` 首次定位为：

```text
txid=HOLDER row=lock:A field=event
```

## 未在当前容器完成的验证

Windows 图形窗口启动需要在 Windows 上执行 README 中的 Tk 自检命令，再运行：

```bat
py -3.11 -m txdiag gui --spec samples\deadlock.json ^
  --ledger ledger\gui.sqlite --snapshot ledger\gui.snapshot.json
```

该限制不影响 CLI、线程锁引擎、SQLite 账本和 GUI 视图模型测试。
