# 实际验证记录

验证时间：2026-11-26 05:14（Asia/Shanghai）。

环境：

- WSL2：`Linux ... 6.18.33.2-microsoft-standard-WSL2 x86_64 GNU/Linux`
- Python：`Python 3.11.0rc1`
- 虚拟环境：`.venv`（由 `virtualenv` 创建，因为系统缺少 `ensurepip/python3.11-venv`）
- NumPy：`2.4.6`（包约束为 `numpy>=1.26`；系统基础解释器中另有 NumPy `1.26.4`）
- SQLite：标准库 `sqlite3`，底层引擎 `3.37.2`
- Tk：dpkg 中 `python3-tk` 为未安装状态；无免密 sudo，因此用 `apt-get download` 将 `libtk8.6`、`libtcl8.6`、`tk8.6`、`tcl8.6`、`libxft2`、`libxss1`、`libfontconfig1`、`libfreetype6` 解包到 `artifacts/tk-root/`。设置 `LD_LIBRARY_PATH/TCL_LIBRARY/TK_LIBRARY` 并使用基础 Python 3.11 站点中的 `_tkinter` 后，`tkinter.TkVersion=8.6`，真实 Tk 根窗口创建/销毁冒烟通过；`scripts/tk-view.sh ... --smoke` 退出码为 0。

## 命令与结果

### 单元与集成测试

```text
$ pytest -q
.............                                                            [100%]
13 passed（最近一次完整 `./scripts/demo.sh` 中为 13 passed in 1.79s）
```

覆盖：

- `E_SPEC` 首个端口非法位置、依赖环、修订父链环；
- 到期/配额竞争的唯一归属与任意 batch size 的逐字节重放；
- 乐观租约版本不符、续约上限、真实 monotonic TTL 过期；
- 失效沿依赖图和节点引用传播；
- 全局预算、节点并发配额、节点租约上限的阻塞式硬约束；
- 取消根分片后派生取消且不调度；
- 真实子进程/TCP 全流程和独立账本校验；
- 静态 `samples/declaration.json` 可加载。

日志：`artifacts/validation/pytest.log`。

### 一键验收

```text
$ cloudshard acceptance --workdir artifacts/acceptance
acceptance_exit=0
ok True
PASS espec_first_invalid_location
PASS arbitrary_batch_boundary_byte_replay
PASS real_tcp_dispatch_and_everify
PASS eport_real_bind_conflict
PASS etimeout_real_socket_timeout
PASS efield_missing_receipt_field
PASS elease_optimistic_node_check
PASS sigkill_sqlite_recovery_and_everify
PASS everify_detects_tampering
```

完整机器可读报告：`artifacts/acceptance/acceptance-report.json`，副本：`docs/acceptance-report.generated.json`。

其中 `SIGKILL` 演练真实启动 `python -m cloudshard.cli run --crash-before-commit s2`，等待 `crash-marker.json` 后对新进程组发送 `SIGKILL`。本次实际恢复出两条已提交分派但无结局事务的记录：`s2/a1` 与 `s3/a1`；二者均标记 `UNKNOWN_RERUN`、先回收占用，随后重跑成功，独立校验 `outcomes_verified=13`、`ok=true`。未提交的 outcome 行由 SQLite 回滚；系统没有生成也没有声称生成强杀瞬间退出报告。

### 可重复脚本

```text
$ ./scripts/demo.sh
demo_script_exit=0
...
13 passed in 1.89s
Demo and acceptance artifacts are under artifacts/demo and artifacts/acceptance.
```

日志：`artifacts/validation/demo-script.log`。

### 固定端口样本

```text
$ cloudshard run --spec samples/declaration.json --workdir artifacts/sample-final
$ cloudshard verify --spec samples/declaration.json --ledger artifacts/sample-final/ledger.sqlite
{
  "ok": true,
  "code": "E_VERIFY_OK",
  "declared_ownership_records": 6,
  "runtime_ownership_records": 26,
  "outcomes_verified": 13,
  "progress": { "completed_count": 8, "topology_cursor": "s6" },
  "recovery": { "marks": 0 }
}
```

首版完成 8 个分片；修订版 2 的稳定失效集合为 `s5,s2,s4,s6,s7`，仅重算这 5 个，因此总共校验 13 个 committed outcomes。`s0/s1/s3` 的首版 canonical outcome bytes 被独立校验确认未改变。

### 等待与配额

固定样本中全局预算为 7，三个根分片需求为 `3+2+3`。稳定优先级调度先预留 `s0(3)` 与 `s1(2)`，合计 5；`s2(3)` 会使总量达到 8，因此进入等待。实际运行记录到：

```text
s2 wait_started_ms -> wait_ended_ms, duration_ms≈34-37
constraint=global_budget_units
```

阻塞式 pytest 还分别制造全局预算、节点 `concurrency_quota=1`、节点 `lease_limit=1` 的持续在途状态，并断言活跃权益未越界。

### Tkinter

```text
tkinter 8.6 smoke=ok
./scripts/tk-view.sh artifacts/demo/ledger.sqlite artifacts/demo/snapshot.json --smoke
tk_smoke_exit=0
```

普通 WSL 使用前仍建议执行 `sudo apt install python3-tk tk8.6`；若 Python 来自其他 apt 源，应安装与之匹配的 `python3.11-tk`。
