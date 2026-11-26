# 实际验证记录

验证时间：2026-11-27 07:05–07:06（Asia/Shanghai）。

## 环境

- WSL2 / `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64`
- Python `3.11.0rc1`（满足 >=3.10）
- NumPy `1.26.4`，`python3 -m pip install numpy` 显示依赖已满足
- SQLite `3.37.2`（标准库 `sqlite3`）
- Tk/Tcl `8.6`，`import tkinter` 成功；额外执行 Tk root 创建/80ms 后销毁成功
- `DISPLAY=:0`，可运行 `python3 -m shard_diag.cli gui`
- 回环高位端口可绑定；示例固定端口为 `39671/TCP`、`39672/TCP`、`39673/TCP`

原始环境输出保存在 `artifacts/environment.txt`。

## 自动化测试

命令：

```bash
python3 -m pytest -v tests/test_diagnostics.py
```

结果：`11 passed in 7.45s`。完整输出见 `artifacts/pytest-output.txt`。

覆盖：

- `E_SPEC` 首个非法条目/字段与不支持缩放因子。
- `E_PORT` 端口占用和 `E_TIMEOUT` 真实套接字超时。
- 不同批次切分、逆序下发、6 并发重复计算的逐位摘要一致。
- 水位误差标注与稳定拓扑/ID 决胜。
- 增量失效传播、未失效字节保持、空白全量重建字段一致。
- SIGKILL 节点、未提交租约回收、未知标记和恢复后六项全提交。
- 到期回收与配额版本换代回收互斥。
- 配额版本变更传播、未知上游冻结、三节点 500ms 延迟分片并行。

## 一键验收

命令：

```bash
python3 -m shard_diag.cli acceptance --output artifacts/acceptance-result.json
```

结果：`ok=true`，`16/16` 项通过。结构化证据见 `artifacts/acceptance-result.json`。

通过的检查项：

1. `e_port`
2. `e_timeout_real_socket`
3. `quota_water_tie_adjudication`
4. `sigkill_interrupted_dispatch`
5. `committed_only_recovery_after_reopen`
6. `all_six_runnable_committed`
7. `missing_dependency_frozen`
8. `replay_after_recovery`
9. `baseline_full_run`
10. `invalidation_propagation`
11. `only_invalidated_rerun`
12. `unaffected_bytes_preserved`
13. `snapshot_digest_full_replace`
14. `blank_full_rebuild_field_match`
15. `e_replay_locates_first_field`
16. `expiry_generation_mutex`

其中强杀链路真实执行：授予 `shard-a` 租约后下发带 800ms 处理延迟的任务，250ms 后对 `node-a` 子进程发送 `SIGKILL`；客户端下发失败。随后关闭并重新打开 SQLite 账本，恢复逻辑恰好把 `shard-a` 标为 `unknown` 并将活动资源占用清零。重启节点后，六个可运行分片全部提交，`shard-missing` 因 `ghost-upstream` 保持冻结。

失效传播验收中，替换 `shard-a` 的第 0 行样本后，失效集合稳定为：

```text
shard-a -> shard-c -> shard-d -> shard-f
```

未失效的 `shard-b`、`shard-e` 的损失摘要、租约 blob 与水位 blob 哈希前后完全相同；增量账本与第二个空白全量重建账本的确定性字段比较为 0 个不一致。

## 正式 CLI 账本

命令：

```bash
python3 -m shard_diag.cli validate
python3 -m shard_diag.cli regenerate-samples
python3 -m shard_diag.cli run --db ledger/diagnostic.sqlite --batch-size 3
python3 -m shard_diag.cli replay --db ledger/diagnostic.sqlite
```

实际结果：

- 声明校验：3 节点、7 分片、3 配额版本、2 初始事件，`ok=true`。
- 调度提交：`shard-a` 至 `shard-f` 共 6 项。
- 冻结：`shard-missing.status=frozen_missing_dependency`，`missing_upstream=["ghost-upstream"]`。
- 重放：`ok=true`，6 个分片结局全部重放一致。

对应输出：

- `artifacts/cli-run.json`
- `artifacts/replay-result.json`
- `ledger/diagnostic.sqlite`
- `ledger/acceptance.sqlite`
- `ledger/session_snapshot.json`
- `ledger/logs/*.log`

## 复核建议

```bash
python3 -m pytest -q
python3 -m shard_diag.cli acceptance
python3 -m shard_diag.cli gui
```

若重复运行验收，命令会重建 `ledger/acceptance.sqlite` 与 `artifacts/acceptance-workspace/` 下的动态端口工作区；不会依赖固定外部端口或远程服务。
