# 实际验证记录

验证时间：2026-11-21（Asia/Shanghai）。

## 环境

| 项目 | 结果 |
|---|---|
| Python | 3.11.0rc1 |
| SQLite | 3.37.2 |
| NumPy | 1.26.4 |
| tkinter | 当前容器不可用（缺少 `_tkinter`）；GUI 完成编译/导入检查，未弹出窗口 |
| 网络/容器 | 未使用 Docker，未访问外部服务 |

## 命令与结果

```text
python -m unittest discover -s tests -v
Ran 16 tests
OK

python demo.py
status: PASS
```

完整演示的检查项全部为 `true`：

```text
fixture_digests_match
first_run_complete
parameter_isolation
snapshot_whole_replacement
snapshot_source_unchanged
historical_replay_exact
restored_estimate_byte_exact
sample_order_invariant
crash_recovery_exact
all_injected_codes_matched
```

## 确定性指纹

固定样本和参数下：

| 场景 | 估算结果指纹 |
|---|---|
| 初始参数 `rate=4, concurrency=2` | `299d7c7b6ef4f97ff33b999d51b3e0cf3f0f62430c825b609d6fb4a7d361cd27` |
| 参数变更 `rate=5, concurrency=3` | `b4128cba1a831647ce38fa3e6fc640b61260b4099cd3ae95bbb447bb0e862398` |
| 快照恢复后重放 | `b4128cba1a831647ce38fa3e6fc640b61260b4099cd3ae95bbb447bb0e862398` |

样本序列摘要：

| 场景 | 样本序列 SHA-256 |
|---|---|
| 初始参数 | `d8fcd81e732181bd4ea5c5db262d9458cea8a6b2c5137bdf48de7399e9eff1fb` |
| 参数变更 | `e42c65aa26a5ba277fca8b85f1bbf3ae9f3990ea9b72c14fe22b8fa54b88304f` |
| 快照恢复后 | `e42c65aa26a5ba277fca8b85f1bbf3ae9f3990ea9b72c14fe22b8fa54b88304f` |

初始与变更参数指纹不同，证明参数版本没有混窗；恢复后与变更场景逐字节一致，证明快照恢复和历史重放确定性。

## 强制终止

- 子进程在提交 2 个完整批次后，于第 3 个批次写入 `RUNNING` 标记后执行 `os._exit(137)`。
- 重启前数据库只有前 2 批共 4 条样本。
- 重启 recovery 将第 3 批标为 `UNKNOWN`，进度为 `REQUIRES_RERUN`。
- `exit_report_available=false`。
- 续跑完成后样本摘要 `d8fcd81e...` 和估算指纹 `299d7c7b...` 与未中断的正常运行完全一致。

## 异常注入

| 注入 | 观察到的业务码 | 状态保护 |
|---|---|---|
| 删除中间 commit_seq，造成日志不连续 | `E_LOG` (`error_seq=6`) | 数据库状态不变 |
| `rate_rps=0` | `E_PARAM` | 数据库状态不变 |
| 复制快照后修改 `estimator.version` | `E_SNAPSHOT` | 恢复事务未开始，数据库状态不变 |
| 复制快照后修改首个样本 `delay_ns` | `E_REPLAY` | 首个差异定位为 `commit_seq=1`, `sample.delay_ns` |

## 产物

- SQLite：`artifacts/workbench/workbench.sqlite3`
- 结构化日志：`artifacts/workbench/logs/workbench.jsonl`
- 快照：`artifacts/workbench/snapshots/*.json`，权限为只读
- 报告：`01_initial_estimate.json`、`02_recomputed_estimate.json`、`03_snapshot_restore_replay.json`、`04_crash_recovery.json`、四类注入报告与 `demo_summary.json`

## Windows GUI 说明

本仓库在 Linux 容器内验证核心与无头演示。Windows 上需使用官方 Python 安装器并勾选 tcl/tk 组件，然后执行：

```powershell
py -3 -c "import numpy, sqlite3, tkinter"
py -3 -m workbench.ui
```

GUI 与核心服务使用同一套 API；界面仅负责加载、触发和稳定顺序渲染，不复制业务计算逻辑。
