# 实际验证结果记录

验证环境：Python 3.11、numpy 1.26、pytest 8、本地回环 TCP、SQLite（WAL）。
所有节点均为真实子进程，无 mock 网络、无外部服务、无容器。

## 1. 自动化测试

```
python -m pytest tests/
41 passed
```

覆盖映射：

| 测试文件 | 覆盖业务要求 |
| --- | --- |
| `tests/test_fixedpoint.py` | 要求 3：网格合法性、向零截断、饱和 MAC、单值/传递误差 |
| `tests/test_content.py` | 固定种子确定性、FNV 摘要、分片连续性 |
| `tests/test_manifest.py` | 要求 1：版本单调、路径/分片/目标节点/策略/派生引用/成环（E_MANIFEST/E_CYCLE） |
| `tests/test_compiler.py` | 要求 4：字节稳定产物、增量编译、拓扑序失效传播与每步依据 |
| `tests/test_e2e.py` | 要求 2/5/6：真实子进程、并发窗口不超限、超时重试、E_PORT/E_REPLY/E_CONVERGE、numpy 一致性 |
| `tests/test_recovery.py` | 要求 7/8：快照整体替换、未提交批次 UNKNOWN、硬杀重跑、E_REPLAY 定位、E_SNAPSHOT、状态机 |

## 2. 端到端 CLI 验收（实测输出）

两次独立运行（独立工作目录、独立 SQLite）的计划摘要完全一致：

```
plan digests identical across runs: True
v1: items=6 all_ok=True outcomes={'success'}
  batches: [('v1-b0', 3, 3), ('v1-b1', 3, 3)]
  error totals: {'single': 0, 'propagated': 0, 'bound': 1}
v2: items=7 all_ok=True outcomes={'success'}
  batches: [('v2-b0', 3, 3), ('v2-b1', 3, 3), ('v2-b2', 1, 1)]
  error totals: {'single': 0, 'propagated': 0, 'bound': 1}
replay matched: 13 / 13 unknown: 0
```

结论：

* 13 个计划项（两个版本）全部并行下发成功并逐字节收敛；
* 派生气泡路径 `media/index.json` 的期望摘要与各节点回执一致；
* 重开账本重放 13/13 复现，无未知批次；
* 误差上界同时给出 `single/propagated/bound`。

## 3. 故障注入实测

```
bad_grid         -> E_GRID    session.scale
corrupt_manifest -> E_MANIFEST inject.corrupt_manifest
force_cycle      -> E_CYCLE   inject.force_cycle
```

运行期故障（测试内断言）：

* 端口被占用：节点子进程退出码 3，平台抛 `E_PORT`；
* 节点 `bad_reply`：结构化日志记录 `E_REPLY`，该节点任务最终在收敛阶段以
  `E_CONVERGE` 偏差列出，差异节点稳定为故障节点；
* 节点 `quant_mismatch`：偏差字段含 `quant_version`；
* 节点 `exit`：`node_exited` 事件记录返回码，后续任务标记
  `task_skipped_dead_node` 并从调度排除；
* 节点 `timeout`：按重试策略产生 `attempt=1/2` 区分回执，错误码
  `E_HANDSHAKE`，结局可判为 timeout/failed。

## 4. 并发上限硬验证

`test_running_windows_never_breach_global_or_node_caps` 从
`task_running/task_finished` 事件重建每个任务的单调时钟运行窗口，统计每个
启动时刻的全局与单节点重叠数，断言全局峰值 `<= global_limit`、单节点峰值
`<= per_node_quota`，证明上限不可绕过。

## 5. 强制终止恢复实测

`test_hard_kill_then_rerun_and_replay_is_consistent`：

1. 以 `crash_after_batches=1` 在首个批次事务提交后模拟 SIGKILL；
2. 重开账本：已提交批次可信、无伪造报告，未完成工作要求重跑；
3. 同一命令重跑后所有项收敛；
4. 权威重放 matched == replayed，无未知批次。

另有 `test_hard_kill_leaves_unknown_batch_requiring_rerun` 直接构造未提交
批次行，验证其被标记为 `RERUN_REQUIRED` 而非误报成功。
